"""Transparent custom drift metrics and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, wasserstein_distance
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

EPSILON = 1e-6

POSTGRES_CREATE_DRIFT_SCORES_TABLE = """
CREATE TABLE IF NOT EXISTS drift_scores (
    id UUID PRIMARY KEY,
    ts timestamptz DEFAULT now(),
    run_id TEXT NOT NULL,
    regime TEXT NOT NULL,
    feature TEXT NOT NULL,
    metric TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL,
    metadata JSONB NOT NULL
)
"""

SQLITE_CREATE_DRIFT_SCORES_TABLE = """
CREATE TABLE IF NOT EXISTS drift_scores (
    id TEXT PRIMARY KEY,
    ts TEXT DEFAULT (datetime('now')),
    run_id TEXT NOT NULL,
    regime TEXT NOT NULL,
    feature TEXT NOT NULL,
    metric TEXT NOT NULL,
    score REAL NOT NULL,
    status TEXT NOT NULL,
    metadata TEXT NOT NULL
)
"""


def _finite(values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def population_stability_index(
    expected: ArrayLike,
    actual: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Compute PSI using expected-distribution quantile bins."""

    expected_values = _finite(expected)
    actual_values = _finite(actual)
    if expected_values.size == 0 or actual_values.size == 0:
        msg = "PSI requires non-empty finite expected and actual arrays"
        raise ValueError(msg)
    if n_bins < 2:
        msg = "n_bins must be at least 2"
        raise ValueError(msg)

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(expected_values, quantiles))
    if edges.size < 2:
        return 0.0 if np.allclose(expected_values[0], actual_values) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected_counts = np.histogram(expected_values, bins=edges)[0].astype(np.float64)
    actual_counts = np.histogram(actual_values, bins=edges)[0].astype(np.float64)
    expected_share = np.clip(expected_counts / expected_counts.sum(), EPSILON, None)
    actual_share = np.clip(actual_counts / actual_counts.sum(), EPSILON, None)
    return float(
        np.sum((actual_share - expected_share) * np.log(actual_share / expected_share))
    )


def ks_drift(reference: ArrayLike, current: ArrayLike) -> dict[str, float | bool]:
    """Return the two-sample KS statistic, p-value, and 5% decision."""

    statistic, p_value = cast(
        tuple[float, float],
        ks_2samp(_finite(reference), _finite(current)),
    )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < 0.05),
    }


def wasserstein_drift(reference: ArrayLike, current: ArrayLike) -> dict[str, float]:
    """Return raw and reference-standard-deviation-scaled Wasserstein distance."""

    reference_values = _finite(reference)
    current_values = _finite(current)
    distance = float(wasserstein_distance(reference_values, current_values))
    scale = float(np.std(reference_values))
    return {
        "distance": distance,
        "normalized_distance": distance / scale if scale > 0.0 else distance,
    }


def maximum_mean_discrepancy(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    gamma: float | None = None,
    max_samples: int = 2_000,
) -> float:
    """Compute squared Gaussian-kernel MMD with a median bandwidth heuristic."""

    x = _finite(reference)[:max_samples].reshape(-1, 1)
    y = _finite(current)[:max_samples].reshape(-1, 1)
    if len(x) == 0 or len(y) == 0:
        msg = "MMD requires non-empty finite reference and current arrays"
        raise ValueError(msg)
    combined = np.vstack([x, y])
    pairwise_squared = cdist(combined, combined, metric="sqeuclidean")
    positive = pairwise_squared[pairwise_squared > 0.0]
    if gamma is None:
        median = float(np.median(positive)) if positive.size else 1.0
        gamma = 1.0 / max(2.0 * median, EPSILON)
    kernel_xx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    kernel_yy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    kernel_xy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    return float(max(0.0, kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()))


def psi_status(score: float, warn: float = 0.10, alert: float = 0.25) -> str:
    """Map PSI to standard operational thresholds."""

    if score > alert:
        return "ALERT"
    if score >= warn:
        return "WARN"
    return "OK"


def numeric_drift_scores(
    reference: ArrayLike,
    current: ArrayLike,
    *,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
) -> dict[str, Any]:
    """Compute the transparent drift score bundle for one numeric feature."""

    psi = population_stability_index(reference, current)
    return {
        "psi": psi,
        "psi_status": psi_status(psi, psi_warn, psi_alert),
        "ks": ks_drift(reference, current),
        "wasserstein": wasserstein_drift(reference, current),
        "mmd": maximum_mean_discrepancy(reference, current),
    }


@dataclass(frozen=True)
class DriftScoreRecord:
    """One persisted custom drift score."""

    run_id: str
    regime: str
    feature: str
    metric: str
    score: float
    status: str
    metadata: dict[str, Any]
    id: UUID = field(default_factory=uuid4)
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


class DriftScoreStore:
    """Persist custom drift scores to PostgreSQL or a SQLite test fallback."""

    def __init__(self, database_url: str) -> None:
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        self.engine: Engine = create_engine(database_url, connect_args=connect_args)

    @property
    def is_sqlite(self) -> bool:
        return self.engine.dialect.name == "sqlite"

    def initialize(self) -> None:
        statement = (
            SQLITE_CREATE_DRIFT_SCORES_TABLE
            if self.is_sqlite
            else POSTGRES_CREATE_DRIFT_SCORES_TABLE
        )
        with self.engine.begin() as connection:
            connection.execute(text(statement))

    def write(self, record: DriftScoreRecord) -> None:
        metadata = json.dumps(record.metadata, sort_keys=True)
        record_id: object = str(record.id)
        timestamp: object = record.ts.isoformat()
        if not self.is_sqlite:
            record_id = record.id
            timestamp = record.ts
        metadata_expression = (
            ":metadata" if self.is_sqlite else "CAST(:metadata AS JSONB)"
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    f"""
                    INSERT INTO drift_scores (
                        id, ts, run_id, regime, feature, metric, score, status, metadata
                    ) VALUES (
                        :id, :ts, :run_id, :regime, :feature, :metric, :score,
                        :status, {metadata_expression}
                    )
                    """
                ),
                {
                    "id": record_id,
                    "ts": timestamp,
                    "run_id": record.run_id,
                    "regime": record.regime,
                    "feature": record.feature,
                    "metric": record.metric,
                    "score": record.score,
                    "status": record.status,
                    "metadata": metadata,
                },
            )

    def fetch_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    text("SELECT * FROM drift_scores WHERE run_id = :run_id"),
                    {"run_id": run_id},
                )
                .mappings()
                .all()
            )
        return [dict(row) for row in rows]


def persist_numeric_drift_scores(
    store: DriftScoreStore,
    *,
    run_id: str,
    regime: str,
    feature: str,
    scores: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist PSI, KS, Wasserstein, and MMD scores for one feature."""

    metadata = metadata or {}
    ks = scores["ks"]
    wasserstein = scores["wasserstein"]
    records = [
        ("psi", float(scores["psi"]), str(scores["psi_status"])),
        (
            "ks",
            float(ks["statistic"]),
            "ALERT" if bool(ks["drift_detected"]) else "OK",
        ),
        ("wasserstein", float(wasserstein["distance"]), "INFO"),
        ("mmd", float(scores["mmd"]), "INFO"),
    ]
    for metric, score, status in records:
        store.write(
            DriftScoreRecord(
                run_id=run_id,
                regime=regime,
                feature=feature,
                metric=metric,
                score=score,
                status=status,
                metadata=metadata,
            )
        )
