"""Synthetic three-regime drift simulation for a credible static-data demo."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from medmlops.monitoring.custom_drift import (
    DriftScoreStore,
    numeric_drift_scores,
    persist_numeric_drift_scores,
)
from medmlops.monitoring.drift import (
    build_drift_report,
    log_drift_snapshot_to_workspace,
    save_evidently_snapshot,
)

SYNTHETIC_NOTICE = (
    "SYNTHETIC DRIFT DEMONSTRATION: injected shifts are not observed real-world drift."
)


def inject_covariate_shift(
    df: pd.DataFrame,
    feature: str,
    magnitude: float,
) -> pd.DataFrame:
    """Shift a numeric P(X) distribution by magnitude standard deviations."""

    if feature not in df.columns:
        raise KeyError(feature)
    output = df.copy()
    values = cast("pd.Series", pd.to_numeric(output[feature], errors="coerce"))
    scale = float(values.std(ddof=0))
    output[feature] = values + magnitude * (scale if scale > 0.0 else 1.0)
    return output


def inject_prior_shift(
    df: pd.DataFrame,
    target_positive_rate: float,
    *,
    target: str = "readmitted_30d",
    seed: int = 42,
) -> pd.DataFrame:
    """Resample rows to synthetically alter P(Y)."""

    if not 0.0 < target_positive_rate < 1.0:
        msg = "target_positive_rate must be between 0 and 1"
        raise ValueError(msg)
    if target not in df.columns:
        raise KeyError(target)
    positives = df.loc[df[target].astype(int) == 1]
    negatives = df.loc[df[target].astype(int) == 0]
    if positives.empty or negatives.empty:
        msg = "Prior shift requires both target classes"
        raise ValueError(msg)
    n_rows = len(df)
    n_positive = round(n_rows * target_positive_rate)
    n_negative = n_rows - n_positive
    shifted = pd.concat(
        [
            positives.sample(n=n_positive, replace=True, random_state=seed),
            negatives.sample(n=n_negative, replace=True, random_state=seed + 1),
        ],
        ignore_index=True,
    )
    return shifted.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def inject_concept_shift(
    df: pd.DataFrame,
    strength: float,
    *,
    target: str = "readmitted_30d",
    feature: str = "num_medications",
) -> pd.DataFrame:
    """Change P(Y|X) while preserving the overall target prevalence."""

    if not 0.0 <= strength <= 1.0:
        msg = "strength must be between 0 and 1"
        raise ValueError(msg)
    if target not in df.columns or feature not in df.columns:
        raise KeyError(target if target not in df.columns else feature)
    output = df.copy()
    labels = output[target].astype(int).to_numpy()
    n_positive = int(labels.sum())
    feature_values = cast(
        "pd.Series", pd.to_numeric(output[feature], errors="coerce")
    ).fillna(0.0)
    ordered = np.argsort(feature_values.to_numpy())
    fully_shifted = np.zeros(len(output), dtype=np.int_)
    if n_positive:
        fully_shifted[ordered[-n_positive:]] = 1
    one_to_zero = np.flatnonzero((labels == 1) & (fully_shifted == 0))
    zero_to_one = np.flatnonzero((labels == 0) & (fully_shifted == 1))
    pairs = min(len(one_to_zero), len(zero_to_one))
    selected = round(strength * pairs)
    labels[one_to_zero[:selected]] = 0
    labels[zero_to_one[:selected]] = 1
    output[target] = labels
    return output


def conditional_shift_score(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature: str = "num_medications",
    target: str = "readmitted_30d",
    n_bins: int = 5,
) -> float:
    """Measure average P(Y|X-bin) difference using reference quantile bins."""

    reference_values = cast(
        "pd.Series", pd.to_numeric(reference[feature], errors="coerce")
    ).to_numpy(dtype=np.float64)
    current_values = cast(
        "pd.Series", pd.to_numeric(current[feature], errors="coerce")
    ).to_numpy(dtype=np.float64)
    reference_target = reference[target].to_numpy(dtype=np.float64)
    current_target = current[target].to_numpy(dtype=np.float64)
    finite_reference = reference_values[np.isfinite(reference_values)]
    edges = np.unique(np.quantile(finite_reference, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    differences: list[float] = []
    for lower, upper in pairwise(edges):
        reference_mask = (reference_values > lower) & (reference_values <= upper)
        current_mask = (current_values > lower) & (current_values <= upper)
        if not reference_mask.any() or not current_mask.any():
            continue
        differences.append(
            abs(
                float(reference_target[reference_mask].mean())
                - float(current_target[current_mask].mean())
            )
        )
    return float(np.mean(differences)) if differences else 0.0


def run_synthetic_drift_demo(
    reference: pd.DataFrame,
    *,
    feature: str = "num_medications",
    target: str = "readmitted_30d",
    database_url: str | None = None,
    report_path: str | Path = "reports/drift_simulation.json",
    html_dir: str | Path = "reports/drift",
    workspace_path: str | Path | None = None,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
) -> Path:
    """Run all synthetic regimes, persist scores, and save Evidently reports."""

    covariate = inject_covariate_shift(reference, feature, magnitude=2.0)
    prior = inject_prior_shift(reference, target_positive_rate=0.50, target=target)
    concept = inject_concept_shift(
        reference, strength=1.0, target=target, feature=feature
    )
    covariate_scores = numeric_drift_scores(
        reference[feature],
        covariate[feature],
        psi_warn=psi_warn,
        psi_alert=psi_alert,
    )
    report: dict[str, Any] = {
        "title": SYNTHETIC_NOTICE,
        "regimes": {
            "covariate_shift": {
                "taxonomy": "P(X)",
                "feature": feature,
                "psi": covariate_scores["psi"],
                "status": covariate_scores["psi_status"],
                "detected": bool(float(covariate_scores["psi"]) > psi_warn),
            },
            "prior_probability_shift": {
                "taxonomy": "P(Y)",
                "reference_positive_rate": float(reference[target].mean()),
                "current_positive_rate": float(prior[target].mean()),
                "detected": bool(
                    abs(float(reference[target].mean()) - float(prior[target].mean()))
                    >= 0.10
                ),
            },
            "concept_shift": {
                "taxonomy": "P(Y|X)",
                "conditional_shift_score": conditional_shift_score(
                    reference,
                    concept,
                    feature=feature,
                    target=target,
                ),
                "detected": bool(
                    conditional_shift_score(
                        reference,
                        concept,
                        feature=feature,
                        target=target,
                    )
                    >= 0.10
                ),
            },
        },
    }

    html_destination = Path(html_dir)
    for name, current in {
        "covariate_shift": covariate,
        "prior_probability_shift": prior,
        "concept_shift": concept,
    }.items():
        run_name = f"SYNTHETIC DRIFT DEMONSTRATION: {name}"
        snapshot = build_drift_report(
            reference,
            current,
            name=run_name,
            tags=["synthetic-drift-demo", name],
            metadata={"notice": SYNTHETIC_NOTICE, "regime": name},
        )
        save_evidently_snapshot(
            snapshot,
            html_path=html_destination / f"synthetic_{name}.html",
        )
        if workspace_path is not None:
            log_drift_snapshot_to_workspace(
                snapshot,
                workspace_path=workspace_path,
                run_name=run_name,
            )

    if database_url is not None:
        store = DriftScoreStore(database_url)
        store.initialize()
        persist_numeric_drift_scores(
            store,
            run_id="synthetic-drift-demo",
            regime="covariate_shift",
            feature=feature,
            scores=covariate_scores,
            metadata={"notice": SYNTHETIC_NOTICE},
        )

    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
