"""Append-only prediction audit logging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

POSTGRES_CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id UUID PRIMARY KEY,
    ts timestamptz DEFAULT now(),
    input JSONB NOT NULL,
    model_version TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    risk_probability DOUBLE PRECISION NOT NULL,
    risk_class TEXT,
    conformal_set TEXT[] NOT NULL,
    abstain BOOLEAN NOT NULL,
    abstain_reason TEXT,
    latency_ms DOUBLE PRECISION NOT NULL,
    ground_truth INTEGER
)
"""

SQLITE_CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY,
    ts TEXT DEFAULT (datetime('now')),
    input TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    risk_probability REAL NOT NULL,
    risk_class TEXT,
    conformal_set TEXT NOT NULL,
    abstain INTEGER NOT NULL,
    abstain_reason TEXT,
    latency_ms REAL NOT NULL,
    ground_truth INTEGER
)
"""


@dataclass(frozen=True)
class PredictionAuditRecord:
    """Prediction audit payload persisted after every API response."""

    request_id: UUID
    input_payload: dict[str, Any]
    model_version: str
    model_alias: str
    data_hash: str
    risk_probability: float
    risk_class: int | None
    conformal_set: list[int]
    abstain: bool
    abstain_reason: str | None
    latency_ms: float
    ground_truth: int | None = None


class PredictionAuditLogger:
    """Small SQLAlchemy wrapper for append-only prediction records."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        connect_args: dict[str, object] = {}
        if database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        self.engine: Engine = create_engine(database_url, connect_args=connect_args)

    @property
    def is_sqlite(self) -> bool:
        return self.engine.dialect.name == "sqlite"

    def initialize(self) -> None:
        """Create the audit table if it does not exist."""

        statement = (
            SQLITE_CREATE_PREDICTIONS_TABLE
            if self.is_sqlite
            else POSTGRES_CREATE_PREDICTIONS_TABLE
        )
        with self.engine.begin() as connection:
            connection.execute(text(statement))

    def healthcheck(self) -> bool:
        """Return whether the audit database is reachable."""

        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception:
            return False
        return True

    def log_prediction(self, record: PredictionAuditRecord) -> None:
        """Insert one prediction audit row."""

        if self.is_sqlite:
            params = {
                "request_id": str(record.request_id),
                "input": json.dumps(record.input_payload, sort_keys=True),
                "model_version": record.model_version,
                "model_alias": record.model_alias,
                "data_hash": record.data_hash,
                "risk_probability": record.risk_probability,
                "risk_class": (
                    None if record.risk_class is None else str(record.risk_class)
                ),
                "conformal_set": json.dumps(
                    [str(value) for value in record.conformal_set]
                ),
                "abstain": int(record.abstain),
                "abstain_reason": record.abstain_reason,
                "latency_ms": record.latency_ms,
                "ground_truth": record.ground_truth,
            }
        else:
            params = {
                "request_id": record.request_id,
                "input": record.input_payload,
                "model_version": record.model_version,
                "model_alias": record.model_alias,
                "data_hash": record.data_hash,
                "risk_probability": record.risk_probability,
                "risk_class": (
                    None if record.risk_class is None else str(record.risk_class)
                ),
                "conformal_set": [str(value) for value in record.conformal_set],
                "abstain": record.abstain,
                "abstain_reason": record.abstain_reason,
                "latency_ms": record.latency_ms,
                "ground_truth": record.ground_truth,
            }

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO predictions (
                        request_id,
                        input,
                        model_version,
                        model_alias,
                        data_hash,
                        risk_probability,
                        risk_class,
                        conformal_set,
                        abstain,
                        abstain_reason,
                        latency_ms,
                        ground_truth
                    )
                    VALUES (
                        :request_id,
                        :input,
                        :model_version,
                        :model_alias,
                        :data_hash,
                        :risk_probability,
                        :risk_class,
                        :conformal_set,
                        :abstain,
                        :abstain_reason,
                        :latency_ms,
                        :ground_truth
                    )
                    """
                ),
                params,
            )

    def fetch_prediction(self, request_id: UUID) -> dict[str, Any] | None:
        """Fetch one row for integration tests and local inspection."""

        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    text("SELECT * FROM predictions WHERE request_id = :request_id"),
                    {"request_id": str(request_id)},
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return dict(row)
