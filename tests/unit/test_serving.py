from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from medmlops.conformal.conformal import ConformalGate
from medmlops.features.pipeline import feature_spec_from_params, load_yaml
from medmlops.serving.app import create_app
from medmlops.serving.audit import PredictionAuditLogger
from medmlops.serving.predict import (
    PredictionService,
    QuantileOODChecker,
    RegistryModelHandle,
    RegistryModelMetadata,
    ServingMetrics,
    ServingSettings,
)

_DEFAULT_REGISTRY_MODEL = object()


class FakeEstimator:
    def __init__(self, probability: float = 0.22) -> None:
        self.probability = probability
        self.last_columns: list[str] = []

    def predict_positive_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self.last_columns = list(frame.columns)
        return np.array([self.probability], dtype=np.float64)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        positive = self.predict_positive_proba(frame)
        return np.column_stack([1.0 - positive, positive])


class FakeGate:
    def __init__(
        self,
        estimator: FakeEstimator | None = None,
        *,
        prediction_set: list[int] | None = None,
        abstain: bool = False,
        reason: str = "single_class_set",
    ) -> None:
        self.estimator = estimator or FakeEstimator()
        self.prediction_set = prediction_set or [0]
        self.abstain = abstain
        self.reason = reason

    def predict_with_abstention(self, frame: pd.DataFrame) -> dict[str, Any]:
        del frame
        return {
            "prediction": self.prediction_set[0]
            if len(self.prediction_set) == 1
            else None,
            "set": self.prediction_set,
            "abstain": self.abstain,
            "reason": self.reason,
        }


class FakeChallenger:
    def __init__(self) -> None:
        self.calls = 0
        self.last_columns: list[str] = []

    def predict(self, frame: pd.DataFrame) -> list[int]:
        self.calls += 1
        self.last_columns = list(frame.columns)
        return [0]


def _payload(sample_diabetes130: pd.DataFrame) -> dict[str, Any]:
    row = sample_diabetes130.iloc[0].to_dict()
    row.pop("readmitted")
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, np.integer):
            payload[key] = int(value)
        else:
            payload[key] = value
    return payload


def _registry_handle(
    *,
    alias: str = "champion",
    model: Any | None = _DEFAULT_REGISTRY_MODEL,
) -> RegistryModelHandle:
    registry_model = object() if model is _DEFAULT_REGISTRY_MODEL else model
    return RegistryModelHandle(
        model=registry_model,
        metadata=RegistryModelMetadata(
            model_name="MedMLOps",
            alias=alias,
            version="7",
            run_id="run-123",
            source="s3://mlflow/artifacts/model",
            model_uri=f"models:/MedMLOps@{alias}",
        ),
    )


def _service(
    tmp_path: Path,
    *,
    gate: FakeGate | None = None,
    ood_checker: QuantileOODChecker | None = None,
    challenger: FakeChallenger | None = None,
    challenger_fraction: float = 0.0,
) -> PredictionService:
    tmp_path.mkdir(parents=True, exist_ok=True)
    params = load_yaml("params.yaml")
    spec = feature_spec_from_params(params)
    audit_logger = PredictionAuditLogger(f"sqlite:///{tmp_path / 'audit.db'}")
    audit_logger.initialize()
    settings = ServingSettings(
        challenger_fraction=challenger_fraction,
        audit_database_url=f"sqlite:///{tmp_path / 'audit.db'}",
    )
    default_ranges = {column: (-1.0, 1_000.0) for column in spec.numeric}
    challenger_handle = (
        _registry_handle(alias="challenger", model=challenger)
        if challenger is not None
        else None
    )
    return PredictionService(
        settings=settings,
        spec=spec,
        gate=cast(ConformalGate, gate or FakeGate()),
        audit_logger=audit_logger,
        registry_model=_registry_handle(),
        challenger_model=challenger_handle,
        ood_checker=ood_checker or QuantileOODChecker(default_ranges),
        training_metrics={"hero": {"test": {"auroc": 0.7}}},
        data_hash="train-hash",
        metrics=ServingMetrics(),
    )


def test_predict_endpoint_logs_audit_and_excludes_demographics(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
) -> None:
    gate = FakeGate()
    service = _service(tmp_path, gate=gate)

    with TestClient(create_app(service=service)) as client:
        response = client.post("/predict", json=_payload(sample_diabetes130))
        assert response.status_code == 200
        body = response.json()
        row = service.audit_logger.fetch_prediction(UUID(body["request_id"]))

    assert body["risk_probability"] == 0.22
    assert body["risk_class"] == 0
    assert body["abstain"] is False
    assert row is not None
    assert json.loads(str(row["input"]))["race"] == "Caucasian"
    assert "race" not in gate.estimator.last_columns
    assert "gender" not in gate.estimator.last_columns
    assert "age" not in gate.estimator.last_columns


def test_predict_rejects_out_of_range_payload(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
) -> None:
    service = _service(tmp_path)
    payload = _payload(sample_diabetes130)
    payload["time_in_hospital"] = 15

    with TestClient(create_app(service=service)) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_abstains_for_ambiguous_and_ood_inputs(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
) -> None:
    ambiguous_service = _service(
        tmp_path / "ambiguous",
        gate=FakeGate(prediction_set=[0, 1], abstain=True, reason="ambiguous_set"),
    )
    with TestClient(create_app(service=ambiguous_service)) as client:
        ambiguous = client.post("/predict", json=_payload(sample_diabetes130))

    ood_payload = _payload(sample_diabetes130)
    ood_payload["num_medications"] = 100
    ood_service = _service(
        tmp_path / "ood",
        ood_checker=QuantileOODChecker({"num_medications": (1.0, 50.0)}),
    )
    with TestClient(create_app(service=ood_service)) as client:
        ood = client.post("/predict", json=ood_payload)

    assert ambiguous.status_code == 200
    assert ambiguous.json()["abstain"] is True
    assert ambiguous.json()["abstain_reason"] == "ambiguous_set"
    assert ood.status_code == 200
    assert ood.json()["abstain"] is True
    assert ood.json()["abstain_reason"].startswith("ood_numeric:num_medications")


def test_batch_health_metrics_and_model_info(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
) -> None:
    service = _service(tmp_path)
    payload = _payload(sample_diabetes130)

    with TestClient(create_app(service=service)) as client:
        batch = client.post("/batch-predict", json=[payload, payload])
        health = client.get("/health")
        metrics = client.get("/metrics")
        model_info = client.get("/model-info")

    assert batch.status_code == 200
    assert len(batch.json()["predictions"]) == 2
    assert health.json()["status"] == "ok"
    assert metrics.text.startswith("# HELP medmlops_predictions_total")
    assert "medmlops_predictions_total 2" in metrics.text
    assert model_info.json()["model_version"] == "7"
    assert model_info.json()["data_hash"] == "train-hash"


def test_shadow_routing_is_configurable(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
) -> None:
    challenger = FakeChallenger()
    service = _service(
        tmp_path,
        challenger=challenger,
        challenger_fraction=1.0,
    )

    with TestClient(create_app(service=service)) as client:
        response = client.post("/predict", json=_payload(sample_diabetes130))
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert challenger.calls == 1
    assert "race" not in challenger.last_columns
    assert "medmlops_shadow_predictions_total 1" in metrics.text


def test_startup_loads_champion_alias_from_registry(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
) -> None:
    params = load_yaml("params.yaml")
    conformal_path = tmp_path / "conformal.pkl"
    joblib.dump(cast(ConformalGate, FakeGate()), conformal_path)
    train_path = tmp_path / "train.parquet"
    sample_diabetes130.to_parquet(train_path, index=False)
    metrics_path = tmp_path / "train_metrics.json"
    metrics_path.write_text(
        json.dumps({"source_train_sha256": "abc123"}),
        encoding="utf-8",
    )
    params["serving"]["conformal_model_path"] = str(conformal_path)
    params["serving"]["audit_database_url"] = f"sqlite:///{tmp_path / 'startup.db'}"
    params["serving"]["training_metrics_path"] = str(metrics_path)
    params["serving"]["ood"]["reference_path"] = str(train_path)
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    calls: list[tuple[str, str, str]] = []

    def loader(
        model_name: str,
        alias: str,
        experiment_name: str,
    ) -> RegistryModelHandle:
        calls.append((model_name, alias, experiment_name))
        return _registry_handle(alias=alias)

    with TestClient(
        create_app(params_path=params_path, registry_loader=loader)
    ) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    assert calls == [("MedMLOps", "champion", "MedMLOps-Lab")]
    assert response.json()["model_version"] == "7"
    assert response.json()["data_hash"] == "abc123"
