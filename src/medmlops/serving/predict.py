"""Prediction orchestration for the FastAPI serving layer."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast
from uuid import uuid4

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from medmlops.conformal.conformal import ConformalGate
from medmlops.features.pipeline import FeatureSpec, feature_spec_from_params, load_yaml
from medmlops.models.tracking import configure_mlflow, sha256_file
from medmlops.serving.audit import PredictionAuditLogger, PredictionAuditRecord
from medmlops.serving.schemas import (
    BatchPredictionResponse,
    ClinicalEncounter,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_CONFORMAL_MODEL_PATH = Path("models/conformal.pkl")
DEFAULT_AUDIT_DATABASE_URL = "sqlite:///audit.db"
DEFAULT_TRAINING_METRICS_PATH = Path("reports/train_metrics.json")
DEFAULT_TRAINING_REFERENCE_PATH = Path("data/processed/train.parquet")


@dataclass(frozen=True)
class ServingSettings:
    """Configuration for the serving layer."""

    model_name: str = "MedMLOps"
    model_alias: str = "champion"
    challenger_alias: str | None = "challenger"
    challenger_fraction: float = 0.0
    conformal_model_path: Path = DEFAULT_CONFORMAL_MODEL_PATH
    audit_database_url: str = DEFAULT_AUDIT_DATABASE_URL
    training_metrics_path: Path = DEFAULT_TRAINING_METRICS_PATH
    ood_reference_path: Path = DEFAULT_TRAINING_REFERENCE_PATH
    ood_lower_quantile: float = 0.01
    ood_upper_quantile: float = 0.99
    experiment_name: str = "MedMLOps-Lab"
    registry_required: bool = False
    random_seed: int = 42


@dataclass(frozen=True)
class RegistryModelMetadata:
    """Metadata for one MLflow alias-resolved model version."""

    model_name: str
    alias: str
    version: str
    run_id: str
    source: str
    model_uri: str


@dataclass(frozen=True)
class RegistryModelHandle:
    """Warm-loaded registry model plus metadata, or the load error."""

    model: Any | None
    metadata: RegistryModelMetadata | None
    error: str | None = None


RegistryLoader = Callable[[str, str, str], RegistryModelHandle]


@dataclass(frozen=True)
class QuantileOODChecker:
    """Numeric OOD checks against training quantile ranges."""

    ranges: dict[str, tuple[float, float]]

    @classmethod
    def from_training_reference(
        cls,
        reference_path: str | Path,
        spec: FeatureSpec,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> QuantileOODChecker:
        """Build numeric quantile ranges from the training split."""

        path = Path(reference_path)
        if not path.exists():
            return cls({})
        frame = pd.read_parquet(path)
        ranges: dict[str, tuple[float, float]] = {}
        for column in spec.numeric:
            if column not in frame.columns:
                continue
            numeric = cast("pd.Series", pd.to_numeric(frame[column], errors="coerce"))
            numeric = numeric.dropna()
            if numeric.empty:
                continue
            low = float(numeric.quantile(lower_quantile))
            high = float(numeric.quantile(upper_quantile))
            ranges[column] = (low, high)
        return cls(ranges)

    @property
    def loaded(self) -> bool:
        return bool(self.ranges)

    def check(self, frame: pd.DataFrame) -> list[str]:
        """Return human-readable OOD flags for one row."""

        if frame.empty:
            return ["empty_feature_frame"]
        flags: list[str] = []
        first_row = frame.iloc[0]
        for column, (lower, upper) in self.ranges.items():
            value = first_row.get(column)
            if pd.isna(value):
                flags.append(f"ood_numeric:{column}:missing")
                continue
            numeric_value = float(value)
            if numeric_value < lower or numeric_value > upper:
                flags.append(
                    f"ood_numeric:{column}:{numeric_value:.6g}"
                    f"_outside_[{lower:.6g},{upper:.6g}]"
                )
        return flags


class ServingMetrics:
    """In-memory counters rendered in Prometheus text format."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.predictions_total = 0
        self.abstentions_total = 0
        self.ood_abstentions_total = 0
        self.latency_ms_sum = 0.0
        self.shadow_predictions_total = 0
        self.shadow_errors_total = 0

    def record_prediction(
        self,
        *,
        latency_ms: float,
        abstain: bool,
        ood_flags: list[str],
    ) -> None:
        with self._lock:
            self.predictions_total += 1
            self.latency_ms_sum += latency_ms
            if abstain:
                self.abstentions_total += 1
            if ood_flags:
                self.ood_abstentions_total += 1

    def record_shadow(self, *, error: bool) -> None:
        with self._lock:
            self.shadow_predictions_total += 1
            if error:
                self.shadow_errors_total += 1

    def render_prometheus(self) -> str:
        """Return Prometheus exposition text."""

        with self._lock:
            lines = [
                "# HELP medmlops_predictions_total Total prediction requests.",
                "# TYPE medmlops_predictions_total counter",
                f"medmlops_predictions_total {self.predictions_total}",
                "# HELP medmlops_abstentions_total Abstained prediction requests.",
                "# TYPE medmlops_abstentions_total counter",
                f"medmlops_abstentions_total {self.abstentions_total}",
                "# HELP medmlops_ood_abstentions_total OOD-triggered abstentions.",
                "# TYPE medmlops_ood_abstentions_total counter",
                f"medmlops_ood_abstentions_total {self.ood_abstentions_total}",
                "# HELP medmlops_prediction_latency_ms_sum Prediction latency sum.",
                "# TYPE medmlops_prediction_latency_ms_sum counter",
                f"medmlops_prediction_latency_ms_sum {self.latency_ms_sum:.6f}",
                "# HELP medmlops_prediction_latency_ms_count Prediction latency count.",
                "# TYPE medmlops_prediction_latency_ms_count counter",
                f"medmlops_prediction_latency_ms_count {self.predictions_total}",
                "# HELP medmlops_shadow_predictions_total Shadow route attempts.",
                "# TYPE medmlops_shadow_predictions_total counter",
                f"medmlops_shadow_predictions_total {self.shadow_predictions_total}",
                "# HELP medmlops_shadow_errors_total Shadow route errors.",
                "# TYPE medmlops_shadow_errors_total counter",
                f"medmlops_shadow_errors_total {self.shadow_errors_total}",
            ]
        return "\n".join(lines) + "\n"


def serving_settings_from_params(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> ServingSettings:
    """Build serving settings from params.yaml and environment overrides."""

    params = load_yaml(params_path)
    model_params = params.get("model", {})
    serving_params = params.get("serving", {})
    if not isinstance(model_params, dict) or not isinstance(serving_params, dict):
        msg = "params.yaml model and serving sections must be mappings"
        raise TypeError(msg)
    ood_params = serving_params.get("ood", {})
    if not isinstance(ood_params, dict):
        msg = "params.yaml serving.ood must be a mapping"
        raise TypeError(msg)

    return ServingSettings(
        model_name=str(
            serving_params.get(
                "model_name", model_params.get("registered_model_name", "MedMLOps")
            )
        ),
        model_alias=str(serving_params.get("model_alias", "champion")),
        challenger_alias=(
            None
            if serving_params.get("challenger_alias") is None
            else str(serving_params.get("challenger_alias", "challenger"))
        ),
        challenger_fraction=float(serving_params.get("challenger_fraction", 0.0)),
        conformal_model_path=Path(
            str(
                serving_params.get("conformal_model_path", DEFAULT_CONFORMAL_MODEL_PATH)
            )
        ),
        audit_database_url=os.environ.get(
            "MEDMLOPS_AUDIT_DATABASE_URL",
            str(serving_params.get("audit_database_url", DEFAULT_AUDIT_DATABASE_URL)),
        ),
        training_metrics_path=Path(
            str(
                serving_params.get(
                    "training_metrics_path", DEFAULT_TRAINING_METRICS_PATH
                )
            )
        ),
        ood_reference_path=Path(
            str(ood_params.get("reference_path", DEFAULT_TRAINING_REFERENCE_PATH))
        ),
        ood_lower_quantile=float(ood_params.get("lower_quantile", 0.01)),
        ood_upper_quantile=float(ood_params.get("upper_quantile", 0.99)),
        experiment_name=str(model_params.get("experiment_name", "MedMLOps-Lab")),
        registry_required=(
            os.environ.get("MEDMLOPS_REGISTRY_REQUIRED", "0").lower()
            in {"1", "true", "yes"}
        ),
        random_seed=int(serving_params.get("random_seed", params.get("seed", 42))),
    )


def default_registry_loader(
    model_name: str,
    alias: str,
    experiment_name: str,
) -> RegistryModelHandle:
    """Warm-load an MLflow model by alias and return model-version metadata."""

    configure_mlflow(experiment_name)
    model_uri = f"models:/{model_name}@{alias}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        version = MlflowClient().get_model_version_by_alias(model_name, alias)
    except Exception as exc:
        return RegistryModelHandle(model=None, metadata=None, error=str(exc))

    metadata = RegistryModelMetadata(
        model_name=model_name,
        alias=alias,
        version=str(version.version),
        run_id=str(version.run_id),
        source=str(version.source),
        model_uri=model_uri,
    )
    return RegistryModelHandle(model=model, metadata=metadata)


def load_training_metrics(path: str | Path) -> dict[str, Any]:
    """Read training metrics if available."""

    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}
    return cast(dict[str, Any], json.loads(metrics_path.read_text(encoding="utf-8")))


def resolve_data_hash(
    *,
    training_metrics: dict[str, Any],
    reference_path: str | Path,
    fallback_path: str | Path,
) -> str:
    """Resolve a provenance hash for audit/model-info responses."""

    source_hash = training_metrics.get("source_train_sha256")
    if isinstance(source_hash, str) and source_hash:
        return source_hash
    for candidate in (Path(reference_path), Path(fallback_path)):
        if candidate.exists():
            return sha256_file(candidate)
    return "unavailable"


def _as_positive_probability(values: object) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return float(array.reshape(-1)[0])


def _as_prediction_set(value: object) -> list[int]:
    return [int(item) for item in cast(list[Any], value)]


class PredictionService:
    """Clinical prediction service with calibration, conformal gate, and audit."""

    def __init__(
        self,
        *,
        settings: ServingSettings,
        spec: FeatureSpec,
        gate: ConformalGate,
        audit_logger: PredictionAuditLogger,
        registry_model: RegistryModelHandle,
        challenger_model: RegistryModelHandle | None,
        ood_checker: QuantileOODChecker,
        training_metrics: dict[str, Any],
        data_hash: str,
        metrics: ServingMetrics | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings
        self.spec = spec
        self.gate = gate
        self.audit_logger = audit_logger
        self.registry_model = registry_model
        self.challenger_model = challenger_model
        self.ood_checker = ood_checker
        self.training_metrics = training_metrics
        self.data_hash = data_hash
        self.metrics = metrics or ServingMetrics()
        self.rng = rng or random.Random(settings.random_seed)

    @classmethod
    def from_params(
        cls,
        params_path: str | Path = DEFAULT_PARAMS_PATH,
        registry_loader: RegistryLoader = default_registry_loader,
    ) -> PredictionService:
        """Load registry alias, conformal gate, OOD reference, and audit DB."""

        params = load_yaml(params_path)
        settings = serving_settings_from_params(params_path)
        spec = feature_spec_from_params(params)
        gate = cast(ConformalGate, joblib.load(settings.conformal_model_path))
        audit_logger = PredictionAuditLogger(settings.audit_database_url)
        audit_logger.initialize()
        registry_model = registry_loader(
            settings.model_name, settings.model_alias, settings.experiment_name
        )
        if settings.registry_required and registry_model.error is not None:
            msg = f"Failed to load registry model alias @{settings.model_alias}"
            raise RuntimeError(msg) from RuntimeError(registry_model.error)

        challenger_model: RegistryModelHandle | None = None
        if settings.challenger_alias and settings.challenger_fraction > 0.0:
            challenger_model = registry_loader(
                settings.model_name,
                settings.challenger_alias,
                settings.experiment_name,
            )
        ood_checker = QuantileOODChecker.from_training_reference(
            settings.ood_reference_path,
            spec,
            settings.ood_lower_quantile,
            settings.ood_upper_quantile,
        )
        training_metrics = load_training_metrics(settings.training_metrics_path)
        data_hash = resolve_data_hash(
            training_metrics=training_metrics,
            reference_path=settings.ood_reference_path,
            fallback_path=settings.conformal_model_path,
        )
        return cls(
            settings=settings,
            spec=spec,
            gate=gate,
            audit_logger=audit_logger,
            registry_model=registry_model,
            challenger_model=challenger_model,
            ood_checker=ood_checker,
            training_metrics=training_metrics,
            data_hash=data_hash,
        )

    @property
    def model_version(self) -> str:
        metadata = self.registry_model.metadata
        if metadata is None:
            return "unavailable"
        return metadata.version

    def predict_one(self, encounter: ClinicalEncounter) -> PredictionResponse:
        """Score one validated encounter and persist its audit record."""

        started = time.perf_counter()
        request_id = uuid4()
        model_frame = encounter.to_model_frame(self.spec)
        ood_flags = self.ood_checker.check(model_frame)
        risk_probability = _as_positive_probability(
            self.gate.estimator.predict_positive_proba(model_frame)
        )
        gate_result = self.gate.predict_with_abstention(model_frame)
        conformal_set = _as_prediction_set(gate_result.get("set", []))
        conformal_abstain = bool(gate_result.get("abstain", False))
        conformal_reason = str(gate_result.get("reason", "single_class_set"))

        abstain = conformal_abstain or bool(ood_flags)
        if ood_flags:
            abstain_reason = ";".join(ood_flags)
        elif conformal_abstain:
            abstain_reason = conformal_reason
        else:
            abstain_reason = None

        raw_prediction = gate_result.get("prediction")
        risk_class = None
        if not abstain:
            risk_class = (
                int(raw_prediction)
                if raw_prediction is not None
                else int(risk_probability >= 0.5)
            )

        self._maybe_shadow_predict(model_frame)
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = PredictionResponse(
            request_id=request_id,
            risk_probability=risk_probability,
            risk_class=risk_class,
            conformal_set=conformal_set,
            abstain=abstain,
            abstain_reason=abstain_reason,
            ood_flags=ood_flags,
            model_alias=self.settings.model_alias,
            model_version=self.model_version,
            data_hash=self.data_hash,
            latency_ms=latency_ms,
        )
        self.audit_logger.log_prediction(
            PredictionAuditRecord(
                request_id=request_id,
                input_payload=encounter.audit_payload(),
                model_version=response.model_version,
                model_alias=response.model_alias,
                data_hash=response.data_hash,
                risk_probability=response.risk_probability,
                risk_class=response.risk_class,
                conformal_set=response.conformal_set,
                abstain=response.abstain,
                abstain_reason=response.abstain_reason,
                latency_ms=response.latency_ms,
            )
        )
        self.metrics.record_prediction(
            latency_ms=response.latency_ms,
            abstain=response.abstain,
            ood_flags=response.ood_flags,
        )
        return response

    def predict_batch(
        self, encounters: list[ClinicalEncounter]
    ) -> BatchPredictionResponse:
        """Score a batch, auditing every row independently."""

        return BatchPredictionResponse(
            predictions=[self.predict_one(encounter) for encounter in encounters]
        )

    def health(self) -> HealthResponse:
        """Return serving health across gate, registry, OOD reference, and DB."""

        audit_ok = self.audit_logger.healthcheck()
        registry_ok = (
            self.registry_model.error is None and self.registry_model.model is not None
        )
        detail: dict[str, str] = {}
        if self.registry_model.error is not None:
            detail["registry"] = self.registry_model.error
        challenger = self.challenger_model
        if challenger is not None and challenger.error is not None:
            detail["challenger_registry"] = challenger.error
        status = "ok" if audit_ok and registry_ok else "degraded"
        return HealthResponse(
            status=status,
            conformal_loaded=True,
            registry_connected=registry_ok,
            audit_db_connected=audit_ok,
            ood_reference_loaded=self.ood_checker.loaded,
            detail=detail,
        )

    def model_info(self) -> ModelInfoResponse:
        """Return registry metadata and model provenance."""

        return ModelInfoResponse(
            model_name=self.settings.model_name,
            model_alias=self.settings.model_alias,
            model_version=self.model_version,
            challenger_alias=self.settings.challenger_alias,
            challenger_fraction=self.settings.challenger_fraction,
            data_hash=self.data_hash,
            training_metrics=self.training_metrics,
        )

    def prometheus_metrics(self) -> str:
        """Return Prometheus metrics text."""

        return self.metrics.render_prometheus()

    def _maybe_shadow_predict(self, model_frame: pd.DataFrame) -> None:
        if (
            self.challenger_model is None
            or self.challenger_model.model is None
            or self.settings.challenger_fraction <= 0.0
            or self.rng.random() >= self.settings.challenger_fraction
        ):
            return
        prediction_callable = getattr(self.challenger_model.model, "predict", None)
        try:
            if callable(prediction_callable):
                prediction_callable(model_frame)
            self.metrics.record_shadow(error=False)
        except Exception:
            self.metrics.record_shadow(error=True)
