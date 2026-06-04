"""MAPIE conformal prediction with abstention."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from numpy.typing import NDArray

from medmlops.calibration.calibrate import CalibratedRiskModel
from medmlops.features.pipeline import feature_spec_from_params, load_yaml, split_xy
from medmlops.models.tracking import (
    configure_mlflow,
    flatten_numeric_metrics,
    flatten_params,
    sha256_file,
)

DEFAULT_CALIBRATED_PATH = Path("models/calibrated.pkl")
DEFAULT_CONFORMAL_PATH = Path("data/processed/conformal.parquet")
DEFAULT_TEST_PATH = Path("data/processed/test.parquet")
DEFAULT_MODEL_PATH = Path("models/conformal.pkl")
DEFAULT_METRICS_PATH = Path("reports/conformal_metrics.json")
DEFAULT_PARAMS_PATH = Path("params.yaml")


def conformal_quantile(scores: NDArray[np.float64], alpha: float) -> float:
    """Finite-sample split-conformal quantile."""

    n = len(scores)
    if n == 0:
        msg = "Conformal calibration requires at least one score"
        raise ValueError(msg)
    quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, quantile_level, method="higher"))


@dataclass
class PredictionSetResult:
    """Batch prediction-set result."""

    prediction: list[int | None]
    prediction_set: list[list[int]]
    abstain: list[bool]
    reason: list[str]


class ConformalGate:
    """Binary conformal prediction-set gate with clinical abstention."""

    def __init__(
        self,
        estimator: CalibratedRiskModel,
        alpha: float = 0.1,
        abstain_on_ambiguous: bool = True,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.abstain_on_ambiguous = abstain_on_ambiguous
        self.classes_ = np.array([0, 1], dtype=np.int_)
        self.quantile_: float | None = None
        self.mapie_: MapieClassifier | None = None
        self.mapie_fit_error_: str | None = None

    def fit(self, x_conf: pd.DataFrame, y_conf: NDArray[np.int_]) -> ConformalGate:
        """Fit MAPIE and split-conformal nonconformity scores."""

        probabilities = self.estimator.predict_proba(x_conf)
        true_probabilities = probabilities[np.arange(len(y_conf)), y_conf]
        self.quantile_ = conformal_quantile(1.0 - true_probabilities, self.alpha)
        try:
            self.mapie_ = MapieClassifier(
                estimator=self.estimator,
                method="lac",
                cv="prefit",
            )
            self.mapie_.fit(x_conf, y_conf)
        except Exception as exc:
            self.mapie_fit_error_ = str(exc)
            self.mapie_ = None
        return self

    def prediction_sets(self, x: pd.DataFrame) -> list[list[int]]:
        """Return class prediction sets from conformal nonconformity."""

        if self.quantile_ is None:
            msg = "ConformalGate must be fitted before prediction"
            raise RuntimeError(msg)
        probabilities = self.estimator.predict_proba(x)
        sets: list[list[int]] = []
        for row in probabilities:
            allowed = [
                int(class_label)
                for class_label, probability in zip(self.classes_, row, strict=True)
                if 1.0 - float(probability) <= self.quantile_
            ]
            sets.append(allowed)
        return sets

    def predict_batch(self, x: pd.DataFrame) -> PredictionSetResult:
        """Return batch prediction sets and abstention decisions."""

        sets = self.prediction_sets(x)
        predictions: list[int | None] = []
        abstain: list[bool] = []
        reasons: list[str] = []
        for prediction_set in sets:
            if len(prediction_set) == 1:
                predictions.append(prediction_set[0])
                abstain.append(False)
                reasons.append("single_class_set")
            elif len(prediction_set) == 0:
                predictions.append(None)
                abstain.append(True)
                reasons.append("empty_set")
            elif self.abstain_on_ambiguous:
                predictions.append(None)
                abstain.append(True)
                reasons.append("ambiguous_set")
            else:
                predictions.append(int(max(prediction_set)))
                abstain.append(False)
                reasons.append("ambiguous_set_allowed")
        return PredictionSetResult(predictions, sets, abstain, reasons)

    def predict_with_abstention(self, x: pd.DataFrame) -> dict[str, Any]:
        """Return prediction-set decisions, scalarized for a single row."""

        result = self.predict_batch(x)
        if len(result.prediction) == 1:
            return {
                "prediction": result.prediction[0],
                "set": result.prediction_set[0],
                "abstain": result.abstain[0],
                "reason": result.reason[0],
            }
        return {
            "prediction": result.prediction,
            "set": result.prediction_set,
            "abstain": result.abstain,
            "reason": result.reason,
        }


def coverage_rate(prediction_sets: list[list[int]], y_true: NDArray[np.int_]) -> float:
    """Return marginal conformal coverage."""

    covered = [
        int(y in prediction_set)
        for y, prediction_set in zip(y_true, prediction_sets, strict=True)
    ]
    return float(np.mean(covered))


def conformalize_phase3(
    calibrated_path: str | Path = DEFAULT_CALIBRATED_PATH,
    conformal_path: str | Path = DEFAULT_CONFORMAL_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> Path:
    """Fit the conformal gate and write metrics."""

    params = load_yaml(params_path)
    conformal_params = params.get("conformal", {})
    model_params = params.get("model", {})
    if not isinstance(conformal_params, dict) or not isinstance(model_params, dict):
        msg = "params.yaml must contain conformal and model mappings"
        raise TypeError(msg)

    alpha = float(conformal_params.get("alpha", 0.1))
    abstain_on_ambiguous = bool(conformal_params.get("abstain_on_ambiguous", True))
    spec = feature_spec_from_params(params)
    estimator: CalibratedRiskModel = joblib.load(calibrated_path)

    conformal_df = pd.read_parquet(conformal_path)
    test_df = pd.read_parquet(test_path)
    x_conf, y_conf_series = split_xy(conformal_df, spec)
    x_test, y_test_series = split_xy(test_df, spec)
    y_conf = y_conf_series.to_numpy(dtype=np.int_)
    y_test = y_test_series.to_numpy(dtype=np.int_)

    gate = ConformalGate(estimator, alpha, abstain_on_ambiguous).fit(x_conf, y_conf)
    batch = gate.predict_batch(x_test)
    coverage = coverage_rate(batch.prediction_set, y_test)
    abstention_rate = float(np.mean(batch.abstain))
    empty_rate = float(
        np.mean([len(prediction_set) == 0 for prediction_set in batch.prediction_set])
    )
    ambiguous_rate = float(
        np.mean([len(prediction_set) == 2 for prediction_set in batch.prediction_set])
    )

    report: dict[str, Any] = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "marginal_coverage": round(coverage, 10),
        "coverage_within_tolerance": bool(abs(coverage - (1.0 - alpha)) <= 0.05),
        "abstention_rate": round(abstention_rate, 10),
        "empty_set_rate": round(empty_rate, 10),
        "ambiguous_set_rate": round(ambiguous_rate, 10),
        "quantile": round(float(gate.quantile_ or 0.0), 10),
        "mapie_classifier_fit": gate.mapie_ is not None,
        "mapie_fit_error": gate.mapie_fit_error_,
        "source_conformal_sha256": sha256_file(conformal_path),
    }

    model_destination = Path(model_path)
    metrics_destination = Path(metrics_path)
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gate, model_destination, compress=3)
    metrics_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    configure_mlflow(str(model_params.get("experiment_name", "MedMLOps-Lab")))
    with mlflow.start_run(run_name="phase3-conformalize"):
        mlflow.log_params(
            {
                key: str(value)
                for key, value in flatten_params(
                    {"conformal": conformal_params}
                ).items()
            }
        )
        mlflow.log_metrics(flatten_numeric_metrics(report, "conformal"))
        mlflow.log_artifact(str(metrics_destination))

    return metrics_destination
