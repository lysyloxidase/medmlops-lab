"""Probability calibration for clinical risk models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from medmlops.features.pipeline import feature_spec_from_params, load_yaml, split_xy
from medmlops.models.tabular_mlp import MLPConfig, TabularMLP, predict_proba_mlp
from medmlops.models.tracking import (
    configure_mlflow,
    flatten_numeric_metrics,
    flatten_params,
    sha256_file,
)
from medmlops.seeds import set_deterministic

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DEFAULT_HERO_PATH = Path("models/hero.pt")
DEFAULT_CALIB_PATH = Path("data/processed/calib.parquet")
DEFAULT_VAL_PATH = Path("data/processed/val.parquet")
DEFAULT_MODEL_PATH = Path("models/calibrated.pkl")
DEFAULT_METRICS_PATH = Path("reports/calibration_metrics.json")
DEFAULT_FIGURE_PATH = Path("reports/figures/reliability.png")
DEFAULT_PARAMS_PATH = Path("params.yaml")

EPSILON = 1e-6


class ProbabilityCalibrator(Protocol):
    """Protocol for probability-only calibration models."""

    def fit(
        self, y_true: NDArray[np.int_], y_probability: NDArray[np.float64]
    ) -> ProbabilityCalibrator:
        """Fit the calibrator."""
        raise NotImplementedError

    def transform(self, y_probability: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform uncalibrated probabilities."""
        raise NotImplementedError


def _clip_probability(y_probability: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(y_probability, EPSILON, 1.0 - EPSILON)


def _logit(y_probability: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = _clip_probability(y_probability)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class IdentityCalibrator:
    """No-op calibrator used before fitting a calibration method."""

    def fit(
        self, y_true: NDArray[np.int_], y_probability: NDArray[np.float64]
    ) -> IdentityCalibrator:
        del y_true, y_probability
        return self

    def transform(self, y_probability: NDArray[np.float64]) -> NDArray[np.float64]:
        return _clip_probability(y_probability)


@dataclass
class IsotonicProbabilityCalibrator:
    """Isotonic regression over raw model probabilities."""

    model: IsotonicRegression | None = None

    def fit(
        self, y_true: NDArray[np.int_], y_probability: NDArray[np.float64]
    ) -> IsotonicProbabilityCalibrator:
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(y_probability, y_true)
        return self

    def transform(self, y_probability: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model is None:
            msg = "Isotonic calibrator must be fitted before transform"
            raise RuntimeError(msg)
        return _clip_probability(np.asarray(self.model.predict(y_probability)))


@dataclass
class PlattProbabilityCalibrator:
    """Platt scaling via logistic regression on logits."""

    model: LogisticRegression | None = None

    def fit(
        self, y_true: NDArray[np.int_], y_probability: NDArray[np.float64]
    ) -> PlattProbabilityCalibrator:
        self.model = LogisticRegression(C=1_000_000.0, solver="lbfgs")
        self.model.fit(_logit(y_probability).reshape(-1, 1), y_true)
        return self

    def transform(self, y_probability: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model is None:
            msg = "Platt calibrator must be fitted before transform"
            raise RuntimeError(msg)
        return _clip_probability(
            np.asarray(self.model.predict_proba(_logit(y_probability).reshape(-1, 1)))[
                :, 1
            ]
        )


@dataclass
class TemperatureProbabilityCalibrator:
    """Temperature scaling over model logits."""

    temperature: float = 1.0

    def fit(
        self, y_true: NDArray[np.int_], y_probability: NDArray[np.float64]
    ) -> TemperatureProbabilityCalibrator:
        logits = _logit(y_probability)
        candidates = np.linspace(0.5, 5.0, 91)
        losses = [
            log_loss(
                y_true,
                _clip_probability(1.0 / (1.0 + np.exp(-(logits / temperature)))),
                labels=[0, 1],
            )
            for temperature in candidates
        ]
        self.temperature = float(candidates[int(np.argmin(losses))])
        return self

    def transform(self, y_probability: NDArray[np.float64]) -> NDArray[np.float64]:
        logits = _logit(y_probability)
        return _clip_probability(1.0 / (1.0 + np.exp(-(logits / self.temperature))))


class CalibratedRiskModel(  # pyright: ignore[reportIncompatibleMethodOverride]
    ClassifierMixin, BaseEstimator
):
    """Sklearn-like wrapper around the PyTorch hero and probability calibrator."""

    classes_: NDArray[np.int_]

    def __init__(
        self,
        transformer: Any,
        model: TabularMLP,
        calibrator: ProbabilityCalibrator | None = None,
    ) -> None:
        self.transformer = transformer
        self.model = model
        self.calibrator = calibrator or IdentityCalibrator()
        self.classes_ = np.array([0, 1], dtype=np.int_)

    def fit(
        self, x: pd.DataFrame, y: NDArray[np.int_] | pd.Series
    ) -> CalibratedRiskModel:
        del x, y
        return self

    def predict_raw_proba(self, x: pd.DataFrame) -> NDArray[np.float64]:
        matrix = self.transformer.transform(x)
        return predict_proba_mlp(self.model, matrix)

    def predict_positive_proba(self, x: pd.DataFrame) -> NDArray[np.float64]:
        return self.calibrator.transform(self.predict_raw_proba(x))

    def predict_proba(self, x: pd.DataFrame) -> NDArray[np.float64]:
        positive = self.predict_positive_proba(x)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, x: pd.DataFrame) -> NDArray[np.int_]:
        return (self.predict_positive_proba(x) >= 0.5).astype(np.int_)


def expected_calibration_error(
    y_true: NDArray[np.int_],
    y_probability: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Compute binary expected calibration error."""

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_probability, bins[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_probability[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return float(ece)


def calibration_slope_intercept(
    y_true: NDArray[np.int_],
    y_probability: NDArray[np.float64],
) -> tuple[float, float]:
    """Estimate calibration slope and intercept on the logit scale."""

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0
    model = LogisticRegression(C=1_000_000.0, solver="lbfgs")
    model.fit(_logit(y_probability).reshape(-1, 1), y_true)
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    intercepts = np.asarray(model.intercept_, dtype=np.float64)
    return float(coefficients[0, 0]), float(intercepts[0])


def build_calibrator(method: str) -> ProbabilityCalibrator:
    """Create a probability calibrator from params."""

    if method == "isotonic":
        return IsotonicProbabilityCalibrator()
    if method == "platt":
        return PlattProbabilityCalibrator()
    if method == "temperature":
        return TemperatureProbabilityCalibrator()
    msg = f"Unsupported calibration method: {method}"
    raise ValueError(msg)


def load_hero_risk_model(
    hero_path: str | Path = DEFAULT_HERO_PATH,
) -> CalibratedRiskModel:
    """Load the PyTorch hero artifact saved by Phase 2 training."""

    bundle = torch.load(hero_path, map_location="cpu", weights_only=False)
    config = bundle["config"]
    if not isinstance(config, MLPConfig):
        config = MLPConfig(**config)
    model = TabularMLP(
        n_features=int(bundle["n_features"]),
        hidden=config.hidden,
        dropout=config.dropout,
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return CalibratedRiskModel(transformer=bundle["transformer"], model=model)


def _metrics(
    y_true: NDArray[np.int_],
    y_probability: NDArray[np.float64],
    n_bins: int,
) -> dict[str, float]:
    slope, intercept = calibration_slope_intercept(y_true, y_probability)
    return {
        "ece": float(expected_calibration_error(y_true, y_probability, n_bins)),
        "brier": float(brier_score_loss(y_true, y_probability)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _stable(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 10) for key, value in sorted(metrics.items())}


def plot_reliability_diagram(
    y_true: NDArray[np.int_],
    y_probability: NDArray[np.float64],
    output_path: str | Path = DEFAULT_FIGURE_PATH,
    n_bins: int = 10,
) -> Path:
    """Save a reliability diagram."""

    prob_true, prob_pred = calibration_curve(
        y_true, y_probability, n_bins=n_bins, strategy="uniform"
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
    ax.plot(prob_pred, prob_true, marker="o", label="Calibrated hero")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Reliability diagram")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def calibrate_phase3(
    hero_path: str | Path = DEFAULT_HERO_PATH,
    calib_path: str | Path = DEFAULT_CALIB_PATH,
    val_path: str | Path = DEFAULT_VAL_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    figure_path: str | Path = DEFAULT_FIGURE_PATH,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> Path:
    """Fit probability calibration and write model, metrics, and plot."""

    params = load_yaml(params_path)
    set_deterministic(int(params.get("seed", 42)))
    calibration_params = params.get("calibration", {})
    model_params = params.get("model", {})
    if not isinstance(calibration_params, dict) or not isinstance(model_params, dict):
        msg = "params.yaml must contain calibration and model mappings"
        raise TypeError(msg)

    method = str(calibration_params.get("method", "isotonic"))
    n_bins = int(calibration_params.get("n_bins", 10))
    max_ece = float(calibration_params.get("max_ece", 0.03))

    risk_model = load_hero_risk_model(hero_path)
    calib_df = pd.read_parquet(calib_path)
    val_df = pd.read_parquet(val_path)
    x_calib, y_calib_series = split_xy(calib_df, feature_spec_from_params(params))
    x_val, y_val_series = split_xy(val_df, feature_spec_from_params(params))
    y_calib = y_calib_series.to_numpy(dtype=np.int_)
    y_val = y_val_series.to_numpy(dtype=np.int_)

    raw_calib = risk_model.predict_raw_proba(x_calib)
    raw_val = risk_model.predict_raw_proba(x_val)
    calibrator = build_calibrator(method).fit(y_calib, raw_calib)
    risk_model.calibrator = calibrator

    calibrated_calib = risk_model.predict_positive_proba(x_calib)
    calibrated_val = risk_model.predict_positive_proba(x_val)

    report: dict[str, Any] = {
        "method": method,
        "max_ece": max_ece,
        "calibration": {
            "raw": _stable(_metrics(y_calib, raw_calib, n_bins)),
            "calibrated": _stable(_metrics(y_calib, calibrated_calib, n_bins)),
        },
        "validation": {
            "raw": _stable(_metrics(y_val, raw_val, n_bins)),
            "calibrated": _stable(_metrics(y_val, calibrated_val, n_bins)),
        },
        "passes_max_ece": bool(
            expected_calibration_error(y_val, calibrated_val, n_bins) <= max_ece
        ),
        "source_calib_sha256": sha256_file(calib_path),
    }

    model_destination = Path(model_path)
    metrics_destination = Path(metrics_path)
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(risk_model, model_destination, compress=3)
    metrics_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_reliability_diagram(y_val, calibrated_val, figure_path, n_bins)

    configure_mlflow(str(model_params.get("experiment_name", "MedMLOps-Lab")))
    with mlflow.start_run(run_name="phase3-calibrate"):
        mlflow.log_params(
            {
                key: str(value)
                for key, value in flatten_params(
                    {"calibration": calibration_params}
                ).items()
            }
        )
        mlflow.log_metrics(flatten_numeric_metrics(report, "calibration"))
        mlflow.log_artifact(str(metrics_destination))
        mlflow.log_artifact(str(figure_path))

    return metrics_destination
