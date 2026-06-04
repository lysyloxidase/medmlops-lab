"""Clinical metrics beyond AUROC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from medmlops.calibration.calibrate import (
    calibration_slope_intercept,
    expected_calibration_error,
)
from medmlops.conformal.conformal import ConformalGate, coverage_rate
from medmlops.features.pipeline import feature_spec_from_params, load_yaml, split_xy
from medmlops.models.tracking import (
    configure_mlflow,
    flatten_numeric_metrics,
    flatten_params,
    sha256_file,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DEFAULT_CONFORMAL_MODEL_PATH = Path("models/conformal.pkl")
DEFAULT_TEST_PATH = Path("data/processed/test.parquet")
DEFAULT_METRICS_PATH = Path("reports/clinical_metrics.json")
DEFAULT_FIGURE_PATH = Path("reports/figures/decision_curve.png")
DEFAULT_PARAMS_PATH = Path("params.yaml")


def _as_float_array(values: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def _as_int_array(values: ArrayLike) -> NDArray[np.int_]:
    return np.asarray(values, dtype=np.int_)


def threshold_metrics(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    threshold: float,
) -> dict[str, float]:
    """Compute clinical binary metrics at a threshold."""

    y = _as_int_array(y_true)
    probability = _as_float_array(y_probability)
    prediction = (probability >= threshold).astype(np.int_)
    tp = float(np.sum((prediction == 1) & (y == 1)))
    fp = float(np.sum((prediction == 1) & (y == 0)))
    tn = float(np.sum((prediction == 0) & (y == 0)))
    fn = float(np.sum((prediction == 0) & (y == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def net_benefit(y_true: ArrayLike, y_probability: ArrayLike, threshold: float) -> float:
    """Compute Vickers-Elkin decision-curve net benefit."""

    if not 0.0 < threshold < 1.0:
        msg = "threshold must be between 0 and 1"
        raise ValueError(msg)
    y = _as_int_array(y_true)
    probability = _as_float_array(y_probability)
    prevalence = float(np.mean(y))
    metrics = threshold_metrics(y, probability, threshold)
    weight = threshold / (1.0 - threshold)
    return float(
        metrics["sensitivity"] * prevalence
        - (1.0 - metrics["specificity"]) * (1.0 - prevalence) * weight
    )


def treat_all_net_benefit(y_true: ArrayLike, threshold: float) -> float:
    """Net benefit if every patient is treated/escalated."""

    y = _as_int_array(y_true)
    prevalence = float(np.mean(y))
    weight = threshold / (1.0 - threshold)
    return float(prevalence - (1.0 - prevalence) * weight)


def decision_curve(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    thresholds: ArrayLike,
) -> dict[str, list[float]]:
    """Compute model, treat-all, and treat-none net-benefit curves."""

    threshold_array = _as_float_array(thresholds)
    return {
        "threshold": [float(threshold) for threshold in threshold_array],
        "model": [
            net_benefit(y_true, y_probability, float(threshold))
            for threshold in threshold_array
        ],
        "treat_all": [
            treat_all_net_benefit(y_true, float(threshold))
            for threshold in threshold_array
        ],
        "treat_none": [0.0 for _threshold in threshold_array],
    }


def plot_decision_curve(curve: dict[str, list[float]], output_path: str | Path) -> Path:
    """Save a decision-curve plot."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["threshold"], curve["model"], label="Model", linewidth=2)
    ax.plot(curve["threshold"], curve["treat_all"], label="Treat all", linestyle="--")
    ax.plot(curve["threshold"], curve["treat_none"], label="Treat none", linestyle=":")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def _stable_float_dict(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 10) for key, value in sorted(metrics.items())}


def evaluate_phase3(
    conformal_model_path: str | Path = DEFAULT_CONFORMAL_MODEL_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    figure_path: str | Path = DEFAULT_FIGURE_PATH,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> Path:
    """Evaluate calibrated/conformal model with clinical metrics."""

    params = load_yaml(params_path)
    evaluation_params = params.get("evaluation", {})
    model_params = params.get("model", {})
    if not isinstance(evaluation_params, dict) or not isinstance(model_params, dict):
        msg = "params.yaml must contain evaluation and model mappings"
        raise TypeError(msg)

    thresholds = evaluation_params.get("clinical_thresholds", [0.05, 0.1, 0.2, 0.3])
    if not isinstance(thresholds, list):
        msg = "evaluation.clinical_thresholds must be a list"
        raise TypeError(msg)

    gate: ConformalGate = joblib.load(conformal_model_path)
    spec = feature_spec_from_params(params)
    test_df = pd.read_parquet(test_path)
    x_test, y_test_series = split_xy(test_df, spec)
    y_test = y_test_series.to_numpy(dtype=np.int_)
    probabilities = gate.estimator.predict_positive_proba(x_test)
    prediction_sets = gate.prediction_sets(x_test)
    batch = gate.predict_batch(x_test)
    slope, intercept = calibration_slope_intercept(y_test, probabilities)
    curve = decision_curve(y_test, probabilities, thresholds)
    threshold_report = {
        str(threshold): _stable_float_dict(
            threshold_metrics(y_test, probabilities, float(threshold))
        )
        for threshold in thresholds
    }

    report: dict[str, Any] = {
        "auroc": round(float(roc_auc_score(y_test, probabilities)), 10),
        "auprc": round(float(average_precision_score(y_test, probabilities)), 10),
        "brier": round(float(brier_score_loss(y_test, probabilities)), 10),
        "ece": round(float(expected_calibration_error(y_test, probabilities)), 10),
        "calibration_slope": round(float(slope), 10),
        "calibration_intercept": round(float(intercept), 10),
        "threshold_metrics": threshold_report,
        "decision_curve": {
            key: [round(float(value), 10) for value in values]
            for key, values in curve.items()
        },
        "conformal": {
            "alpha": gate.alpha,
            "marginal_coverage": round(coverage_rate(prediction_sets, y_test), 10),
            "abstention_rate": round(float(np.mean(batch.abstain)), 10),
            "ambiguous_set_rate": round(
                float(
                    np.mean(
                        [len(prediction_set) == 2 for prediction_set in prediction_sets]
                    )
                ),
                10,
            ),
            "empty_set_rate": round(
                float(
                    np.mean(
                        [len(prediction_set) == 0 for prediction_set in prediction_sets]
                    )
                ),
                10,
            ),
        },
        "source_test_sha256": sha256_file(test_path),
    }

    destination = Path(metrics_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_decision_curve(curve, figure_path)

    configure_mlflow(str(model_params.get("experiment_name", "MedMLOps-Lab")))
    with mlflow.start_run(run_name="phase3-evaluate"):
        mlflow.log_params(
            {
                key: str(value)
                for key, value in flatten_params(
                    {"evaluation": evaluation_params}
                ).items()
            }
        )
        mlflow.log_metrics(flatten_numeric_metrics(report, "clinical"))
        mlflow.log_artifact(str(destination))
        mlflow.log_artifact(str(figure_path))

    return destination
