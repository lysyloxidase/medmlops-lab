"""Fairlearn subgroup audit for protected attributes kept out of the model."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from medmlops.calibration.calibrate import (
    calibration_slope_intercept,
    expected_calibration_error,
)
from medmlops.conformal.conformal import ConformalGate
from medmlops.features.pipeline import feature_spec_from_params, load_yaml, split_xy
from medmlops.models.tracking import (
    configure_mlflow,
    flatten_numeric_metrics,
    sha256_file,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DEFAULT_MODEL_PATH = Path("models/conformal.pkl")
DEFAULT_TEST_PATH = Path("data/processed/test.parquet")
DEFAULT_REPORT_PATH = Path("reports/fairness.json")
DEFAULT_FIGURE_PATH = Path("reports/figures/subgroup_auroc.png")
DEFAULT_PARAMS_PATH = Path("params.yaml")

IMPOSSIBILITY_TRADEOFF = (
    "When outcome base rates differ across groups, a calibrated score generally cannot "
    "also satisfy equalized odds except in special cases. This audit exposes the "
    "trade-off; it does not claim that the model is fully fair."
)
EGFR_LESSON = (
    "Race is a social and political construct, not a biological correction factor. "
    "The removal of race coefficients from eGFR equations is a cautionary lesson: "
    "race is retained here only to audit inequity and is never used as a predictor."
)


def _as_binary(values: pd.Series | NDArray[np.int_]) -> NDArray[np.int_]:
    return np.asarray(values, dtype=np.int_).reshape(-1)


def _safe_discrimination(
    y_true: NDArray[np.int_], y_probability: NDArray[np.float64]
) -> tuple[float | None, float | None]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None, None
    return (
        float(roc_auc_score(y_true, y_probability)),
        float(average_precision_score(y_true, y_probability)),
    )


def _group_calibration(
    y_true: NDArray[np.int_], y_probability: NDArray[np.float64]
) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"brier": None, "ece": None, "slope": None, "intercept": None}
    slope: float | None = None
    intercept: float | None = None
    if len(np.unique(y_true)) >= 2:
        slope, intercept = calibration_slope_intercept(y_true, y_probability)
    return {
        "brier": float(brier_score_loss(y_true, y_probability)),
        "ece": expected_calibration_error(y_true, y_probability),
        "slope": slope,
        "intercept": intercept,
    }


def _group_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_probability: NDArray[np.float64],
    groups: pd.Series,
) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    normalized_groups = groups.astype("string").fillna("Missing")
    for group in sorted(str(value) for value in normalized_groups.unique()):
        mask = np.asarray(normalized_groups == group)
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        group_probability = y_probability[mask]
        auroc, auprc = _safe_discrimination(group_true, group_probability)
        metrics[group] = {
            "n": len(group_true),
            "prevalence": float(np.mean(group_true)),
            "selection_rate": float(np.mean(group_pred)),
            "true_positive_rate": float(true_positive_rate(group_true, group_pred)),
            "false_positive_rate": float(false_positive_rate(group_true, group_pred)),
            "auroc": auroc,
            "auprc": auprc,
            **_group_calibration(group_true, group_probability),
        }
    return metrics


def _metric_frame_differences(
    y_true: NDArray[np.int_], y_pred: NDArray[np.int_], groups: pd.Series
) -> dict[str, float]:
    frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    )
    differences = frame.difference(method="between_groups")
    return {
        "demographic_parity_difference": float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
        ),
        "demographic_parity_ratio": float(
            demographic_parity_ratio(y_true, y_pred, sensitive_features=groups)
        ),
        "equalized_odds_difference": float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=groups)
        ),
        "equal_opportunity_difference": float(
            cast(Any, differences["true_positive_rate"])
        ),
    }


def _gap(
    group_metrics: dict[str, dict[str, float | int | None]], metric: str
) -> float | None:
    values: list[float] = []
    for metrics in group_metrics.values():
        value = metrics.get(metric)
        if isinstance(value, int | float):
            values.append(float(value))
    return float(max(values) - min(values)) if len(values) >= 2 else None


def audit_subgroups(
    y_true: pd.Series | NDArray[np.int_],
    y_probability: NDArray[np.float64],
    sensitive_features: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Audit each sensitive feature with Fairlearn and clinical metrics."""

    true = _as_binary(y_true)
    probability = np.asarray(y_probability, dtype=np.float64).reshape(-1)
    if len(true) != len(probability) or len(true) != len(sensitive_features):
        msg = "Targets, probabilities, and sensitive features must have equal length"
        raise ValueError(msg)
    prediction = (probability >= threshold).astype(np.int_)

    feature_reports: dict[str, Any] = {}
    for feature in sensitive_features.columns:
        groups = cast(
            pd.Series, sensitive_features[feature].astype("string").fillna("Missing")
        )
        per_group = _group_metrics(true, prediction, probability, groups)
        feature_reports[str(feature)] = {
            "fairlearn_disparities": _metric_frame_differences(
                true, prediction, groups
            ),
            "subgroup_auroc_gap": _gap(per_group, "auroc"),
            "subgroup_auprc_gap": _gap(per_group, "auprc"),
            "calibration_ece_gap": _gap(per_group, "ece"),
            "groups": per_group,
        }

    overall_auroc, overall_auprc = _safe_discrimination(true, probability)
    return {
        "audit_scope": {
            "sensitive_features_audit_only": [
                str(column) for column in sensitive_features.columns
            ],
            "sensitive_features_used_as_predictors": False,
            "threshold": threshold,
        },
        "overall": {
            "n": len(true),
            "prevalence": float(np.mean(true)),
            "auroc": overall_auroc,
            "auprc": overall_auprc,
            **_group_calibration(true, probability),
        },
        "features": feature_reports,
        "impossibility_tradeoff": IMPOSSIBILITY_TRADEOFF,
        "egfr_race_lesson": EGFR_LESSON,
    }


def plot_subgroup_auroc(report: dict[str, Any], output_path: str | Path) -> Path:
    """Plot subgroup AUROC values, omitting groups without both outcome classes."""

    labels: list[str] = []
    values: list[float] = []
    features = report.get("features", {})
    if isinstance(features, dict):
        for feature, feature_report in features.items():
            groups = feature_report.get("groups", {})
            if not isinstance(groups, dict):
                continue
            for group, metrics in groups.items():
                auroc = metrics.get("auroc")
                if isinstance(auroc, int | float):
                    labels.append(f"{feature}: {group}")
                    values.append(float(auroc))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, max(4.0, 0.3 * len(labels))))
    positions = np.arange(len(labels))
    axis.barh(positions, values, color="#167d8d")
    axis.set_yticks(positions, labels=labels)
    axis.set_xlim(0.0, 1.0)
    axis.axvline(0.5, color="#a33b20", linestyle="--", label="Chance")
    axis.set_xlabel("AUROC")
    axis.set_title("Subgroup discrimination audit (audit-only attributes)")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return destination


def fairness_audit_phase6(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    figure_path: str | Path = DEFAULT_FIGURE_PATH,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> Path:
    """Run and persist the Phase 6 fairness audit."""

    params = load_yaml(params_path)
    spec = feature_spec_from_params(params)
    fairness_params = params.get("fairness", {})
    model_params = params.get("model", {})
    if not isinstance(fairness_params, dict) or not isinstance(model_params, dict):
        raise TypeError("fairness and model params must be mappings")
    requested = [
        str(feature)
        for feature in fairness_params.get("sensitive_features", spec.audit_only)
    ]
    invalid = sorted(set(requested).intersection(spec.model_feature_columns))
    if invalid:
        msg = "Sensitive audit-only features leaked into model features: " + ", ".join(
            invalid
        )
        raise RuntimeError(msg)

    test_df = pd.read_parquet(test_path)
    missing = sorted(set(requested).difference(test_df.columns))
    if missing:
        raise ValueError("Missing sensitive audit columns: " + ", ".join(missing))
    x_test, y_test = split_xy(test_df, spec)
    gate = joblib.load(model_path)
    if not isinstance(gate, ConformalGate):
        raise TypeError("Expected a ConformalGate artifact")
    probability = gate.estimator.predict_positive_proba(x_test)
    report = audit_subgroups(
        y_test,
        probability,
        test_df.loc[:, requested],
        threshold=float(fairness_params.get("threshold", 0.5)),
    )
    report["provenance"] = {
        "model_sha256": sha256_file(model_path),
        "test_data_sha256": sha256_file(test_path),
    }

    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_subgroup_auroc(report, figure_path)

    configure_mlflow(str(model_params.get("experiment_name", "MedMLOps-Lab")))
    with mlflow.start_run(run_name="phase6-fairness-audit"):
        mlflow.log_params(
            {
                "sensitive_features": ",".join(requested),
                "audit_only": True,
                "threshold": float(fairness_params.get("threshold", 0.5)),
            }
        )
        metrics = flatten_numeric_metrics(report, "fairness")
        mlflow.log_metrics(
            {
                re.sub(r"[^A-Za-z0-9_.:/ -]", "_", key)[:250]: value
                for key, value in metrics.items()
            }
        )
        mlflow.log_artifact(str(destination))
        mlflow.log_artifact(str(figure_path))
    return destination
