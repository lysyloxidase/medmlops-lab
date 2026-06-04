"""Delayed-label performance monitoring from the prediction audit log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset
from evidently.ui.workspace import Workspace
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sqlalchemy import create_engine, text

from medmlops.calibration.calibrate import (
    calibration_slope_intercept,
    expected_calibration_error,
)
from medmlops.metrics.clinical import net_benefit
from medmlops.serving.audit import PredictionAuditLogger

DEFAULT_REPORT_PATH = Path("reports/performance_monitoring.json")
DEFAULT_HTML_PATH = Path("reports/drift/delayed_label_performance.html")
DEFAULT_WORKSPACE_PATH = Path("reports/evidently-workspace")
DEFAULT_PROJECT_NAME = "MedMLOps delayed-label performance"
LABELED_PREDICTION_COLUMNS = [
    "request_id",
    "ts",
    "model_version",
    "model_alias",
    "risk_probability",
    "risk_class",
    "abstain",
    "ground_truth",
]


def load_labeled_predictions(database_url: str) -> pd.DataFrame:
    """Query the labeled subset of the Phase 4 prediction audit table."""

    engine = create_engine(database_url)
    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                    SELECT
                        request_id,
                        ts,
                        model_version,
                        model_alias,
                        risk_probability,
                        risk_class,
                        abstain,
                        ground_truth
                    FROM predictions
                    WHERE ground_truth IS NOT NULL
                    ORDER BY ts
                    """
                )
            )
            .mappings()
            .all()
        )
    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=pd.Index(LABELED_PREDICTION_COLUMNS),
    )


def join_delayed_ground_truth(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join later-arriving labels onto prediction records by request_id."""

    required = {"request_id", "ground_truth"}
    if not required.issubset(labels.columns):
        msg = "labels must contain request_id and ground_truth"
        raise ValueError(msg)
    prediction_source = predictions.drop(columns=["ground_truth"], errors="ignore")
    return prediction_source.merge(
        labels.loc[:, ["request_id", "ground_truth"]],
        on="request_id",
        how="inner",
        validate="one_to_one",
    )


def write_ground_truth_updates(database_url: str, labels: pd.DataFrame) -> int:
    """Attach delayed ground truth to existing prediction audit rows."""

    required = {"request_id", "ground_truth"}
    if not required.issubset(labels.columns):
        msg = "labels must contain request_id and ground_truth"
        raise ValueError(msg)
    engine = create_engine(database_url)
    updated = 0
    with engine.begin() as connection:
        for row in labels.loc[:, ["request_id", "ground_truth"]].to_dict("records"):
            result = connection.execute(
                text(
                    """
                    UPDATE predictions
                    SET ground_truth = :ground_truth
                    WHERE request_id = :request_id
                    """
                ),
                {
                    "request_id": str(row["request_id"]),
                    "ground_truth": int(row["ground_truth"]),
                },
            )
            updated += int(result.rowcount or 0)
    return updated


def delayed_label_metrics(
    labeled_predictions: pd.DataFrame,
    *,
    auroc_floor: float = 0.60,
    clinical_threshold: float = 0.10,
) -> dict[str, Any]:
    """Recompute clinical performance on the currently labeled subset."""

    required = {"ground_truth", "risk_probability"}
    if not required.issubset(labeled_predictions.columns):
        msg = "labeled predictions must contain ground_truth and risk_probability"
        raise ValueError(msg)
    y_true = labeled_predictions["ground_truth"].to_numpy(dtype=np.int_)
    y_probability = labeled_predictions["risk_probability"].to_numpy(dtype=np.float64)
    if len(y_true) == 0:
        return {
            "status": "INSUFFICIENT_LABELS",
            "alert": False,
            "labeled_count": 0,
            "auroc_floor": auroc_floor,
        }
    if len(np.unique(y_true)) < 2:
        return {
            "status": "INSUFFICIENT_CLASSES",
            "alert": False,
            "labeled_count": len(y_true),
            "auroc_floor": auroc_floor,
        }

    auroc = float(roc_auc_score(y_true, y_probability))
    slope, intercept = calibration_slope_intercept(y_true, y_probability)
    alert = auroc < auroc_floor
    return {
        "status": "ALERT" if alert else "OK",
        "alert": alert,
        "alert_reason": "auroc_below_floor" if alert else None,
        "labeled_count": len(y_true),
        "positive_rate": float(np.mean(y_true)),
        "auroc": auroc,
        "auroc_floor": auroc_floor,
        "auprc": float(average_precision_score(y_true, y_probability)),
        "brier": float(brier_score_loss(y_true, y_probability)),
        "ece": float(expected_calibration_error(y_true, y_probability)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "clinical_threshold": clinical_threshold,
        "net_benefit": float(net_benefit(y_true, y_probability, clinical_threshold)),
    }


def build_performance_report(labeled_predictions: pd.DataFrame) -> Any:
    """Build an Evidently classification report for delayed-label performance."""

    frame = pd.DataFrame(
        {
            "target": labeled_predictions["ground_truth"].astype(int).astype(str),
            "prediction": labeled_predictions["risk_probability"].astype(float),
            "prediction_label": (
                labeled_predictions["risk_probability"].astype(float) >= 0.5
            )
            .astype(int)
            .astype(str),
        }
    )
    definition = DataDefinition(
        classification=cast(
            Any,
            [
                BinaryClassification(
                    target="target",
                    prediction_labels="prediction_label",
                    prediction_probas="prediction",
                    pos_label="1",
                )
            ],
        ),
        categorical_columns=["target", "prediction_label"],
        numerical_columns=["prediction"],
    )
    dataset = Dataset.from_pandas(frame, data_definition=definition)
    report = Report(
        [ClassificationPreset()],
        include_tests=True,
        tags=["delayed-label", "retrospective"],
        metadata={
            "caveat": "Performance drift is detectable only retrospectively.",
        },
    )
    return report.run(current_data=dataset, reference_data=None)


def log_snapshot_to_workspace(
    snapshot: Any,
    *,
    workspace_path: str | Path = DEFAULT_WORKSPACE_PATH,
    project_name: str = DEFAULT_PROJECT_NAME,
) -> str:
    """Log an Evidently snapshot into a local self-hosted workspace."""

    workspace = Workspace.create(str(workspace_path))
    project = next(
        (
            candidate
            for candidate in workspace.list_projects()
            if candidate.name == project_name
        ),
        None,
    )
    if project is None:
        project = workspace.create_project(
            project_name,
            description="Retrospective performance monitoring with delayed labels.",
        )
    workspace.add_run(project.id, snapshot, include_data=False)
    return str(project.id)


def monitor_delayed_labels(
    database_url: str,
    *,
    auroc_floor: float = 0.60,
    clinical_threshold: float = 0.10,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    html_path: str | Path = DEFAULT_HTML_PATH,
    workspace_path: str | Path = DEFAULT_WORKSPACE_PATH,
) -> Path:
    """Run the nightly delayed-label monitoring job."""

    PredictionAuditLogger(database_url).initialize()
    labeled = load_labeled_predictions(database_url)
    metrics = delayed_label_metrics(
        labeled,
        auroc_floor=auroc_floor,
        clinical_threshold=clinical_threshold,
    )
    metrics["caveat"] = "Performance drift is detectable only retrospectively."

    if metrics["status"] not in {"INSUFFICIENT_LABELS", "INSUFFICIENT_CLASSES"}:
        snapshot = build_performance_report(labeled)
        html_destination = Path(html_path)
        html_destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_destination))
        metrics["evidently_project_id"] = log_snapshot_to_workspace(
            snapshot,
            workspace_path=workspace_path,
        )

    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def performance_report_dict(snapshot: Any) -> dict[str, Any]:
    """Return a typed dictionary representation of an Evidently snapshot."""

    return cast(dict[str, Any], snapshot.dict())
