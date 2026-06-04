"""Evidently AI drift detection and reference snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.ui.workspace import Workspace

from medmlops.features.pipeline import (
    FeatureSpec,
    feature_spec_from_params,
    load_yaml,
)

DEFAULT_TRAIN_PATH = Path("data/processed/train.parquet")
DEFAULT_REFERENCE_PATH = Path("reports/drift/reference_snapshot.parquet")
DEFAULT_PARAMS_PATH = Path("params.yaml")


def monitoring_columns(spec: FeatureSpec) -> list[str]:
    """Return predictor and audit-only columns monitored for input drift."""

    return list(dict.fromkeys((*spec.model_feature_columns, *spec.audit_only)))


def prepare_monitoring_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Select and normalize a dataframe for Evidently."""

    columns = monitoring_columns(spec)
    missing = sorted(column for column in columns if column not in df.columns)
    if missing:
        msg = "Missing monitoring columns: " + ", ".join(missing)
        raise ValueError(msg)

    output = df.loc[:, columns].copy()
    for column in spec.numeric:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    categorical = set(spec.model_categorical).union(spec.audit_only)
    for column in categorical:
        if column in output.columns:
            output[column] = output[column].astype("object").fillna("?")
    return output


def _data_definition(frame: pd.DataFrame) -> DataDefinition:
    numerical = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical = [
        str(column) for column in frame.columns if str(column) not in numerical
    ]
    return DataDefinition(
        numerical_columns=numerical,
        categorical_columns=categorical,
    )


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> Any:
    """Build and run an Evidently drift report with auto-generated tests."""

    common = [column for column in reference_df.columns if column in current_df.columns]
    if not common:
        msg = "Reference and current data must share at least one column"
        raise ValueError(msg)
    reference = reference_df.loc[:, common].copy()
    current = current_df.loc[:, common].copy()
    definition = _data_definition(reference)
    reference_dataset = Dataset.from_pandas(reference, data_definition=definition)
    current_dataset = Dataset.from_pandas(current, data_definition=definition)
    report = Report(
        [DataDriftPreset(), DataSummaryPreset()],
        include_tests=True,
        tags=tags or [],
        metadata=cast(Any, metadata or {}),
    )
    return report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
        name=name,
    )


def build_drift_baseline(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    output_path: str | Path = DEFAULT_REFERENCE_PATH,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> Path:
    """Write the training-distribution snapshot used as the drift baseline."""

    params = load_yaml(params_path)
    spec = feature_spec_from_params(params)
    reference = prepare_monitoring_frame(pd.read_parquet(train_path), spec)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(destination, index=False)
    return destination


def save_evidently_snapshot(
    snapshot: Any,
    *,
    html_path: str | Path,
    json_path: str | Path | None = None,
) -> Path:
    """Persist an Evidently snapshot as HTML and optionally JSON."""

    html_destination = Path(html_path)
    html_destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_destination))
    if json_path is not None:
        json_destination = Path(json_path)
        json_destination.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_json(str(json_destination))
    return html_destination


def snapshot_dict(snapshot: Any) -> dict[str, Any]:
    """Return a typed dictionary representation of an Evidently snapshot."""

    return cast(dict[str, Any], snapshot.dict())


def log_drift_snapshot_to_workspace(
    snapshot: Any,
    *,
    workspace_path: str | Path,
    project_name: str = "MedMLOps synthetic drift demo",
    run_name: str | None = None,
) -> str:
    """Log a drift snapshot into a reusable local Evidently project."""

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
            description="Clearly labeled synthetic three-regime drift demonstrations.",
        )
    workspace.add_run(project.id, snapshot, include_data=False, name=run_name)
    return str(project.id)
