"""Run Diabetes 130 data contracts and write validated artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from medmlops.data.contracts import (
    BINARY_TARGET_COLUMN,
    TARGET_COLUMN,
    binarize_readmission,
    readmission_positive_rate,
    validate_diabetes130_frame,
)
from medmlops.data.ingest import sha256_file

DEFAULT_INPUT_PATH = Path("data/raw/diabetes130.parquet")
DEFAULT_OUTPUT_PATH = Path("data/interim/validated.parquet")
DEFAULT_METRICS_PATH = Path("reports/data_quality.json")


def read_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a CSV or parquet dataframe."""

    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix == ".csv":
        return pd.read_csv(source, keep_default_na=False)
    msg = f"Unsupported tabular input format: {source.suffix}"
    raise ValueError(msg)


def build_data_quality_report(
    df: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build DVC metrics for the validated dataset."""

    positive_rate = readmission_positive_rate(df, target_col=TARGET_COLUMN)
    string_columns = [
        column
        for column in df.columns
        if pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ]
    report: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "target_col": TARGET_COLUMN,
        "binary_target_col": BINARY_TARGET_COLUMN,
        "positive_class": "<30",
        "positive_rate_30d": positive_rate,
        "readmitted_counts": {
            str(key): int(value)
            for key, value in df[TARGET_COLUMN].value_counts(dropna=False).items()
        },
        "missing_token_counts": {
            str(column): int((df[column] == "?").sum()) for column in string_columns
        },
    }
    if source_path is not None:
        report["source_sha256"] = sha256_file(source_path)
    return report


def validate_diabetes130(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    *,
    check_statistics: bool = True,
) -> Path:
    """Validate raw Diabetes 130 data and persist the Phase 1 artifact."""

    source = Path(input_path)
    destination = Path(output_path)
    metrics_destination = Path(metrics_path)

    df = read_dataframe(source)
    validated = validate_diabetes130_frame(df, check_statistics=check_statistics)
    validated = binarize_readmission(validated)

    destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)

    validated.to_parquet(destination, index=False)
    report = build_data_quality_report(validated, source_path=source)
    metrics_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
