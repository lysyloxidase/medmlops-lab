from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from medmlops.data.contracts import (
    BINARY_TARGET_COLUMN,
    assert_no_feature_leakage,
    binarize_readmission,
    readmission_positive_rate,
    validate_diabetes130_frame,
)
from medmlops.data.validate import validate_diabetes130


def test_sample_fixture_satisfies_schema_and_statistical_contract(
    sample_diabetes130: pd.DataFrame,
) -> None:
    validated = validate_diabetes130_frame(sample_diabetes130)

    assert len(validated) == 10
    assert readmission_positive_rate(validated) == pytest.approx(0.1)


def test_binarize_readmission_adds_int_target(
    sample_diabetes130: pd.DataFrame,
) -> None:
    output = binarize_readmission(sample_diabetes130)

    assert BINARY_TARGET_COLUMN in output.columns
    assert output[BINARY_TARGET_COLUMN].sum() == 1


def test_invalid_gender_fails_contract(sample_diabetes130: pd.DataFrame) -> None:
    invalid = sample_diabetes130.copy()
    invalid.loc[0, "gender"] = "Prefer not to say"

    with pytest.raises(SchemaErrors):
        validate_diabetes130_frame(invalid)


def test_feature_leakage_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="Leakage columns"):
        assert_no_feature_leakage(["encounter_id", "time_in_hospital"])


def test_validation_writes_parquet_and_metrics(
    tmp_path: Path, sample_diabetes130_path: Path
) -> None:
    output_path = tmp_path / "validated.parquet"
    metrics_path = tmp_path / "data_quality.json"

    validate_diabetes130(
        input_path=sample_diabetes130_path,
        output_path=output_path,
        metrics_path=metrics_path,
    )

    assert output_path.exists()
    assert metrics_path.exists()
    validated = pd.read_parquet(output_path)
    assert BINARY_TARGET_COLUMN in validated.columns
