"""Pandera data contracts for Diabetes 130-US Hospitals.

The contracts are intentionally code-first and lightweight. They validate known
columns, categorical domains, numeric ranges, readmission target values, leakage
feature exclusions, and the headline 30-day readmission-rate sanity band.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover - compatibility for older pandera builds
    import pandera as pa
from pandera.typing import Series

TARGET_COLUMN = "readmitted"
BINARY_TARGET_COLUMN = "readmitted_30d"
POSITIVE_CLASS = "<30"
POSITIVE_RATE_BOUNDS = (0.08, 0.14)

LEAKAGE_COLUMNS = ("encounter_id", "patient_nbr")

RACE_CATEGORIES = (
    "Caucasian",
    "AfricanAmerican",
    "Hispanic",
    "Asian",
    "Other",
    "?",
)
GENDER_CATEGORIES = ("Male", "Female", "Unknown/Invalid")
AGE_CATEGORIES = (
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
)
READMITTED_CATEGORIES = ("NO", ">30", "<30")
MEDICATION_STATES = ("No", "Steady", "Up", "Down")

EXPECTED_COLUMNS = (
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
    "readmitted",
)


class Diabetes130Schema(pa.DataFrameModel):  # pyright: ignore[reportGeneralTypeIssues]
    """Known Diabetes 130 schema fields.

    `strict = False` allows extra columns so Phase 2 can add derived fields
    without weakening Phase 1 raw-data checks.
    """

    encounter_id: Series[int] = pa.Field(ge=0)
    patient_nbr: Series[int] = pa.Field(ge=0)
    race: Series[str] = pa.Field(isin=RACE_CATEGORIES)
    gender: Series[str] = pa.Field(isin=GENDER_CATEGORIES)
    age: Series[str] = pa.Field(isin=AGE_CATEGORIES)
    admission_type_id: Series[int] = pa.Field(ge=1, le=8)
    discharge_disposition_id: Series[int] = pa.Field(ge=1, le=30)
    admission_source_id: Series[int] = pa.Field(ge=1, le=25)
    time_in_hospital: Series[int] = pa.Field(ge=1, le=14)
    num_lab_procedures: Series[int] = pa.Field(ge=0, le=200)
    num_procedures: Series[int] = pa.Field(ge=0, le=10)
    num_medications: Series[int] = pa.Field(ge=1, le=100)
    number_outpatient: Series[int] = pa.Field(ge=0, le=100)
    number_emergency: Series[int] = pa.Field(ge=0, le=100)
    number_inpatient: Series[int] = pa.Field(ge=0, le=100)
    number_diagnoses: Series[int] = pa.Field(ge=1, le=16)
    max_glu_serum: Series[str] = pa.Field(isin=("None", ">200", ">300", "Norm"))
    A1Cresult: Series[str] = pa.Field(isin=("None", ">7", ">8", "Norm"))
    metformin: Series[str] = pa.Field(isin=MEDICATION_STATES)
    repaglinide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    nateglinide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    chlorpropamide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    glimepiride: Series[str] = pa.Field(isin=MEDICATION_STATES)
    acetohexamide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    glipizide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    glyburide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    tolbutamide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    pioglitazone: Series[str] = pa.Field(isin=MEDICATION_STATES)
    rosiglitazone: Series[str] = pa.Field(isin=MEDICATION_STATES)
    acarbose: Series[str] = pa.Field(isin=MEDICATION_STATES)
    miglitol: Series[str] = pa.Field(isin=MEDICATION_STATES)
    troglitazone: Series[str] = pa.Field(isin=MEDICATION_STATES)
    tolazamide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    examide: Series[str] = pa.Field(isin=MEDICATION_STATES)
    citoglipton: Series[str] = pa.Field(isin=MEDICATION_STATES)
    insulin: Series[str] = pa.Field(isin=MEDICATION_STATES)
    glyburide_metformin: Series[str] = pa.Field(
        alias="glyburide-metformin", isin=MEDICATION_STATES
    )
    glipizide_metformin: Series[str] = pa.Field(
        alias="glipizide-metformin", isin=MEDICATION_STATES
    )
    glimepiride_pioglitazone: Series[str] = pa.Field(
        alias="glimepiride-pioglitazone", isin=MEDICATION_STATES
    )
    metformin_rosiglitazone: Series[str] = pa.Field(
        alias="metformin-rosiglitazone", isin=MEDICATION_STATES
    )
    metformin_pioglitazone: Series[str] = pa.Field(
        alias="metformin-pioglitazone", isin=MEDICATION_STATES
    )
    change: Series[str] = pa.Field(isin=("No", "Ch"))
    diabetesMed: Series[str] = pa.Field(isin=("No", "Yes"))
    readmitted: Series[str] = pa.Field(isin=READMITTED_CATEGORIES)

    class Config:
        strict = False
        coerce = True


def missing_expected_columns(df: pd.DataFrame) -> list[str]:
    """Return required raw columns absent from a dataframe."""

    return [column for column in EXPECTED_COLUMNS if column not in df.columns]


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise when the raw Diabetes 130 column set is incomplete."""

    missing = missing_expected_columns(df)
    if missing:
        msg = "Missing required Diabetes 130 columns: " + ", ".join(missing)
        raise ValueError(msg)


def assert_no_feature_leakage(feature_columns: Iterable[str]) -> None:
    """Ensure encounter and patient identifiers are not used as predictors."""

    leakage = sorted(set(feature_columns).intersection(LEAKAGE_COLUMNS))
    if leakage:
        msg = "Leakage columns must not be predictors: " + ", ".join(leakage)
        raise ValueError(msg)


def binarize_readmission(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    output_col: str = BINARY_TARGET_COLUMN,
    positive_class: str = POSITIVE_CLASS,
) -> pd.DataFrame:
    """Add an integer 30-day readmission target column."""

    output = df.copy()
    output[output_col] = (output[target_col] == positive_class).astype("int8")
    return output


def readmission_positive_rate(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    positive_class: str = POSITIVE_CLASS,
) -> float:
    """Calculate the 30-day readmission positive rate."""

    if len(df) == 0:
        msg = "Cannot compute positive rate for an empty dataframe"
        raise ValueError(msg)
    return float((df[target_col] == positive_class).mean())


def validate_statistical_contracts(
    df: pd.DataFrame,
    positive_rate_bounds: Sequence[float] = POSITIVE_RATE_BOUNDS,
) -> None:
    """Validate statistical sanity checks for the raw cohort."""

    lower, upper = positive_rate_bounds
    rate = readmission_positive_rate(df)
    if not lower <= rate <= upper:
        msg = (
            "30-day readmission positive rate "
            f"{rate:.4f} is outside [{lower:.2f}, {upper:.2f}]"
        )
        raise ValueError(msg)


def validate_diabetes130_frame(
    df: pd.DataFrame,
    *,
    check_statistics: bool = True,
) -> pd.DataFrame:
    """Run all Phase 1 data contracts and return a coerced dataframe."""

    validate_required_columns(df)
    validated = Diabetes130Schema.validate(df, lazy=True)
    if check_statistics:
        validate_statistical_contracts(validated)
    return validated
