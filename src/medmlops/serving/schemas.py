"""Pydantic v2 schemas for clinical serving."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator

from medmlops.data.contracts import (
    AGE_CATEGORIES,
    GENDER_CATEGORIES,
    MEDICATION_STATES,
    RACE_CATEGORIES,
)
from medmlops.features.pipeline import FeatureSpec, add_diagnosis_groups

MedicationState = Literal["No", "Steady", "Up", "Down"]
GlucoseResult = Literal["None", ">200", ">300", "Norm"]
A1CResult = Literal["None", ">7", ">8", "Norm"]
ChangeState = Literal["No", "Ch"]
DiabetesMedicationState = Literal["No", "Yes"]

MEDICATION_COLUMNS = (
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
)


class ClinicalEncounter(BaseModel):
    """Validated single-encounter payload.

    Race, gender, and age are accepted for audit logging and OOD/fairness
    monitoring, but the Phase 2 feature spec excludes them from model features.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    encounter_id: NonNegativeInt | None = None
    patient_nbr: NonNegativeInt | None = None
    race: str = Field(description="Audit-only race category")
    gender: str = Field(description="Audit-only gender category")
    age: str = Field(description="Audit-only age bracket")
    weight: str = "?"
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(ge=1, le=30)
    admission_source_id: int = Field(ge=1, le=25)
    time_in_hospital: int = Field(ge=1, le=14)
    payer_code: str = "?"
    medical_specialty: str = "?"
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=10)
    num_medications: int = Field(ge=1, le=100)
    number_outpatient: int = Field(ge=0, le=100)
    number_emergency: int = Field(ge=0, le=100)
    number_inpatient: int = Field(ge=0, le=100)
    diag_1: str
    diag_2: str
    diag_3: str
    number_diagnoses: int = Field(ge=1, le=16)
    max_glu_serum: GlucoseResult
    A1Cresult: A1CResult
    metformin: MedicationState
    repaglinide: MedicationState
    nateglinide: MedicationState
    chlorpropamide: MedicationState
    glimepiride: MedicationState
    acetohexamide: MedicationState
    glipizide: MedicationState
    glyburide: MedicationState
    tolbutamide: MedicationState
    pioglitazone: MedicationState
    rosiglitazone: MedicationState
    acarbose: MedicationState
    miglitol: MedicationState
    troglitazone: MedicationState
    tolazamide: MedicationState
    examide: MedicationState
    citoglipton: MedicationState
    insulin: MedicationState
    glyburide_metformin: MedicationState = Field(alias="glyburide-metformin")
    glipizide_metformin: MedicationState = Field(alias="glipizide-metformin")
    glimepiride_pioglitazone: MedicationState = Field(alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedicationState = Field(alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedicationState = Field(alias="metformin-pioglitazone")
    change: ChangeState
    diabetesMed: DiabetesMedicationState

    @field_validator("race")
    @classmethod
    def validate_race(cls, value: str) -> str:
        if value not in RACE_CATEGORIES:
            msg = f"race must be one of {RACE_CATEGORIES}"
            raise ValueError(msg)
        return value

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        if value not in GENDER_CATEGORIES:
            msg = f"gender must be one of {GENDER_CATEGORIES}"
            raise ValueError(msg)
        return value

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: str) -> str:
        if value not in AGE_CATEGORIES:
            msg = f"age must be one of {AGE_CATEGORIES}"
            raise ValueError(msg)
        return value

    @field_validator(*MEDICATION_COLUMNS, mode="before", check_fields=False)
    @classmethod
    def validate_medication_state(cls, value: object) -> object:
        if value not in MEDICATION_STATES:
            msg = f"medication state must be one of {MEDICATION_STATES}"
            raise ValueError(msg)
        return value

    @field_validator("diag_1", "diag_2", "diag_3", mode="before")
    @classmethod
    def normalize_diagnosis_code(cls, value: object) -> str:
        if value is None or value == "":
            msg = "diagnosis code must be non-empty"
            raise ValueError(msg)
        return str(value)

    @field_validator("diag_1", "diag_2", "diag_3", "weight", "payer_code")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value:
            msg = "value must be non-empty"
            raise ValueError(msg)
        return value

    def audit_payload(self) -> dict[str, Any]:
        """Return the original payload shape for append-only audit logging."""

        return self.model_dump(mode="json", by_alias=True)

    def to_model_frame(self, spec: FeatureSpec) -> pd.DataFrame:
        """Return a one-row Phase 2 model feature frame."""

        raw = pd.DataFrame([self.audit_payload()])
        with_groups = add_diagnosis_groups(raw)
        required_columns = list(spec.model_feature_columns)
        missing = sorted(
            column for column in required_columns if column not in with_groups
        )
        if missing:
            msg = "Missing model feature columns after serving transform: "
            raise ValueError(msg + ", ".join(missing))

        frame = with_groups.loc[:, required_columns].copy()
        for column in spec.numeric:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in spec.model_categorical:
            if column in frame.columns:
                frame[column] = frame[column].astype("string").fillna("?")
        return frame


class PredictionResponse(BaseModel):
    """Single-prediction API response."""

    request_id: UUID
    risk_probability: float = Field(ge=0.0, le=1.0)
    risk_class: int | None
    conformal_set: list[int]
    abstain: bool
    abstain_reason: str | None
    ood_flags: list[str]
    model_alias: str
    model_version: str
    data_hash: str
    latency_ms: float = Field(ge=0.0)


class BatchPredictionResponse(BaseModel):
    """Batch-prediction API response."""

    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Health-check response."""

    status: Literal["ok", "degraded"]
    conformal_loaded: bool
    registry_connected: bool
    audit_db_connected: bool
    ood_reference_loaded: bool
    detail: dict[str, str]


class ModelInfoResponse(BaseModel):
    """Registry and provenance metadata returned by /model-info."""

    model_name: str
    model_alias: str
    model_version: str
    challenger_alias: str | None
    challenger_fraction: float = Field(ge=0.0, le=1.0)
    data_hash: str
    training_metrics: dict[str, Any]
