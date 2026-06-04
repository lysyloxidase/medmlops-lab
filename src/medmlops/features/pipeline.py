"""sklearn Pipeline and ColumnTransformer for clinical tabular features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from medmlops.data.contracts import BINARY_TARGET_COLUMN, TARGET_COLUMN
from medmlops.data.ids_mapping import group_icd9_diagnosis

DEFAULT_VALIDATED_PATH = Path("data/interim/validated.parquet")
DEFAULT_FEATURES_PATH = Path("data/processed/features.parquet")
DEFAULT_PARAMS_PATH = Path("params.yaml")
DEFAULT_SPLIT_INPUT_PATH = Path("data/processed/features.parquet")
DEFAULT_SPLIT_DIR = Path("data/processed")

DIAGNOSIS_COLUMNS = ("diag_1", "diag_2", "diag_3")
DIAGNOSIS_GROUP_COLUMNS = ("diag_1_group", "diag_2_group", "diag_3_group")


@dataclass(frozen=True)
class FeatureSpec:
    """Column groups used by the Phase 2 feature matrix."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    audit_only: tuple[str, ...]
    drop_leakage: tuple[str, ...]
    target_col: str
    binary_target_col: str

    @property
    def model_feature_columns(self) -> tuple[str, ...]:
        excluded = set(self.audit_only).union(self.drop_leakage)
        return tuple(
            column
            for column in (*self.numeric, *self.categorical)
            if column not in excluded
        )

    @property
    def model_categorical(self) -> tuple[str, ...]:
        return tuple(
            column
            for column in self.categorical
            if column in self.model_feature_columns
        )


def load_yaml(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load project YAML as a mapping."""

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        msg = f"Expected YAML mapping at {path}"
        raise TypeError(msg)
    return data


def feature_spec_from_params(params: dict[str, Any]) -> FeatureSpec:
    """Build a feature specification from `params.yaml`."""

    features = params.get("features", {})
    data = params.get("data", {})
    if not isinstance(features, dict) or not isinstance(data, dict):
        msg = "params.yaml must contain mapping sections for features and data"
        raise TypeError(msg)

    return FeatureSpec(
        numeric=tuple(str(column) for column in features.get("numeric", [])),
        categorical=tuple(str(column) for column in features.get("categorical", [])),
        audit_only=tuple(str(column) for column in features.get("audit_only", [])),
        drop_leakage=tuple(str(column) for column in features.get("drop_leakage", [])),
        target_col=str(data.get("target_col", TARGET_COLUMN)),
        binary_target_col=str(data.get("binary_target_col", BINARY_TARGET_COLUMN)),
    )


def add_diagnosis_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add coarse ICD-9 diagnosis groups used by the model matrix."""

    output = df.copy()
    for source, destination in zip(
        DIAGNOSIS_COLUMNS, DIAGNOSIS_GROUP_COLUMNS, strict=True
    ):
        if source in output.columns:
            output[destination] = output[source].map(group_icd9_diagnosis)
    return output


def prepare_feature_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Return the cleaned Phase 2 feature source table."""

    output = add_diagnosis_groups(df)
    required = set(spec.model_feature_columns).union(
        spec.audit_only, {spec.target_col, spec.binary_target_col}
    )
    missing = sorted(column for column in required if column not in output.columns)
    if missing:
        msg = "Missing required feature columns: " + ", ".join(missing)
        raise ValueError(msg)

    selected_columns = [
        *spec.model_feature_columns,
        *spec.audit_only,
        spec.target_col,
        spec.binary_target_col,
    ]
    selected = output.loc[:, selected_columns].copy()

    for column in spec.numeric:
        if column in selected.columns:
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
    for column in spec.model_categorical:
        if column in selected.columns:
            selected[column] = selected[column].astype("string").fillna("?")

    return selected


def build_feature_transformer(spec: FeatureSpec) -> ColumnTransformer:
    """Build the sklearn preprocessing transformer for model features."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(spec.numeric)),
            ("categorical", categorical_pipeline, list(spec.model_categorical)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def split_xy(
    df: pd.DataFrame,
    spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split model feature source columns from the binary target."""

    return (
        df.loc[:, list(spec.model_feature_columns)],
        df.loc[:, spec.binary_target_col].astype("int8"),
    )


def get_transformed_feature_names(transformer: ColumnTransformer) -> list[str]:
    """Return stable transformed feature names for model interpretation."""

    return [str(name) for name in transformer.get_feature_names_out()]


def ensure_sparse_matrix(matrix: object) -> spmatrix:
    """Return a scipy sparse matrix from sklearn transformer output."""

    return sparse.csr_matrix(matrix)


def preprocess_features(
    input_path: str | Path = DEFAULT_VALIDATED_PATH,
    output_path: str | Path = DEFAULT_FEATURES_PATH,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> Path:
    """Write the cleaned Phase 2 feature source parquet."""

    params = load_yaml(params_path)
    spec = feature_spec_from_params(params)
    df = pd.read_parquet(input_path)
    features = prepare_feature_frame(df, spec)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
    return destination


def _split_once(
    df: pd.DataFrame,
    *,
    test_size: float,
    target_col: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one deterministic stratified split."""

    left_raw, right_raw = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=df[target_col],
    )
    left = cast(pd.DataFrame, left_raw)
    right = cast(pd.DataFrame, right_raw)
    return left.sort_index(), right.sort_index()


def deterministic_split(
    df: pd.DataFrame,
    *,
    target_col: str = BINARY_TARGET_COLUMN,
    validation_size: float = 0.1,
    calibration_size: float = 0.1,
    conformal_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create deterministic train/validation/calibration/conformal/test splits."""

    split_sizes = {
        "validation_size": validation_size,
        "calibration_size": calibration_size,
        "conformal_size": conformal_size,
        "test_size": test_size,
    }
    if sum(split_sizes.values()) >= 1.0:
        msg = "Holdout split sizes must sum to less than 1.0"
        raise ValueError(msg)

    train_pool, test = _split_once(
        df, test_size=test_size, target_col=target_col, seed=seed
    )
    remaining_fraction = 1.0 - test_size

    train_pool, conformal = _split_once(
        train_pool,
        test_size=conformal_size / remaining_fraction,
        target_col=target_col,
        seed=seed + 1,
    )
    remaining_fraction -= conformal_size

    train_pool, calib = _split_once(
        train_pool,
        test_size=calibration_size / remaining_fraction,
        target_col=target_col,
        seed=seed + 2,
    )
    remaining_fraction -= calibration_size

    train, val = _split_once(
        train_pool,
        test_size=validation_size / remaining_fraction,
        target_col=target_col,
        seed=seed + 3,
    )

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "calib": calib.reset_index(drop=True),
        "conformal": conformal.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def write_splits(
    input_path: str | Path = DEFAULT_SPLIT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_SPLIT_DIR,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> dict[str, Path]:
    """Write deterministic split parquet files for the DVC split stage."""

    params = load_yaml(params_path)
    data = params.get("data", {})
    if not isinstance(data, dict):
        msg = "params.yaml data section must be a mapping"
        raise TypeError(msg)

    seed = int(params.get("seed", 42))
    target_col = str(data.get("binary_target_col", BINARY_TARGET_COLUMN))
    splits = deterministic_split(
        pd.read_parquet(input_path),
        target_col=target_col,
        validation_size=float(data.get("validation_size", 0.1)),
        calibration_size=float(data.get("calibration_size", 0.1)),
        conformal_size=float(data.get("conformal_size", 0.1)),
        test_size=float(data.get("test_size", 0.2)),
        seed=seed,
    )

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, split_df in splits.items():
        path = destination_dir / f"{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        paths[split_name] = path
    return paths
