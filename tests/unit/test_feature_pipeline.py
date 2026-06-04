from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import sparse

from medmlops.data.contracts import binarize_readmission
from medmlops.features.pipeline import (
    build_feature_transformer,
    deterministic_split,
    ensure_sparse_matrix,
    feature_spec_from_params,
    load_yaml,
    prepare_feature_frame,
    split_xy,
)


def test_transformer_excludes_audit_columns_and_handles_unknown_categories(
    sample_diabetes130: pd.DataFrame,
) -> None:
    params = load_yaml(Path("params.yaml"))
    spec = feature_spec_from_params(params)
    frame = prepare_feature_frame(binarize_readmission(sample_diabetes130), spec)
    x, _y = split_xy(frame, spec)

    transformer = build_feature_transformer(spec)
    matrix = ensure_sparse_matrix(transformer.fit_transform(x))

    assert sparse.issparse(matrix)
    assert "race" not in spec.model_feature_columns
    assert "gender" not in spec.model_feature_columns
    assert "age" not in spec.model_feature_columns

    unknown = x.iloc[[0]].copy()
    unknown.loc[:, "payer_code"] = "NEVER_SEEN"
    transformed_unknown = ensure_sparse_matrix(transformer.transform(unknown))
    assert transformed_unknown.get_shape()[1] == matrix.get_shape()[1]


def test_diagnosis_groups_are_created(sample_diabetes130: pd.DataFrame) -> None:
    params = load_yaml(Path("params.yaml"))
    spec = feature_spec_from_params(params)
    frame = prepare_feature_frame(binarize_readmission(sample_diabetes130), spec)

    assert {"diag_1_group", "diag_2_group", "diag_3_group"}.issubset(frame.columns)
    assert "diag_1" not in spec.model_feature_columns


def test_deterministic_split_is_stable() -> None:
    df = pd.DataFrame(
        {
            "feature": range(200),
            "readmitted_30d": [0, 1] * 100,
        }
    )

    first = deterministic_split(df, seed=123)
    second = deterministic_split(df, seed=123)

    assert first.keys() == second.keys()
    for split_name in first:
        pd.testing.assert_frame_equal(first[split_name], second[split_name])
    assert len(first["test"]) == 40
    assert len(first["train"]) == 100
