from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import yaml
from pytest import MonkeyPatch

from medmlops.calibration.calibrate import (
    CalibratedRiskModel,
    MLPConfig,
    calibrate_phase3,
    calibration_slope_intercept,
    expected_calibration_error,
)
from medmlops.conformal.conformal import (
    ConformalGate,
    conformal_quantile,
    conformalize_phase3,
    coverage_rate,
)
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
from medmlops.metrics.clinical import decision_curve, evaluate_phase3, net_benefit
from medmlops.models.tabular_mlp import train_tabular_mlp


class ConstantEstimator:
    classes_ = np.array([0, 1], dtype=np.int_)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> ConstantEstimator:
        del x, y
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([[0.5, 0.5]]), (len(x), 1))

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.ones(len(x), dtype=np.int_)


def test_calibration_metrics_helpers_are_finite() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int_)
    probability = np.array([0.05, 0.2, 0.8, 0.95])

    slope, intercept = calibration_slope_intercept(y, probability)

    assert expected_calibration_error(y, probability, n_bins=2) < 0.2
    assert slope > 0
    assert np.isfinite(intercept)


def test_conformal_gate_abstains_on_ambiguous_sets() -> None:
    x = pd.DataFrame({"a": [0, 1, 2, 3]})
    y = np.array([0, 1, 0, 1], dtype=np.int_)
    gate = ConformalGate(cast(CalibratedRiskModel, ConstantEstimator()), alpha=0.1)

    gate.fit(x, y)
    result = gate.predict_with_abstention(x.iloc[[0]])

    assert conformal_quantile(np.array([0.1, 0.2, 0.3]), alpha=0.1) == 0.3
    assert coverage_rate([[0, 1], [1]], np.array([0, 1])) == 1.0
    assert result["abstain"] is True
    assert result["reason"] == "ambiguous_set"


def test_decision_curve_net_benefit() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int_)
    probability = np.array([0.1, 0.3, 0.6, 0.9])
    curve = decision_curve(y, probability, [0.2, 0.5])

    assert net_benefit(y, probability, 0.5) == 0.5
    assert curve["treat_none"] == [0.0, 0.0]
    assert len(curve["model"]) == 2


def test_phase3_small_flow(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
    monkeypatch: MonkeyPatch,
) -> None:
    params = load_yaml(Path("params.yaml"))
    params["model"]["experiment_name"] = "unit-phase3"
    params["tracking"]["register_model"] = False
    params["calibration"]["max_ece"] = 1.0
    params["calibration"]["method"] = "isotonic"

    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    spec = feature_spec_from_params(params)
    raw = pd.concat([sample_diabetes130] * 12, ignore_index=True)
    frame = prepare_feature_frame(binarize_readmission(raw), spec)
    splits = deterministic_split(frame, seed=7)
    for name, split_df in splits.items():
        split_df.to_parquet(tmp_path / f"{name}.parquet", index=False)

    train_df = splits["train"]
    x_train_raw, y_train_series = split_xy(train_df, spec)
    transformer = build_feature_transformer(spec)
    x_train = ensure_sparse_matrix(transformer.fit_transform(x_train_raw))
    y_train = y_train_series.to_numpy(dtype=np.int_)
    mlp_config = MLPConfig(hidden=(4,), dropout=0.0, epochs=1, batch_size=16, seed=3)
    hero = train_tabular_mlp(x_train, y_train, x_train, y_train, mlp_config)
    hero_path = tmp_path / "hero.pt"
    torch.save(
        {
            "state_dict": hero.state_dict(),
            "config": mlp_config,
            "n_features": x_train.shape[1],
            "feature_names": transformer.get_feature_names_out().tolist(),
            "transformer": transformer,
        },
        hero_path,
    )

    calibrate_phase3(
        hero_path=hero_path,
        calib_path=tmp_path / "calib.parquet",
        val_path=tmp_path / "val.parquet",
        model_path=tmp_path / "calibrated.pkl",
        metrics_path=tmp_path / "calibration_metrics.json",
        figure_path=tmp_path / "reliability.png",
        params_path=params_path,
    )
    conformalize_phase3(
        calibrated_path=tmp_path / "calibrated.pkl",
        conformal_path=tmp_path / "conformal.parquet",
        test_path=tmp_path / "test.parquet",
        model_path=tmp_path / "conformal.pkl",
        metrics_path=tmp_path / "conformal_metrics.json",
        params_path=params_path,
    )
    evaluate_phase3(
        conformal_model_path=tmp_path / "conformal.pkl",
        test_path=tmp_path / "test.parquet",
        metrics_path=tmp_path / "clinical_metrics.json",
        figure_path=tmp_path / "decision_curve.png",
        params_path=params_path,
    )

    clinical = yaml.safe_load((tmp_path / "clinical_metrics.json").read_text())
    assert "auprc" in clinical
    assert (tmp_path / "reliability.png").exists()
    assert (tmp_path / "decision_curve.png").exists()
