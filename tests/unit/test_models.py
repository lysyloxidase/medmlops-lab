from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pytest import MonkeyPatch

from medmlops.data.contracts import binarize_readmission
from medmlops.data.ids_mapping import group_icd9_diagnosis
from medmlops.features.pipeline import (
    deterministic_split,
    feature_spec_from_params,
    load_yaml,
    prepare_feature_frame,
)
from medmlops.models.baseline import GBDTBaseline, GBDTBaselineConfig
from medmlops.models.tabular_mlp import (
    MLPConfig,
    predict_proba_mlp,
    train_tabular_mlp,
)
from medmlops.models.tracking import flatten_params, sha256_file
from medmlops.models.train import (
    classification_metrics,
    expected_calibration_error,
    train_phase2,
)


def test_baseline_fallback_fit_predicts_probabilities() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1])
    baseline = GBDTBaseline(
        GBDTBaselineConfig(backend="sklearn", n_estimators=3, max_depth=2)
    )

    baseline.fit(x, y)
    probabilities = baseline.predict_proba(x)

    assert baseline.effective_backend == "sklearn_hist_gradient_boosting"
    assert probabilities.shape == (6,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_tabular_mlp_trains_and_predicts_probabilities() -> None:
    x = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1])
    config = MLPConfig(hidden=(4,), dropout=0.0, epochs=1, batch_size=2, seed=7)

    model = train_tabular_mlp(x, y, x, y, config)
    probabilities = predict_proba_mlp(model, x)

    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_metric_helpers_are_stable() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = classification_metrics(y, probability)

    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert expected_calibration_error(y, probability, n_bins=2) == 0.25


def test_icd9_grouping_covers_major_branches() -> None:
    assert group_icd9_diagnosis(None) == "missing"
    assert group_icd9_diagnosis("?") == "missing"
    assert group_icd9_diagnosis("V45") == "supplemental"
    assert group_icd9_diagnosis("410") == "circulatory"
    assert group_icd9_diagnosis("486") == "respiratory"
    assert group_icd9_diagnosis("530") == "digestive"
    assert group_icd9_diagnosis("250.13") == "diabetes"
    assert group_icd9_diagnosis("820") == "injury"
    assert group_icd9_diagnosis("715") == "musculoskeletal"
    assert group_icd9_diagnosis("585") == "genitourinary"
    assert group_icd9_diagnosis("174") == "neoplasms"
    assert group_icd9_diagnosis("text") == "other"


def test_tracking_helpers_flatten_and_hash(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.txt"
    file_path.write_text("medmlops\n", encoding="utf-8")

    flat = flatten_params({"a": {"b": [1, 2]}, "c": True})

    assert flat == {"a.b": "1,2", "c": True}
    assert len(sha256_file(file_path)) == 64


def test_train_phase2_small_dataset(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
    monkeypatch: MonkeyPatch,
) -> None:
    params = load_yaml(Path("params.yaml"))
    params["model"]["baseline"].update(
        {"backend": "sklearn", "n_estimators": 2, "max_depth": 2}
    )
    params["model"]["mlp"].update(
        {"hidden": [4], "dropout": 0.0, "epochs": 1, "batch_size": 16}
    )
    params["model"]["experiment_name"] = "unit-train"
    params["model"]["registered_model_name"] = "UnitMedMLOps"
    params["tracking"]["register_model"] = False

    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")

    spec = feature_spec_from_params(params)
    raw = pd.concat([sample_diabetes130] * 10, ignore_index=True)
    frame = prepare_feature_frame(binarize_readmission(raw), spec)
    splits = deterministic_split(frame, seed=11)
    for name, split_df in splits.items():
        split_df.to_parquet(tmp_path / f"{name}.parquet", index=False)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    report_path = tmp_path / "train_metrics.json"

    train_phase2(
        train_path=tmp_path / "train.parquet",
        val_path=tmp_path / "val.parquet",
        test_path=tmp_path / "test.parquet",
        models_dir=tmp_path / "models",
        report_path=report_path,
        params_path=params_path,
    )

    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    assert report["baseline"]["backend"] == "sklearn_hist_gradient_boosting"
    assert report["features"]["audit_only_excluded"] == ["race", "gender", "age"]
    assert (tmp_path / "models" / "hero.pt").exists()
    assert (tmp_path / "models" / "baseline.pkl").exists()
