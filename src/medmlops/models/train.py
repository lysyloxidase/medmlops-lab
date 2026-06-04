"""Reproducible Phase 2 training orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.utils.class_weight import compute_sample_weight

from medmlops.features.pipeline import (
    build_feature_transformer,
    ensure_sparse_matrix,
    feature_spec_from_params,
    get_transformed_feature_names,
    load_yaml,
    split_xy,
)
from medmlops.models.baseline import GBDTBaseline, GBDTBaselineConfig
from medmlops.models.tabular_mlp import (
    MLPConfig,
    predict_proba_mlp,
    train_tabular_mlp,
)
from medmlops.models.tracking import (
    configure_mlflow,
    current_git_commit,
    flatten_params,
    register_champion,
    sha256_file,
)
from medmlops.seeds import set_deterministic

DEFAULT_TRAIN_PATH = Path("data/processed/train.parquet")
DEFAULT_VAL_PATH = Path("data/processed/val.parquet")
DEFAULT_TEST_PATH = Path("data/processed/test.parquet")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_REPORT_PATH = Path("reports/train_metrics.json")
DEFAULT_PARAMS_PATH = Path("params.yaml")


def _section(params: dict[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key, {})
    if not isinstance(value, dict):
        msg = f"params.yaml section must be a mapping: {key}"
        raise TypeError(msg)
    return value


def _baseline_config(params: dict[str, Any], seed: int) -> GBDTBaselineConfig:
    model_params = _section(params, "model")
    baseline = model_params.get("baseline", {})
    if not isinstance(baseline, dict):
        msg = "model.baseline params must be a mapping"
        raise TypeError(msg)
    return GBDTBaselineConfig(
        backend=str(baseline.get("backend", "xgboost")),
        n_estimators=int(baseline.get("n_estimators", 80)),
        max_depth=int(baseline.get("max_depth", 3)),
        learning_rate=float(baseline.get("learning_rate", 0.08)),
        min_child_weight=float(baseline.get("min_child_weight", 5.0)),
        subsample=float(baseline.get("subsample", 1.0)),
        colsample_bytree=float(baseline.get("colsample_bytree", 1.0)),
        reg_lambda=float(baseline.get("reg_lambda", 1.0)),
        seed=seed,
    )


def _mlp_config(params: dict[str, Any], seed: int) -> MLPConfig:
    model_params = _section(params, "model")
    mlp = model_params.get("mlp", {})
    if not isinstance(mlp, dict):
        msg = "model.mlp params must be a mapping"
        raise TypeError(msg)
    hidden = mlp.get("hidden", [256, 128, 64])
    if not isinstance(hidden, list):
        msg = "model.mlp.hidden must be a list"
        raise TypeError(msg)
    return MLPConfig(
        hidden=tuple(int(width) for width in hidden),
        dropout=float(mlp.get("dropout", 0.3)),
        epochs=int(mlp.get("epochs", 6)),
        batch_size=int(mlp.get("batch_size", 1024)),
        learning_rate=float(mlp.get("learning_rate", 0.001)),
        weight_decay=float(mlp.get("weight_decay", 0.0001)),
        use_class_weight=bool(mlp.get("use_class_weight", True)),
        seed=seed,
    )


def expected_calibration_error(
    y_true: NDArray[np.int_],
    y_probability: NDArray[np.float64],
    *,
    n_bins: int = 10,
) -> float:
    """Compute binary expected calibration error."""

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_probability, bins[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        confidence = float(np.mean(y_probability[mask]))
        accuracy = float(np.mean(y_true[mask]))
        ece += float(np.mean(mask)) * abs(accuracy - confidence)
    return float(ece)


def classification_metrics(
    y_true: NDArray[np.int_],
    y_probability: NDArray[np.float64],
) -> dict[str, float]:
    """Return core binary classification metrics."""

    return {
        "auroc": float(roc_auc_score(y_true, y_probability)),
        "auprc": float(average_precision_score(y_true, y_probability)),
        "brier": float(brier_score_loss(y_true, y_probability)),
        "ece": expected_calibration_error(y_true, y_probability),
    }


def subgroup_aurocs(
    df: pd.DataFrame,
    y_true: NDArray[np.int_],
    y_probability: NDArray[np.float64],
    audit_columns: tuple[str, ...],
) -> dict[str, float]:
    """Compute subgroup AUROCs where both classes are present."""

    metrics: dict[str, float] = {}
    for column in audit_columns:
        if column not in df.columns:
            continue
        values = df[column].astype("string").fillna("?")
        for group in sorted(values.unique()):
            mask = values == group
            group_y = y_true[mask.to_numpy()]
            if len(group_y) < 25 or len(np.unique(group_y)) < 2:
                continue
            key = f"{column}={group}"
            metrics[key] = float(roc_auc_score(group_y, y_probability[mask.to_numpy()]))
    return metrics


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _stable_float_dict(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 10) for key, value in sorted(metrics.items())}


def _log_flattened_params(params: dict[str, Any]) -> None:
    for key, value in flatten_params(params).items():
        string_value = str(value)
        mlflow.log_param(key, string_value[:500])


def _docker_base_image_digest(dockerfile_path: Path = Path("docker/Dockerfile")) -> str:
    if not dockerfile_path.exists():
        return "unavailable"
    for line in dockerfile_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("FROM "):
            image = line.split()[1]
            if "@sha256:" in image:
                return image.split("@", maxsplit=1)[1]
            return f"unresolved:{image}"
    return "unavailable"


def _log_provenance(params: dict[str, Any], train_path: Path) -> None:
    mlflow.set_tag("git_commit", current_git_commit())
    mlflow.set_tag("dvc_train_data_sha256", sha256_file(train_path))
    if Path("dvc.lock").exists():
        mlflow.set_tag("dvc_lock_sha256", sha256_file("dvc.lock"))
    if Path("uv.lock").exists():
        mlflow.set_tag("uv_lock_sha256", sha256_file("uv.lock"))
    mlflow.set_tag("docker_base_image_digest", _docker_base_image_digest())
    mlflow.set_tag("tracking_backend", mlflow.get_tracking_uri())
    mlflow.set_tag("provenance_links", "dvc,git,uv,docker")
    _log_flattened_params(params)


def _metric_prefix(model_name: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{model_name}_{key}": value for key, value in metrics.items()}


def train_phase2(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    val_path: str | Path = DEFAULT_VAL_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> Path:
    """Run the Phase 2 training pipeline."""

    params = load_yaml(params_path)
    seed = int(params.get("seed", 42))
    set_deterministic(seed)

    spec = feature_spec_from_params(params)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    transformer = build_feature_transformer(spec)
    x_train_raw, y_train_series = split_xy(train_df, spec)
    x_val_raw, y_val_series = split_xy(val_df, spec)
    x_test_raw, y_test_series = split_xy(test_df, spec)

    x_train = ensure_sparse_matrix(transformer.fit_transform(x_train_raw))
    x_val = ensure_sparse_matrix(transformer.transform(x_val_raw))
    x_test = ensure_sparse_matrix(transformer.transform(x_test_raw))

    y_train = y_train_series.to_numpy(dtype=np.int_)
    y_val = y_val_series.to_numpy(dtype=np.int_)
    y_test = y_test_series.to_numpy(dtype=np.int_)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    baseline = GBDTBaseline(_baseline_config(params, seed))
    baseline.fit(x_train, y_train, sample_weight=sample_weight)
    baseline_val_prob = baseline.predict_proba(x_val)
    baseline_test_prob = baseline.predict_proba(x_test)

    mlp_config = _mlp_config(params, seed)
    hero = train_tabular_mlp(x_train, y_train, x_val, y_val, mlp_config)
    hero_val_prob = predict_proba_mlp(hero, x_val)
    hero_test_prob = predict_proba_mlp(hero, x_test)

    baseline_val_metrics = classification_metrics(y_val, baseline_val_prob)
    baseline_test_metrics = classification_metrics(y_test, baseline_test_prob)
    hero_val_metrics = classification_metrics(y_val, hero_val_prob)
    hero_test_metrics = classification_metrics(y_test, hero_test_prob)

    champion = (
        "hero"
        if hero_val_metrics["auroc"] >= baseline_val_metrics["auroc"]
        else "baseline"
    )
    beat_delta = hero_val_metrics["auroc"] - baseline_val_metrics["auroc"]
    feature_names = get_transformed_feature_names(transformer)

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    baseline_path = models_path / "baseline.pkl"
    hero_path = models_path / "hero.pt"

    sklearn_baseline_pipeline = SklearnPipeline(
        steps=[("features", transformer), ("model", baseline.model)]
    )
    joblib.dump(sklearn_baseline_pipeline, baseline_path, compress=3)
    torch.save(
        {
            "state_dict": hero.state_dict(),
            "config": mlp_config,
            "n_features": x_train.shape[1],
            "feature_names": feature_names,
            "transformer": transformer,
        },
        hero_path,
    )

    report: dict[str, Any] = {
        "baseline": {
            "backend": baseline.effective_backend,
            "validation": _stable_float_dict(baseline_val_metrics),
            "test": _stable_float_dict(baseline_test_metrics),
            "subgroup_test_auroc": _stable_float_dict(
                subgroup_aurocs(test_df, y_test, baseline_test_prob, spec.audit_only)
            ),
        },
        "hero": {
            "validation": _stable_float_dict(hero_val_metrics),
            "test": _stable_float_dict(hero_test_metrics),
            "subgroup_test_auroc": _stable_float_dict(
                subgroup_aurocs(test_df, y_test, hero_test_prob, spec.audit_only)
            ),
        },
        "beat_baseline_delta_auroc": round(float(beat_delta), 10),
        "champion": champion,
        "class_imbalance": {
            "positive_rate_train": round(float(np.mean(y_train)), 10),
            "positive_rate_validation": round(float(np.mean(y_val)), 10),
            "positive_rate_test": round(float(np.mean(y_test)), 10),
            "class_weighted_loss": mlp_config.use_class_weight,
        },
        "features": {
            "n_raw_model_features": len(spec.model_feature_columns),
            "n_transformed_features": len(feature_names),
            "audit_only_excluded": list(spec.audit_only),
        },
        "flavors_logged": ["pytorch", "sklearn"],
        "source_train_sha256": sha256_file(train_path),
    }
    _write_metrics(Path(report_path), report)

    model_params = _section(params, "model")
    tracking_params = _section(params, "tracking")
    experiment_name = str(model_params.get("experiment_name", "MedMLOps-Lab"))
    registered_model_name = str(model_params.get("registered_model_name", "MedMLOps"))
    configure_mlflow(experiment_name)

    with mlflow.start_run(run_name="phase2-train") as run:
        _log_provenance(params, Path(train_path))
        mlflow.log_metrics(_metric_prefix("baseline_val", baseline_val_metrics))
        mlflow.log_metrics(_metric_prefix("baseline_test", baseline_test_metrics))
        mlflow.log_metrics(_metric_prefix("hero_val", hero_val_metrics))
        mlflow.log_metrics(_metric_prefix("hero_test", hero_test_metrics))
        mlflow.log_metric("beat_baseline_delta_auroc", float(beat_delta))
        mlflow.log_metric("positive_rate_train", float(np.mean(y_train)))
        mlflow.log_artifact(str(report_path))
        mlflow.sklearn.log_model(sklearn_baseline_pipeline, name="baseline_model")
        mlflow.pytorch.log_model(hero, name="hero_model")
        if champion == "baseline":
            champion_info = mlflow.sklearn.log_model(
                sklearn_baseline_pipeline, name="model"
            )
        else:
            champion_info = mlflow.pytorch.log_model(hero, name="model")
        if bool(tracking_params.get("register_model", True)):
            register_champion(
                run.info.run_id,
                registered_model_name,
                "model",
                model_uri=str(champion_info.model_uri),
            )

    return Path(report_path)
