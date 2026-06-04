"""MLflow tracking and alias-based registry helpers."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient


def sha256_file(path: str | Path) -> str:
    """Return SHA-256 for a provenance file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    """Return the current Git commit or a stable uncommitted sentinel."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "uncommitted"
    return result.stdout.strip()


def flatten_params(
    params: dict[str, Any],
    prefix: str = "",
) -> dict[str, str | int | float | bool]:
    """Flatten nested params for MLflow logging."""

    flat: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_params(value, name))
        elif isinstance(value, list):
            flat[name] = ",".join(str(item) for item in value)
        elif isinstance(value, str | int | float | bool):
            flat[name] = value
        elif value is None:
            flat[name] = "null"
        else:
            flat[name] = str(value)
    return flat


def configure_mlflow(experiment_name: str) -> None:
    """Configure MLflow with an env-driven tracking URI."""

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def register_champion(
    run_id: str,
    model_name: str = "MedMLOps",
    artifact_path: str = "model",
    model_uri: str | None = None,
) -> None:
    """Register a model version and set the @champion alias."""

    client = MlflowClient()
    source = model_uri or f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(source, model_name)
    client.set_registered_model_alias(model_name, "champion", model_version.version)


def get_champion_model_version(model_name: str = "MedMLOps") -> Any:
    """Resolve the @champion alias with the MLflow alias API."""

    client = MlflowClient()
    return client.get_model_version_by_alias(model_name, "champion")
