"""Application configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseModel):
    """Top-level project metadata."""

    name: str = "medmlops-lab"
    environment: str = "local"
    seed: int = 42


class PathSettings(BaseModel):
    """Filesystem paths used by the Phase 1 pipeline."""

    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    reports: Path = Path("reports")


class DataSettings(BaseModel):
    """Data configuration indirection."""

    config: Path = Path("configs/data/diabetes130.yaml")


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    json_output: bool = Field(default=False, alias="json")


class AppSettings(BaseSettings):
    """Settings loaded from YAML with environment overrides."""

    model_config = SettingsConfigDict(
        env_prefix="MEDMLOPS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        msg = f"Config file must contain a mapping: {path}"
        raise TypeError(msg)
    return data


def load_settings(path: str | Path | None = None) -> AppSettings:
    """Load application settings from YAML and environment variables."""

    config_path = Path(
        path or os.environ.get("MEDMLOPS_CONFIG_PATH", "configs/config.yaml")
    )
    return AppSettings(**_load_yaml(config_path))
