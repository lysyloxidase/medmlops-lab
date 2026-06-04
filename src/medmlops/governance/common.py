"""Shared helpers for deterministic governance-document generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medmlops.models.tracking import current_git_commit, sha256_file


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON evidence artifact, returning an empty mapping when absent."""

    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {source}")
    return payload


def nested(payload: dict[str, Any], *keys: str, default: Any = "not available") -> Any:
    """Read a nested evidence value."""

    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def evidence_bundle(report_dir: str | Path = "reports") -> dict[str, dict[str, Any]]:
    """Load the project evidence artifacts used by governance documents."""

    root = Path(report_dir)
    names = (
        "data_quality",
        "train_metrics",
        "calibration_metrics",
        "conformal_metrics",
        "clinical_metrics",
        "drift_simulation",
        "performance_monitoring",
        "fairness",
    )
    return {name: read_json(root / f"{name}.json") for name in names}


def provenance_lines() -> list[str]:
    """Return linked build provenance for the current checkout."""

    lines = [f"- Git commit: `{current_git_commit()}`"]
    for path in ("uv.lock", "dvc.lock"):
        source = Path(path)
        value = sha256_file(source) if source.exists() else "not available"
        suffix = " at generation time" if path == "dvc.lock" else ""
        lines.append(f"- `{path}` SHA-256{suffix}: `{value}`")
    return lines


def write_markdown(path: str | Path, content: str) -> Path:
    """Write a generated Markdown governance artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content.rstrip() + "\n", encoding="utf-8")
    return destination
