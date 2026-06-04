"""Canonical metric hashes for the independent-reproduction release gate."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

DEFAULT_REPORT_DIR = Path("reports")
DEFAULT_REFERENCE_PATH = Path("reports/reference_hashes.json")
METRIC_FILES = (
    "train_metrics.json",
    "calibration_metrics.json",
    "conformal_metrics.json",
    "clinical_metrics.json",
    "fairness.json",
)
EXCLUDED_NON_METRIC_FIELDS = {
    "fairness.json": ("provenance.model_sha256",),
}
REPRODUCIBILITY_CLAIM = (
    "Bit-reproducible metric reports within the pinned Docker image on the "
    "reference CPU architecture; cross-architecture equality is not guaranteed."
)


class ReproductionMismatchError(RuntimeError):
    """Raised when produced metric hashes diverge from the release reference."""


def _normalized_architecture(value: str) -> str:
    aliases = {
        "aarch64": "arm64",
        "arm64": "arm64",
        "amd64": "amd64",
        "x86_64": "amd64",
    }
    return aliases.get(value.lower(), value.lower())


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def canonical_json_sha256(path: str | Path) -> str:
    """Hash semantic JSON content independently of whitespace and key order."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_metric_sha256(path: str | Path) -> str:
    """Hash metric content while excluding declared volatile non-metric metadata."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if source.name == "fairness.json" and isinstance(payload, dict):
        provenance = payload.get("provenance")
        if isinstance(provenance, dict):
            provenance.pop("model_sha256", None)
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def metric_hashes(
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    metric_files: tuple[str, ...] = METRIC_FILES,
) -> dict[str, str]:
    """Return canonical SHA-256 hashes for all release metric reports."""

    root = Path(report_dir)
    return {name: canonical_metric_sha256(root / name) for name in metric_files}


def _quality_baseline(report_dir: Path) -> dict[str, Any]:
    clinical = _read_json(report_dir / "clinical_metrics.json")
    fairness = _read_json(report_dir / "fairness.json")
    features = fairness.get("features", {})
    subgroup_gaps: dict[str, float] = {}
    if isinstance(features, dict):
        for feature, values in features.items():
            if not isinstance(values, dict):
                continue
            gap = values.get("subgroup_auroc_gap")
            if isinstance(gap, int | float):
                subgroup_gaps[str(feature)] = float(gap)
    return {
        "clinical_auroc": clinical.get("auroc"),
        "clinical_ece": clinical.get("ece"),
        "subgroup_auroc_gaps": subgroup_gaps,
    }


def build_reference_manifest(
    report_dir: str | Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    """Build a release reference manifest from current metric reports."""

    root = Path(report_dir)
    return {
        "schema_version": 1,
        "algorithm": "sha256(canonical-metric-json-v1)",
        "excluded_non_metric_fields": EXCLUDED_NON_METRIC_FIELDS,
        "reproducibility_claim": REPRODUCIBILITY_CLAIM,
        "reference_environment": {
            "cpu_architecture": _normalized_architecture(platform.machine()),
            "python_version": platform.python_version(),
        },
        "quality_baseline": _quality_baseline(root),
        "metrics": metric_hashes(root),
    }


def write_reference_hashes(
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    output_path: str | Path = DEFAULT_REFERENCE_PATH,
) -> Path:
    """Write the committed metric-hash reference manifest."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_reference_manifest(report_dir), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination


def verify_reproduction(
    produced_dir: str | Path = DEFAULT_REPORT_DIR,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    *,
    require_architecture: bool = False,
) -> dict[str, Any]:
    """Verify produced metric hashes against a committed reference manifest."""

    reference = _read_json(Path(reference_path))
    expected = reference.get("metrics")
    if not isinstance(expected, dict) or not expected:
        raise TypeError("Reference manifest must contain a non-empty metrics mapping")

    architecture = _normalized_architecture(platform.machine())
    environment = reference.get("reference_environment", {})
    reference_architecture = (
        environment.get("cpu_architecture") if isinstance(environment, dict) else None
    )
    normalized_reference = (
        _normalized_architecture(str(reference_architecture))
        if reference_architecture is not None
        else None
    )
    architecture_matches = normalized_reference in {None, architecture}
    if require_architecture and not architecture_matches:
        raise ReproductionMismatchError(
            "Reference CPU architecture "
            f"{reference_architecture!r} does not match produced architecture "
            f"{architecture!r}"
        )

    actual: dict[str, str] = {}
    missing: list[str] = []
    mismatches: dict[str, dict[str, str]] = {}
    root = Path(produced_dir)
    for name, expected_hash in expected.items():
        path = root / str(name)
        if not path.exists():
            missing.append(str(name))
            continue
        actual_hash = canonical_metric_sha256(path)
        actual[str(name)] = actual_hash
        if actual_hash != expected_hash:
            mismatches[str(name)] = {
                "expected": str(expected_hash),
                "actual": actual_hash,
            }

    result: dict[str, Any] = {
        "passed": not missing and not mismatches,
        "architecture_matches": architecture_matches,
        "reference_architecture": reference_architecture,
        "produced_architecture": architecture,
        "verified": sorted(actual),
        "missing": missing,
        "mismatches": mismatches,
    }
    if not result["passed"]:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if mismatches:
            details.append("mismatched=" + ",".join(sorted(mismatches)))
        raise ReproductionMismatchError(
            "Independent reproduction diverged: " + "; ".join(details)
        )
    return result
