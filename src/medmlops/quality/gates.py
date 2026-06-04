"""Release quality gates for performance, calibration, and fairness regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medmlops.fairness.gate import fairness_regression_gate

DEFAULT_CLINICAL_REPORT = Path("reports/clinical_metrics.json")
DEFAULT_FAIRNESS_REPORT = Path("reports/fairness.json")
DEFAULT_REFERENCE_PATH = Path("reports/reference_hashes.json")


class QualityGateError(RuntimeError):
    """Raised when a release quality threshold is violated."""


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {source}")
    return payload


def _number(payload: dict[str, Any], key: str, source: str | Path) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise TypeError(f"{source} must contain numeric {key}")
    return float(value)


def check_performance(
    report_path: str | Path = DEFAULT_CLINICAL_REPORT,
    *,
    auroc_floor: float = 0.62,
) -> dict[str, float | bool]:
    """Fail when held-out clinical AUROC falls below the release floor."""

    auroc = _number(_read_json(report_path), "auroc", report_path)
    passed = auroc >= auroc_floor
    if not passed:
        raise QualityGateError(
            f"Performance gate failed: AUROC {auroc:.6f} < floor {auroc_floor:.6f}"
        )
    return {"passed": passed, "auroc": auroc, "auroc_floor": auroc_floor}


def check_calibration(
    report_path: str | Path = DEFAULT_CLINICAL_REPORT,
    *,
    max_ece: float = 0.03,
) -> dict[str, float | bool]:
    """Fail when held-out expected calibration error exceeds the release limit."""

    ece = _number(_read_json(report_path), "ece", report_path)
    passed = ece <= max_ece
    if not passed:
        raise QualityGateError(
            f"Calibration gate failed: ECE {ece:.6f} > limit {max_ece:.6f}"
        )
    return {"passed": passed, "ece": ece, "max_ece": max_ece}


def _reference_fairness_report(reference_path: str | Path) -> dict[str, Any]:
    reference = _read_json(reference_path)
    baseline = reference.get("quality_baseline", {})
    if not isinstance(baseline, dict):
        raise TypeError("Reference manifest quality_baseline must be a mapping")
    gaps = baseline.get("subgroup_auroc_gaps", {})
    if not isinstance(gaps, dict):
        raise TypeError("Reference manifest subgroup_auroc_gaps must be a mapping")
    return {
        "features": {
            str(feature): {"subgroup_auroc_gap": float(gap)}
            for feature, gap in gaps.items()
            if isinstance(gap, int | float)
        }
    }


def check_fairness(
    report_path: str | Path = DEFAULT_FAIRNESS_REPORT,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    *,
    max_gap: float = 0.10,
) -> dict[str, Any]:
    """Fail when any subgroup AUROC gap widens beyond the committed baseline."""

    current = _read_json(report_path)
    reference = _reference_fairness_report(reference_path)
    result = fairness_regression_gate(current, reference, max_widening=max_gap)
    if not result["passed"]:
        failures = ", ".join(str(value) for value in result["failures"])
        raise QualityGateError(
            "Fairness regression gate failed: subgroup AUROC gap widened by more "
            f"than {max_gap:.6f} for {failures}"
        )
    return result
