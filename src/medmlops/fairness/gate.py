"""Regression gate for widening subgroup discrimination gaps."""

from __future__ import annotations

from typing import Any


class FairnessRegressionError(RuntimeError):
    """Raised when a subgroup AUROC gap widens beyond the allowed tolerance."""


def subgroup_auroc_gaps(report: dict[str, Any]) -> dict[str, float]:
    """Extract available per-feature subgroup AUROC gaps from an audit report."""

    gaps: dict[str, float] = {}
    features = report.get("features", {})
    if not isinstance(features, dict):
        return gaps
    for feature, values in features.items():
        if not isinstance(values, dict):
            continue
        gap = values.get("subgroup_auroc_gap")
        if isinstance(gap, int | float):
            gaps[str(feature)] = float(gap)
    return gaps


def fairness_regression_gate(
    current: dict[str, Any],
    reference: dict[str, Any],
    *,
    max_widening: float = 0.10,
) -> dict[str, Any]:
    """Compare current and reference AUROC gaps and return a gate result."""

    current_gaps = subgroup_auroc_gaps(current)
    reference_gaps = subgroup_auroc_gaps(reference)
    comparisons: dict[str, dict[str, float | bool]] = {}
    failures: list[str] = []
    for feature, current_gap in current_gaps.items():
        reference_gap = reference_gaps.get(feature, 0.0)
        widening = current_gap - reference_gap
        passed = widening <= max_widening
        comparisons[feature] = {
            "reference_gap": reference_gap,
            "current_gap": current_gap,
            "widening": widening,
            "passed": passed,
        }
        if not passed:
            failures.append(feature)
    return {
        "passed": not failures,
        "max_widening": max_widening,
        "failures": failures,
        "comparisons": comparisons,
    }


def assert_fairness_regression(
    current: dict[str, Any],
    reference: dict[str, Any],
    *,
    max_widening: float = 0.10,
) -> dict[str, Any]:
    """Raise when the fairness regression gate fails."""

    result = fairness_regression_gate(current, reference, max_widening=max_widening)
    if not result["passed"]:
        failed = ", ".join(str(value) for value in result["failures"])
        raise FairnessRegressionError(f"Subgroup AUROC gap widened for: {failed}")
    return result
