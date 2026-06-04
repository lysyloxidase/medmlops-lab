"""Responsible-AI fairness auditing and regression gates."""

from medmlops.fairness.audit import audit_subgroups, fairness_audit_phase6
from medmlops.fairness.gate import (
    FairnessRegressionError,
    assert_fairness_regression,
    fairness_regression_gate,
)

__all__ = [
    "FairnessRegressionError",
    "assert_fairness_regression",
    "audit_subgroups",
    "fairness_audit_phase6",
    "fairness_regression_gate",
]
