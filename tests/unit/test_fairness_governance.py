from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from pytest import MonkeyPatch

from medmlops.calibration.calibrate import CalibratedRiskModel
from medmlops.conformal.conformal import ConformalGate
from medmlops.data.contracts import binarize_readmission
from medmlops.fairness.audit import (
    audit_subgroups,
    fairness_audit_phase6,
    plot_subgroup_auroc,
)
from medmlops.fairness.gate import (
    FairnessRegressionError,
    assert_fairness_regression,
    fairness_regression_gate,
    subgroup_auroc_gaps,
)
from medmlops.features.pipeline import (
    feature_spec_from_params,
    load_yaml,
    prepare_feature_frame,
)
from medmlops.governance.common import evidence_bundle, nested, read_json
from medmlops.governance.generate import generate_governance_docs
from medmlops.governance.probast_ai import PROBAST_AI_DOMAINS
from medmlops.governance.regulatory import FDA_GMLP_PRINCIPLES
from medmlops.governance.tripod_ai import TRIPOD_AI_ITEMS


class AuditEstimator:
    """Pickle-friendly probability source for the Phase 6 orchestration test."""

    def predict_positive_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.linspace(0.05, 0.95, len(x), dtype=np.float64)


def _synthetic_audit() -> dict[str, object]:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int_)
    probability = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.7, 0.6, 0.3])
    sensitive = pd.DataFrame(
        {
            "race": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "gender": ["F", "F", "M", "M", "F", "F", "M", "M"],
            "age": ["young", "old", "young", "old", "young", "old", "young", "old"],
        }
    )
    return audit_subgroups(y_true, probability, sensitive)


def test_fairlearn_audit_reports_disparity_calibration_and_caveats(
    tmp_path: Path,
) -> None:
    report = _synthetic_audit()
    features = cast(dict[str, object], report["features"])
    race = cast(dict[str, object], features["race"])
    disparities = cast(dict[str, float], race["fairlearn_disparities"])
    groups = cast(dict[str, dict[str, float]], race["groups"])

    assert set(disparities) == {
        "demographic_parity_difference",
        "demographic_parity_ratio",
        "equalized_odds_difference",
        "equal_opportunity_difference",
    }
    assert {"auroc", "auprc", "ece", "slope", "intercept"}.issubset(groups["A"])
    scope = cast(dict[str, object], report["audit_scope"])
    assert scope["sensitive_features_used_as_predictors"] is False
    assert "cannot" in str(report["impossibility_tradeoff"])
    assert "social and political construct" in str(report["egfr_race_lesson"])

    figure = plot_subgroup_auroc(report, tmp_path / "subgroup.png")
    assert figure.exists()

    with pytest.raises(ValueError, match="equal length"):
        audit_subgroups(
            np.array([0, 1], dtype=np.int_),
            np.array([0.5]),
            pd.DataFrame({"race": ["A", "B"]}),
        )


def test_race_gender_age_are_audit_only_and_fairlearn_is_pinned() -> None:
    params = load_yaml("params.yaml")
    spec = feature_spec_from_params(params)
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert set(params["fairness"]["sensitive_features"]) == {"race", "gender", "age"}
    assert set(spec.audit_only).isdisjoint(spec.model_feature_columns)
    assert "fairlearn==0.13.0" in pyproject["project"]["dependencies"]


def test_fairness_regression_gate_passes_and_fails() -> None:
    reference = {"features": {"race": {"subgroup_auroc_gap": 0.04}}}
    current = {"features": {"race": {"subgroup_auroc_gap": 0.16}}}

    assert subgroup_auroc_gaps(reference) == {"race": 0.04}
    assert fairness_regression_gate(reference, reference, max_widening=0.10)["passed"]
    with pytest.raises(FairnessRegressionError, match="race"):
        assert_fairness_regression(current, reference, max_widening=0.10)


def test_phase6_fairness_orchestration(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
    monkeypatch: MonkeyPatch,
) -> None:
    params = load_yaml("params.yaml")
    params["model"]["experiment_name"] = "unit-phase6"
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.safe_dump(params), encoding="utf-8")
    spec = feature_spec_from_params(params)
    frame = prepare_feature_frame(
        binarize_readmission(pd.concat([sample_diabetes130] * 4, ignore_index=True)),
        spec,
    )
    test_path = tmp_path / "test.parquet"
    frame.to_parquet(test_path, index=False)
    gate = ConformalGate(cast(CalibratedRiskModel, AuditEstimator()))
    model_path = tmp_path / "conformal.pkl"
    joblib.dump(gate, model_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")

    report_path = fairness_audit_phase6(
        model_path=model_path,
        test_path=test_path,
        report_path=tmp_path / "fairness.json",
        figure_path=tmp_path / "subgroup.png",
        params_path=params_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["audit_scope"]["sensitive_features_used_as_predictors"] is False
    assert report["provenance"]["model_sha256"]
    assert (tmp_path / "subgroup.png").exists()


def test_governance_generators_have_required_structures(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "clinical_metrics.json").write_text(
        json.dumps({"auroc": 0.68, "auprc": 0.22, "ece": 0.01}),
        encoding="utf-8",
    )
    (reports / "fairness.json").write_text(
        json.dumps(_synthetic_audit()),
        encoding="utf-8",
    )
    docs = generate_governance_docs(reports, tmp_path / "docs")

    model_card = docs["model_card"].read_text(encoding="utf-8")
    datasheet = docs["datasheet"].read_text(encoding="utf-8")
    tripod = docs["tripod_ai"].read_text(encoding="utf-8")
    probast = docs["probast_ai"].read_text(encoding="utf-8")
    regulatory = docs["regulatory_framing"].read_text(encoding="utf-8")

    assert len(TRIPOD_AI_ITEMS) == 27
    assert tripod.count("| Addressed |") == 27
    assert len(PROBAST_AI_DOMAINS) == 4
    assert sum(len(questions) for questions in PROBAST_AI_DOMAINS.values()) == 16
    assert probast.count("| Yes / concern noted |") == 16
    assert len(FDA_GMLP_PRINCIPLES) == 10
    assert regulatory.count("\n1.") == 1
    assert "EU AI Act" in regulatory
    assert "not certified or approved" in regulatory
    assert "social and political" in model_card
    assert "not a certification" in datasheet


def test_governance_common_evidence_helpers(tmp_path: Path) -> None:
    assert read_json(tmp_path / "missing.json") == {}
    source = tmp_path / "value.json"
    source.write_text('{"outer": {"inner": 3}}', encoding="utf-8")
    assert nested(read_json(source), "outer", "inner") == 3
    assert nested({}, "missing") == "not available"
    assert set(evidence_bundle(tmp_path)) == {
        "data_quality",
        "train_metrics",
        "calibration_metrics",
        "conformal_metrics",
        "clinical_metrics",
        "drift_simulation",
        "performance_monitoring",
        "fairness",
    }
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        read_json(source)
