from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from medmlops.cli import app
from medmlops.quality.commits import is_conventional_commit_message
from medmlops.quality.gates import (
    QualityGateError,
    check_calibration,
    check_fairness,
    check_performance,
)
from medmlops.quality.reproduction import (
    METRIC_FILES,
    ReproductionMismatchError,
    build_reference_manifest,
    canonical_json_sha256,
    canonical_metric_sha256,
    metric_hashes,
    verify_reproduction,
    write_reference_hashes,
)


def _write_reports(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {
        "train_metrics.json": {"hero": {"test": {"auroc": 0.68}}},
        "calibration_metrics.json": {"method": "isotonic"},
        "conformal_metrics.json": {"marginal_coverage": 0.9},
        "clinical_metrics.json": {"auroc": 0.68, "ece": 0.01},
        "fairness.json": {
            "features": {
                "race": {"subgroup_auroc_gap": 0.15},
                "gender": {"subgroup_auroc_gap": 0.01},
            }
        },
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_metric_hash_ignores_json_formatting(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"b": 2, "a": 1}', encoding="utf-8")
    second.write_text('{\n  "a": 1,\n  "b": 2\n}\n', encoding="utf-8")

    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_metric_hash_excludes_only_volatile_fairness_model_digest(
    tmp_path: Path,
) -> None:
    fairness = tmp_path / "fairness.json"
    fairness.write_text(
        '{"features":{"race":{"subgroup_auroc_gap":0.15}},'
        '"provenance":{"model_sha256":"first","test_data_sha256":"stable"}}',
        encoding="utf-8",
    )
    first = canonical_metric_sha256(fairness)
    fairness.write_text(
        '{"features":{"race":{"subgroup_auroc_gap":0.15}},'
        '"provenance":{"model_sha256":"second","test_data_sha256":"stable"}}',
        encoding="utf-8",
    )
    assert canonical_metric_sha256(fairness) == first

    fairness.write_text(
        '{"features":{"race":{"subgroup_auroc_gap":0.16}},'
        '"provenance":{"model_sha256":"second","test_data_sha256":"stable"}}',
        encoding="utf-8",
    )
    assert canonical_metric_sha256(fairness) != first


def test_reference_manifest_and_reproduction_verification(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_reports(reports)
    reference = write_reference_hashes(reports, reports / "reference_hashes.json")
    manifest = build_reference_manifest(reports)

    assert set(metric_hashes(reports)) == set(METRIC_FILES)
    assert manifest["algorithm"] == "sha256(canonical-metric-json-v1)"
    assert manifest["quality_baseline"]["subgroup_auroc_gaps"]["race"] == 0.15
    assert verify_reproduction(reports, reference)["passed"] is True

    (reports / "clinical_metrics.json").write_text(
        '{"auroc": 0.60, "ece": 0.01}', encoding="utf-8"
    )
    with pytest.raises(ReproductionMismatchError, match=r"clinical_metrics\.json"):
        verify_reproduction(reports, reference)


def test_reproduction_verification_handles_missing_bad_reference_and_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports = tmp_path / "reports"
    _write_reports(reports)
    reference = write_reference_hashes(reports, reports / "reference_hashes.json")
    (reports / "fairness.json").unlink()
    with pytest.raises(ReproductionMismatchError, match=r"missing=fairness\.json"):
        verify_reproduction(reports, reference)

    bad_reference = tmp_path / "bad.json"
    bad_reference.write_text("{}", encoding="utf-8")
    with pytest.raises(TypeError, match="non-empty metrics"):
        verify_reproduction(reports, bad_reference)

    _write_reports(reports)
    payload = json.loads(reference.read_text(encoding="utf-8"))
    payload["reference_environment"]["cpu_architecture"] = "different-architecture"
    reference.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("platform.machine", lambda: "current-architecture")
    with pytest.raises(ReproductionMismatchError, match="CPU architecture"):
        verify_reproduction(reports, reference, require_architecture=True)

    payload["reference_environment"]["cpu_architecture"] = "aarch64"
    reference.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert verify_reproduction(reports, reference, require_architecture=True)["passed"]


def test_performance_and_calibration_quality_gates(tmp_path: Path) -> None:
    clinical = tmp_path / "clinical.json"
    clinical.write_text('{"auroc": 0.68, "ece": 0.01}', encoding="utf-8")

    assert check_performance(clinical, auroc_floor=0.62)["passed"] is True
    assert check_calibration(clinical, max_ece=0.03)["passed"] is True
    with pytest.raises(QualityGateError, match="Performance gate failed"):
        check_performance(clinical, auroc_floor=0.70)
    with pytest.raises(QualityGateError, match="Calibration gate failed"):
        check_calibration(clinical, max_ece=0.005)

    clinical.write_text('{"auroc": "bad"}', encoding="utf-8")
    with pytest.raises(TypeError, match="numeric auroc"):
        check_performance(clinical)


def test_fairness_gate_enforces_gap_widening_not_false_absolute_claim(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    _write_reports(reports)
    reference = write_reference_hashes(reports, reports / "reference_hashes.json")
    fairness = reports / "fairness.json"

    assert check_fairness(fairness, reference, max_gap=0.10)["passed"] is True
    fairness.write_text(
        json.dumps(
            {
                "features": {
                    "race": {"subgroup_auroc_gap": 0.26},
                    "gender": {"subgroup_auroc_gap": 0.01},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(QualityGateError, match="race"):
        check_fairness(fairness, reference, max_gap=0.10)


def test_quality_gate_cli_commands(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_reports(reports)
    reference = write_reference_hashes(reports, reports / "reference_hashes.json")
    runner = CliRunner()

    assert (
        runner.invoke(
            app,
            [
                "verify-reproduction",
                "--produced",
                str(reports),
                "--reference",
                str(reference),
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            [
                "check-performance",
                "--report-path",
                str(reports / "clinical_metrics.json"),
                "--auroc-floor",
                "0.70",
            ],
        ).exit_code
        == 1
    )


def test_conventional_commit_validation() -> None:
    assert is_conventional_commit_message("feat: add release gate")
    assert is_conventional_commit_message("fix(api)!: change response contract")
    assert is_conventional_commit_message("Merge branch 'main'")
    assert not is_conventional_commit_message("added a release gate")
