from __future__ import annotations

import json
import tomllib
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from medmlops.data.contracts import binarize_readmission
from medmlops.features.pipeline import (
    feature_spec_from_params,
    load_yaml,
    prepare_feature_frame,
)
from medmlops.monitoring.custom_drift import (
    DriftScoreRecord,
    DriftScoreStore,
    ks_drift,
    maximum_mean_discrepancy,
    numeric_drift_scores,
    persist_numeric_drift_scores,
    population_stability_index,
    psi_status,
    wasserstein_drift,
)
from medmlops.monitoring.drift import (
    build_drift_baseline,
    build_drift_report,
    prepare_monitoring_frame,
    snapshot_dict,
)
from medmlops.monitoring.performance import (
    delayed_label_metrics,
    join_delayed_ground_truth,
    load_labeled_predictions,
    monitor_delayed_labels,
    write_ground_truth_updates,
)
from medmlops.monitoring.simulate import (
    SYNTHETIC_NOTICE,
    conditional_shift_score,
    inject_concept_shift,
    inject_covariate_shift,
    inject_prior_shift,
    run_synthetic_drift_demo,
)
from medmlops.serving.audit import PredictionAuditLogger, PredictionAuditRecord


def _simulation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_medications": np.tile(np.arange(1, 21), 10),
            "readmitted_30d": np.tile([0, 0, 0, 1], 50),
        }
    )


def _audit_records(probabilities: list[float]) -> list[PredictionAuditRecord]:
    truths = [0, 0, 0, 1, 1, 1]
    return [
        PredictionAuditRecord(
            request_id=uuid4(),
            input_payload={"time_in_hospital": index + 1},
            model_version="8",
            model_alias="champion",
            data_hash="train-hash",
            risk_probability=probability,
            risk_class=int(probability >= 0.5),
            conformal_set=[int(probability >= 0.5)],
            abstain=False,
            abstain_reason=None,
            latency_ms=12.0,
            ground_truth=truth,
        )
        for index, (probability, truth) in enumerate(
            zip(probabilities, truths, strict=True)
        )
    ]


def test_evidently_is_exactly_pinned_and_report_includes_tests() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    params = load_yaml("params.yaml")
    reference = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]})
    current = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})

    snapshot = snapshot_dict(build_drift_report(reference, current))

    assert "evidently==0.7.21" in dependencies
    assert params["monitoring"]["evidently_version_pin"] == "0.7.21"
    assert snapshot["metrics"]
    assert snapshot["tests"]


def test_drift_baseline_selects_model_and_audit_columns(
    tmp_path: Path,
    sample_diabetes130: pd.DataFrame,
) -> None:
    params = load_yaml("params.yaml")
    spec = feature_spec_from_params(params)
    feature_frame = prepare_feature_frame(
        binarize_readmission(sample_diabetes130),
        spec,
    )
    input_path = tmp_path / "train.parquet"
    output_path = tmp_path / "reference.parquet"
    feature_frame.to_parquet(input_path, index=False)

    build_drift_baseline(input_path, output_path)
    reference = pd.read_parquet(output_path)

    assert set(spec.audit_only).issubset(reference.columns)
    assert spec.binary_target_col not in reference.columns
    assert list(reference.columns) == list(
        prepare_monitoring_frame(feature_frame, spec).columns
    )


def test_custom_psi_matches_hand_computation_and_thresholds() -> None:
    expected = np.array([0, 0, 1, 1], dtype=np.float64)
    actual = np.array([0, 0, 0, 1], dtype=np.float64)
    hand_computed = (0.75 - 0.50) * np.log(0.75 / 0.50) + (0.25 - 0.50) * np.log(
        0.25 / 0.50
    )

    assert population_stability_index(expected, actual, n_bins=2) == hand_computed
    assert psi_status(0.05) == "OK"
    assert psi_status(0.10) == "WARN"
    assert psi_status(0.251) == "ALERT"


def test_custom_drift_metrics_identical_vs_shifted() -> None:
    reference = np.linspace(0.0, 1.0, 100)
    identical = reference.copy()
    shifted = reference + 5.0

    assert ks_drift(reference, identical)["statistic"] == 0.0
    assert wasserstein_drift(reference, identical)["distance"] == 0.0
    assert maximum_mean_discrepancy(reference, identical) == 0.0
    shifted_scores = numeric_drift_scores(reference, shifted)
    assert shifted_scores["psi_status"] == "ALERT"
    assert bool(shifted_scores["ks"]["drift_detected"]) is True
    assert float(shifted_scores["wasserstein"]["distance"]) > 0.0
    assert float(shifted_scores["mmd"]) > 0.0


def test_drift_score_store_persists_warn_and_alert(tmp_path: Path) -> None:
    store = DriftScoreStore(f"sqlite:///{tmp_path / 'drift.db'}")
    store.initialize()
    for score in (0.15, 0.40):
        store.write(
            DriftScoreRecord(
                run_id="run-1",
                regime="covariate_shift",
                feature="num_medications",
                metric="psi",
                score=score,
                status=psi_status(score),
                metadata={"synthetic": True},
            )
        )

    rows = store.fetch_run("run-1")

    assert {row["status"] for row in rows} == {"WARN", "ALERT"}
    persist_numeric_drift_scores(
        store,
        run_id="run-2",
        regime="covariate_shift",
        feature="num_medications",
        scores=numeric_drift_scores(np.arange(10), np.arange(10) + 10),
    )
    assert {row["metric"] for row in store.fetch_run("run-2")} == {
        "psi",
        "ks",
        "wasserstein",
        "mmd",
    }


def test_all_three_synthetic_drift_regimes_trip_and_are_labeled(
    tmp_path: Path,
) -> None:
    reference = _simulation_frame()
    covariate = inject_covariate_shift(reference, "num_medications", 2.0)
    prior = inject_prior_shift(reference, 0.60)
    concept = inject_concept_shift(reference, 1.0)

    assert (
        population_stability_index(
            reference["num_medications"], covariate["num_medications"]
        )
        > 0.25
    )
    assert abs(float(prior["readmitted_30d"].mean()) - 0.60) < 0.01
    assert concept["readmitted_30d"].mean() == reference["readmitted_30d"].mean()
    assert conditional_shift_score(reference, concept) > 0.10

    report_path = run_synthetic_drift_demo(
        reference,
        database_url=f"sqlite:///{tmp_path / 'scores.db'}",
        report_path=tmp_path / "synthetic.json",
        html_dir=tmp_path / "html",
        workspace_path=tmp_path / "workspace",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["title"] == SYNTHETIC_NOTICE
    assert all(regime["detected"] for regime in report["regimes"].values())
    assert len(list((tmp_path / "html").glob("*.html"))) == 3
    assert (tmp_path / "workspace").exists()


def test_delayed_label_monitoring_queries_metrics_and_alerts(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'audit.db'}"
    audit = PredictionAuditLogger(database_url)
    audit.initialize()
    for record in _audit_records([0.05, 0.10, 0.20, 0.70, 0.80, 0.95]):
        audit.log_prediction(record)

    labeled = load_labeled_predictions(database_url)
    healthy = delayed_label_metrics(labeled, auroc_floor=0.60)
    regressed = labeled.copy()
    regressed["risk_probability"] = 1.0 - regressed["risk_probability"]
    alert = delayed_label_metrics(regressed, auroc_floor=0.60)
    report_path = monitor_delayed_labels(
        database_url,
        report_path=tmp_path / "performance.json",
        html_path=tmp_path / "performance.html",
        workspace_path=tmp_path / "workspace",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert healthy["status"] == "OK"
    assert "auprc" in healthy
    assert alert["status"] == "ALERT"
    assert alert["alert_reason"] == "auroc_below_floor"
    assert report["status"] == "OK"
    assert (tmp_path / "performance.html").exists()
    assert (tmp_path / "workspace").exists()


def test_delayed_ground_truth_join_and_database_update(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'audit.db'}"
    audit = PredictionAuditLogger(database_url)
    audit.initialize()
    record = _audit_records([0.10, 0.20, 0.30, 0.70, 0.80, 0.90])[0]
    unlabeled = PredictionAuditRecord(
        request_id=record.request_id,
        input_payload=record.input_payload,
        model_version=record.model_version,
        model_alias=record.model_alias,
        data_hash=record.data_hash,
        risk_probability=record.risk_probability,
        risk_class=record.risk_class,
        conformal_set=record.conformal_set,
        abstain=record.abstain,
        abstain_reason=record.abstain_reason,
        latency_ms=record.latency_ms,
    )
    audit.log_prediction(unlabeled)
    labels = pd.DataFrame({"request_id": [str(record.request_id)], "ground_truth": [1]})
    predictions = pd.DataFrame(
        {"request_id": [str(record.request_id)], "risk_probability": [0.1]}
    )

    joined = join_delayed_ground_truth(predictions, labels)
    updated = write_ground_truth_updates(database_url, labels)

    assert joined["ground_truth"].tolist() == [1]
    assert updated == 1
    assert load_labeled_predictions(database_url)["ground_truth"].tolist() == [1]


def test_delayed_label_monitor_handles_empty_labeled_subset(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'audit.db'}"
    audit = PredictionAuditLogger(database_url)
    audit.initialize()

    report_path = monitor_delayed_labels(
        database_url,
        report_path=tmp_path / "performance.json",
        workspace_path=tmp_path / "workspace",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "INSUFFICIENT_LABELS"
    assert report["labeled_count"] == 0
