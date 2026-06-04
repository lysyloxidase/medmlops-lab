from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.environ.get("MEDMLOPS_INTEGRATION_BASE_URL")
pytestmark = pytest.mark.skipif(
    BASE_URL is None,
    reason="Set MEDMLOPS_INTEGRATION_BASE_URL to test a running stack.",
)


def test_full_stack_exposes_api_health_metrics_and_model_provenance() -> None:
    assert BASE_URL is not None
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        health = client.get("/health")
        metrics = client.get("/metrics")
        model_info = client.get("/model-info")

    assert health.status_code == 200
    assert health.json()["conformal_loaded"] is True
    assert health.json()["audit_db_connected"] is True
    assert health.json()["ood_reference_loaded"] is True
    assert metrics.status_code == 200
    assert "medmlops_predictions_total" in metrics.text
    assert model_info.status_code == 200
    assert model_info.json()["data_hash"] != "unavailable"
