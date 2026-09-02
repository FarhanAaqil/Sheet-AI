# =============================================================================
# SheetSense AI — Evaluation Harness Unit Tests
# =============================================================================

import pytest
from fastapi.testclient import TestClient

from eval_harness import EvaluationHarness, run_evaluation
from main import app
import database


@pytest.fixture
def client():
    return TestClient(app)


def test_eval_harness_execution():
    """Verify evaluation harness runs across all 30 benchmark cases and calculates metrics."""
    summary = run_evaluation()

    assert summary["total_cases"] == 30
    assert summary["benchmark_routing_accuracy"] >= 90.0
    assert summary["tsa"] >= 90.0, f"TSA below threshold: {summary['tsa']}%"
    assert summary["ea"] >= 90.0, f"EA below threshold: {summary['ea']}%"
    assert summary["cga"] == 100.0, f"CGA must be 100%: {summary['cga']}%"
    assert summary["ibr"] == 100.0, f"IBR must be 100%: {summary['ibr']}%"
    assert summary["latency_p50_ms"] > 0
    assert summary["latency_p95_ms"] >= summary["latency_p50_ms"]


def test_eval_endpoint_and_sqlite_persistence(client):
    """Verify POST /eval/run endpoint executes and logs metrics into SQLite."""
    headers = {"X-API-Key": "dev-key-123"}
    resp = client.post("/eval/run", headers=headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["total_cases"] == 30
    assert "benchmark_routing_accuracy" in data
    assert "tsa" in data
    assert "ea" in data
    assert "cga" in data
    assert "ibr" in data

    # Verify run logged in SQLite
    runs = database.get_recent_eval_runs(limit=5)
    assert len(runs) >= 1
    latest_run = runs[0]
    assert latest_run["total_queries"] == 30
    assert latest_run["tool_selection_accuracy"] == data["tsa"]

