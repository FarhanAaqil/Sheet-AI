# =============================================================================
# SheetSense AI — End-to-End System Integration Tests (Day 6)
# =============================================================================
# Exercises the entire integrated system across:
# 1. Health and operational metrics (/health, /metrics)
# 2. ReAct chat endpoint with session tracking (/chat)
# 3. Complete destructive staging, confirmation, and replay prevention lifecycle
# 4. Automated evaluation suite endpoint (/eval/run)
# =============================================================================

import uuid
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import database
from main import app, get_agent


@pytest.fixture
def client():
    return TestClient(app)


def test_e2e_health_and_metrics_flow(client):
    """Verify health and metrics endpoints return valid operational state."""
    mock_agent = MagicMock()
    mock_agent.get_sheet_names.return_value = ["Orders", "Customers"]
    app.dependency_overrides[get_agent] = lambda: mock_agent

    try:
        # 1. Health check
        h_resp = client.get("/health", headers={"X-API-Key": "dev-key-123"})
        assert h_resp.status_code == 200
        h_data = h_resp.json()
        assert h_data["status"] == "ok"
        assert h_data["sheets_connected"] is True
        assert h_data["sheets_count"] == 2

        # 2. Operational metrics
        m_resp = client.get("/metrics", headers={"X-API-Key": "dev-key-123"})
        assert m_resp.status_code == 200
        m_data = m_resp.json()
        assert "actions" in m_data
        assert "confirmed" in m_data["actions"]
    finally:
        app.dependency_overrides.pop(get_agent, None)


def test_e2e_chat_read_flow(client):
    """Verify read query flow through /chat returns structured response."""
    mock_agent = MagicMock()
    mock_agent.run.return_value = {
        "answer": "There are 20 orders in the Orders sheet with total revenue of $14,250.00.",
        "tools_used": ["read_sheet", "filter_and_aggregate"],
        "intermediate_steps": [{"tool": "read_sheet", "observation": "20 rows"}],
        "pending_action": None,
    }
    app.dependency_overrides[get_agent] = lambda: mock_agent

    try:
        resp = client.post(
            "/chat",
            json={"message": "What is the total revenue?", "session_id": "sess-e2e-1"},
            headers={"X-API-Key": "dev-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "14,250.00" in data["answer"]
        assert data["session_id"] == "sess-e2e-1"
        assert data["tool_calls_made"] == ["read_sheet", "filter_and_aggregate"]
        assert data["pending_action"] is None
    finally:
        app.dependency_overrides.pop(get_agent, None)


def test_e2e_destructive_confirmation_lifecycle(client, monkeypatch):
    """Verify end-to-end confirmation gate: staging -> confirm -> execute -> replay block."""
    mock_writer = MagicMock()
    mock_writer.execute_action.return_value = {
        "updated": True,
        "sheet": "Orders",
        "row": 5,
        "col": 4,
        "new_value": 89.99,
    }
    monkeypatch.setattr("main.sheets_writer", mock_writer)

    mock_agent = MagicMock()
    app.dependency_overrides[get_agent] = lambda: mock_agent

    action_id = str(uuid.uuid4())
    database.create_pending_action(
        action_id=action_id,
        tool_name="update_cell",
        target={"sheet_name": "Orders", "id_column": "order_id", "id_value": "ORD-1005"},
        proposed_change={"update_column": "price", "new_value": 89.99},
        ttl_minutes=5,
    )

    try:
        # Step 1: Confirm action
        resp = client.post(
            f"/actions/{action_id}/confirm",
            headers={"X-API-Key": "dev-key-123"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "confirmed"
        assert body["action_id"] == action_id
        assert body["result"]["updated"] is True

        # Step 2: Replay attack block (action already confirmed)
        replay_resp = client.post(
            f"/actions/{action_id}/confirm",
            headers={"X-API-Key": "dev-key-123"},
        )
        assert replay_resp.status_code == 410
        assert "already been confirmed" in replay_resp.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_agent, None)


def test_e2e_eval_run_endpoint(client):
    """Verify /eval/run endpoint executes benchmark harness and logs to SQLite."""
    resp = client.post(
        "/eval/run",
        headers={"X-API-Key": "dev-key-123"},
    )
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["total_cases"] == 30
    assert summary["tsa"] >= 90.0
    assert summary["cga"] == 100.0
    assert summary["ibr"] == 100.0
    assert "run_id" in summary


