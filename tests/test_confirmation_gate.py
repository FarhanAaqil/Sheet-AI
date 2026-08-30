# =============================================================================
# SheetSense AI — Confirmation Gate Runtime Regression Tests
# =============================================================================
# Ensures that NO destructive action can reach Google Sheets write methods
# without passing through POST /actions/{action_id}/confirm.
# =============================================================================

import os
import uuid
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

import database
import sheets_writer
from agent import SheetTools, SheetSenseAgent
from main import app, get_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Ensure tests run against a fresh isolated SQLite database."""
    test_db = str(tmp_path / "test_sheetsense.db")
    monkeypatch.setattr(database, "DB_PATH", test_db)
    database.init_db(test_db)
    yield test_db


@pytest.fixture
def mock_spreadsheet():
    """Mock gspread spreadsheet with spy write methods to catch any direct write leaks."""
    ws = MagicMock()
    ws.title = "Orders"
    ws.row_values.return_value = ["OrderID", "Customer", "Price", "Status"]
    
    # Target row mock cell
    mock_cell = MagicMock()
    mock_cell.row = 3
    ws.find.return_value = mock_cell
    
    # Spy methods on writes
    ws.update_cell = MagicMock(return_value={"updated": True})
    ws.update = MagicMock(return_value={"updated": True})
    ws.delete_rows = MagicMock(return_value={"deleted": True})
    ws.get_all_records.return_value = [
        {"OrderID": "ORD-1001", "Customer": "Alice", "Price": 100, "Status": "completed"},
        {"OrderID": "ORD-4471", "Customer": "Bob", "Price": 250, "Status": "pending"},
    ]

    spreadsheet = MagicMock()
    spreadsheet.worksheets.return_value = [ws]
    spreadsheet.worksheet.return_value = ws
    return spreadsheet


@pytest.fixture
def client(mock_spreadsheet, monkeypatch):
    """FastAPI TestClient with mock agent."""
    class MockTools:
        def _refresh_sheet(self, name):
            pass

    class MockAgent:
        spreadsheet = mock_spreadsheet
        sheet_tools = MockTools()
        def get_sheet_names(self):
            return ["Orders"]
        def get_sheet_schema(self, sheet_name):
            return {"sheet_name": "Orders", "columns": ["OrderID", "Customer", "Price", "Status"]}

    mock_agent = MockAgent()
    monkeypatch.setattr("main._agent", mock_agent)
    app.dependency_overrides[get_agent] = lambda: mock_agent
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_direct_tool_calls_never_reach_gspread_write_methods(mock_spreadsheet):
    """
    CRITICAL REGRESSION TEST: Calling update_cell or delete_row tools directly MUST NOT
    call any gspread write or delete method. It must only stage a pending action.
    """
    st = SheetTools(mock_spreadsheet)
    ws = mock_spreadsheet.worksheet("Orders")

    # 1. Test update_cell
    update_input = json.dumps({
        "sheet_name": "Orders",
        "id_column": "OrderID",
        "id_value": "ORD-4471",
        "update_column": "Price",
        "new_value": 300,
    })
    res_upd = st.update_cell(update_input)

    # Assert gspread write methods were NEVER called
    assert ws.update_cell.call_count == 0
    assert ws.update.call_count == 0
    assert ws.delete_rows.call_count == 0

    # Assert pending action created
    assert st.last_pending_action is not None
    assert st.last_pending_action["tool_name"] == "update_cell"
    assert st.last_pending_action["requires_confirmation"] is True
    assert "CONFIRMATION REQUIRED" in res_upd

    # 2. Test delete_row
    delete_input = json.dumps({
        "sheet_name": "Orders",
        "id_column": "OrderID",
        "id_value": "ORD-4471",
    })
    res_del = st.delete_row(delete_input)

    # Assert gspread write methods were STILL NEVER called
    assert ws.update_cell.call_count == 0
    assert ws.update.call_count == 0
    assert ws.delete_rows.call_count == 0

    # Assert pending action created
    assert st.last_pending_action is not None
    assert st.last_pending_action["tool_name"] == "delete_row"
    assert st.last_pending_action["requires_confirmation"] is True
    assert "CONFIRMATION REQUIRED" in res_del


def test_confirm_endpoint_executes_valid_pending_action(client, mock_spreadsheet):
    """
    Valid pending actions can only be executed via POST /actions/{action_id}/confirm.
    """
    ws = mock_spreadsheet.worksheet("Orders")
    action_id = str(uuid.uuid4())

    # Stage pending delete action in database
    database.create_pending_action(
        action_id=action_id,
        tool_name="delete_row",
        target={"sheet_name": "Orders", "id_column": "OrderID", "id_value": "ORD-4471"},
        proposed_change={"action": "delete"},
        ttl_minutes=5,
    )

    # Confirm action
    resp = client.post(f"/actions/{action_id}/confirm")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "confirmed"
    assert data["action_id"] == action_id
    assert data["tool_name"] == "delete_row"

    # Verify that gspread delete was called EXACTLY ONCE here
    assert ws.delete_rows.call_count == 1

    # Verify status in database
    db_action = database.get_pending_action(action_id)
    assert db_action["status"] == "confirmed"


def test_confirm_endpoint_blocks_replay_attacks(client):
    """
    Attempting to confirm an already-confirmed action must return 410 Gone.
    """
    action_id = str(uuid.uuid4())
    database.create_pending_action(
        action_id=action_id,
        tool_name="delete_row",
        target={"sheet_name": "Orders", "id_column": "OrderID", "id_value": "ORD-4471"},
        proposed_change={"action": "delete"},
        ttl_minutes=5,
    )

    # First confirm succeeds
    resp1 = client.post(f"/actions/{action_id}/confirm")
    assert resp1.status_code == 200

    # Second confirm must fail with 410 Gone
    resp2 = client.post(f"/actions/{action_id}/confirm")
    assert resp2.status_code == 410
    assert "already been confirmed" in resp2.json()["detail"]


def test_confirm_endpoint_blocks_expired_actions(client):
    """
    Attempting to confirm an action past its 5-minute TTL must return 410 Gone.
    """
    action_id = str(uuid.uuid4())
    database.create_pending_action(
        action_id=action_id,
        tool_name="update_cell",
        target={"sheet_name": "Orders", "id_column": "OrderID", "id_value": "ORD-4471"},
        proposed_change={"update_column": "Price", "new_value": 500},
        ttl_minutes=-2,  # Created 2 minutes in the past -> expired
    )

    resp = client.post(f"/actions/{action_id}/confirm")
    assert resp.status_code == 410
    assert "expired" in resp.json()["detail"]

    # Verify database status marked expired
    db_action = database.get_pending_action(action_id)
    assert db_action["status"] == "expired"


def test_confirm_endpoint_rejects_nonexistent_action(client):
    """
    Non-existent action IDs must return 404 Not Found.
    """
    resp = client.post(f"/actions/{uuid.uuid4()}/confirm")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_reject_endpoint_cancels_pending_action(client):
    """
    POST /actions/{action_id}/reject cancels the pending action.
    """
    action_id = str(uuid.uuid4())
    database.create_pending_action(
        action_id=action_id,
        tool_name="delete_row",
        target={"sheet_name": "Orders", "id_column": "OrderID", "id_value": "ORD-4471"},
        proposed_change={"action": "delete"},
        ttl_minutes=5,
    )

    resp = client.post(f"/actions/{action_id}/reject")
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"

    # Confirming rejected action must now fail with 410
    resp_confirm = client.post(f"/actions/{action_id}/confirm")
    assert resp_confirm.status_code == 410
    assert "rejected" in resp_confirm.json()["detail"]
