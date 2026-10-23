"""Phase 2 acceptance tests for concurrency, tenancy, freshness, and tool limits."""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pandas as pd
import pytest

import database
from agent import SheetTools
from retrieval import SchemaIndex
from session_store import InMemorySessionStore
from spreadsheet_registry import SpreadsheetRegistry


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    path = str(tmp_path / "phase2.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db(path)
    return path


def test_one_hundred_parallel_confirmations_claim_exactly_once(isolated_db):
    action_id = str(uuid.uuid4())
    database.create_pending_action(
        action_id=action_id,
        tool_name="delete_row",
        target={"sheet_name": "Orders", "id_column": "id", "id_value": "1"},
        proposed_change={"action": "delete"},
        principal_id="owner",
    )

    def claim(_):
        return database.claim_pending_action(action_id, "owner", "same-request")["claim_status"]

    with ThreadPoolExecutor(max_workers=20) as pool:
        outcomes = list(pool.map(claim, range(100)))

    assert outcomes.count("claimed") == 1
    assert set(outcomes) <= {"claimed", "in_progress"}
    assert database.complete_action(action_id, {"deleted": True}) is True
    replay = database.claim_pending_action(action_id, "owner", "same-request")
    assert replay["claim_status"] == "idempotent_replay"
    assert json.loads(replay["result_json"]) == {"deleted": True}


def test_actions_are_owner_scoped(isolated_db):
    action_id = str(uuid.uuid4())
    database.create_pending_action(
        action_id=action_id,
        tool_name="delete_row",
        target={"sheet_name": "Orders"},
        proposed_change={"action": "delete"},
        principal_id="owner-a",
    )
    assert (
        database.claim_pending_action(action_id, "owner-b", "foreign")["claim_status"]
        == "forbidden"
    )
    assert database.reject_pending_action(action_id, "owner-b") == "forbidden"
    assert database.get_pending_action(action_id)["status"] == "pending"


def test_same_session_id_is_isolated_by_principal():
    store = InMemorySessionStore()
    store.save_turn("alice", "shared-id", "Alice question", "Alice answer")
    store.save_turn("bob", "shared-id", "Bob question", "Bob answer")

    alice = store.get_history("alice", "shared-id")
    bob = store.get_history("bob", "shared-id")
    assert [message["content"] for message in alice] == ["Alice question", "Alice answer"]
    assert [message["content"] for message in bob] == ["Bob question", "Bob answer"]


def test_connection_registry_prevents_cross_tenant_resolution(isolated_db):
    client = MagicMock()
    sheet = MagicMock(title="Tenant A Sheet")
    client.open_by_key.return_value = sheet
    registry = SpreadsheetRegistry(client)
    connection = registry.register(
        "tenant-a",
        "https://docs.google.com/spreadsheets/d/abcdefghij1234567890/edit",
    )

    with pytest.raises(PermissionError):
        registry.resolve("tenant-b", connection["connection_id"])
    resolved, resolved_sheet = registry.resolve("tenant-a", connection["connection_id"])
    assert resolved["principal_id"] == "tenant-a"
    assert resolved_sheet is sheet


def _spreadsheet_with_frames(frames):
    worksheets = {}
    for name, frame in frames.items():
        ws = MagicMock()
        ws.title = name
        ws.get_all_records.return_value = frame.to_dict("records")
        worksheets[name] = ws
    spreadsheet = MagicMock()
    spreadsheet.worksheets.return_value = list(worksheets.values())
    spreadsheet.worksheet.side_effect = worksheets.__getitem__
    return spreadsheet, worksheets


def test_stale_cache_refreshes_and_reports_freshness():
    initial = pd.DataFrame([{"id": 1, "amount": 10}])
    spreadsheet, worksheets = _spreadsheet_with_frames({"Orders": initial})
    tools = SheetTools(spreadsheet)
    entry = tools.cache.get("Orders")
    entry.loaded_monotonic -= tools.cache.ttl_seconds + 1
    worksheets["Orders"].get_all_records.return_value = [
        {"id": 1, "amount": 10},
        {"id": 2, "amount": 20},
    ]

    result = tools.read_sheet(json.dumps({"sheet_name": "Orders"}))
    assert '"matched_rows": 2' in result
    metadata = tools.cache_metadata("Orders")
    assert metadata["cache_version"] == 2
    assert metadata["cache_age_seconds"] < 1


def test_anomalies_keep_row_alignment_with_null_values():
    frame = pd.DataFrame(
        [
            {"id": "a", "price": 10},
            {"id": "missing", "price": None},
            {"id": "b", "price": 11},
            {"id": "outlier", "price": 1000},
        ]
    )
    spreadsheet, _ = _spreadsheet_with_frames({"Orders": frame})
    result = SheetTools(spreadsheet).find_anomalies(
        json.dumps({"sheet_name": "Orders", "column_name": "price"})
    )
    assert "outlier" in result
    assert '"excluded_value_count": 1' in result
    assert "missing" not in result


def test_join_cardinality_limit_rejects_explosive_output():
    left = pd.DataFrame({"key": ["same"] * 100, "left": range(100)})
    right = pd.DataFrame({"key": ["same"] * 100, "right": range(100)})
    spreadsheet, _ = _spreadsheet_with_frames({"Left": left, "Right": right})
    result = SheetTools(spreadsheet).cross_sheet_join(
        json.dumps(
            {
                "sheet1": "Left",
                "sheet2": "Right",
                "on_column": "key",
                "max_output_rows": 1000,
            }
        )
    )
    assert "rejected" in result
    assert "10000 rows" in result


def test_sensitive_schema_values_are_not_indexed():
    index = SchemaIndex()
    index.build_from_dataframes(
        {
            "Customers": pd.DataFrame(
                {
                    "email": ["private@example.com"],
                    "region": ["North"],
                }
            )
        }
    )
    email = next(entry for entry in index.get_entries() if entry.column_name == "email")
    assert email.sensitive is True
    assert email.sample_values == []
    assert "private@example.com" not in email.searchable_text
