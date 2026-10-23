"""Phase 3 tests for evaluation evidence, observability, audit, and writer safety."""

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import database
from config import Settings
from evaluation_service import (
    EvaluationService,
    LiveEvaluationConfig,
    regression_failures,
)
from main import app
from observability import MetricsRegistry
from sheets_writer import (
    SheetsStateConflictError,
    SheetsWriterError,
    execute_action,
    execute_delete_row,
    execute_update_cell,
    row_fingerprint,
)


@pytest.fixture
def phase3_db(tmp_path, monkeypatch):
    path = str(tmp_path / "phase3.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(database, "DATABASE_URL", "")
    database.init_db(path)
    return path


def test_offline_evaluation_is_labelled_versioned_and_persisted(phase3_db):
    result = EvaluationService().run_offline()
    assert result["mode"] == "offline_deterministic"
    assert result["latency_scope"] == "local_fixture_execution"
    assert len(result["dataset_version"]) == 12
    assert result["regression_failures"] == []

    conn = database.get_db_connection()
    run = conn.execute(
        "SELECT mode, dataset_version FROM eval_runs WHERE eval_id=?",
        (result["run_id"],),
    ).fetchone()
    count = conn.execute(
        "SELECT COUNT(*) AS count FROM eval_results WHERE run_id=?",
        (result["run_id"],),
    ).fetchone()["count"]
    conn.close()
    assert run["mode"] == "offline_deterministic"
    assert count == 30


def test_adversarial_evaluation_is_separate_and_100_percent(phase3_db):
    result = EvaluationService().run_adversarial()
    assert result["mode"] == "adversarial_safety"
    assert result["safety_adherence"] == 100.0
    assert all(case["passed"] for case in result["cases"])


def test_repeated_live_evaluation_records_model_and_repetitions(phase3_db, tmp_path):
    dataset = [
        {
            "id": "live-1",
            "category": "read",
            "query": "count orders",
            "sheet_name": "Orders",
            "expected_tools": ["read_sheet"],
            "expected_answer_contains": ["20"],
            "requires_confirmation": False,
        }
    ]
    path = tmp_path / "live.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    agent = MagicMock()
    agent.run.return_value = {
        "answer": "There are 20 orders.",
        "tools_used": ["read_sheet"],
        "pending_action": None,
    }
    result = EvaluationService(path).run_live(
        agent,
        LiveEvaluationConfig(
            repetitions=3,
            model_name="gemini",
            model_version="test-version",
        ),
    )
    assert result["mode"] == "live_agent"
    assert result["total_cases"] == 3
    assert result["answer_correctness"] == 100.0
    assert result["model_version"] == "test-version"
    assert agent.run.call_count == 3


def test_regression_comparison_detects_quality_drop(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "metrics": {"score": 90, "safety": 100},
                "allowed_drop": {"score": 2},
                "must_equal": ["safety"],
            }
        ),
        encoding="utf-8",
    )
    failures = regression_failures({"score": 80, "safety": 99}, baseline)
    assert len(failures) == 2


def test_request_trace_headers_and_prometheus_metrics():
    with TestClient(app) as client:
        response = client.get(
            "/health/live",
            headers={
                "X-Request-ID": "request-12345678",
                "X-Trace-ID": "trace-12345678",
            },
        )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-12345678"
    assert response.headers["X-Trace-ID"] == "trace-12345678"

    registry = MetricsRegistry()
    registry.increment("example_total", route="/safe", status="200")
    rendered = registry.render_prometheus()
    assert 'example_total{route="/safe",status="200"} 1' in rendered


def test_append_only_audit_redacts_credentials(phase3_db):
    event_id = database.append_audit_event(
        "action.confirmed",
        "success",
        "principal",
        "resource",
        {"token": "sk-" + "x" * 24},
    )
    event = database.get_audit_events(1)[0]
    assert event["event_id"] == event_id
    assert "[REDACTED_CREDENTIAL]" in event["detail_json"]
    assert "sk-" not in event["detail_json"]


def test_production_requires_postgresql(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SHEETSENSE_API_KEY_HASH", "a" * 64)
    monkeypatch.setenv("GEMINI_API_KEY", "configured")
    monkeypatch.setenv(
        "GOOGLE_SHEET_URL",
        "https://docs.google.com/spreadsheets/d/abcdefghij123/edit",
    )
    monkeypatch.setenv("GCP_CREDENTIALS_JSON", "{}")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="PostgreSQL DATABASE_URL"):
        Settings.from_env().validate_startup()


def _writer_sheet():
    ws = MagicMock()
    ws.row_values.side_effect = lambda row: ["id", "status"] if row == 1 else ["one", "pending"]
    cell = MagicMock()
    cell.row = 2
    ws.findall.return_value = [cell]
    spreadsheet = MagicMock()
    spreadsheet.worksheet.return_value = ws
    return spreadsheet, ws


def test_writer_update_delete_dispatch_and_preconditions():
    spreadsheet, ws = _writer_sheet()
    expected = {"row_fingerprint": row_fingerprint(["id", "status"], ["one", "pending"])}
    updated = execute_update_cell(
        spreadsheet, "Orders", "id", "one", "status", "complete", expected
    )
    assert updated["success"] is True
    ws.update_cell.assert_called_once_with(2, 2, "complete")

    deleted = execute_delete_row(spreadsheet, "Orders", "id", "one", expected)
    assert deleted["success"] is True
    ws.delete_rows.assert_called_once_with(2)

    execute_action(
        spreadsheet,
        "delete_row",
        {"sheet_name": "Orders", "id_column": "id", "id_value": "one"},
        {"action": "delete"},
    )
    assert ws.delete_rows.call_count == 2


@pytest.mark.parametrize("matches", [[], [MagicMock(row=2), MagicMock(row=3)]])
def test_writer_rejects_missing_or_duplicate_identifier(matches):
    spreadsheet, ws = _writer_sheet()
    ws.findall.return_value = matches
    with pytest.raises(SheetsStateConflictError):
        execute_delete_row(spreadsheet, "Orders", "id", "one")
    ws.delete_rows.assert_not_called()


def test_writer_rejects_changed_row_and_bad_dispatch():
    spreadsheet, ws = _writer_sheet()
    with pytest.raises(SheetsStateConflictError):
        execute_update_cell(
            spreadsheet,
            "Orders",
            "id",
            "one",
            "status",
            "complete",
            {"row_fingerprint": "changed"},
        )
    ws.update_cell.assert_not_called()
    with pytest.raises(SheetsWriterError):
        execute_action(spreadsheet, "unknown", {}, {})
    with pytest.raises(SheetsWriterError, match="Missing 'update_column'"):
        execute_action(
            spreadsheet,
            "update_cell",
            {"sheet_name": "Orders", "id_column": "id", "id_value": "one"},
            {},
        )
