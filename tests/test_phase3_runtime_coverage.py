"""High-value runtime branch tests for orchestration, API services, and auth."""

import asyncio
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage

import main
from agent import SheetSenseAgent
from config import Settings
from domain_errors import (
    AuthorizationError,
    ConflictError,
    DependencyUnavailableError,
    NotFoundError,
    SheetSenseError,
)
from security import (
    APIKeyAuthenticator,
    APIPrincipal,
    build_auth_dependencies,
    hash_api_key,
    load_key_records,
    redact_sensitive,
)
from session_store import InMemorySessionStore, RedisSessionStore, get_session_store
from sheets_writer import SheetsWriterError, execute_delete_row, execute_update_cell
from spreadsheet_registry import SpreadsheetRegistry, normalize_spreadsheet_id


def _settings(**overrides):
    values = {
        "app_env": "test",
        "allowed_origins": ["http://test"],
        "allow_insecure_no_auth": False,
        "api_key": "",
        "api_key_hash": "",
        "api_keys_json": "",
        "gemini_api_key": "",
        "google_sheet_url": "",
        "gcp_credentials_json": "",
        "google_creds_path": "",
        "database_url": "",
    }
    values.update(overrides)
    return Settings(**values)


def test_authentication_records_roles_and_failure_paths():
    key = "valid-secret"
    settings = _settings(
        api_keys_json=(
            '[{"id":"viewer","key_sha256":"' + hash_api_key(key) + '","roles":["viewer"]}]'
        )
    )
    records = load_key_records(settings)
    assert records[0].principal_id == "viewer"
    auth = APIKeyAuthenticator(settings)
    assert auth.authenticate(key).principal_id == "viewer"
    with pytest.raises(HTTPException) as missing:
        auth.authenticate(None)
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as invalid:
        auth.authenticate("wrong")
    assert invalid.value.status_code == 403

    _, dependency, require_roles = build_auth_dependencies(settings)
    assert asyncio.run(dependency(key)).principal_id == "viewer"
    with pytest.raises(HTTPException) as forbidden:
        asyncio.run(require_roles("admin")(key))
    assert forbidden.value.status_code == 403


@pytest.mark.parametrize(
    "json_value",
    [
        '["not-an-object"]',
        '[{"id":"","key_sha256":"bad","roles":["viewer"]}]',
        '[{"id":"x","key_sha256":"' + "a" * 64 + '","roles":["invalid"]}]',
    ],
)
def test_invalid_auth_records_fail_closed(json_value):
    with pytest.raises(RuntimeError):
        load_key_records(_settings(api_keys_json=json_value))
    with pytest.raises(RuntimeError):
        load_key_records(_settings(api_key_hash="bad"))


def test_unconfigured_auth_and_development_escape_hatch():
    with pytest.raises(HTTPException) as exc:
        APIKeyAuthenticator(_settings()).authenticate(None)
    assert exc.value.status_code == 503
    principal = APIKeyAuthenticator(_settings(allow_insecure_no_auth=True)).authenticate(None)
    assert principal.principal_id == "insecure-development"
    assert "[REDACTED_CREDENTIAL]" in redact_sensitive("postgresql://user:password@localhost/db")


def _agent_with_graph(graph):
    agent = SheetSenseAgent.__new__(SheetSenseAgent)
    agent.session_store = InMemorySessionStore()
    retriever = MagicMock()
    retriever.retrieve_top_k.return_value = [
        {
            "sheet_name": "Orders",
            "column_name": "price",
            "dtype": "float64",
            "description": "order price",
            "similarity_score": 0.9,
        }
    ]
    agent._request_services = MagicMock(
        return_value=(
            {"connection_id": "connection-1"},
            MagicMock(),
            MagicMock(),
            retriever,
            graph,
        )
    )
    return agent


def test_agent_run_tracks_tools_history_and_connection():
    tool_message = AIMessage(
        content="working",
        tool_calls=[{"name": "read_sheet", "args": {"sheet_name": "Orders"}, "id": "1"}],
    )
    graph = MagicMock()
    graph.invoke.return_value = {
        "messages": [tool_message, AIMessage(content="There are 20 orders.")]
    }
    agent = _agent_with_graph(graph)
    result = agent.run(
        "count orders",
        "session-1",
        principal_id="alice",
        sheet_name="Orders",
    )
    assert result["answer"] == "There are 20 orders."
    assert result["tools_used"] == ["read_sheet"]
    assert result["connection_id"] == "connection-1"
    assert len(agent.session_store.get_history("alice", "session-1")) == 2
    with pytest.raises(ValueError):
        agent.run("x", "")


def test_agent_run_sanitizes_graph_failure():
    graph = MagicMock()
    graph.invoke.side_effect = RuntimeError("secret internal value")
    agent = _agent_with_graph(graph)
    result = agent.run("count", "session", principal_id="alice")
    assert "secret internal value" not in result["answer"]
    assert "Reference:" in result["answer"]


def test_agent_utility_methods_use_registry_and_fresh_tools(monkeypatch):
    agent = SheetSenseAgent.__new__(SheetSenseAgent)
    sheet = MagicMock()
    sheet.worksheets.return_value = [MagicMock(title="Orders")]
    agent.registry = MagicMock()
    agent.registry.resolve.return_value = ({"connection_id": "c1"}, sheet)
    agent.registry.register.return_value = {"connection_id": "c2"}
    agent.registry.list.return_value = [{"connection_id": "c1"}]
    assert agent.resolve_spreadsheet("alice") == ({"connection_id": "c1"}, sheet)
    assert agent.get_sheet_names("alice") == ["Orders"]
    assert agent.list_connections("alice") == [{"connection_id": "c1"}]

    tools = MagicMock()
    tools._get_df.return_value = pd.DataFrame([{"id": 1}])
    tools.cache_metadata.return_value = {"cache_version": 1}
    monkeypatch.setattr(agent, "get_sheet_tools", lambda *args: ({"connection_id": "c1"}, tools))
    schema = agent.get_sheet_schema("Orders", "alice")
    assert schema["row_count"] == 1
    tools._get_df.return_value = None
    assert agent.get_sheet_schema("Missing", "alice") is None
    monkeypatch.setattr(agent, "get_sheet_names", lambda *args: ["Orders"])
    connected = agent.connect_spreadsheet("alice", "https://example.invalid")
    assert connected["sheets"] == ["Orders"]


def test_api_service_functions_cover_success_and_failure_paths():
    principal = APIPrincipal("alice", frozenset({"admin"}))
    agent = MagicMock()
    agent.get_sheet_names.return_value = ["Orders"]
    agent.session_store = InMemorySessionStore()

    assert main.readiness(agent, principal)["status"] == "ready"
    assert main.health(agent, principal, True).sheets_count == 1
    assert main.list_sheets(None, agent, principal, True)["sheets"] == ["Orders"]

    agent.get_sheet_schema.return_value = {"columns": ["id"]}
    assert main.sheet_schema("Orders", None, False, agent, principal, True)["columns"] == ["id"]
    agent.get_sheet_schema.return_value = None
    with pytest.raises(HTTPException) as missing:
        main.sheet_schema("Missing", None, False, agent, principal, True)
    assert missing.value.status_code == 404

    agent.connect_spreadsheet.return_value = {"connection_id": "c1"}
    connected = main.connect_sheet(
        main.ConnectSheetRequest(
            sheet_url="https://docs.google.com/spreadsheets/d/abcdefghij123/edit"
        ),
        agent,
        principal,
        True,
    )
    assert connected["status"] == "connected"
    agent.connect_spreadsheet.side_effect = RuntimeError("private")
    with pytest.raises(HTTPException) as bad_connect:
        main.connect_sheet(
            main.ConnectSheetRequest(
                sheet_url="https://docs.google.com/spreadsheets/d/abcdefghij123/edit"
            ),
            agent,
            principal,
            True,
        )
    assert bad_connect.value.status_code == 400
    agent.connect_spreadsheet.side_effect = None

    tools = MagicMock()
    tools._get_df.return_value = pd.DataFrame([{"id": 1}, {"id": 2}])
    tools.cache_metadata.return_value = {"cache_version": 1}
    agent.get_sheet_tools.return_value = ({"connection_id": "c1"}, tools)
    data = main.get_sheet_data("Orders", None, 1, 0, False, agent, principal, True)
    assert data["truncated"] is True
    tools._get_df.return_value = None
    with pytest.raises(HTTPException):
        main.get_sheet_data("Missing", None, 1, 0, False, agent, principal, True)

    agent.list_connections.return_value = [{"connection_id": "c1"}]
    assert main.list_connections(agent, principal, True)["connections"]
    created = main.create_session(agent, principal, True)
    agent.session_store.save_turn("alice", created.session_id, "q", "a")
    assert created.session_id in main.list_sessions(agent, principal, True)["sessions"]
    assert main.clear_session(created.session_id, agent, principal, True)["status"] == "cleared"
    assert main.delete_session(created.session_id, agent, principal, True)["status"] == "deleted"
    with pytest.raises(HTTPException):
        main.clear_session("missing", agent, principal, True)


def test_readiness_dependency_failure_is_sanitized():
    principal = APIPrincipal("alice", frozenset({"admin"}))
    agent = MagicMock()
    agent.get_sheet_names.side_effect = RuntimeError("redis password")
    with pytest.raises(HTTPException) as exc:
        main.readiness(agent, principal)
    assert exc.value.status_code == 503
    assert "redis password" not in str(exc.value.detail)


def test_writer_additional_error_branches():
    spreadsheet = MagicMock()
    ws = spreadsheet.worksheet.return_value
    ws.row_values.return_value = ["id", "status"]
    ws.findall.return_value = MagicMock()
    ws.find.return_value = None
    with pytest.raises(SheetsWriterError):
        execute_delete_row(spreadsheet, "Orders", "id", "one")

    ws.findall.side_effect = None
    ws.findall.return_value = [MagicMock(row=2)]
    with pytest.raises(SheetsWriterError, match="Update column"):
        execute_update_cell(spreadsheet, "Orders", "id", "one", "missing", "x")
    ws.row_values.side_effect = RuntimeError("google unavailable")
    with pytest.raises(SheetsWriterError, match="execution failed"):
        execute_delete_row(spreadsheet, "Orders", "id", "one")


def test_typed_domain_errors_expose_stable_codes():
    assert SheetSenseError.code == "SHEETSENSE_ERROR"
    assert NotFoundError.status_code == 404
    assert AuthorizationError.status_code == 403
    assert ConflictError.status_code == 409
    assert DependencyUnavailableError.status_code == 503


def test_redis_session_management_and_lock(monkeypatch):
    client = MagicMock()
    client.ping.return_value = True
    client.smembers.return_value = {"b", "a"}
    lock = client.lock.return_value
    lock.acquire.return_value = True
    monkeypatch.setattr("redis.from_url", lambda *args, **kwargs: client)
    store = RedisSessionStore("redis://test")
    session_id = store.create_session("alice")
    assert session_id
    assert store.list_sessions("alice") == ["a", "b"]
    store.clear_history("alice", session_id)
    store.clear_session("alice", session_id)
    with store.turn_lock("alice", session_id):
        pass
    lock.release.assert_called_once()


def test_redis_lock_timeout(monkeypatch):
    client = MagicMock()
    client.ping.return_value = True
    client.lock.return_value.acquire.return_value = False
    monkeypatch.setattr("redis.from_url", lambda *args, **kwargs: client)
    store = RedisSessionStore("redis://test")
    with pytest.raises(TimeoutError):
        with store.turn_lock("alice", "session"):
            pass


def test_session_factory_fallback_and_production_failure(monkeypatch):
    monkeypatch.setattr(
        "redis.from_url",
        MagicMock(side_effect=ConnectionError("unavailable")),
    )
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("ENV", raising=False)
    assert isinstance(get_session_store(), InMemorySessionStore)
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="Redis is required"):
        get_session_store()


def test_registry_default_connection_and_url_validation(tmp_path, monkeypatch):
    import database

    path = str(tmp_path / "registry.db")
    monkeypatch.setattr(database, "DB_PATH", path)
    monkeypatch.setattr(database, "DATABASE_URL", "")
    database.init_db(path)
    client = MagicMock()
    client.open_by_key.return_value = MagicMock(title="Default")
    url = "https://docs.google.com/spreadsheets/d/abcdefghij123456/edit"
    registry = SpreadsheetRegistry(client, url)
    created = registry.ensure_default("alice")
    existing = registry.ensure_default("alice")
    assert created["connection_id"] == existing["connection_id"]
    assert normalize_spreadsheet_id(url) == "abcdefghij123456"
    with pytest.raises(ValueError):
        normalize_spreadsheet_id("http://docs.google.com/spreadsheets/d/abcdefghij123456")
    with pytest.raises(ValueError):
        normalize_spreadsheet_id("https://example.com/spreadsheets/d/abcdefghij123456")
    with pytest.raises(ValueError):
        SpreadsheetRegistry(client).register("alice", url, "owner")
