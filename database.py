# SheetSense AI — SQLite Database Layer
# Manages tool_calls audit log, pending_actions confirmation gate store,
# and eval_runs history.

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from security import redact_sensitive

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(os.path.dirname(__file__), "sheetsense.db"))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


class _PostgresCursor:
    def __init__(self, cursor):
        self._cursor = cursor
        self.lastrowid = None

    @staticmethod
    def _sql(query: str) -> str:
        return query.replace("?", "%s").replace("BEGIN IMMEDIATE", "BEGIN")

    def execute(self, query: str, params=()):
        self._cursor.execute(self._sql(query), params)
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def rowcount(self):
        return self._cursor.rowcount


class _PostgresConnection:
    def __init__(self, connection):
        self._connection = connection

    def cursor(self):
        return _PostgresCursor(self._connection.cursor())

    def execute(self, query: str, params=()):
        return self.cursor().execute(query, params)

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


def get_db_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database with Row factory."""
    if DATABASE_URL:
        import psycopg
        from psycopg.rows import dict_row

        url = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1)
        return _PostgresConnection(psycopg.connect(url, row_factory=dict_row))
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None):
    """Create all required tables if they do not exist."""
    if DATABASE_URL and db_path is None:
        conn = get_db_connection()
        try:
            row = conn.execute(
                "SELECT to_regclass('public.pending_actions') AS table_name"
            ).fetchone()
            if not row or not row["table_name"]:
                raise RuntimeError("PostgreSQL schema is not migrated. Run 'alembic upgrade head'.")
        finally:
            conn.close()
        return
    path = db_path or DB_PATH
    conn = sqlite3.connect(path, timeout=20.0)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS tool_calls (
        call_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        tool_name TEXT NOT NULL,
        input_json TEXT,
        output_json TEXT,
        success BOOLEAN NOT NULL DEFAULT 1,
        latency_ms INTEGER DEFAULT 0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        request_id TEXT,
        trace_id TEXT
    );

    CREATE TABLE IF NOT EXISTS pending_actions (
        action_id TEXT PRIMARY KEY,
        session_id TEXT,
        principal_id TEXT NOT NULL DEFAULT 'legacy-admin',
        connection_id TEXT,
        tool_name TEXT NOT NULL,
        target_json TEXT NOT NULL,
        proposed_change_json TEXT NOT NULL,
        expected_state_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        claimed_at TIMESTAMP,
        executed_at TIMESTAMP,
        failure_code TEXT,
        idempotency_key TEXT,
        result_json TEXT
    );

    CREATE TABLE IF NOT EXISTS spreadsheet_connections (
        connection_id TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        spreadsheet_id TEXT NOT NULL,
        display_name TEXT NOT NULL,
        permission_level TEXT NOT NULL DEFAULT 'editor',
        created_at TIMESTAMP NOT NULL,
        last_sync_at TIMESTAMP,
        UNIQUE(principal_id, spreadsheet_id)
    );

    CREATE TABLE IF NOT EXISTS eval_runs (
        eval_id TEXT PRIMARY KEY,
        total_queries INTEGER NOT NULL,
        tool_selection_accuracy REAL NOT NULL,
        answer_correctness_rate REAL NOT NULL,
        guardrail_compliance_rate REAL NOT NULL,
        run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        mode TEXT NOT NULL DEFAULT 'offline_deterministic',
        model_name TEXT,
        model_version TEXT,
        dataset_version TEXT,
        metrics_json TEXT,
        baseline_delta_json TEXT
    );

    CREATE TABLE IF NOT EXISTS eval_results (
        run_id TEXT NOT NULL,
        case_id TEXT NOT NULL,
        category TEXT NOT NULL,
        passed BOOLEAN NOT NULL,
        latency_ms REAL NOT NULL,
        result_json TEXT NOT NULL,
        PRIMARY KEY (run_id, case_id),
        FOREIGN KEY (run_id) REFERENCES eval_runs(eval_id)
    );

    CREATE TABLE IF NOT EXISTS audit_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        principal_id TEXT,
        request_id TEXT,
        trace_id TEXT,
        resource_id TEXT,
        outcome TEXT NOT NULL,
        detail_json TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    );
    """)

    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(pending_actions)").fetchall()
    }
    migrations = {
        "principal_id": "TEXT NOT NULL DEFAULT 'legacy-admin'",
        "connection_id": "TEXT",
        "expected_state_json": "TEXT",
        "claimed_at": "TIMESTAMP",
        "executed_at": "TIMESTAMP",
        "failure_code": "TEXT",
        "idempotency_key": "TEXT",
        "result_json": "TEXT",
    }
    for column, definition in migrations.items():
        if column not in existing_columns:
            cursor.execute(f"ALTER TABLE pending_actions ADD COLUMN {column} {definition}")
    tool_columns = {row[1] for row in cursor.execute("PRAGMA table_info(tool_calls)").fetchall()}
    for column in ("request_id", "trace_id"):
        if column not in tool_columns:
            cursor.execute(f"ALTER TABLE tool_calls ADD COLUMN {column} TEXT")
    eval_columns = {row[1] for row in cursor.execute("PRAGMA table_info(eval_runs)").fetchall()}
    eval_migrations = {
        "mode": "TEXT NOT NULL DEFAULT 'offline_deterministic'",
        "model_name": "TEXT",
        "model_version": "TEXT",
        "dataset_version": "TEXT",
        "metrics_json": "TEXT",
        "baseline_delta_json": "TEXT",
    }
    for column, definition in eval_migrations.items():
        if column not in eval_columns:
            cursor.execute(f"ALTER TABLE eval_runs ADD COLUMN {column} {definition}")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_owner "
        "ON pending_actions(principal_id, status, expires_at)"
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_action_idempotency "
        "ON pending_actions(principal_id, idempotency_key) "
        "WHERE idempotency_key IS NOT NULL"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_connections_owner "
        "ON spreadsheet_connections(principal_id, created_at)"
    )

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at '{path}'.")


# Credential Scrubbing (Architecture §7)
SENSITIVE_PATTERNS = [
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{33}"),  # Google API key
    re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"),  # OpenAI API key
    re.compile(
        r"-----BEGIN PRIVATE KEY-----[^-]+-----END PRIVATE KEY-----", re.DOTALL
    ),  # RSA/Private key
    re.compile(r'"private_key":\s*"[^"]+"'),  # Service account private key
    re.compile(r'"client_email":\s*"[^"]+"'),  # Service account email
    re.compile(r"xox[baprs]-[0-9a-zA-Z\-]+"),  # Slack tokens
]


def scrub_credentials(text: Any) -> str:
    """Scrub sensitive keys, passwords, and private tokens from string or JSON."""
    return redact_sensitive(text)


# Tool Call Logging Helper
def log_tool_call(
    tool_name: str,
    input_data: Any,
    output_data: Any,
    success: bool = True,
    latency_ms: int = 0,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> int:
    """Insert a scrubbed tool call audit record."""
    clean_input = scrub_credentials(input_data)
    clean_output = scrub_credentials(output_data)

    conn = get_db_connection()
    cursor = conn.cursor()
    if request_id is None or trace_id is None:
        from observability import request_id as current_request_id
        from observability import trace_id as current_trace_id

        request_id = request_id or current_request_id()
        trace_id = trace_id or current_trace_id()
    insert_sql = """
        INSERT INTO tool_calls (
            session_id, tool_name, input_json, output_json, success, latency_ms,
            timestamp, request_id, trace_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """ + (" RETURNING call_id" if DATABASE_URL else "")
    cursor.execute(
        insert_sql,
        (
            session_id,
            tool_name,
            clean_input,
            clean_output,
            1 if success else 0,
            latency_ms,
            datetime.now(timezone.utc).isoformat(),
            request_id,
            trace_id,
        ),
    )
    call_id = cursor.fetchone()["call_id"] if DATABASE_URL else cursor.lastrowid
    conn.commit()
    conn.close()
    from observability import record_tool

    record_tool(tool_name, success, latency_ms)
    return call_id


# Pending Actions Helper (Architecture §2, §6)
def create_pending_action(
    action_id: str,
    tool_name: str,
    target: Dict[str, Any],
    proposed_change: Dict[str, Any],
    session_id: Optional[str] = None,
    principal_id: str = "legacy-admin",
    connection_id: Optional[str] = None,
    expected_state: Optional[Dict[str, Any]] = None,
    ttl_minutes: int = 5,
) -> Dict[str, Any]:
    """Create and persist a pending destructive action with a 5-minute TTL."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ttl_minutes)

    target_json = json.dumps(target)
    proposed_change_json = json.dumps(proposed_change)
    expected_state_json = json.dumps(expected_state) if expected_state is not None else None

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO pending_actions (
            action_id, session_id, principal_id, connection_id, tool_name,
            target_json, proposed_change_json, expected_state_json,
            created_at, expires_at, status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            action_id,
            session_id,
            principal_id,
            connection_id,
            tool_name,
            target_json,
            proposed_change_json,
            expected_state_json,
            now.isoformat(),
            expires_at.isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    return {
        "action_id": action_id,
        "tool_name": tool_name,
        "target": target,
        "proposed_change": proposed_change,
        "principal_id": principal_id,
        "connection_id": connection_id,
        "requires_confirmation": True,
        "expires_at": expires_at.isoformat(),
    }


def get_pending_action(action_id: str) -> Optional[Dict[str, Any]]:
    """Fetch pending action by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pending_actions WHERE action_id = ?", (action_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def update_action_status(action_id: str, status: str) -> bool:
    """Update the status of a pending action (e.g. confirmed, expired, rejected)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE pending_actions SET status = ? WHERE action_id = ?",
        (status, action_id),
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def claim_pending_action(
    action_id: str,
    principal_id: str,
    idempotency_key: str,
    stale_after_seconds: int = 120,
) -> Dict[str, Any]:
    """Atomically claim one unexpired pending action for one execution attempt."""
    now = datetime.now(timezone.utc)
    stale_before = now - timedelta(seconds=stale_after_seconds)
    conn = get_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
        ).fetchone()
        if row is None:
            conn.commit()
            return {"claim_status": "not_found"}
        action = dict(row)
        if action.get("principal_id") != principal_id:
            conn.commit()
            return {"claim_status": "forbidden"}

        if action["status"] == "executing" and action.get("claimed_at"):
            claimed_at = datetime.fromisoformat(action["claimed_at"])
            if claimed_at.tzinfo is None:
                claimed_at = claimed_at.replace(tzinfo=timezone.utc)
            if claimed_at <= stale_before:
                conn.execute(
                    "UPDATE pending_actions SET status='failed', "
                    "failure_code='EXECUTION_STATE_UNKNOWN', executed_at=? "
                    "WHERE action_id=? AND status='executing'",
                    (now.isoformat(), action_id),
                )
                action["status"] = "failed"
                action["failure_code"] = "EXECUTION_STATE_UNKNOWN"

        if action["status"] == "confirmed":
            action["claim_status"] = (
                "idempotent_replay"
                if action.get("idempotency_key") == idempotency_key
                else "already_confirmed"
            )
            conn.commit()
            return action
        if action["status"] in {"rejected", "expired", "failed"}:
            action["claim_status"] = action["status"]
            conn.commit()
            return action
        if action["status"] == "executing":
            action["claim_status"] = (
                "in_progress"
                if action.get("idempotency_key") == idempotency_key
                else "idempotency_conflict"
            )
            conn.commit()
            return action

        expires_at = datetime.fromisoformat(action["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            conn.execute(
                "UPDATE pending_actions SET status='expired', executed_at=? "
                "WHERE action_id=? AND status='pending'",
                (now.isoformat(), action_id),
            )
            conn.commit()
            action["claim_status"] = "expired"
            return action

        cursor = conn.execute(
            "UPDATE pending_actions SET status='executing', claimed_at=?, "
            "idempotency_key=? WHERE action_id=? AND principal_id=? "
            "AND status='pending' AND expires_at>?",
            (now.isoformat(), idempotency_key, action_id, principal_id, now.isoformat()),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            return {"claim_status": "in_progress"}
        conn.commit()
        claimed = get_pending_action(action_id) or {}
        claimed["claim_status"] = "claimed"
        return claimed
    except sqlite3.IntegrityError:
        conn.rollback()
        return {"claim_status": "idempotency_conflict"}
    finally:
        conn.close()


def complete_action(action_id: str, result: Dict[str, Any]) -> bool:
    """Transition executing -> confirmed and persist the idempotent result."""
    conn = get_db_connection()
    cursor = conn.execute(
        "UPDATE pending_actions SET status='confirmed', executed_at=?, "
        "result_json=?, failure_code=NULL WHERE action_id=? AND status='executing'",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(result, default=str),
            action_id,
        ),
    )
    conn.commit()
    updated = cursor.rowcount == 1
    conn.close()
    return updated


def fail_action(
    action_id: str,
    failure_code: str,
    result: Optional[Dict[str, Any]] = None,
) -> bool:
    """Transition executing -> failed. Failed actions are never auto-retried."""
    conn = get_db_connection()
    cursor = conn.execute(
        "UPDATE pending_actions SET status='failed', executed_at=?, "
        "failure_code=?, result_json=? WHERE action_id=? AND status='executing'",
        (
            datetime.now(timezone.utc).isoformat(),
            failure_code,
            json.dumps(result or {}, default=str),
            action_id,
        ),
    )
    conn.commit()
    updated = cursor.rowcount == 1
    conn.close()
    return updated


def reject_pending_action(action_id: str, principal_id: str) -> str:
    """Atomically reject an owned pending action, or expire it if its TTL elapsed."""
    now = datetime.now(timezone.utc)
    conn = get_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT principal_id, status, expires_at FROM pending_actions WHERE action_id=?",
            (action_id,),
        ).fetchone()
        if row is None:
            conn.commit()
            return "not_found"
        if row["principal_id"] != principal_id:
            conn.commit()
            return "forbidden"
        if row["status"] != "pending":
            conn.commit()
            return row["status"]
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        status = "expired" if expires_at <= now else "rejected"
        cursor = conn.execute(
            "UPDATE pending_actions SET status=?, executed_at=? "
            "WHERE action_id=? AND status='pending'",
            (status, now.isoformat(), action_id),
        )
        conn.commit()
        return status if cursor.rowcount == 1 else "conflict"
    finally:
        conn.close()


def upsert_spreadsheet_connection(
    connection_id: str,
    principal_id: str,
    spreadsheet_id: str,
    display_name: str,
    permission_level: str = "editor",
) -> Dict[str, Any]:
    """Create or refresh a tenant-owned spreadsheet connection."""
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO spreadsheet_connections (
            connection_id, principal_id, spreadsheet_id, display_name,
            permission_level, created_at, last_sync_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(principal_id, spreadsheet_id) DO UPDATE SET
            display_name=excluded.display_name,
            permission_level=excluded.permission_level,
            last_sync_at=excluded.last_sync_at
        """,
        (
            connection_id,
            principal_id,
            spreadsheet_id,
            display_name,
            permission_level,
            now,
            now,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM spreadsheet_connections WHERE principal_id=? AND spreadsheet_id=?",
        (principal_id, spreadsheet_id),
    ).fetchone()
    conn.close()
    return dict(row)


def get_spreadsheet_connection(connection_id: str, principal_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    row = conn.execute(
        "SELECT * FROM spreadsheet_connections WHERE connection_id=? AND principal_id=?",
        (connection_id, principal_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_spreadsheet_connections(principal_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM spreadsheet_connections WHERE principal_id=? ORDER BY created_at",
        (principal_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def touch_spreadsheet_connection(connection_id: str, principal_id: str) -> None:
    conn = get_db_connection()
    conn.execute(
        "UPDATE spreadsheet_connections SET last_sync_at=? "
        "WHERE connection_id=? AND principal_id=?",
        (datetime.now(timezone.utc).isoformat(), connection_id, principal_id),
    )
    conn.commit()
    conn.close()


# Metrics Aggregation Helper (Architecture §2)
def get_metrics_summary() -> Dict[str, Any]:
    """Calculate aggregated metrics for the GET /metrics endpoint."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Tool usage counts & average latency
    cursor.execute("""
        SELECT 
            tool_name,
            COUNT(*) as total_calls,
            SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_calls,
            AVG(latency_ms) as avg_latency
        FROM tool_calls
        GROUP BY tool_name
    """)
    tool_rows = cursor.fetchall()

    tool_usage = {}
    error_rate = {}
    avg_latency_ms = {}

    for row in tool_rows:
        t_name = row["tool_name"]
        total = row["total_calls"]
        failed = row["failed_calls"]
        tool_usage[t_name] = total
        error_rate[t_name] = round(failed / total, 4) if total > 0 else 0.0
        avg_latency_ms[t_name] = round(row["avg_latency"] or 0, 2)

    # Action counts
    cursor.execute("""
        SELECT status, COUNT(*) as count
        FROM pending_actions
        GROUP BY status
    """)
    action_rows = cursor.fetchall()
    actions = {
        "pending": 0,
        "executing": 0,
        "confirmed": 0,
        "failed": 0,
        "expired": 0,
        "rejected": 0,
    }
    for row in action_rows:
        status = row["status"]
        if status in actions:
            actions[status] = row["count"]

    conn.close()

    return {
        "tool_usage": tool_usage,
        "error_rate": error_rate,
        "avg_latency_ms": avg_latency_ms,
        "actions": actions,
    }


# Evaluation Run Logging (Architecture §6, PRD FR-8)
def log_eval_run(
    eval_id: str,
    total_queries: int,
    tool_selection_accuracy: float,
    answer_correctness_rate: float,
    guardrail_compliance_rate: float,
    mode: str = "offline_deterministic",
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    dataset_version: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    baseline_delta: Optional[Dict[str, Any]] = None,
    cases: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Persist aggregate evaluation evidence and every individual case result."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO eval_runs (
            eval_id, total_queries, tool_selection_accuracy,
            answer_correctness_rate, guardrail_compliance_rate, run_at,
            mode, model_name, model_version, dataset_version, metrics_json,
            baseline_delta_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            eval_id,
            total_queries,
            tool_selection_accuracy,
            answer_correctness_rate,
            guardrail_compliance_rate,
            datetime.now(timezone.utc).isoformat(),
            mode,
            model_name,
            model_version,
            dataset_version,
            json.dumps(metrics or {}, default=str),
            json.dumps(baseline_delta or {}, default=str),
        ),
    )
    for case in cases or []:
        cursor.execute(
            """
            INSERT INTO eval_results (
                run_id, case_id, category, passed, latency_ms, result_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id,
                str(case.get("id", "")),
                str(case.get("category", "unknown")),
                1 if case.get("passed", case.get("ea_pass", False)) else 0,
                float(case.get("latency_ms", 0)),
                json.dumps(case, default=str),
            ),
        )
    conn.commit()
    conn.close()


def get_recent_eval_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent benchmark evaluation run records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM eval_runs ORDER BY run_at DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def append_audit_event(
    event_type: str,
    outcome: str,
    principal_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> int:
    """Append a redacted business audit event; no update/delete API is exposed."""
    from observability import request_id, trace_id

    conn = get_db_connection()
    audit_sql = """
        INSERT INTO audit_events (
            event_type, principal_id, request_id, trace_id, resource_id,
            outcome, detail_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """ + (" RETURNING event_id" if DATABASE_URL else "")
    cursor = conn.execute(
        audit_sql,
        (
            event_type,
            principal_id,
            request_id(),
            trace_id(),
            resource_id,
            outcome,
            scrub_credentials(detail or {}),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    event_id = int(cursor.fetchone()["event_id"]) if DATABASE_URL else int(cursor.lastrowid)
    conn.close()
    return event_id


def get_audit_events(limit: int = 100) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT event_id, event_type, principal_id, request_id, trace_id,
               resource_id, outcome, detail_json, created_at
        FROM audit_events ORDER BY event_id DESC LIMIT ?
        """,
        (min(max(limit, 1), 500),),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
