# =============================================================================
# SheetSense AI — SQLite Database Layer
# =============================================================================
# Manages tool_calls audit log, pending_actions confirmation gate store,
# and eval_runs history.
# =============================================================================

import os
import re
import json
import sqlite3
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SQLITE_DB_PATH", os.path.join(os.path.dirname(__file__), "sheetsense.db"))


def get_db_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None):
    """Create all required tables if they do not exist."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS tool_calls (
        call_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        tool_name TEXT NOT NULL,
        input_json TEXT,
        output_json TEXT,
        success BOOLEAN NOT NULL DEFAULT 1,
        latency_ms INTEGER DEFAULT 0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS pending_actions (
        action_id TEXT PRIMARY KEY,
        session_id TEXT,
        tool_name TEXT NOT NULL,
        target_json TEXT NOT NULL,
        proposed_change_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expires_at TIMESTAMP NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending' -- pending | confirmed | expired | rejected
    );

    CREATE TABLE IF NOT EXISTS eval_runs (
        eval_id TEXT PRIMARY KEY,
        total_queries INTEGER NOT NULL,
        tool_selection_accuracy REAL NOT NULL,
        answer_correctness_rate REAL NOT NULL,
        guardrail_compliance_rate REAL NOT NULL,
        run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at '{path}'.")


# ---------------------------------------------------------------------------
# Credential Scrubbing (Architecture §7)
# ---------------------------------------------------------------------------
SENSITIVE_PATTERNS = [
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{33}"),                            # Google API key
    re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"),                             # OpenAI API key
    re.compile(r"-----BEGIN PRIVATE KEY-----[^-]+-----END PRIVATE KEY-----", re.DOTALL), # RSA/Private key
    re.compile(r'"private_key":\s*"[^"]+"'),                           # Service account private key
    re.compile(r'"client_email":\s*"[^"]+"'),                          # Service account email
    re.compile(r'xox[baprs]-[0-9a-zA-Z\-]+'),                          # Slack tokens
]


def scrub_credentials(text: Any) -> str:
    """Scrub sensitive keys, passwords, and private tokens from string or JSON."""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, default=str)
        except Exception:
            text = str(text)

    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED_CREDENTIAL]", text)
    return text


# ---------------------------------------------------------------------------
# Tool Call Logging Helper
# ---------------------------------------------------------------------------
def log_tool_call(
    tool_name: str,
    input_data: Any,
    output_data: Any,
    success: bool = True,
    latency_ms: int = 0,
    session_id: Optional[str] = None,
) -> int:
    """Insert a scrubbed tool call audit record."""
    clean_input = scrub_credentials(input_data)
    clean_output = scrub_credentials(output_data)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO tool_calls (session_id, tool_name, input_json, output_json, success, latency_ms, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            tool_name,
            clean_input,
            clean_output,
            1 if success else 0,
            latency_ms,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    call_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return call_id


# ---------------------------------------------------------------------------
# Pending Actions Helper (Architecture §2, §6)
# ---------------------------------------------------------------------------
def create_pending_action(
    action_id: str,
    tool_name: str,
    target: Dict[str, Any],
    proposed_change: Dict[str, Any],
    session_id: Optional[str] = None,
    ttl_minutes: int = 5,
) -> Dict[str, Any]:
    """Create and persist a pending destructive action with a 5-minute TTL."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=ttl_minutes)

    target_json = json.dumps(target)
    proposed_change_json = json.dumps(proposed_change)

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO pending_actions (action_id, session_id, tool_name, target_json, proposed_change_json, created_at, expires_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            action_id,
            session_id,
            tool_name,
            target_json,
            proposed_change_json,
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


# ---------------------------------------------------------------------------
# Metrics Aggregation Helper (Architecture §2)
# ---------------------------------------------------------------------------
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
    actions = {"pending": 0, "confirmed": 0, "expired": 0, "rejected": 0}
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


# ---------------------------------------------------------------------------
# Evaluation Run Logging (Architecture §6, PRD FR-8)
# ---------------------------------------------------------------------------
def log_eval_run(
    eval_id: str,
    total_queries: int,
    tool_selection_accuracy: float,
    answer_correctness_rate: float,
    guardrail_compliance_rate: float,
) -> None:
    """Insert a benchmark evaluation run record into SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO eval_runs (
            eval_id, total_queries, tool_selection_accuracy,
            answer_correctness_rate, guardrail_compliance_rate, run_at
        )
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        """,
        (
            eval_id,
            total_queries,
            tool_selection_accuracy,
            answer_correctness_rate,
            guardrail_compliance_rate,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_eval_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent benchmark evaluation run records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM eval_runs ORDER BY run_at DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


