# SheetSense AI — FastAPI Model Deployment Layer
# Exposes the LangChain agent as a REST API with a single /chat endpoint.
# Any frontend, Postman, or third-party workflow can integrate via HTTP.

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import database
from agent import SheetSenseAgent  # Our LangChain agent module

# Bootstrap
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from config import Settings
from observability import (
    RequestTracingMiddleware,
    configure_opentelemetry,
    record_action,
    request_id,
    trace_id,
)
from observability import metrics as runtime_metrics
from security import APIPrincipal, SecurityHeadersMiddleware, build_auth_dependencies

settings = Settings.from_env()
_authenticator, get_api_key, require_roles = build_auth_dependencies(settings)

# Process-wide immutable service holder, initialized by the application lifespan.
_agent: Optional[SheetSenseAgent] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _agent
    settings.validate_startup()
    database.init_db()
    if _agent is None:
        try:
            logger.info("Initializing SheetSense AI agent")
            _agent = SheetSenseAgent()
        except Exception as exc:
            logger.warning("Agent startup failed: exception_type=%s", type(exc).__name__)
            if settings.is_production:
                raise
    yield


app = FastAPI(
    title="SheetSense AI",
    description=(
        "Conversational AI Agent over Live Google Sheets data. "
        "Send plain-English commands and receive live spreadsheet insights."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
configure_opentelemetry(app)

# Allow all origins for development (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestTracingMiddleware)
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from rate_limiter import rate_limit


# Global Exception Handlers (Architecture §7, PRD FR-9)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Pass-through for standard HTTP exceptions with headers (e.g. 429 Retry-After)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers or {},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Sanitized global error handler: logs stack trace securely with an error ID
    and returns a user-safe message without leaking system internals.
    """
    error_id = str(uuid.uuid4())[:8]
    logger.error(
        "[UnhandledException] error_id=%s request_id=%s trace_id=%s path=%s exception_type=%s",
        error_id,
        request_id(),
        trace_id(),
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
            "error_id": error_id,
        },
    )


def get_agent() -> SheetSenseAgent:
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "AGENT_UNAVAILABLE", "message": "The agent is not ready."},
        )
    return _agent


# Request / Response models
class ChatRequest(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "What is the total revenue for Q1?"})
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation memory.",
        json_schema_extra={"example": "user-abc-123"},
    )
    connection_id: Optional[str] = Field(
        default=None,
        description="Tenant-owned spreadsheet connection ID. Defaults to the configured sheet.",
    )
    sheet_name: Optional[str] = Field(
        default=None,
        description="Which worksheet to target. Defaults to the first sheet.",
        json_schema_extra={"example": "Sales"},
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    connection_id: str
    tool_calls_made: list[str] = Field(default_factory=list)
    raw_steps: Optional[list[Dict[str, Any]]] = None
    pending_action: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    sheets_connected: bool
    sheets_count: int


class ConnectSheetRequest(BaseModel):
    sheet_url: str = Field(
        ...,
        description="External Google Sheets URL",
        json_schema_extra={"example": "https://docs.google.com/spreadsheets/d/..."},
    )


class SessionResponse(BaseModel):
    session_id: str


# Endpoints


@app.get("/", tags=["UI"], include_in_schema=False)
def root():
    """Serve the canonical React client built by the frontend pipeline."""
    index_file = os.path.join(os.path.dirname(__file__), "frontend", "dist", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(
        status_code=503,
        content={
            "detail": {
                "code": "FRONTEND_NOT_BUILT",
                "message": "Build the frontend with 'npm run build --prefix frontend'.",
            }
        },
    )


@app.get("/favicon.svg", tags=["UI"], include_in_schema=False)
def favicon():
    """Serve the icon emitted by the frontend build."""
    icon_file = os.path.join(os.path.dirname(__file__), "frontend", "dist", "favicon.svg")
    if os.path.exists(icon_file):
        return FileResponse(icon_file, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found.")


@app.get("/health/live", tags=["Meta"], include_in_schema=False)
def liveness():
    return {"status": "ok"}


@app.get("/health/ready", tags=["Meta"])
def readiness(
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(get_api_key),
):
    try:
        sheet_names = agent.get_sheet_names(principal.principal_id)
        store = getattr(agent, "session_store", None)
        redis_client = getattr(store, "client", None)
        if redis_client is not None:
            redis_client.ping()
        return {
            "status": "ready",
            "dependencies": {
                "google_sheets": bool(sheet_names),
                "session_store": type(store).__name__ if store else "unknown",
            },
        }
    except Exception as exc:
        logger.warning("Readiness check failed: exception_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "SERVICE_NOT_READY",
                "message": "A required dependency is unavailable.",
            },
        ) from None


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health(
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(get_api_key),
    _rate_limit=Depends(rate_limit(60, 60, "health")),
):
    """Returns service health and Google Sheets connection status."""
    sheet_names = agent.get_sheet_names(principal.principal_id)
    return HealthResponse(
        status="ok",
        sheets_connected=len(sheet_names) > 0,
        sheets_count=len(sheet_names),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(
    request: ChatRequest,
    _rate_limit: bool = Depends(
        rate_limit(max_requests=60, window_seconds=60, endpoint_group="chat")
    ),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    agent: SheetSenseAgent = Depends(get_agent),
):
    """
    Send a plain-English command to the SheetSense AI agent.

    The agent will:
    1. Retrieve live data from Google Sheets via tool calls
    2. Reason step-by-step using a ReAct loop (LangChain)
    3. Return a structured, human-readable answer

    This single endpoint enables integration with any frontend or workflow.
    """
    logger.info(
        "Chat request received: session_present=%s message_length=%s",
        bool(request.session_id),
        len(request.message),
    )
    session_id = request.session_id or agent.session_store.create_session(principal.principal_id)
    try:
        result = agent.run(
            user_message=request.message,
            session_id=session_id,
            principal_id=principal.principal_id,
            connection_id=request.connection_id,
            request_id=str(uuid.uuid4()),
            sheet_name=request.sheet_name,
        )
    except PermissionError:
        raise HTTPException(
            status_code=404,
            detail={"code": "CONNECTION_NOT_FOUND", "message": "Spreadsheet connection not found."},
        ) from None
    return ChatResponse(
        answer=str(result["answer"]),
        session_id=session_id,
        connection_id=str(result.get("connection_id") or request.connection_id or "default"),
        tool_calls_made=result.get("tools_used", []),
        raw_steps=result.get("intermediate_steps"),
        pending_action=result.get("pending_action"),
    )


@app.get("/sheets", tags=["Agent"])
def list_sheets(
    connection_id: Optional[str] = Query(default=None),
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(60, 60, "sheets")),
):
    """Returns all worksheet names in the connected spreadsheet."""
    return {
        "connection_id": connection_id,
        "sheets": agent.get_sheet_names(principal.principal_id, connection_id),
    }


@app.get("/sheets/{sheet_name}/schema", tags=["Agent"])
def sheet_schema(
    sheet_name: str,
    connection_id: Optional[str] = Query(default=None),
    refresh: bool = Query(default=False),
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(60, 60, "sheets")),
):
    """Returns column names and row count for a specific worksheet."""
    schema = agent.get_sheet_schema(
        sheet_name,
        principal.principal_id,
        connection_id,
        refresh=refresh,
    )
    if not schema:
        raise HTTPException(status_code=404, detail=f"Sheet '{sheet_name}' not found.")
    return schema


@app.post("/sheets/connect", tags=["Agent"])
def connect_sheet(
    request: ConnectSheetRequest,
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("admin")),
    _rate_limit=Depends(rate_limit(10, 60, "connect")),
):
    """Dynamically connects the agent to an external Google Sheet URL."""
    try:
        result = agent.connect_spreadsheet(principal.principal_id, request.sheet_url)
        conn = result.get("connection", {})
        return {
            "status": "connected",
            "connection_id": conn.get("connection_id") or result.get("connection_id"),
            "display_name": conn.get("display_name"),
            **result,
        }
    except Exception as exc:
        error_id = str(uuid.uuid4())[:8]
        logger.error(
            "Sheet connection failed: error_id=%s exception_type=%s",
            error_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "code": "SHEET_CONNECTION_FAILED",
                "message": "The spreadsheet could not be connected.",
                "error_id": error_id,
            },
        ) from None


@app.get("/sheets/{sheet_name}/data", tags=["Agent"])
def get_sheet_data(
    sheet_name: str,
    connection_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    refresh: bool = Query(default=False),
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(60, 60, "sheets")),
):
    """Returns row records and columns for a worksheet."""
    connection, tools = agent.get_sheet_tools(principal.principal_id, connection_id)
    df = tools._get_df(sheet_name, refresh=refresh)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Sheet '{sheet_name}' not found.")
    page = df.iloc[offset : offset + limit]
    data = json.loads(page.to_json(orient="records", date_format="iso"))
    next_offset = offset + len(page) if offset + len(page) < len(df) else None
    response = {
        "sheet_name": sheet_name,
        "connection_id": connection["connection_id"],
        "total_rows": len(df),
        "columns": list(df.columns),
        "data": data,
        "offset": offset,
        "limit": limit,
        "returned_rows": len(page),
        "truncated": next_offset is not None,
        "next_offset": next_offset,
        "source": "google_sheets",
    }
    response.update(tools.cache_metadata(sheet_name))
    return response


@app.get("/connections", tags=["Agent"])
def list_connections(
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(60, 60, "connections")),
):
    """List only spreadsheet connections owned by the authenticated principal."""
    return {"connections": agent.list_connections(principal.principal_id)}


@app.post("/sessions", response_model=SessionResponse, tags=["Sessions"])
def create_session(
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(30, 60, "sessions")),
):
    return SessionResponse(session_id=agent.session_store.create_session(principal.principal_id))


@app.get("/sessions", tags=["Sessions"])
def list_sessions(
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(60, 60, "sessions")),
):
    return {"sessions": agent.session_store.list_sessions(principal.principal_id)}


@app.post("/sessions/{session_id}/clear", tags=["Sessions"])
def clear_session(
    session_id: str,
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(30, 60, "sessions")),
):
    if session_id not in agent.session_store.list_sessions(principal.principal_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    agent.session_store.clear_history(principal.principal_id, session_id)
    return {"status": "cleared", "session_id": session_id}


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def delete_session(
    session_id: str,
    agent: SheetSenseAgent = Depends(get_agent),
    principal: APIPrincipal = Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(30, 60, "sessions")),
):
    if session_id not in agent.session_store.list_sessions(principal.principal_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    agent.session_store.clear_session(principal.principal_id, session_id)
    return {"status": "deleted", "session_id": session_id}


# Confirmation Gate Endpoints (Architecture §2, §6)
import time
from datetime import datetime, timezone

import sheets_writer


class ConfirmActionResponse(BaseModel):
    status: str
    action_id: str
    tool_name: str
    result: Dict[str, Any]
    executed_at: str


def _execute_confirmed_action(
    action_id: str,
    idempotency_key: str,
    principal: APIPrincipal,
    agent: SheetSenseAgent,
) -> ConfirmActionResponse:
    claim = database.claim_pending_action(
        action_id=action_id,
        principal_id=principal.principal_id,
        idempotency_key=idempotency_key,
    )
    outcome = claim.get("claim_status")
    record_action(str(outcome or "unknown"))
    if outcome == "not_found":
        raise HTTPException(status_code=404, detail="Action not found.")
    if outcome == "forbidden":
        raise HTTPException(status_code=404, detail="Action not found.")
    if outcome == "expired":
        raise HTTPException(status_code=410, detail="Action has expired.")
    if outcome == "rejected":
        raise HTTPException(status_code=410, detail="Action was rejected.")
    if outcome == "already_confirmed":
        raise HTTPException(status_code=410, detail="Action has already been confirmed.")
    if outcome in {"in_progress", "idempotency_conflict"}:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ACTION_IN_PROGRESS",
                "message": "This action or idempotency key is already in use.",
            },
        )
    if outcome == "failed":
        raise HTTPException(
            status_code=409,
            detail={
                "code": claim.get("failure_code") or "ACTION_FAILED",
                "message": "The prior execution failed and will not be retried automatically.",
            },
        )
    if outcome == "idempotent_replay":
        result = json.loads(claim.get("result_json") or "{}")
        return ConfirmActionResponse(
            status="confirmed",
            action_id=action_id,
            tool_name=claim["tool_name"],
            result=result,
            executed_at=claim.get("executed_at") or datetime.now(timezone.utc).isoformat(),
        )
    if outcome != "claimed":
        raise HTTPException(status_code=409, detail="Action could not be claimed.")

    start_time = time.time()
    try:
        target = json.loads(claim["target_json"])
        proposed_change = json.loads(claim["proposed_change_json"])
        expected_state = (
            json.loads(claim["expected_state_json"]) if claim.get("expected_state_json") else None
        )
        connection_id = claim.get("connection_id")
        if connection_id == "default":
            connection_id = None
        _, spreadsheet = agent.resolve_spreadsheet(principal.principal_id, connection_id)
        write_result = sheets_writer.execute_action(
            spreadsheet=spreadsheet,
            tool_name=claim["tool_name"],
            target=target,
            proposed_change=proposed_change,
            expected_state=expected_state,
        )
        if not database.complete_action(action_id, write_result):
            raise RuntimeError("Action state changed before completion.")
        database.append_audit_event(
            "action.confirmed",
            "success",
            principal.principal_id,
            action_id,
            {"tool_name": claim["tool_name"]},
        )
        latency_ms = int((time.time() - start_time) * 1000)
        database.log_tool_call(
            tool_name=claim["tool_name"],
            input_data={"target": target, "proposed_change": proposed_change},
            output_data=write_result,
            success=True,
            latency_ms=latency_ms,
            session_id=claim.get("session_id"),
        )
        return ConfirmActionResponse(
            status="confirmed",
            action_id=action_id,
            tool_name=claim["tool_name"],
            result=write_result,
            executed_at=datetime.now(timezone.utc).isoformat(),
        )
    except sheets_writer.SheetsStateConflictError:
        database.fail_action(action_id, "PRECONDITION_FAILED")
        database.append_audit_event(
            "action.confirmation",
            "conflict",
            principal.principal_id,
            action_id,
            {"failure_code": "PRECONDITION_FAILED"},
        )
        raise HTTPException(
            status_code=409,
            detail={
                "code": "TARGET_CHANGED",
                "message": "The target changed after approval was requested.",
            },
        ) from None
    except Exception as exc:
        error_id = str(uuid.uuid4())[:8]
        database.fail_action(
            action_id,
            "WRITE_STATE_UNKNOWN",
            {"error_id": error_id},
        )
        database.append_audit_event(
            "action.confirmation",
            "unknown",
            principal.principal_id,
            action_id,
            {"failure_code": "WRITE_STATE_UNKNOWN", "error_id": error_id},
        )
        database.log_tool_call(
            tool_name=claim.get("tool_name", "unknown"),
            input_data=claim.get("target_json"),
            output_data={"error_id": error_id, "code": "WRITE_STATE_UNKNOWN"},
            success=False,
            latency_ms=int((time.time() - start_time) * 1000),
            session_id=claim.get("session_id"),
        )
        logger.error(
            "Confirmed action failed: error_id=%s action_id=%s exception_type=%s",
            error_id,
            action_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "WRITE_STATE_UNKNOWN",
                "message": "The write outcome could not be verified; manual review is required.",
                "error_id": error_id,
            },
        ) from None


@app.post(
    "/actions/{action_id}/confirm",
    response_model=ConfirmActionResponse,
    tags=["Safety Guardrail"],
)
def confirm_action(
    action_id: str,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    _rate_limit: bool = Depends(
        rate_limit(max_requests=20, window_seconds=60, endpoint_group="confirm")
    ),
    principal: APIPrincipal = Depends(require_roles("approver", "admin")),
    agent: SheetSenseAgent = Depends(get_agent),
):
    """
    CONFIRMATION GATE: The ONLY code path that executes destructive Google Sheets writes.

    1. Validates that the action exists, is in 'pending' status, and has not expired (5-min TTL).
    2. Calls isolated sheets_writer to execute the write/delete.
    3. Updates action status to 'confirmed' and writes an audit log to tool_calls.
    """
    return _execute_confirmed_action(
        action_id=action_id,
        idempotency_key=idempotency_key or f"action:{action_id}",
        principal=principal,
        agent=agent,
    )


@app.post("/actions/{action_id}/reject", tags=["Safety Guardrail"])
def reject_action(
    action_id: str,
    principal: APIPrincipal = Depends(require_roles("approver", "admin")),
    _rate_limit=Depends(rate_limit(20, 60, "confirm")),
):
    """Cancel / reject a pending destructive action."""
    outcome = database.reject_pending_action(action_id, principal.principal_id)
    if outcome in {"not_found", "forbidden"}:
        raise HTTPException(status_code=404, detail="Action not found.")
    if outcome != "rejected":
        raise HTTPException(
            status_code=409 if outcome in {"executing", "conflict"} else 410,
            detail=f"Action cannot be rejected in state '{outcome}'.",
        )
    return {"status": "rejected", "action_id": action_id}


# Observability & Metrics (Architecture §2, PRD FR-8)
@app.get("/metrics", tags=["Observability"])
def metrics(
    _key=Depends(require_roles("admin")),
    _rate_limit=Depends(rate_limit(60, 60, "metrics")),
):
    """Returns tool usage counts, error rates, average latency, and action counts."""
    return database.get_metrics_summary()


@app.get("/admin/audit", tags=["Observability"])
def audit_events(
    limit: int = Query(default=100, ge=1, le=500),
    _key=Depends(require_roles("admin")),
):
    return {"events": database.get_audit_events(limit)}


@app.get("/metrics/prometheus", tags=["Observability"], include_in_schema=False)
def prometheus_metrics(_key=Depends(require_roles("admin"))):
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(
        runtime_metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# Evaluation Endpoints (Architecture §6, PRD FR-8)
from evaluation_service import EvaluationService, LiveEvaluationConfig


@app.post("/eval/run", tags=["Evaluation"])
def trigger_eval_run(
    mode: str = Query(default="offline", pattern="^(offline|live|adversarial)$"),
    repetitions: int = Query(default=3, ge=2, le=10),
    _key=Depends(require_roles("admin")),
    _rate_limit=Depends(rate_limit(5, 60, "evaluation")),
):
    """
    Triggers the benchmark evaluation harness across 30 test cases,
    computes TSA, EA, CGA, IBR, and latency percentiles, and logs the run to SQLite.
    """
    try:
        service = EvaluationService()
        if mode == "offline":
            return service.run_offline()
        if mode == "adversarial":
            return service.run_adversarial()
        if _agent is None:
            raise HTTPException(status_code=503, detail="Live agent is unavailable.")
        return service.run_live(
            _agent,
            LiveEvaluationConfig(repetitions=repetitions),
        )
    except HTTPException:
        raise
    except Exception as exc:
        error_id = str(uuid.uuid4())[:8]
        logger.error(
            "Evaluation run failed: error_id=%s exception_type=%s",
            error_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EVALUATION_FAILED",
                "message": "The evaluation could not be completed.",
                "error_id": error_id,
            },
        ) from None


_frontend_assets = os.path.join(os.path.dirname(__file__), "frontend", "dist", "assets")
if os.path.isdir(_frontend_assets):
    from starlette.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_frontend_assets), name="frontend-assets")
