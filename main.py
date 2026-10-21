# SheetSense AI — FastAPI Model Deployment Layer
# Exposes the LangChain agent as a REST API with a single /chat endpoint.
# Any frontend, Postman, or third-party workflow can integrate via HTTP.

import os
import json
import uuid
import logging

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Request

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from starlette.staticfiles import StaticFiles

from agent import SheetSenseAgent  # Our LangChain agent module

# Bootstrap
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from config import Settings
from security import SecurityHeadersMiddleware, build_auth_dependencies

settings = Settings.from_env()
_authenticator, get_api_key, require_roles = build_auth_dependencies(settings)

# FastAPI App
app = FastAPI(
    title="SheetSense AI",
    description=(
        "Conversational AI Agent over Live Google Sheets data. "
        "Send plain-English commands and receive live spreadsheet insights."
    ),
    version="1.0.0",
)

# Allow all origins for development (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

import database
from rate_limiter import rate_limit
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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
        "[UnhandledException] error_id=%s path=%s exception_type=%s",
        error_id,
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

# Singleton agent (loaded once at startup)
_agent: Optional[SheetSenseAgent] = None

@app.on_event("startup")
async def startup_event():
    global _agent
    settings.validate_startup()
    logger.info("Initializing SQLite database tables …")
    database.init_db()
    if _agent is None:
        try:
            logger.info("Initializing SheetSense AI Agent …")
            _agent = SheetSenseAgent()
            logger.info("Agent ready.")
        except Exception as exc:
            logger.warning("Agent startup failed: exception_type=%s", type(exc).__name__)
            if settings.is_production:
                raise

def get_agent() -> SheetSenseAgent:
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "AGENT_UNAVAILABLE", "message": "The agent is not ready."},
        )
    return _agent


# Request / Response models
class ChatRequest(BaseModel):
    message: str = Field(..., example="What is the total revenue for Q1?")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation memory.",
        example="user-abc-123",
    )
    sheet_name: Optional[str] = Field(
        default=None,
        description="Which worksheet to target. Defaults to the first sheet.",
        example="Sales",
    )

class ChatResponse(BaseModel):
    answer: str
    session_id: Optional[str]
    tool_calls_made: list[str] = []
    raw_steps: Optional[list[Dict[str, Any]]] = None
    pending_action: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    sheets_connected: bool
    sheets_count: int

class ConnectSheetRequest(BaseModel):
    sheet_url: str = Field(..., description="External Google Sheets URL", example="https://docs.google.com/spreadsheets/d/...")


# Endpoints

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def root():
    """Serves the Stitch-designed interactive desktop dashboard."""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        with open(static_file, "r", encoding="utf-8") as f:
            return f.read()
    return '{"message": "SheetSense AI is running. POST to /chat to query your spreadsheet."}'

@app.get("/health/live", tags=["Meta"], include_in_schema=False)
def liveness():
    return {"status": "ok"}

@app.get("/health/ready", tags=["Meta"])
def readiness(
    agent: SheetSenseAgent = Depends(get_agent),
    _key=Depends(get_api_key),
):
    try:
        sheet_names = agent.get_sheet_names()
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
            detail={"code": "SERVICE_NOT_READY", "message": "A required dependency is unavailable."},
        ) from None


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health(
    agent: SheetSenseAgent = Depends(get_agent),
    _key=Depends(get_api_key),
    _rate_limit=Depends(rate_limit(60, 60, "health")),
):
    """Returns service health and Google Sheets connection status."""
    sheet_names = agent.get_sheet_names()
    return HealthResponse(
        status="ok",
        sheets_connected=len(sheet_names) > 0,
        sheets_count=len(sheet_names),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(
    request: ChatRequest,
    _rate_limit: bool = Depends(rate_limit(max_requests=60, window_seconds=60, endpoint_group="chat")),
    _key=Depends(require_roles("viewer", "editor", "approver", "admin")),
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
    result = agent.run(
        user_message=request.message,
        session_id=request.session_id,
        sheet_name=request.sheet_name,
    )
    return ChatResponse(
        answer=str(result["answer"]),
        session_id=request.session_id,
        tool_calls_made=result.get("tools_used", []),
        raw_steps=result.get("intermediate_steps"),
        pending_action=result.get("pending_action"),
    )



@app.get("/sheets", tags=["Agent"])
def list_sheets(
    agent: SheetSenseAgent = Depends(get_agent),
    _key=Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(60, 60, "sheets")),
):
    """Returns all worksheet names in the connected spreadsheet."""
    return {"sheets": agent.get_sheet_names()}


@app.get("/sheets/{sheet_name}/schema", tags=["Agent"])
def sheet_schema(
    sheet_name: str,
    agent: SheetSenseAgent = Depends(get_agent),
    _key=Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(60, 60, "sheets")),
):
    """Returns column names and row count for a specific worksheet."""
    schema = agent.get_sheet_schema(sheet_name)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Sheet '{sheet_name}' not found.")
    return schema


@app.post("/sheets/connect", tags=["Agent"])
def connect_sheet(
    request: ConnectSheetRequest,
    agent: SheetSenseAgent = Depends(get_agent),
    _key=Depends(require_roles("admin")),
    _rate_limit=Depends(rate_limit(10, 60, "connect")),
):
    """Dynamically connects the agent to an external Google Sheet URL."""
    try:
        sheets = agent.connect_spreadsheet(request.sheet_url)
        return {"status": "connected", "sheet_url": request.sheet_url, "sheets": sheets}
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
    limit: int = 150,
    agent: SheetSenseAgent = Depends(get_agent),
    _key=Depends(require_roles("viewer", "editor", "approver", "admin")),
    _rate_limit=Depends(rate_limit(60, 60, "sheets")),
):
    """Returns row records and columns for a worksheet."""
    df = agent.sheet_tools._get_df(sheet_name)
    if df is None:
        raise HTTPException(status_code=404, detail=f"Sheet '{sheet_name}' not found.")
    data = df.head(limit).fillna("").to_dict(orient="records")
    return {
        "sheet_name": sheet_name,
        "total_rows": len(df),
        "columns": list(df.columns),
        "data": data,
    }



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


@app.post(
    "/actions/{action_id}/confirm",
    response_model=ConfirmActionResponse,
    tags=["Safety Guardrail"],
)
def confirm_action(
    action_id: str,
    _rate_limit: bool = Depends(rate_limit(max_requests=20, window_seconds=60, endpoint_group="confirm")),
    _key=Depends(require_roles("approver", "admin")),
    agent: SheetSenseAgent = Depends(get_agent),
):


    """
    CONFIRMATION GATE: The ONLY code path that executes destructive Google Sheets writes.

    1. Validates that the action exists, is in 'pending' status, and has not expired (5-min TTL).
    2. Calls isolated sheets_writer to execute the write/delete.
    3. Updates action status to 'confirmed' and writes an audit log to tool_calls.
    """
    action = database.get_pending_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found.")

    current_status = action["status"]
    if current_status == "confirmed":
        raise HTTPException(
            status_code=410,
            detail=f"Action '{action_id}' has already been confirmed and executed.",
        )
    if current_status == "rejected":
        raise HTTPException(
            status_code=410,
            detail=f"Action '{action_id}' was previously rejected.",
        )
    if current_status != "pending":
        raise HTTPException(
            status_code=410,
            detail=f"Action '{action_id}' is no longer pending (current status: '{current_status}').",
        )

    # Validate 5-minute TTL expiration
    expires_at_str = action["expires_at"]
    try:
        expires_at = datetime.fromisoformat(expires_at_str)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires_at:
            database.update_action_status(action_id, "expired")
            raise HTTPException(
                status_code=410,
                detail=f"Action '{action_id}' has expired. Destructive actions must be confirmed within 5 minutes.",
            )
    except ValueError:
        logger.error("Invalid pending-action expiration timestamp: action_id=%s", action_id)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid expiration timestamp on action '{action_id}'.",
        )

    # Execute the write operation through sheets_writer
    start_time = time.time()
    try:
        target = json.loads(action["target_json"])
        proposed_change = json.loads(action["proposed_change_json"])
        tool_name = action["tool_name"]

        write_result = sheets_writer.execute_action(
            spreadsheet=agent.spreadsheet,
            tool_name=tool_name,
            target=target,
            proposed_change=proposed_change,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        # Update action status and write audit log
        database.update_action_status(action_id, "confirmed")
        database.log_tool_call(
            tool_name=tool_name,
            input_data={"target": target, "proposed_change": proposed_change},
            output_data=write_result,
            success=True,
            latency_ms=latency_ms,
            session_id=action.get("session_id"),
        )

        # Refresh in-memory DataFrame cache for target sheet
        sheet_name = target.get("sheet_name")
        if sheet_name:
            agent.sheet_tools._refresh_sheet(sheet_name)

        return ConfirmActionResponse(
            status="confirmed",
            action_id=action_id,
            tool_name=tool_name,
            result=write_result,
            executed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        latency_ms = int((time.time() - start_time) * 1000)
        error_id = str(uuid.uuid4())[:8]
        database.log_tool_call(
            tool_name=action.get("tool_name", "unknown"),
            input_data=action.get("target_json"),
            output_data={"error_id": error_id, "code": "WRITE_FAILED"},
            success=False,
            latency_ms=latency_ms,
            session_id=action.get("session_id"),
        )
        logger.error(
            "Confirmed action failed: error_id=%s action_id=%s exception_type=%s",
            error_id,
            action_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "WRITE_FAILED",
                "message": "The confirmed write could not be completed.",
                "error_id": error_id,
            },
        ) from None


@app.post("/actions/{action_id}/reject", tags=["Safety Guardrail"])
def reject_action(
    action_id: str,
    _key=Depends(require_roles("approver", "admin")),
    _rate_limit=Depends(rate_limit(20, 60, "confirm")),
):
    """Cancel / reject a pending destructive action."""
    action = database.get_pending_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found.")
    if action["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reject action '{action_id}' in state '{action['status']}'.",
        )
    database.update_action_status(action_id, "rejected")
    return {"status": "rejected", "action_id": action_id}


# Observability & Metrics (Architecture §2, PRD FR-8)
@app.get("/metrics", tags=["Observability"])
def metrics(
    _key=Depends(require_roles("admin")),
    _rate_limit=Depends(rate_limit(60, 60, "metrics")),
):
    """Returns tool usage counts, error rates, average latency, and action counts."""
    return database.get_metrics_summary()


# Evaluation Endpoints (Architecture §6, PRD FR-8)
import eval_harness


@app.post("/eval/run", tags=["Evaluation"])
def trigger_eval_run(
    _key=Depends(require_roles("admin")),
    _rate_limit=Depends(rate_limit(5, 60, "evaluation")),
):
    """
    Triggers the benchmark evaluation harness across 30 test cases,
    computes TSA, EA, CGA, IBR, and latency percentiles, and logs the run to SQLite.
    """
    try:
        summary = eval_harness.run_evaluation()
        return summary
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
