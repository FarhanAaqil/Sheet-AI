# =============================================================================
# SheetSense AI — FastAPI Model Deployment Layer
# =============================================================================
# Exposes the LangChain agent as a REST API with a single /chat endpoint.
# Any frontend, Postman, or third-party workflow can integrate via HTTP.
# =============================================================================

import os
import json
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent import SheetSenseAgent  # Our LangChain agent module

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optional: API Key auth (set SHEETSENSE_API_KEY in .env to enable)
# ---------------------------------------------------------------------------
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("SHEETSENSE_API_KEY")  # None means auth is disabled

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(key: Optional[str] = Security(api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return key

from database import init_db

# ---------------------------------------------------------------------------
# Singleton agent (loaded once at startup)
# ---------------------------------------------------------------------------
_agent: Optional[SheetSenseAgent] = None

@app.on_event("startup")
async def startup_event():
    global _agent
    logger.info("Initializing SQLite database tables …")
    init_db()
    logger.info("Initializing SheetSense AI Agent …")
    _agent = SheetSenseAgent()
    logger.info("Agent ready.")

def get_agent() -> SheetSenseAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet.")
    return _agent

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
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

class HealthResponse(BaseModel):
    status: str
    sheets_connected: bool
    sheets_count: int

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Meta"])
def root():
    return {"message": "SheetSense AI is running. POST to /chat to query your spreadsheet."}


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health(agent: SheetSenseAgent = Depends(get_agent), _key=Depends(get_api_key)):
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
    agent: SheetSenseAgent = Depends(get_agent),
    _key: str = Depends(get_api_key),
):
    """
    Send a plain-English command to the SheetSense AI agent.

    The agent will:
    1. Retrieve live data from Google Sheets via tool calls
    2. Reason step-by-step using a ReAct loop (LangChain)
    3. Return a structured, human-readable answer

    This single endpoint enables integration with any frontend or workflow.
    """
    logger.info(f"[{request.session_id}] Query: {request.message}")
    try:
        result = agent.run(
            user_message=request.message,
            session_id=request.session_id,
            sheet_name=request.sheet_name,
        )
        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id,
            tool_calls_made=result.get("tools_used", []),
            raw_steps=result.get("intermediate_steps"),
        )
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/sheets", tags=["Agent"])
def list_sheets(
    agent: SheetSenseAgent = Depends(get_agent),
    _key: str = Depends(get_api_key),
):
    """Returns all worksheet names in the connected spreadsheet."""
    return {"sheets": agent.get_sheet_names()}


@app.get("/sheets/{sheet_name}/schema", tags=["Agent"])
def sheet_schema(
    sheet_name: str,
    agent: SheetSenseAgent = Depends(get_agent),
    _key: str = Depends(get_api_key),
):
    """Returns column names and row count for a specific worksheet."""
    schema = agent.get_sheet_schema(sheet_name)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Sheet '{sheet_name}' not found.")
    return schema
