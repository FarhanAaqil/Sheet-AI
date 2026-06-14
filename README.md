# SheetSense AI — Conversational AI Agent over Live Spreadsheet Data

> **Python · LangChain · Google Sheets API · FastAPI**

A production-grade AI agent that lets users retrieve and update live Google Sheets data through **plain English commands** — no SQL, no formulas required.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Clients                        │
│  (Streamlit UI  /  REST API  /  Third-party)    │
└────────────────────┬────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────┐
│              FastAPI Layer (main.py)             │
│  POST /chat  │  GET /health  │  GET /sheets      │
└────────────────────┬────────────────────────────┘
                     │ calls
┌────────────────────▼────────────────────────────┐
│         LangChain ReAct Agent (agent.py)         │
│  • create_react_agent (Gemini 1.5 Pro)           │
│  • ConversationBufferWindowMemory (per session)  │
│  • AgentExecutor (max_iterations=5)              │
└────────────────────┬────────────────────────────┘
                     │ tool calls
┌────────────────────▼────────────────────────────┐
│              SheetTools (agent.py)               │
│  read_sheet │ filter_and_aggregate               │
│  update_cell │ delete_row │ summarize_sheet       │
│  list_sheets │ find_anomalies │ cross_sheet_join  │
└────────────────────┬────────────────────────────┘
                     │ gspread
┌────────────────────▼────────────────────────────┐
│           Google Sheets API (Live Data)          │
└─────────────────────────────────────────────────┘
```

---

## Project Structure

```
sheet-ai-agent/
├── agent.py          # LangChain ReAct agent + SheetTools (core logic)
├── main.py           # FastAPI deployment layer (REST API)
├── app.py            # Streamlit conversational UI
├── requirements.txt  # All dependencies
├── Dockerfile        # Container build
├── docker-compose.yml
└── .env              # Secrets (never commit)
```

---

## Key Features

| Feature | Description |
|---|---|
| **Plain-English queries** | "Show me all orders above $500 from last month" |
| **Live CRUD** | Update cells, delete rows directly in Google Sheets |
| **Multi-step reasoning** | ReAct loop: Thought → Action → Observation → repeat |
| **Tool calling** | 8 purpose-built tools the LLM can invoke |
| **Conversation memory** | Per-session context window (last 10 turns) |
| **REST API** | Single `/chat` endpoint, any client can integrate |
| **Cross-sheet joins** | Merge data across multiple worksheets |
| **Anomaly detection** | IQR-based statistical outlier detection |

---

## Quick Start

### 1. Set up environment

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/YOUR_ID/edit
GCP_CREDENTIALS_JSON={"type": "service_account", ...}
GEMINI_MODEL=gemini-1.5-pro-latest
SHEETSENSE_API_KEY=optional_api_key_for_fastapi
```

### 3. Run FastAPI server

```bash
uvicorn main:app --reload --port 8000
```

### 4. Run Streamlit UI

```bash
streamlit run app.py
```

---

## API Reference

### `POST /chat`

Send a plain-English command to the agent.

**Request:**
```json
{
  "message": "What is the total revenue for Q1?",
  "session_id": "user-123",
  "sheet_name": "Sales"
}
```

**Response:**
```json
{
  "answer": "The total Q1 revenue is $284,500.",
  "session_id": "user-123",
  "tool_calls_made": ["filter_and_aggregate"],
  "raw_steps": [...]
}
```

### `GET /health`
Returns service status and Sheets connection info.

### `GET /sheets`
Lists all available worksheet names.

### `GET /sheets/{sheet_name}/schema`
Returns columns, row count, and data types.

---

## Docker

```bash
docker-compose up --build
```

---

## Tech Stack

- **LangChain** — Agent orchestration, tool-calling, ReAct prompting, memory
- **Google Gemini 1.5 Pro** — LLM reasoning backbone  
- **Google Sheets API** (gspread) — Live data read/write
- **FastAPI** — REST API deployment layer  
- **Streamlit** — Conversational web UI  
- **Pandas** — Data manipulation engine for tool outputs  
