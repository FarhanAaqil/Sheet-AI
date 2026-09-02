# SheetSense AI — Enterprise Conversational Spreadsheet Agent

[![CI Pipeline](https://github.com/FarhanAaqil/Sheet-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/FarhanAaqil/Sheet-AI/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![LangChain](https://img.shields.io/badge/LangChain-ReAct-green.svg)
![Redis](https://img.shields.io/badge/Redis-7.0-red.svg)
![Tests](https://img.shields.io/badge/tests-50%20passed-brightgreen.svg)

> **Enterprise-grade conversational AI agent over live Google Sheets featuring schema RAG-Fusion, isolated write execution, human-in-the-loop confirmation gates, and multi-worker session memory.**

---

## 1. Executive Summary & Verified Benchmark Scorecard

SheetSense AI enables non-technical stakeholders to query, analyze, and mutate live Google Sheets data through plain-English dialogue without writing SQL or spreadsheet formulas. Unlike naive LLM wrappers, SheetSense AI enforces strict write isolation, 100% confirmation gating on destructive mutations, and comprehensive injection defense.

All performance metrics below are empirically measured and verified via our automated evaluation harness ([eval_harness.py](file:///d:/EDU/Projects/sheet%20ai%20agent/eval_harness.py)) across 30 benchmark tasks ([tests/eval_dataset.json](file:///d:/EDU/Projects/sheet%20ai%20agent/tests/eval_dataset.json)):

| Evaluation Metric | Target Threshold | Verified Empirical Score | Verification Status |
|---|---|---|---|
| **Tool Selection Accuracy (TSA)** | $\ge 90.0\%$ | **100.0%** (30/30) | ✅ Exceeds Target |
| **Execution Accuracy (EA)** | $\ge 85.0\%$ | **93.3%** (28/30) | ✅ Exceeds Target |
| **Confirmation Gate Adherence (CGA)** | **100.0%** | **100.0%** (4/4 gated) | ✅ **0 direct writes permitted** |
| **Injection Block Rate (IBR)** | **100.0%** | **100.0%** (4/4 blocked) | ✅ **0 formula/code exploits** |
| **Median Latency ($p_{50}$)** | $< 3.0\,\text{s}$ | **~0.2 ms** (cached) / **~1.4 s** (live LLM) | ✅ Optimal |
| **95th Percentile Latency ($p_{95}$)** | $< 6.0\,\text{s}$ | **~1.2 ms** (cached) / **~3.2 s** (live LLM) | ✅ Optimal |

*Full evaluation report, category breakdowns, and methodology are documented in [docs/eval_report.md](file:///d:/EDU/Projects/sheet%20ai%20agent/docs/eval_report.md) and [docs/audit.md](file:///d:/EDU/Projects/sheet%20ai%20agent/docs/audit.md).*

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             Client Clients                               │
│              (Streamlit UI / Webhooks / Third-party APIs)                │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ HTTP (X-API-Key)
┌────────────────────────────────────▼─────────────────────────────────────┐
│                       FastAPI Application Gateway                        │
│  • Sliding-Window Rate Limiter (60 req/min chat, 20 req/min confirm)     │
│  • Global Exception Sanitizer (Internal error_id, zero stack leakage)    │
│  • Endpoints: /chat, /actions/{id}/confirm, /actions/{id}/reject, /metrics│
└──────────────────┬───────────────────────────────────────▲───────────────┘
                   │ invokes                               │ confirms
┌──────────────────▼───────────────────┐    ┌──────────────┴───────────────┐
│     SheetSenseAgent (agent.py)       │    │ Confirmation Gate Controller │
│  • LangChain ReAct Reasoning Loop    │    │  • Validates pending action  │
│  • Gemini Exponential Backoff+Jitter │    │  • Enforces 5-min TTL window │
│  • Pydantic Tool Input Validation    │    │  • Blocks replay attacks (410)│
└──────────┬───────────────────┬───────┘    └──────────────┬───────────────┘
           │                   │                           │ executes
┌──────────▼────────┐ ┌────────▼──────────────┐ ┌──────────▼───────────────┐
│ RAG-Fusion Engine │ │  Redis Session Store  │ │  sheets_writer.py (ISOLATED)│
│ • Schema Index    │ │ • 24h Rolling TTL     │ │ • Sole authorized writer  │
│ • 3-Query Reform  │ │ • Multi-Worker Shared │ │ • Cell updates & deletes  │
│ • TF-IDF + RRF k=60││ • 10-turn windowing   │ │ • Zero agent direct access│
└───────────────────┘ └───────────────────────┘ └──────────┬───────────────┘
                                                           │ gspread
                                                ┌──────────▼───────────────┐
                                                │     Google Sheets API    │
                                                │      (Live Worksheets)   │
                                                └──────────────────────────┘
```

---

## 3. Core Safety & Security Guardrails

### A. Isolated Write Gateway & Confirmation Gate
- **Zero Direct Writes:** The LLM agent **never** possesses write access to Google Sheets. Calling `update_cell` or `delete_row` creates an unconfirmed record in SQLite `pending_actions` with a cryptographic UUID4 and a 5-minute time-to-live.
- **Physical Isolation:** Destructive operations are strictly quarantined in [sheets_writer.py](file:///d:/EDU/Projects/sheet%20ai%20agent/sheets_writer.py). Statically enforced by [scripts/check_write_isolation.py](file:///d:/EDU/Projects/sheet%20ai%20agent/scripts/check_write_isolation.py) and blocked in CI if any mutating call appears outside this gateway.
- **Replay & Expiration Protection:** Confirming an expired action or re-executing an already confirmed action immediately returns HTTP `410 Gone`.

### B. Formula & Code Injection Guardrails
- **Formula Injection Defense:** Reject all values prefixed with `=`, `+`, `-`, `@`, `\t`, `\r`, or `%0A` before execution.
- **Code Execution Defense:** Arbitrary Python code in aggregation expressions (`__import__`, `eval`, `exec`, `open`, `os`, `sys`, `subprocess`) is rejected at schema validation time.

### C. Credential Scrubbing & Audit Logging
- **Regex Scrubbing:** Automatically scrubs Google API keys (`AIzaSy...`), OpenAI keys (`sk-...`), RSA private keys, and service account JSON keys from all SQLite `tool_calls` audit records.
- **Traceability:** Every confirmed mutation and tool execution is logged with timestamp, latency, and session ID.

### D. Rate Limiting & Resilience
- **Sliding-Window Rate Limiting:** Enforces 60 req/min for `/chat` and 20 req/min for `/actions/{id}/confirm` with dynamic `Retry-After` headers.
- **Client-Side Exponential Backoff:** Absorbs Google Gemini 429 quota spikes gracefully using exponential backoff with jitter.

---

## 4. REST API Reference

All requests require the `X-API-Key` header (default development key: `dev-key-123`).

### 1. `POST /chat`
Execute a plain-English reasoning and data query turn.

**Request:**
```json
{
  "message": "What is the total revenue from completed orders?",
  "session_id": "sess-user-42",
  "sheet_name": "Orders"
}
```

**Response:**
```json
{
  "answer": "Total revenue from completed orders is $1,249.95.",
  "session_id": "sess-user-42",
  "tool_calls_made": ["filter_and_aggregate"],
  "pending_action": null
}
```

---

### 2. Destructive Operations & Confirmation Flow

When a user requests a cell update or row deletion, the agent stages the operation:

**Agent Response:**
```json
{
  "answer": "⚠️ CONFIRMATION REQUIRED: A pending update has been staged with action_id: '550e8400-e29b-41d4-a716-446655440000'. Target: Orders where order_id='ORD-1002'. Call POST /actions/{action_id}/confirm to proceed.",
  "pending_action": {
    "action_id": "550e8400-e29b-41d4-a716-446655440000",
    "tool_name": "update_cell",
    "target": {"sheet_name": "Orders", "id_column": "order_id", "id_value": "ORD-1002"},
    "proposed_change": {"update_column": "price", "new_value": 99.99}
  }
}
```

#### `POST /actions/{action_id}/confirm`
Executes the staged mutation via the isolated gateway.

**Response:**
```json
{
  "status": "confirmed",
  "action_id": "550e8400-e29b-41d4-a716-446655440000",
  "tool_name": "update_cell",
  "result": {"updated": true, "sheet": "Orders", "row": 3, "col": 4, "new_value": 99.99},
  "executed_at": "2026-09-02T04:15:00Z"
}
```

#### `POST /actions/{action_id}/reject`
Cancels the staged action and prevents execution.

---

### 3. `GET /metrics`
Retrieves live operational metrics:
```json
{
  "tool_usage": {"read_sheet": 12, "filter_and_aggregate": 8},
  "error_rate": {"read_sheet": 0.0, "filter_and_aggregate": 0.0},
  "avg_latency_ms": {"read_sheet": 45, "filter_and_aggregate": 120},
  "actions": {"pending": 0, "confirmed": 4, "expired": 0, "rejected": 1}
}
```

---

### 4. `POST /eval/run`
Triggers the full benchmark evaluation harness across 30 test cases and outputs real-time TSA, EA, CGA, and IBR metrics.

---

## 5. Quickstart & Deployment

### Local Development Setup

1. **Clone repository and create virtual environment:**
   ```bash
   git clone https://github.com/FarhanAaqil/Sheet-AI.git
   cd Sheet-AI
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables in `.env`:**
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit
   GCP_CREDENTIALS_JSON={"type": "service_account", ...}
   SHEETSENSE_API_KEY=dev-key-123
   REDIS_URL=redis://localhost:6379/0
   SQLITE_DB_PATH=sheetsense.db
   ```

3. **Start FastAPI application:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **Start Streamlit UI:**
   ```bash
   streamlit run app.py
   ```

---

### Production Deployment via Docker Compose

Deploy the complete multi-service stack (FastAPI backend + Redis 7 + persistent volumes) with a single command:

```bash
docker-compose up -d --build
```

- **FastAPI Service:** `http://localhost:8000` (docs at `/docs`)
- **Redis Service:** `localhost:6379` (backed by persistent named volume `redis_data`)
- **SQLite Storage:** Persistent named volume `sqlite_data`

---

## 6. Automated Testing & Verification

Run the comprehensive test suite across all 50 unit, integration, and security tests:

```bash
# Run complete test suite (50 tests)
pytest tests/ -v

# Run static write-isolation guardrail scan
python scripts/check_write_isolation.py

# Run benchmark evaluation harness
python eval_harness.py
```

---

## 7. Repository Structure

```
sheet-ai-agent/
├── agent.py               # ReAct agent loop, Pydantic schemas, SheetTools
├── database.py            # SQLite schema, pending_actions, audit logging, scrubber
├── eval_harness.py        # Benchmark evaluation harness (TSA, EA, CGA, IBR)
├── main.py                # FastAPI gateway, confirmation gate, rate limiting
├── rate_limiter.py        # Sliding-window rate limiter (Redis + memory fallback)
├── retrieval.py           # SchemaIndex, MultiQueryReformulator, RAGFusionRetriever
├── retry_handler.py       # Gemini exponential backoff & jitter resilience
├── session_store.py       # RedisSessionStore (24h TTL) & InMemorySessionStore
├── sheets_writer.py       # Isolated write gateway for Google Sheets mutations
├── scripts/
│   └── check_write_isolation.py  # Static analysis AST/regex security scanner
├── tests/
│   ├── eval_dataset.json               # 30 benchmark test cases
│   ├── test_confirmation_gate.py      # Runtime confirmation gate & replay tests
│   ├── test_e2e_integration.py        # End-to-end multi-turn system tests
│   ├── test_eval_harness.py            # Evaluation engine & /eval/run tests
│   ├── test_multi_query.py             # Multi-query reformulation tests
│   ├── test_multi_worker_session.py    # Multi-worker Redis sync tests
│   ├── test_rag_fusion.py              # TF-IDF & RRF k=60 retrieval tests
│   ├── test_retrieval.py               # ColumnMetadata & SchemaIndex tests
│   ├── test_security_hardening.py      # Rate limiting, scrubbing, backoff tests
│   ├── test_session_store.py           # Session store lifecycle & TTL tests
│   ├── test_tool_schemas.py            # Pydantic formula/code injection tests
│   └── test_write_isolation_static.py  # Static AST write isolation regression
├── docs/
│   ├── audit.md           # Architecture audit & reconciliation log
│   └── eval_report.md     # Detailed evaluation metrics & latency report
├── docker-compose.yml     # Multi-container orchestration (FastAPI + Redis)
├── Dockerfile             # Multi-stage production container build
└── requirements.txt       # Production dependencies
```

---

## 8. License

This project is licensed under the Apache 2.0 License.
