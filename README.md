# SheetSense AI — Conversational Spreadsheet Agent

[![CI Pipeline](https://github.com/FarhanAaqil/Sheet-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/FarhanAaqil/Sheet-AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![LangChain](https://img.shields.io/badge/LangChain-ReAct-green.svg)
![Redis](https://img.shields.io/badge/Redis-7.0-red.svg)
![Tests](https://img.shields.io/badge/tests-84%20passed-brightgreen.svg)

> **Production-ready conversational AI agent over live Google Sheets featuring schema RAG-Fusion, physically isolated write execution, cryptographic human-in-the-loop confirmation gates, and distributed session memory.**

---

### Quick Navigation
[⚡ Quickstart](#1-quickstart-under-2-minutes) • [✨ Key Capabilities](#2-key-capabilities) • [🏗️ System Architecture](#3-system-architecture) • [🛡️ Safety & Security](#4-safety--security-guardrails) • [📡 REST API Reference](#5-rest-api-reference) • [🖥️ Web Dashboards](#6-web-dashboards) • [📊 Benchmark Scorecard](#7-verified-benchmark-scorecard) • [🧪 Testing](#8-testing--verification) • [📁 Project Structure](#9-repository-structure) • [📄 License](#10-license)

---

## 1. Quickstart (Under 2 Minutes)

Get SheetSense AI running locally in under two minutes using either Docker Compose or a standard Python virtual environment.

### Option A: Docker Compose (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/FarhanAaqil/Sheet-AI.git
cd Sheet-AI

# 2. Configure environment
cp .env.example .env  # or edit .env directly with your keys

# 3. Launch backend + Redis multi-service stack
docker-compose up -d --build
```
- **Interactive Web Dashboard & REST API:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redis Service:** `localhost:6379` (backed by persistent named volume)

---

### Option B: Local Python Environment

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start FastAPI server
uvicorn main:app --reload --port 8000

# 4. (Optional) Launch Streamlit companion UI in a second terminal
streamlit run app.py
```

### Required Configuration (`.env`)

```env
# LLM Provider
GEMINI_API_KEY=your_gemini_api_key

# Google Sheets Integration
GOOGLE_SHEET_URL=https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit
GCP_CREDENTIALS_JSON={"type": "service_account", ...}

# Gateway Security & Session Storage
SHEETSENSE_API_KEY=replace-with-a-random-secret
REDIS_URL=redis://localhost:6379/0
SQLITE_DB_PATH=sheetsense.db
```

---

## 2. Key Capabilities

- 💬 **Natural Language Querying:** Query, filter, and summarize complex spreadsheet data without writing SQL or formulas.
- 🛡️ **Zero-Bypass Isolated Write Gateway:** The LLM reasoning agent is physically stripped of write capabilities. All mutations are quarantined to [sheets_writer.py](sheets_writer.py) and verified by static AST analysis in CI.
- 🔐 **Human-in-the-Loop Confirmation Gate:** Destructive actions (`update_cell`, `delete_row`) generate an unconfirmed staging record with a cryptographic UUID4 and an enforced 5-minute TTL.
- 🔍 **Schema RAG-Fusion:** Automatically extracts column semantics, performs 3-query expansion, and ranks schemas using TF-IDF and Reciprocal Rank Fusion ($k=60$).
- 🛡️ **Zero-Eval AST & Formula Shield:** Completely eliminates Python `eval()` and `exec()`. Blocks formula injection prefixes (`=`, `+`, `-`, `@`, `\t`, `\r`) while preserving legitimate negative numbers.
- 🧠 **Multi-Worker Redis Session Memory:** Production-grade distributed conversation memory with 24-hour rolling TTL, sliding 10-turn window, and automatic in-memory fallback.
- 📊 **Real-time Observability & Evaluation:** Built-in metrics (`/metrics`) and an automated 30-task evaluation benchmark (`eval_harness.py`).

---

## 3. System Architecture

SheetSense AI strictly isolates analytical reasoning from write execution:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             Client Interfaces                            │
│              (Embedded Dashboard / Streamlit UI / REST APIs)             │
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

## 4. Safety & Security Guardrails

### 1. Isolated Write Gateway & Confirmation Gate
- **Zero Direct Writes:** The LLM agent **never** has direct write access to Google Sheets. Calling `update_cell` or `delete_row` stages an unconfirmed record in SQLite `pending_actions` with a UUID4 and 5-minute TTL.
- **Physical Isolation:** Destructive operations are strictly quarantined in [sheets_writer.py](sheets_writer.py). Statically enforced by [scripts/check_write_isolation.py](scripts/check_write_isolation.py) and blocked in CI if any mutating call appears outside this gateway.
- **Replay Protection:** Attempting to re-execute an already confirmed or expired action returns HTTP `410 Gone`.

### 2. Formula & Code Injection Guardrails
- **Formula Injection Defense:** Rejects spreadsheet formula injection prefixes (`=`, `+`, `-`, `@`, `\t`, `\r`, `%0A`) on text/formula values, while correctly permitting legitimate positive and negative numbers (e.g. `-100`, `"-100"`, `-42.5`).
- **Zero-Eval AST Safe Execution:** Completely eliminates Python `eval()` and `exec()`. Math expressions execute via a strict AST evaluator that whitelists safe operators and blocks arbitrary code execution, imports, builtins, and comprehensions.

### 3. Credential Scrubbing & Audit Logging
- **Regex Scrubbing:** Automatically redacts Google API keys (`AIzaSy...`), OpenAI keys (`sk-...`), RSA private keys, and service account secrets from all `tool_calls` audit records.
- **Traceability:** Every confirmed mutation and tool execution is recorded in SQLite with timestamp, execution latency, and session ID.

### 4. Rate Limiting & Resilience
- **Sliding-Window Limiter:** Enforces 60 req/min for `/chat` and 20 req/min for `/actions/{id}/confirm` with dynamic `Retry-After` headers.
- **Exponential Backoff:** Absorbs Gemini API 429 quota spikes gracefully using exponential backoff with randomized jitter.

---

## 5. REST API Reference

All requests accept the `X-API-Key` header configured in `.env`.

### 1. Execute Query (`POST /chat`)
Submits a plain-English query or command to the agent.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace-with-a-random-secret" \
  -d '{
    "message": "What is the total revenue from completed orders?",
    "session_id": "sess-user-42",
    "sheet_name": "Orders"
  }'
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

### 2. Staged Mutation & Confirmation Flow

When a user requests a cell modification or row deletion, the agent stages the operation:

**Agent Response (Staged):**
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

#### Confirm Action (`POST /actions/{action_id}/confirm`)
```bash
curl -X POST http://localhost:8000/actions/550e8400-e29b-41d4-a716-446655440000/confirm \
  -H "X-API-Key: replace-with-a-random-secret"
```

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

#### Reject Action (`POST /actions/{action_id}/reject`)
Cancels the staged action and prevents execution.

---

### 3. Observability & Operational Metrics (`GET /metrics`)
```bash
curl http://localhost:8000/metrics -H "X-API-Key: replace-with-a-random-secret"
```

**Response:**
```json
{
  "tool_usage": {"read_sheet": 12, "filter_and_aggregate": 8},
  "error_rate": {"read_sheet": 0.0, "filter_and_aggregate": 0.0},
  "avg_latency_ms": {"read_sheet": 45, "filter_and_aggregate": 120},
  "actions": {"pending": 0, "confirmed": 4, "expired": 0, "rejected": 1}
}
```

---

### 4. Benchmark Runner (`POST /eval/run`)
Executes the automated 30-task evaluation harness and outputs real-time TSA, EA, CGA, and IBR metrics.

---

## 6. Web Dashboards

SheetSense AI provides two frontend interfaces out of the box:

1. **Embedded Desktop Dashboard (`GET /`):**
   - Served directly by FastAPI at `http://localhost:8000`.
   - Features real-time conversational chat, interactive sheet data grid, staged action confirmation dialogs, and live operational metrics.
2. **Streamlit Companion UI (`app.py`):**
   - Run via `streamlit run app.py` for rapid exploration and developer testing.

---

## 7. Verified Benchmark Scorecard

Empirically measured across 30 benchmark tasks ([tests/eval_dataset.json](tests/eval_dataset.json)) using the evaluation harness ([eval_harness.py](eval_harness.py)):

| Metric | Target | Verified Score | Verification Status |
|---|---|---|---|
| **Benchmark Routing Accuracy (BRA)** | $\ge 90.0\%$ | **100.0%** (30/30) | ✅ Exceeds Target |
| **Execution Accuracy (EA)** | $\ge 85.0\%$ | **93.3%** (28/30) | ✅ Exceeds Target |
| **Confirmation Gate Adherence (CGA)** | **100.0%** | **100.0%** (4/4 gated) | ✅ **0 direct writes permitted** |
| **Injection Block Rate (IBR)** | **100.0%** | **100.0%** (4/4 blocked) | ✅ **0 exploits executed** |
| **Median Latency ($p_{50}$)** | $< 3.0\,\text{s}$ | **~2.2 ms** (offline) | ✅ Optimal |
| **95th Percentile Latency ($p_{95}$)** | $< 6.0\,\text{s}$ | **~7.0 ms** (offline) | ✅ Optimal |

*Detailed category breakdowns and analysis are available in [docs/eval_report.md](docs/eval_report.md) and [docs/audit.md](docs/audit.md).*

---

## 8. Testing & Verification

Run the complete test suite spanning unit tests, integration tests, and static security scans:

```bash
# Run all 84 test cases
pytest tests/ -v

# Run static AST write-isolation guardrail scanner
python scripts/check_write_isolation.py

# Run end-to-end benchmark evaluation harness
python eval_harness.py
```

---

## 9. Repository Structure

```
sheet-ai-agent/
├── agent.py               # LangChain ReAct agent loop, Pydantic schemas, SheetTools
├── database.py            # SQLite schema, pending_actions, audit logging, scrubber
├── eval_harness.py        # Benchmark evaluation harness (BRA, EA, CGA, IBR)
├── main.py                # FastAPI gateway, confirmation gate, rate limiting, metrics
├── rate_limiter.py        # Sliding-window rate limiter (Redis + memory fallback)
├── retrieval.py           # SchemaIndex, MultiQueryReformulator, RAGFusionRetriever
├── retry_handler.py       # Gemini exponential backoff & jitter resilience
├── session_store.py       # RedisSessionStore (24h TTL) & InMemorySessionStore
├── sheets_writer.py       # Isolated write gateway for Google Sheets mutations
├── scripts/
│   └── check_write_isolation.py  # Static analysis AST/regex security scanner
├── static/
│   └── index.html         # Interactive web dashboard interface
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
├── requirements.txt       # Production dependencies
└── LICENSE                # MIT License
```

---

## 10. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
