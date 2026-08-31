# SheetSense AI — Verification & Audit Log

**Date:** 2026-08-31  
**Auditor:** SheetSense Pair Programmer  
**Status:** In Progress (Day 1)

---

## 1. Environment & Service Bootstrap (Task D1-1)

### Baseline System Findings
- **OS:** Windows / PowerShell environment.
- **Python:** 3.12 with all dependencies pre-installed in user environment.
- **Docker Status:** Docker Desktop daemon was not running locally (`npipe:////./pipe/dockerDesktopLinuxEngine` not found). Service verified and run natively via Uvicorn.
- **Google Sheets API:** Service account configured in `GCP_CREDENTIALS_JSON` connects successfully to Google Sheets via `gspread`.
- **Health Check:** `GET /health` returns `200 OK` with `{"status": "ok", "sheets_connected": True, "sheets_count": 1}`.

### Issues Discovered & Fixed
1. **LangChain 1.x Breaking Change (AgentExecutor / create_react_agent deprecation)**
   - *Problem:* `from langchain.agents import AgentExecutor, create_react_agent` raised `ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'`. In LangChain 1.x / LangGraph, `AgentExecutor` was removed in favor of `langgraph.prebuilt.create_react_agent`.
   - *Fix:* Refactored `agent.py` to use `langgraph.prebuilt.create_react_agent` with structured tools, maintaining the exact same public API (`run()`, `get_sheet_names()`, `get_sheet_schema()`).
2. **Gemini Model Deprecation (`gemini-2.5-flash` / `gemini-2.0-flash`)**
   - *Problem:* Google Generative AI API returned `404 NOT_FOUND` for `gemini-2.5-flash` ("no longer available to new users") and `gemini-2.0-flash`.
   - *Fix:* Tested available models and updated `.env` to `gemini-3.6-flash`, which functions properly with Google GenAI / LangChain.

---

## 2. Feature Audit Summary (Task D1-2)

| Feature Claimed in README | Implementation Status | Verified Behavior | Gap / Action Needed |
|---|---|---|---|
| **Simple Reads (`read_sheet`)** | ✅ Implemented | Reads sheet into DataFrame cache, returns markdown table (capped at 50 rows). | None. |
| **Data Aggregation (`filter_and_aggregate`)** | ✅ Implemented | Executes sandboxed pandas expressions on DataFrame cache. | Needs strict Pydantic input validation. |
| **Destructive Update (`update_cell`)** | ⚠️ Hazard (Unsafe) | Writes directly to live Google Sheet via `gspread` without user confirmation gate. | **Critical:** Must be gated through `pending_actions` + `POST /actions/{id}/confirm` (Day 2). |
| **Destructive Delete (`delete_row`)** | ⚠️ Hazard (Unsafe) | Deletes live sheet row directly without confirmation gate. | **Critical:** Must be gated through `pending_actions` + `POST /actions/{id}/confirm` (Day 2). |
| **Summary Stats (`summarize_sheet`)** | ✅ Implemented | Computes missing values, row counts, and numeric statistics. | None. |
| **Sheet Listing (`list_sheets`)** | ✅ Implemented | Returns worksheet names from connected spreadsheet. | None. |
| **IQR Anomaly Detection (`find_anomalies`)** | ✅ Implemented | Calculates Q1, Q3, IQR = Q3 - Q1, flags rows outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$. | Fixed pandas Boolean Series index alignment warning. |
| **Cross-Sheet Join (`cross_sheet_join`)** | ✅ Implemented | Merges two sheets in DataFrame cache on common key. | Needs multi-sheet test fixture (orders + customers). |
| **Session Memory** | ✅ Migrated to Redis | Migrated to `SessionStore` with `RedisSessionStore` (24h rolling TTL) and `InMemorySessionStore` fallback. Multi-worker state loss resolved. | Completed (Day 3). |
| **RAG-Fusion** | ⏳ In Progress | Schema metadata index + 3-variant multi-query TF-IDF with Reciprocal Rank Fusion ($k=60$). | Tasks D3-5 to D3-7. |

---

## 3. RAG-Fusion Architecture Decision (Task D1-3)

- **Vector Store Check:** No external vector database (`chromadb`, `pinecone`, `qdrant`, `faiss`) is present in `requirements.txt`.
- **Existing ML Dependencies:** `scikit-learn>=1.3.0`, `pandas`, `numpy` are already in `requirements.txt`.
- **Decision:** Implement **Metadata Schema RAG-Fusion** using:
  1. 3-variant query reformulation via LLM (literal, conceptual, synonym-expanded).
  2. TF-IDF scoring over column descriptions, sheet names, sample values, and data types (leveraging `scikit-learn.feature_extraction.text.TfidfVectorizer`).
  3. Reciprocal Rank Fusion (RRF) with $k=60$ to combine the 3 ranked lists into candidate sheet/column hints.
- **Rationale:** Avoids heavyweight dependencies, runs 100% locally and deterministically, adds zero external API cost or latency for embeddings, and perfectly matches the structured schema disambiguation use case.

---

## 4. Live Audit Execution Run Log

The live test run verified that the ReAct agent successfully routes to and invokes the following tools:
- `list_sheets`
- `summarize_sheet`
- `filter_and_aggregate`
- `find_anomalies`

### Observations & Mitigations for Subsequent Days:
1. **Free Tier Rate Limit (5 RPM):** Fast consecutive requests trigger Gemini 429 quota exhaustion. The client retry backoff in Day 5 will ensure requests buffer cleanly.
2. **ReAct Step Recursion:** Multi-tool chaining requires prompt refinement so the agent emits final answers concisely without exhausting iteration limits.

---

## 5. Multi-Worker Session Synchronization Verification (Task D3-4)

- **Test Suite:** `tests/test_multi_worker_session.py`
- **Result:** **PASSED**
- **Findings:**
  - Multiple distinct worker processes reading/writing from shared Redis keys (`session:{session_id}`) successfully share and extend multi-turn dialogue histories.
  - Rolling 24-hour TTL (`86400` seconds) is properly refreshed on each read and write.
  - History windowing is capped at 10 turns (20 messages), preventing unbounded memory growth.

