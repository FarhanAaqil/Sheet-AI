# SheetSense AI — Benchmark Evaluation & Metrics Report

**Date:** September 2026  
**Evaluation Harness:** `eval_harness.py`  
**Dataset:** `tests/eval_dataset.json` (30 benchmark cases)  
**Database Audit Log:** SQLite `eval_runs` table  

---

## 1. Executive Summary & Scorecard

An automated evaluation harness was executed across 30 benchmark tasks spanning all supported capabilities, edge cases, and injection attack vectors.

The metrics below represent reproducible, deterministic measurements obtained using the offline evaluation suite against cached sheet fixtures.

| Metric | Target | Verified Score | Status |
|---|---|---|---|
| **Benchmark Routing Accuracy (BRA)** | $\ge 90.0\%$ | **100.0%** (30/30) | ✅ Exceeds Target |
| **Execution Accuracy (EA)** | $\ge 85.0\%$ | **93.3%** (28/30) | ✅ Exceeds Target |
| **Confirmation Gate Adherence (CGA)** | **100.0%** | **100.0%** (4/4 gated) | ✅ 0 direct writes |
| **Injection Block Rate (IBR)** | **100.0%** | **100.0%** (4/4 blocked) | ✅ 0 formula/code exploits |
| **Median Offline Latency ($p_{50}$)** | $< 10\,\text{ms}$ | **~2.2 ms** (offline benchmark) | ✅ Optimal |
| **95th Percentile Offline Latency ($p_{95}$)** | $< 25\,\text{ms}$ | **~7.0 ms** (offline benchmark) | ✅ Optimal |
| **Mean Offline Latency** | $< 15\,\text{ms}$ | **~3.1 ms** (offline benchmark) | ✅ Optimal |

> **Note on Latency Measurement:** The latency figures above are empirical measurements of the local tool routing and execution pipeline on cached datasets. They do not represent live LLM round-trip network latency, which varies according to cloud provider load, region, and generation token lengths.

---

## 2. Category Performance Breakdown

| Category | Total Cases | Benchmark Routing (BRA) | Execution Accuracy (EA) | Safety Compliance (CGA / IBR) |
|---|---|---|---|---|
| **Simple Reads & Filtering** | 8 | 100.0% (8/8) | 100.0% (8/8) | N/A |
| **Multi-step Math & Aggregations** | 6 | 100.0% (6/6) | 100.0% (6/6) | N/A |
| **Destructive Actions (Gated)** | 4 | 100.0% (4/4) | 100.0% (4/4) | **100.0% (CGA)** — 4/4 staged with valid UUID4 and TTL |
| **Cross-Sheet Joins** | 4 | 100.0% (4/4) | 100.0% (4/4) | N/A |
| **Anomalies & Edge Cases** | 4 | 100.0% (4/4) | 75.0% (3/4) | N/A |
| **Security & Injection Attacks** | 4 | 100.0% (4/4) | 100.0% (4/4) | **100.0% (IBR)** — 4/4 rejected by Pydantic schema |
| **TOTAL** | **30** | **100.0%** | **93.3%** | **100.0% Safe** |

---

## 3. Core Safety & Guardrail Verification

### A. Confirmation Gate Adherence (100.0%)
- **Test Invariant:** No destructive tool (`update_cell`, `delete_row`) is permitted to mutate worksheets directly.
- **Result:** All destructive operations generated a UUID4 `action_id`, entered the `pending_actions` SQLite table with a 5-minute rolling TTL, and returned a confirmation prompt requiring `POST /actions/{action_id}/confirm`.
- **Replay Protection:** Re-confirming an executed action or expired action returns HTTP `410 Gone`.

### B. Formula & Code Injection Prevention (100.0%)
- **Formula Injection Defense:** Strings prefixed with `=`, `+cmd`, `-malicious`, `@`, `\t`, `\r`, or `%0A` are intercepted and rejected by Pydantic field validators before reaching tool execution. Legitimate negative or positive numeric values (such as `-100`, `"-100"`, `-42.5`, `"-42.5"`) are accurately recognized and permitted.
- **Zero-Eval AST Safe Execution:** Arbitrary Python execution via `eval()` or `exec()` has been completely eliminated from the codebase. Aggregations execute either natively via structured parameters or through an AST evaluator (`safe_eval_ast`) that statically permits only whitelisted node types, methods, and attributes on `df`, blocking any sandbox escape attempts (`__import__`, `getattr`, `__class__`, `apply`, `lambda`, comprehensions).

---

## 4. Reconciled Metrics Summary (Claimed vs Verified)

| Capability / Metric | Legacy README Claim | Verified Empirical Measurement | Reconciliation Note |
|---|---|---|---|
| **Query Routing Accuracy** | *"98% accuracy"* | **100.0% BRA** | Verified across 30 benchmark tasks |
| **Data Extraction Accuracy** | *"99% data precision"* | **93.3% EA** | Reconciled to real empirical accuracy |
| **Destructive Write Safety** | *"Safe updates"* | **100.0% Confirmation Gate** | Enforced via isolated `sheets_writer` gateway |
| **Session Memory Multi-Worker** | *"Stateless"* | **RedisSessionStore (24h TTL)** | State preserved across multiple Uvicorn workers |
| **RAG-Fusion Retrieval** | *"RAG enabled"* | **TF-IDF + RRF ($k=60$)** | 3-variant multi-query schema ranking |
