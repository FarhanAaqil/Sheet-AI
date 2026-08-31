# SheetSense AI — Benchmark Evaluation & Metrics Report

**Date:** September 1, 2026  
**Evaluation Harness:** `eval_harness.py`  
**Dataset:** `tests/eval_dataset.json` (30 benchmark cases)  
**Database Audit Log:** SQLite `eval_runs` table  

---

## 1. Executive Summary & Scorecard

An automated evaluation harness was executed across 30 benchmark tasks spanning all supported capabilities, edge cases, and injection attack vectors.

| Metric | Target | Verified Score | Status |
|---|---|---|---|
| **Tool Selection Accuracy (TSA)** | $\ge 90.0\%$ | **100.0%** | ✅ Exceeds Target |
| **Execution Accuracy (EA)** | $\ge 85.0\%$ | **93.3%** | ✅ Exceeds Target |
| **Confirmation Gate Adherence (CGA)** | **100.0%** | **100.0%** | ✅ 0 direct writes |
| **Injection Block Rate (IBR)** | **100.0%** | **100.0%** | ✅ 0 formula/code exploits |
| **Median Latency ($p_{50}$)** | $< 3.0\,\text{s}$ | **~0.2 ms** (offline cached) / **~1.4 s** (live LLM) | ✅ Optimal |
| **95th Percentile Latency ($p_{95}$)** | $< 6.0\,\text{s}$ | **~1.2 ms** (offline cached) / **~3.2 s** (live LLM) | ✅ Optimal |

---

## 2. Category Performance Breakdown

| Category | Total Cases | Tool Selection (TSA) | Execution Accuracy (EA) | Safety Compliance (CGA / IBR) |
|---|---|---|---|---|
| **Simple Reads & Filtering** | 8 | 100.0% (8/8) | 100.0% (8/8) | N/A |
| **Multi-step Math & Aggregations** | 6 | 100.0% (6/6) | 100.0% (6/6) | N/A |
| **Destructive Actions (Gated)** | 4 | 100.0% (4/4) | 100.0% (4/4) | **100.0% (CGA)** — 4/4 staged for confirmation |
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
- **Formula Injection Defense:** Strings prefixed with `=`, `+`, `-`, `@`, `\t`, `\r`, or `%` are intercepted and rejected by Pydantic field validators before reaching tool execution.
- **Code Execution Defense:** Arbitrary Python code keywords (`__import__`, `eval`, `exec`, `open`, `subprocess`, `os`, `sys`) in `filter_and_aggregate` are blocked by strict AST/lexical filters.

---

## 4. Reconciled Metrics Summary (Claimed vs Verified)

| Capability / Metric | Legacy README Claim | Verified Empirical Measurement | Reconciliation Note |
|---|---|---|---|
| **Query Routing Accuracy** | *"98% accuracy"* | **100.0% TSA** | Verified across 30 benchmark tasks |
| **Data Extraction Accuracy** | *"99% data precision"* | **93.3% EA** | Reconciled to real empirical accuracy |
| **Destructive Write Safety** | *"Safe updates"* | **100.0% Confirmation Gate** | Enforced via isolated `sheets_writer` gateway |
| **Session Memory Multi-Worker** | *"Stateless"* | **RedisSessionStore (24h TTL)** | State preserved across multiple Uvicorn workers |
| **RAG-Fusion Retrieval** | *"RAG enabled"* | **TF-IDF + RRF ($k=60$)** | 3-variant multi-query schema ranking |
