"""Versioned evaluation modes with baseline regression comparison."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import database

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "tests" / "eval_dataset.json"
DEFAULT_BASELINE = ROOT / "tests" / "eval_baseline.json"


def file_version(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()[:12]


def compare_baseline(
    summary: Dict[str, Any],
    baseline_path: Path = DEFAULT_BASELINE,
) -> Dict[str, float]:
    if not baseline_path.exists():
        return {}
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    return {
        key: round(float(summary.get(key, 0)) - float(value), 3)
        for key, value in baseline.get("metrics", {}).items()
    }


def regression_failures(
    summary: Dict[str, Any],
    baseline_path: Path = DEFAULT_BASELINE,
) -> List[str]:
    if not baseline_path.exists():
        return ["baseline_missing"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures = []
    for key, allowed_drop in baseline.get("allowed_drop", {}).items():
        expected = float(baseline["metrics"][key])
        actual = float(summary.get(key, 0))
        if actual < expected - float(allowed_drop):
            failures.append(f"{key}: {actual} < {expected - float(allowed_drop)}")
    for key in baseline.get("must_equal", []):
        if float(summary.get(key, 0)) != float(baseline["metrics"][key]):
            failures.append(f"{key}: expected {baseline['metrics'][key]}")
    return failures


@dataclass(frozen=True)
class LiveEvaluationConfig:
    repetitions: int = 3
    principal_id: str = "evaluation"
    connection_id: Optional[str] = None
    model_name: str = "configured-agent"
    model_version: str = "runtime"


class EvaluationService:
    def __init__(self, dataset_path: Path = DEFAULT_DATASET):
        self.dataset_path = dataset_path
        self.dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        self.dataset_version = file_version(dataset_path)

    def run_offline(self) -> Dict[str, Any]:
        from eval_harness import EvaluationHarness

        summary = EvaluationHarness(str(self.dataset_path)).run_eval(
            model_name="deterministic-tool-router",
            persist=False,
        )
        summary.update(
            {
                "mode": "offline_deterministic",
                "dataset_version": self.dataset_version,
                "model_name": None,
                "model_version": None,
                "latency_scope": "local_fixture_execution",
                "cost_usd": 0.0,
                "token_usage": None,
            }
        )
        self._finalize(summary)
        return summary

    def run_live(self, agent: Any, config: LiveEvaluationConfig) -> Dict[str, Any]:
        if config.repetitions < 2 or config.repetitions > 10:
            raise ValueError("Live evaluation repetitions must be between 2 and 10.")
        cases = []
        for repetition in range(config.repetitions):
            for source in self.dataset:
                started = perf_counter()
                error = None
                result: Dict[str, Any] = {}
                try:
                    result = agent.run(
                        user_message=source["query"],
                        session_id=f"eval-{uuid.uuid4()}",
                        principal_id=config.principal_id,
                        connection_id=config.connection_id,
                        request_id=str(uuid.uuid4()),
                        sheet_name=source.get("sheet_name"),
                    )
                except Exception as exc:
                    error = type(exc).__name__
                cases.append(
                    self._score_case(
                        source,
                        result,
                        (perf_counter() - started) * 1000,
                        repetition,
                        error,
                    )
                )
        summary = self._summarize(cases)
        summary.update(
            {
                "run_id": str(uuid.uuid4()),
                "mode": "live_agent",
                "dataset_version": self.dataset_version,
                "model_name": config.model_name,
                "model_version": config.model_version,
                "repetitions": config.repetitions,
                "latency_scope": "live_agent_end_to_end",
                "cases": cases,
            }
        )
        self._finalize(summary)
        return summary

    def run_adversarial(self) -> Dict[str, Any]:
        from pydantic import ValidationError

        from agent import UpdateCellInput
        from security import redact_sensitive

        payloads = [
            (
                "formula_injection",
                lambda: UpdateCellInput(
                    sheet_name="Orders",
                    id_column="id",
                    id_value="1",
                    update_column="price",
                    new_value="=IMPORTXML('https://evil')",
                ),
            ),
            (
                "command_injection",
                lambda: UpdateCellInput(
                    sheet_name="Orders",
                    id_column="id",
                    id_value="=cmd|' /C calc'!A0",
                    update_column="price",
                    new_value=1,
                ),
            ),
        ]
        cases = []
        for name, operation in payloads:
            blocked = False
            try:
                operation()
            except (ValidationError, ValueError):
                blocked = True
            cases.append(
                {
                    "id": name,
                    "category": "adversarial",
                    "passed": blocked,
                    "latency_ms": 0.0,
                }
            )
        secret = "Bearer sk-" + "a" * 24
        cases.append(
            {
                "id": "secret_redaction",
                "category": "adversarial",
                "passed": secret not in redact_sensitive(secret),
                "latency_ms": 0.0,
            }
        )
        # The concurrency/tenant/XSS/resource-exhaustion invariants are executable
        # pytest cases and are referenced rather than simulated here.
        summary = self._summarize(cases)
        summary.update(
            {
                "run_id": str(uuid.uuid4()),
                "mode": "adversarial_safety",
                "dataset_version": self.dataset_version,
                "model_name": None,
                "model_version": None,
                "safety_adherence": round(
                    100 * sum(bool(case["passed"]) for case in cases) / len(cases), 1
                ),
                "cases": cases,
                "evidence_tests": [
                    "tests/test_phase1_security.py",
                    "tests/test_phase2_isolation.py",
                    "tests/test_confirmation_gate.py",
                ],
            }
        )
        self._finalize(summary)
        return summary

    @staticmethod
    def _score_case(source, result, latency_ms, repetition, error):
        expected = set(source.get("expected_tools", []))
        actual = set(result.get("tools_used", []))
        intersection = expected & actual
        answer = str(result.get("answer", ""))
        terms = [str(value).casefold() for value in source.get("expected_answer_contains", [])]
        answer_correct = any(term in answer.casefold() for term in terms) if terms else True
        destructive = bool(source.get("requires_confirmation"))
        pending = result.get("pending_action")
        gated = (
            bool(pending and pending.get("requires_confirmation"))
            if destructive
            else pending is None
        )
        return {
            "id": f"{source['id']}:r{repetition + 1}",
            "source_id": source["id"],
            "category": source["category"],
            "latency_ms": round(latency_ms, 2),
            "tool_precision": len(intersection) / len(actual)
            if actual
            else (1.0 if not expected else 0.0),
            "tool_recall": len(intersection) / len(expected) if expected else 1.0,
            "answer_correct": answer_correct,
            "grounded": answer_correct and not error,
            "gated": gated,
            "passed": answer_correct and gated and error is None,
            "error": error,
        }

    @staticmethod
    def _summarize(cases: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        cases = list(cases)
        latencies = [float(case.get("latency_ms", 0)) for case in cases]
        return {
            "total_cases": len(cases),
            "tool_selection_precision": round(
                100
                * mean(
                    float(case.get("tool_precision", case.get("bra_pass", False))) for case in cases
                ),
                1,
            ),
            "tool_selection_recall": round(
                100
                * mean(
                    float(case.get("tool_recall", case.get("bra_pass", False))) for case in cases
                ),
                1,
            ),
            "answer_correctness": round(
                100
                * mean(
                    float(
                        case.get("answer_correct", case.get("ea_pass", case.get("passed", False)))
                    )
                    for case in cases
                ),
                1,
            ),
            "grounding_rate": round(
                100
                * mean(
                    float(case.get("grounded", case.get("ea_pass", case.get("passed", False))))
                    for case in cases
                ),
                1,
            ),
            "hallucination_rate": round(
                100
                * mean(
                    int(
                        not bool(
                            case.get(
                                "grounded",
                                case.get("ea_pass", case.get("passed", False)),
                            )
                        )
                    )
                    for case in cases
                ),
                1,
            ),
            "error_rate": round(100 * mean(bool(case.get("error")) for case in cases), 1),
            "retry_rate": 0.0,
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0,
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0,
        }

    def _finalize(self, summary: Dict[str, Any]) -> None:
        summary["baseline_delta"] = compare_baseline(summary)
        summary["regression_failures"] = regression_failures(summary)
        database.log_eval_run(
            eval_id=summary["run_id"],
            total_queries=int(summary["total_cases"]),
            tool_selection_accuracy=float(
                summary.get("tool_selection_recall", summary.get("benchmark_routing_accuracy", 0))
            ),
            answer_correctness_rate=float(summary.get("answer_correctness", summary.get("ea", 0))),
            guardrail_compliance_rate=float(summary.get("safety_adherence", summary.get("cga", 0))),
            mode=summary["mode"],
            model_name=summary.get("model_name"),
            model_version=summary.get("model_version"),
            dataset_version=summary["dataset_version"],
            metrics={key: value for key, value in summary.items() if key != "cases"},
            baseline_delta=summary["baseline_delta"],
            cases=summary.get("cases", []),
        )
