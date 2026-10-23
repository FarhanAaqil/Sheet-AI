"""Low-dependency request tracing, metrics, and redacted audit instrumentation."""

import logging
import os
import re
import uuid
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)
_request_id: ContextVar[str] = ContextVar("request_id", default="")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


def request_id() -> str:
    return _request_id.get()


def trace_id() -> str:
    return _trace_id.get()


@dataclass
class Counter:
    value: float = 0


class MetricsRegistry:
    """Thread-safe bounded-label metrics registry with Prometheus rendering."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._counters: Dict[tuple[str, tuple[tuple[str, str], ...]], Counter] = defaultdict(
            Counter
        )

    def increment(self, name: str, value: float = 1, **labels: str) -> None:
        safe_labels = tuple(sorted((key, str(val)[:80]) for key, val in labels.items()))
        with self._lock:
            self._counters[(name, safe_labels)].value += value

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {
                self._format_key(name, labels): counter.value
                for (name, labels), counter in self._counters.items()
            }

    def render_prometheus(self) -> str:
        lines = []
        with self._lock:
            for (name, labels), counter in sorted(self._counters.items()):
                label_text = ""
                if labels:
                    encoded = ",".join(
                        f'{key}="{value.replace(chr(34), chr(92) + chr(34))}"'
                        for key, value in labels
                    )
                    label_text = "{" + encoded + "}"
                lines.append(f"{name}{label_text} {counter.value:g}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _format_key(name: str, labels: tuple[tuple[str, str], ...]) -> str:
        suffix = ",".join(f"{key}={value}" for key, value in labels)
        return f"{name}[{suffix}]" if suffix else name


metrics = MetricsRegistry()
_otel_configured = False


def configure_opentelemetry(app) -> bool:
    """Enable OTLP tracing when an exporter endpoint is configured."""
    global _otel_configured
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint or _otel_configured:
        return _otel_configured
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OTLP endpoint configured but OpenTelemetry packages are unavailable")
        return False

    provider = TracerProvider(
        resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "sheetsense-api")})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    _otel_configured = True
    logger.info("OpenTelemetry tracing enabled")
    return True


class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        supplied_request = request.headers.get("X-Request-ID", "")
        supplied_trace = request.headers.get("X-Trace-ID", "")
        rid = supplied_request if _SAFE_ID.fullmatch(supplied_request) else str(uuid.uuid4())
        tid = supplied_trace if _SAFE_ID.fullmatch(supplied_trace) else str(uuid.uuid4())
        request_token = _request_id.set(rid)
        trace_token = _trace_id.set(tid)
        started = perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            response.headers["X-Trace-ID"] = tid
            return response
        finally:
            elapsed = perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            metrics.increment(
                "sheetsense_http_requests_total",
                method=request.method,
                route=path,
                status=str(status),
            )
            metrics.increment(
                "sheetsense_http_request_duration_seconds_total",
                elapsed,
                method=request.method,
                route=path,
            )
            _trace_id.reset(trace_token)
            _request_id.reset(request_token)


def record_tool(tool_name: str, success: bool, latency_ms: int) -> None:
    metrics.increment(
        "sheetsense_tool_calls_total",
        tool=tool_name,
        outcome="success" if success else "failure",
    )
    metrics.increment(
        "sheetsense_tool_latency_ms_total",
        latency_ms,
        tool=tool_name,
    )


def record_cache(hit: bool) -> None:
    metrics.increment("sheetsense_cache_access_total", result="hit" if hit else "miss")


def record_action(outcome: str) -> None:
    metrics.increment("sheetsense_action_outcomes_total", outcome=outcome)


def record_retrieval(confidence: Optional[float]) -> None:
    metrics.increment("sheetsense_retrieval_queries_total")
    if confidence is not None:
        metrics.increment("sheetsense_retrieval_confidence_total", confidence)
