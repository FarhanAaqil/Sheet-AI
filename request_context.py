"""Concurrency-safe request context for agent and tool execution."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RequestContext:
    principal_id: str
    session_id: str
    connection_id: str
    request_id: str


_request_context: ContextVar[Optional[RequestContext]] = ContextVar(
    "sheetsense_request_context", default=None
)
_pending_action: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "sheetsense_pending_action", default=None
)


def set_request_context(context: RequestContext):
    return _request_context.set(context)


def reset_request_context(token) -> None:
    _request_context.reset(token)


def get_request_context() -> Optional[RequestContext]:
    return _request_context.get()


def set_pending_action(action: Dict[str, Any]) -> None:
    _pending_action.set(action)


def clear_pending_action():
    return _pending_action.set(None)


def get_pending_action() -> Optional[Dict[str, Any]]:
    return _pending_action.get()


def reset_pending_action(token) -> None:
    _pending_action.reset(token)
