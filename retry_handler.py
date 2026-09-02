# =============================================================================
# SheetSense AI — Gemini API Retry & Resilience Handler (Architecture §4, PRD FR-9)
# =============================================================================
# Provides client-side exponential backoff with jitter on Gemini LLM calls to
# gracefully handle free-tier 429 (ResourceExhausted) quota rate limit spikes.
# =============================================================================

import time
import random
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a Google GenAI rate limit (429 / ResourceExhausted)."""
    exc_str = str(exc).lower()
    type_name = type(exc).__name__.lower()

    keywords = [
        "429",
        "resourceexhausted",
        "quota exceeded",
        "rate limit",
        "toomanyrequests",
        "resource_exhausted",
    ]
    return any(kw in exc_str or kw in type_name for kw in keywords)


def execute_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 15.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    **kwargs: Any,
) -> T:
    """
    Executes a callable with exponential backoff and jitter if a rate limit error occurs.

    Args:
        func: The function to call (e.g. agent.invoke or llm.invoke).
        max_retries: Maximum number of retry attempts before propagating the error.
        initial_delay: Initial sleep duration in seconds.
        max_delay: Maximum sleep duration cap in seconds.
        backoff_factor: Multiplier applied per attempt (default 2.0).
        jitter: If True, adds random jitter to prevent thundering herd.

    Returns:
        The result of func(*args, **kwargs).
    """
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not is_rate_limit_error(e) or attempt == max_retries:
                raise e

            sleep_duration = min(delay, max_delay)
            if jitter:
                sleep_duration += random.uniform(0.1, 0.5)

            logger.warning(
                f"[RateLimitBackoff] Gemini 429 quota hit on attempt {attempt}/{max_retries}. "
                f"Backing off for {sleep_duration:.2f}s before retry. Error: {e}"
            )
            time.sleep(sleep_duration)
            delay *= backoff_factor

    return func(*args, **kwargs)
