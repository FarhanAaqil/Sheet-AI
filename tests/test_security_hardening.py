# =============================================================================
# SheetSense AI — Security Hardening & Resilience Tests (Day 5)
# =============================================================================
# Verifies:
# 1. Sliding-window rate limiting with 429 and Retry-After header.
# 2. Credential scrubbing preventing secret leakage into SQLite logs.
# 3. Gemini API 429 exponential backoff with jitter.
# 4. Global exception sanitization preventing stack trace leakage.
# =============================================================================

import time
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from rate_limiter import RateLimiter
from retry_handler import execute_with_retry, is_rate_limit_error
from database import scrub_credentials, log_tool_call, get_db_connection
from main import app


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Rate Limiter Tests
# ---------------------------------------------------------------------------
def test_sliding_window_rate_limiter_in_memory():
    """Verify rate limiter blocks after max_requests and provides Retry-After."""
    limiter = RateLimiter(redis_url="redis://localhost:6379/15")
    # Force in-memory fallback for deterministic testing
    limiter.redis_client = None

    ident = "test-client-ip"
    group = "test_chat"

    # Allow 3 requests per 10 seconds
    for _ in range(3):
        allowed, retry_after = limiter.check_rate_limit(ident, group, max_requests=3, window_seconds=10)
        assert allowed is True
        assert retry_after == 0

    # 4th request must be blocked
    allowed, retry_after = limiter.check_rate_limit(ident, group, max_requests=3, window_seconds=10)
    assert allowed is False
    assert retry_after > 0


def test_rate_limiter_endpoint_429(client, monkeypatch):
    """Verify endpoint returns HTTP 429 with Retry-After when rate limit is tripped."""
    from rate_limiter import get_rate_limiter
    limiter = get_rate_limiter()
    limiter.redis_client = None  # in-memory test

    api_key = "rate-limit-test-key"
    headers = {"X-API-Key": api_key}

    # Artificially consume the 60 req/min limit
    for _ in range(60):
        limiter.check_rate_limit(api_key, "chat", max_requests=60, window_seconds=60)

    # Next request must return 429
    resp = client.post(
        "/chat",
        json={"message": "ping"},
        headers=headers,
    )
    assert resp.status_code == 429
    assert "Rate limit" in resp.json()["detail"]
    assert "Retry-After" in resp.headers


# ---------------------------------------------------------------------------
# 2. Credential Scrubber Tests
# ---------------------------------------------------------------------------
def test_credential_scrubber_masks_sensitive_patterns():
    """Verify regex scrubber redacts Google keys, OpenAI keys, and service account keys."""
    raw_payload = {
        "google_key": "AIzaSyB12345678901234567890123456789012",
        "openai_key": "sk-12345678901234567890123456789012",
        "service_account": {
            "client_email": "sheets-agent@project.iam.gserviceaccount.com",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASC...fake...==\n-----END PRIVATE KEY-----",
        },
    }

    scrubbed = scrub_credentials(raw_payload)

    assert "AIzaSy" not in scrubbed
    assert "sk-" not in scrubbed
    assert "BEGIN PRIVATE KEY" not in scrubbed
    assert "[REDACTED_CREDENTIAL]" in scrubbed


def test_credential_scrubber_sqlite_audit_log():
    """Verify logged tool calls never persist raw secrets into SQLite."""
    secret_key = "AIzaSyC99999999999999999999999999999999"
    call_id = log_tool_call(
        tool_name="test_tool",
        input_data={"api_key": secret_key, "query": "hello"},
        output_data={"result": secret_key},
        success=True,
    )

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT input_json, output_json FROM tool_calls WHERE call_id = ?", (call_id,))
    row = cursor.fetchone()
    conn.close()

    assert secret_key not in row["input_json"]
    assert secret_key not in row["output_json"]
    assert "[REDACTED_CREDENTIAL]" in row["input_json"]
    assert "[REDACTED_CREDENTIAL]" in row["output_json"]


# ---------------------------------------------------------------------------
# 3. Gemini Retry & Resilience Tests
# ---------------------------------------------------------------------------
def test_gemini_retry_backoff_on_429():
    """Verify retry handler catches 429 rate limits, sleeps with backoff, and succeeds."""
    mock_fn = MagicMock()
    # Fail twice with 429, then succeed on 3rd attempt
    mock_fn.side_effect = [
        Exception("429 ResourceExhausted: Quota exceeded for gemini-flash"),
        Exception("ResourceExhausted: 429 Too Many Requests"),
        {"answer": "Success after backoff"},
    ]

    result = execute_with_retry(
        mock_fn,
        "test query",
        max_retries=3,
        initial_delay=0.01,
        max_delay=0.05,
        jitter=False,
    )

    assert result == {"answer": "Success after backoff"}
    assert mock_fn.call_count == 3


def test_gemini_retry_propagates_non_rate_limit_errors():
    """Verify non-rate-limit errors fail fast without redundant retries."""
    mock_fn = MagicMock()
    mock_fn.side_effect = ValueError("Invalid prompt format")

    with pytest.raises(ValueError):
        execute_with_retry(mock_fn, max_retries=3, initial_delay=0.01)

    assert mock_fn.call_count == 1


# ---------------------------------------------------------------------------
# 4. Global Sanitized Exception Handler Tests
# ---------------------------------------------------------------------------
def test_sanitized_global_exception_handler(client, monkeypatch):
    """Verify unhandled exceptions return 500 with error_id and no leaked stack traces."""
    def crash_endpoint():
        raise RuntimeError("Internal DB connection failed at /var/secrets/keys.json line 42")


    from main import get_agent
    app.dependency_overrides[get_agent] = crash_endpoint

    safe_client = TestClient(app, raise_server_exceptions=False)

    try:
        resp = safe_client.post(
            "/chat",
            json={"message": "cause crash"},
            headers={"X-API-Key": "dev-key-123"},
        )

        assert resp.status_code == 500
        data = resp.json()
        assert data["detail"] == "An internal server error occurred."
        assert "error_id" in data
        assert "/var/secrets" not in resp.text
        assert "RuntimeError" not in resp.text
    finally:
        app.dependency_overrides.pop(get_agent, None)
