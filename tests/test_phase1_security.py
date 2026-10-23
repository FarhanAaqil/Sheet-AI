import base64
import hashlib
import re
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import Settings
from database import scrub_credentials
from main import app
from security import SecurityHeadersMiddleware


def test_liveness_is_public_and_has_security_headers():
    response = TestClient(app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors" in response.headers["content-security-policy"]


def test_protected_endpoint_requires_valid_api_key():
    client = TestClient(app)
    missing = client.get("/metrics")
    invalid = client.get("/metrics", headers={"X-API-Key": "wrong-key"})
    valid = client.get("/metrics", headers={"X-API-Key": "dev-key-123"})
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "AUTH_REQUIRED"
    assert invalid.status_code == 403
    assert invalid.json()["detail"]["code"] == "AUTH_INVALID"
    assert valid.status_code == 200


def test_production_configuration_fails_closed():
    unsafe = replace(
        Settings.from_env(),
        app_env="production",
        api_key="",
        api_key_hash="",
        api_keys_json="",
        gemini_api_key="",
        google_sheet_url="",
        gcp_credentials_json="",
        google_creds_path="",
    )
    with pytest.raises(RuntimeError, match="Invalid production configuration"):
        unsafe.validate_startup()


def test_extended_redaction_masks_connection_strings_and_passwords():
    payload = {
        "redis": "redis://worker:super-secret@redis.internal:6379/0",
        "database": "postgresql://service:db-password@db.internal/app",
        "password": "correct-horse-battery-staple",
        "authorization": "Bearer private-token-value",
    }
    scrubbed = scrub_credentials(payload)
    for secret in ("super-secret", "db-password", "correct-horse", "private-token-value"):
        assert secret not in scrubbed
    assert "[REDACTED_CREDENTIAL]" in scrubbed


def test_browser_ui_has_no_placeholder_secret_or_fake_offline_execution():
    source = Path("static/index.html").read_text(encoding="utf-8")
    assert 'value="replace-with-a-random-secret"' not in source
    assert 'session_id: "sess-user-42"' not in source
    assert "fallbackLocalQuery" not in source
    assert "has been staged in SQLite" not in source
    assert "htmlContent}" not in source
    assert 'onclick="sortTable' not in source


def test_csp_hashes_match_every_inline_script():
    source = Path("static/index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", source, re.DOTALL)
    assert scripts
    for script in scripts:
        digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in SecurityHeadersMiddleware.CSP
