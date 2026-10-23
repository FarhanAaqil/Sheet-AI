"""Authentication, authorization, redaction, and HTTP security controls."""

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import FrozenSet, Optional

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from config import Settings

ALL_ROLES = frozenset({"viewer", "editor", "approver", "admin"})
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class APIPrincipal:
    principal_id: str
    roles: FrozenSet[str]


@dataclass(frozen=True)
class APIKeyRecord:
    principal_id: str
    key_hash: str
    roles: FrozenSet[str]


def hash_api_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_key_records(settings: Settings) -> list[APIKeyRecord]:
    records = []
    if settings.api_keys_json:
        for item in json.loads(settings.api_keys_json):
            if not isinstance(item, dict):
                raise RuntimeError("Each API key record must be an object.")
            principal = str(item.get("id", "")).strip()
            key_hash = str(item.get("key_sha256", "")).strip().lower()
            roles = frozenset(str(r).strip().lower() for r in item.get("roles", []))
            if not principal or not re.fullmatch(r"[0-9a-f]{64}", key_hash):
                raise RuntimeError("API key records require id and SHA-256 key_sha256.")
            if not roles or not roles.issubset(ALL_ROLES):
                raise RuntimeError("API key records contain unsupported roles.")
            records.append(APIKeyRecord(principal, key_hash, roles))
    if settings.api_key_hash:
        if not re.fullmatch(r"[0-9a-f]{64}", settings.api_key_hash):
            raise RuntimeError("SHEETSENSE_API_KEY_HASH must be a SHA-256 digest.")
        records.append(APIKeyRecord("default-admin", settings.api_key_hash, ALL_ROLES))
    if settings.api_key:
        records.append(APIKeyRecord("legacy-admin", hash_api_key(settings.api_key), ALL_ROLES))
    return records


class APIKeyAuthenticator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.records = load_key_records(settings)

    def authenticate(self, presented_key: Optional[str]) -> APIPrincipal:
        if not self.records:
            if self.settings.allow_insecure_no_auth and not self.settings.is_production:
                return APIPrincipal("insecure-development", ALL_ROLES)
            raise HTTPException(
                503,
                detail={
                    "code": "AUTH_NOT_CONFIGURED",
                    "message": "API authentication is not configured.",
                },
            )
        if not presented_key:
            raise HTTPException(
                401,
                detail={
                    "code": "AUTH_REQUIRED",
                    "message": "A valid X-API-Key header is required.",
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )
        presented_hash = hash_api_key(presented_key)
        for record in self.records:
            if hmac.compare_digest(presented_hash, record.key_hash):
                return APIPrincipal(record.principal_id, record.roles)
        raise HTTPException(
            403,
            detail={
                "code": "AUTH_INVALID",
                "message": "The supplied API key is invalid.",
            },
        )


def build_auth_dependencies(settings: Settings):
    authenticator = APIKeyAuthenticator(settings)

    async def get_api_principal(
        key: Optional[str] = Security(API_KEY_HEADER),
    ) -> APIPrincipal:
        return authenticator.authenticate(key)

    def require_roles(*required_roles: str):
        required = frozenset(required_roles)

        async def dependency(
            key: Optional[str] = Security(API_KEY_HEADER),
        ) -> APIPrincipal:
            principal = authenticator.authenticate(key)
            if not principal.roles.intersection(required):
                raise HTTPException(
                    403,
                    detail={
                        "code": "AUTH_FORBIDDEN",
                        "message": "This API key lacks the required role.",
                    },
                )
            return principal

        return dependency

    return authenticator, get_api_principal, require_roles


SENSITIVE_PATTERNS = [
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"xox[baprs]-[0-9a-zA-Z\-]+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(
        r"(?i)(authorization[\x22\x27]?\s*[:=]\s*[\x22\x27]?(?:bearer\s+)?)[^\x22\x27\s,;}]+"
    ),
    re.compile(
        r"(?i)((?:api[_-]?key|token|password|secret)\s*[\x22\x27]?\s*[:=]\s*[\x22\x27]?)[^\x22\x27\s,}]+"
    ),
    re.compile(r"(?i)\b(?:redis|rediss|postgres(?:ql)?|mysql)://[^\s/@:]+(?::[^\s/@]*)?@"),
]


def redact_sensitive(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, default=str)
        except Exception:
            value = type(value).__name__
    text = value
    for index, pattern in enumerate(SENSITIVE_PATTERNS):
        if index in {4, 5}:
            text = pattern.sub(r"\1[REDACTED_CREDENTIAL]", text)
        elif index == 6:
            text = pattern.sub(
                lambda match: match.group(0).split("://", 1)[0] + "://[REDACTED_CREDENTIAL]@",
                text,
            )
        else:
            text = pattern.sub("[REDACTED_CREDENTIAL]", text)
    return text


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    CSP = "; ".join(
        [
            """default-src 'self' """,
            """script-src 'self' https://cdn.tailwindcss.com https://cdn.jsdelivr.net 'unsafe-eval' 'sha256-YoMSsnwv4n0ghqbN3klQxu71BhEu5k5kV2d0QoPCMxs=' 'sha256-GB2394cGq2L0VyowYQWe9w6OGIv/k4dX4qlJ/OrOJeQ=' """,
            """style-src 'self' 'unsafe-inline' https://fonts.googleapis.com""",
            """font-src 'self' https://fonts.gstatic.com data:""",
            """img-src 'self' data:""",
            """connect-src 'self' http://localhost:* http://127.0.0.1:* https:""",
            """object-src 'none' """,
            """base-uri 'self' """,
            """form-action 'self' """,
            """frame-ancestors 'none' """,
        ]
    )

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = self.CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
