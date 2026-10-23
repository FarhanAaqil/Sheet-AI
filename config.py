import json
import os
from dataclasses import dataclass

PLACEHOLDERS = {
    "",
    "change-me",
    "dev-key-123",
    "replace-with-a-random-secret",
    "replace-with-a-secure-random-token",
    "your_gemini_api_key_here",
}


def env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_env: str
    allowed_origins: list[str]
    allow_insecure_no_auth: bool
    api_key: str
    api_key_hash: str
    api_keys_json: str
    gemini_api_key: str
    google_sheet_url: str
    gcp_credentials_json: str
    google_creds_path: str
    database_url: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.getenv("APP_ENV", os.getenv("ENV", "development")).strip().lower()
        if env == "local_no_redis":
            env = "development"
        defaults = (
            "http://localhost:8000,http://127.0.0.1:8000,"
            "http://localhost:8501,http://127.0.0.1:8501"
        )
        origins = os.getenv("ALLOWED_ORIGINS", defaults)
        return cls(
            app_env=env,
            allowed_origins=[v.strip() for v in origins.split(",") if v.strip()],
            allow_insecure_no_auth=env_bool("ALLOW_INSECURE_NO_AUTH"),
            api_key=os.getenv("SHEETSENSE_API_KEY", "").strip(),
            api_key_hash=os.getenv("SHEETSENSE_API_KEY_HASH", "").strip().lower(),
            api_keys_json=os.getenv("SHEETSENSE_API_KEYS_JSON", "").strip(),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            google_sheet_url=os.getenv("GOOGLE_SHEET_URL", "").strip(),
            gcp_credentials_json=os.getenv("GCP_CREDENTIALS_JSON", "").strip(),
            google_creds_path=os.getenv("GOOGLE_CREDS_PATH", "").strip(),
            database_url=os.getenv("DATABASE_URL", "").strip(),
        )

    def validate_startup(self) -> None:
        if self.app_env not in {"development", "test", "staging", "production"}:
            raise RuntimeError(f"Unsupported APP_ENV {self.app_env!r}.")
        if "*" in self.allowed_origins:
            raise RuntimeError("Wildcard CORS origins are not permitted.")
        if self.api_keys_json and not isinstance(json.loads(self.api_keys_json), list):
            raise RuntimeError("SHEETSENSE_API_KEYS_JSON must be a JSON list.")
        if not self.is_production:
            return
        missing = []
        if not (self.api_key_hash or self.api_keys_json):
            missing.append("hashed API key configuration")
        if self.api_key and self.api_key.lower() in PLACEHOLDERS:
            missing.append("non-placeholder API key configuration")
        if self.gemini_api_key.lower() in PLACEHOLDERS:
            missing.append("GEMINI_API_KEY")
        if not self.google_sheet_url:
            missing.append("GOOGLE_SHEET_URL")
        if not (self.gcp_credentials_json or self.google_creds_path):
            missing.append("Google credentials")
        if not self.allowed_origins:
            missing.append("ALLOWED_ORIGINS")
        if self.allow_insecure_no_auth:
            missing.append("ALLOW_INSECURE_NO_AUTH=false")
        if not self.database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            missing.append("PostgreSQL DATABASE_URL")
        if missing:
            raise RuntimeError("Invalid production configuration: " + ", ".join(missing))
