"""Tenant-scoped Google Sheets connection registry."""

import re
import uuid
from threading import RLock
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import database


SHEET_URL_PATTERN = re.compile(r"^/spreadsheets/d/([A-Za-z0-9_-]{10,})")
PERMISSIONS = {"viewer", "editor"}


def normalize_spreadsheet_id(sheet_url: str) -> str:
    parsed = urlparse(sheet_url.strip())
    if parsed.scheme != "https" or parsed.netloc != "docs.google.com":
        raise ValueError("A valid HTTPS docs.google.com spreadsheet URL is required.")
    match = SHEET_URL_PATTERN.match(parsed.path)
    if not match:
        raise ValueError("The Google Sheets URL does not contain a valid spreadsheet ID.")
    return match.group(1)


class SpreadsheetRegistry:
    def __init__(self, gspread_client: Any, default_sheet_url: str = ""):
        self.client = gspread_client
        self.default_spreadsheet_id = (
            normalize_spreadsheet_id(default_sheet_url) if default_sheet_url else ""
        )
        self._instances: Dict[tuple[str, str], Any] = {}
        self._lock = RLock()

    def _connection_id(self, principal_id: str, spreadsheet_id: str) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"sheetsense:{principal_id}:{spreadsheet_id}",
            )
        )

    def register(
        self,
        principal_id: str,
        sheet_url: str,
        permission_level: str = "editor",
    ) -> Dict[str, Any]:
        if permission_level not in PERMISSIONS:
            raise ValueError("Unsupported spreadsheet permission level.")
        spreadsheet_id = normalize_spreadsheet_id(sheet_url)
        spreadsheet = self.client.open_by_key(spreadsheet_id)
        display_name = str(getattr(spreadsheet, "title", "Google Spreadsheet"))[:200]
        connection_id = self._connection_id(principal_id, spreadsheet_id)
        connection = database.upsert_spreadsheet_connection(
            connection_id=connection_id,
            principal_id=principal_id,
            spreadsheet_id=spreadsheet_id,
            display_name=display_name,
            permission_level=permission_level,
        )
        with self._lock:
            self._instances[(principal_id, connection_id)] = spreadsheet
        return connection

    def ensure_default(self, principal_id: str) -> Dict[str, Any]:
        if not self.default_spreadsheet_id:
            raise LookupError("No default spreadsheet is configured.")
        connection_id = self._connection_id(principal_id, self.default_spreadsheet_id)
        existing = database.get_spreadsheet_connection(connection_id, principal_id)
        if existing:
            return existing
        spreadsheet = self.client.open_by_key(self.default_spreadsheet_id)
        display_name = str(getattr(spreadsheet, "title", "Default spreadsheet"))[:200]
        connection = database.upsert_spreadsheet_connection(
            connection_id,
            principal_id,
            self.default_spreadsheet_id,
            display_name,
            "editor",
        )
        with self._lock:
            self._instances[(principal_id, connection_id)] = spreadsheet
        return connection

    def resolve(
        self, principal_id: str, connection_id: Optional[str] = None
    ) -> tuple[Dict[str, Any], Any]:
        connection = (
            database.get_spreadsheet_connection(connection_id, principal_id)
            if connection_id
            else self.ensure_default(principal_id)
        )
        if not connection:
            raise PermissionError("Spreadsheet connection not found.")
        key = (principal_id, connection["connection_id"])
        with self._lock:
            spreadsheet = self._instances.get(key)
        if spreadsheet is None:
            spreadsheet = self.client.open_by_key(connection["spreadsheet_id"])
            with self._lock:
                self._instances[key] = spreadsheet
        database.touch_spreadsheet_connection(connection["connection_id"], principal_id)
        return connection, spreadsheet

    def list(self, principal_id: str) -> List[Dict[str, Any]]:
        return database.list_spreadsheet_connections(principal_id)
