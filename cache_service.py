"""Thread-safe worksheet cache with explicit freshness metadata."""

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from time import monotonic
from typing import Dict, Optional

import pandas as pd


@dataclass
class CacheEntry:
    frame: pd.DataFrame
    loaded_monotonic: float
    refreshed_at: str
    schema_signature: tuple
    version: int


class WorksheetCache:
    def __init__(self, ttl_seconds: int = 30):
        self.ttl_seconds = max(1, ttl_seconds)
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = RLock()

    def get(self, sheet_name: str) -> Optional[CacheEntry]:
        with self._lock:
            return self._entries.get(sheet_name)

    def is_stale(self, sheet_name: str) -> bool:
        entry = self.get(sheet_name)
        return entry is None or monotonic() - entry.loaded_monotonic >= self.ttl_seconds

    def put(self, sheet_name: str, frame: pd.DataFrame) -> tuple[CacheEntry, bool]:
        signature = tuple((str(column), str(dtype)) for column, dtype in frame.dtypes.items())
        with self._lock:
            previous = self._entries.get(sheet_name)
            schema_changed = previous is None or previous.schema_signature != signature
            entry = CacheEntry(
                frame=frame,
                loaded_monotonic=monotonic(),
                refreshed_at=datetime.now(timezone.utc).isoformat(),
                schema_signature=signature,
                version=(previous.version + 1) if previous else 1,
            )
            self._entries[sheet_name] = entry
            return entry, schema_changed

    def invalidate(self, sheet_name: str) -> None:
        with self._lock:
            self._entries.pop(sheet_name, None)

    def metadata(self, sheet_name: str) -> Dict[str, object]:
        entry = self.get(sheet_name)
        if entry is None:
            return {
                "last_refreshed_at": None,
                "cache_age_seconds": None,
                "cache_version": None,
                "cache_ttl_seconds": self.ttl_seconds,
            }
        return {
            "last_refreshed_at": entry.refreshed_at,
            "cache_age_seconds": round(max(0.0, monotonic() - entry.loaded_monotonic), 3),
            "cache_version": entry.version,
            "cache_ttl_seconds": self.ttl_seconds,
        }

    def frames(self) -> Dict[str, pd.DataFrame]:
        with self._lock:
            return {name: entry.frame.copy() for name, entry in self._entries.items()}
