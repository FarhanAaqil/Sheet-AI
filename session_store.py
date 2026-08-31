# =============================================================================
# SheetSense AI — Multi-Turn Session Store (Architecture §1, §3)
# =============================================================================
# Manages multi-turn conversation memory with a 24-hour rolling TTL.
# Backed by Redis to support multi-worker Uvicorn deployments without state loss.
# Falls back to in-process memory ONLY when ENV=local_no_redis is explicitly set.
# =============================================================================

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 86400  # 24 hours rolling TTL
MAX_TURNS = 10               # 10 turns = 20 messages


class SessionStore(ABC):
    """Abstract interface for session message history storage."""

    @abstractmethod
    def get_history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        """Retrieve the last turns as a list of {'role': 'human'|'ai', 'content': str}."""
        pass

    @abstractmethod
    def save_turn(self, session_id: Optional[str], human: str, ai: str) -> None:
        """Append a completed turn and reset the rolling 24h TTL."""
        pass

    @abstractmethod
    def clear_session(self, session_id: Optional[str]) -> None:
        """Clear history for a specific session."""
        pass


class RedisSessionStore(SessionStore):
    """
    Production-grade Redis-backed session store.
    Key format: session:{session_id} -> JSON string list
    """

    def __init__(self, redis_url: str, ttl_seconds: int = SESSION_TTL_SECONDS):
        import redis
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.client = redis.from_url(redis_url, decode_responses=True)
        # Test connection immediately
        self.client.ping()
        logger.info(f"RedisSessionStore connected successfully to '{redis_url}'.")

    def _key(self, session_id: Optional[str]) -> str:
        sid = session_id.strip() if session_id and session_id.strip() else "__default__"
        return f"session:{sid}"

    def get_history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        key = self._key(session_id)
        raw = self.client.get(key)
        if not raw:
            return []
        try:
            messages = json.loads(raw)
            # Refresh rolling TTL on access
            self.client.expire(key, self.ttl_seconds)
            return messages[- (MAX_TURNS * 2):]
        except Exception as e:
            logger.error(f"Error deserializing session history for key '{key}': {e}")
            return []

    def save_turn(self, session_id: Optional[str], human: str, ai: str) -> None:
        key = self._key(session_id)
        history = self.get_history(session_id)
        history.append({"role": "human", "content": human})
        history.append({"role": "ai", "content": ai})
        # Keep only the last 10 turns (20 messages)
        trimmed = history[- (MAX_TURNS * 2):]
        self.client.set(key, json.dumps(trimmed), ex=self.ttl_seconds)

    def clear_session(self, session_id: Optional[str]) -> None:
        key = self._key(session_id)
        self.client.delete(key)


class InMemorySessionStore(SessionStore):
    """
    In-memory fallback session store.
    Permitted ONLY when ENV=local_no_redis is explicitly configured.
    """

    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        logger.warning(
            "InMemorySessionStore active (ENV=local_no_redis). "
            "Note: In-memory session store does NOT support multi-worker synchronization."
        )

    def _key(self, session_id: Optional[str]) -> str:
        return session_id.strip() if session_id and session_id.strip() else "__default__"

    def get_history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        key = self._key(session_id)
        return self._store.get(key, [])[- (MAX_TURNS * 2):]

    def save_turn(self, session_id: Optional[str], human: str, ai: str) -> None:
        key = self._key(session_id)
        if key not in self._store:
            self._store[key] = []
        self._store[key].append({"role": "human", "content": human})
        self._store[key].append({"role": "ai", "content": ai})
        self._store[key] = self._store[key][- (MAX_TURNS * 2):]

    def clear_session(self, session_id: Optional[str]) -> None:
        key = self._key(session_id)
        self._store.pop(key, None)


def get_session_store() -> SessionStore:
    """
    Factory creating the configured SessionStore instance.
    Defaults to RedisSessionStore. Falls back to InMemorySessionStore ONLY if ENV=local_no_redis.
    """
    env = os.getenv("ENV", "").lower()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    if env == "local_no_redis":
        return InMemorySessionStore()

    try:
        return RedisSessionStore(redis_url=redis_url)
    except Exception as e:
        if env == "local_no_redis":
            return InMemorySessionStore()
        logger.warning(
            f"Could not connect to Redis at '{redis_url}': {e}. "
            "Falling back to InMemorySessionStore for local execution. "
            "Set REDIS_URL or run redis container for multi-worker support."
        )
        return InMemorySessionStore()
