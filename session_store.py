# SheetSense AI — Multi-Turn Session Store (Architecture §1, §3)
# Manages multi-turn conversation memory with a 24-hour rolling TTL.
# Backed by Redis to support multi-worker Uvicorn deployments without state loss.
# Falls back to in-process memory ONLY when ENV=local_no_redis is explicitly set.

import os
import json
import logging
import secrets
from contextlib import contextmanager
from abc import ABC, abstractmethod
from threading import RLock
from typing import Dict, List

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 86400  # 24 hours rolling TTL
MAX_TURNS = 10               # 10 turns = 20 messages


class SessionStore(ABC):
    """Abstract interface for session message history storage."""

    @abstractmethod
    def get_history(self, principal_id: str, session_id: str) -> List[Dict[str, str]]:
        """Retrieve the last turns as a list of {'role': 'human'|'ai', 'content': str}."""
        pass

    @abstractmethod
    def save_turn(
        self, principal_id: str, session_id: str, human: str, ai: str
    ) -> None:
        """Append a completed turn and reset the rolling 24h TTL."""
        pass

    @abstractmethod
    def clear_session(self, principal_id: str, session_id: str) -> None:
        """Delete a specific session and its history."""
        pass

    @abstractmethod
    def clear_history(self, principal_id: str, session_id: str) -> None:
        """Clear messages while retaining the session."""
        pass

    def create_session(self, principal_id: str) -> str:
        session_id = secrets.token_urlsafe(24)
        self.clear_session(principal_id, session_id)
        return session_id

    @abstractmethod
    def list_sessions(self, principal_id: str) -> List[str]:
        pass

    @contextmanager
    def turn_lock(self, principal_id: str, session_id: str):
        yield


class RedisSessionStore(SessionStore):
    """
    Production-grade Redis-backed session store.
    Key format: session:{principal_id}:{session_id} -> JSON string list
    """

    def __init__(self, redis_url: str, ttl_seconds: int = SESSION_TTL_SECONDS):
        import redis
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.client = redis.from_url(redis_url, decode_responses=True)
        # Test connection immediately
        self.client.ping()
        logger.info("RedisSessionStore connected successfully.")

    def _key(self, principal_id: str, session_id: str) -> str:
        if not principal_id.strip() or not session_id.strip():
            raise ValueError("principal_id and session_id are required.")
        return f"session:{principal_id}:{session_id}"

    def _index_key(self, principal_id: str) -> str:
        return f"sessions:{principal_id}"

    def get_history(self, principal_id: str, session_id: str) -> List[Dict[str, str]]:
        key = self._key(principal_id, session_id)
        raw = self.client.get(key)
        if not raw:
            return []
        try:
            messages = json.loads(raw)
            # Refresh rolling TTL on access
            self.client.expire(key, self.ttl_seconds)
            return messages[- (MAX_TURNS * 2):]
        except Exception as exc:
            logger.error(
                "Session history deserialization failed: exception_type=%s",
                type(exc).__name__,
            )
            return []

    def save_turn(
        self, principal_id: str, session_id: str, human: str, ai: str
    ) -> None:
        key = self._key(principal_id, session_id)
        script = """
        local raw = redis.call('GET', KEYS[1])
        local history = {}
        if raw then history = cjson.decode(raw) end
        table.insert(history, {role='human', content=ARGV[1]})
        table.insert(history, {role='ai', content=ARGV[2]})
        while #history > tonumber(ARGV[3]) do table.remove(history, 1) end
        redis.call('SET', KEYS[1], cjson.encode(history), 'EX', ARGV[4])
        redis.call('SADD', KEYS[2], ARGV[5])
        redis.call('EXPIRE', KEYS[2], ARGV[4])
        return #history
        """
        self.client.eval(
            script,
            2,
            key,
            self._index_key(principal_id),
            human,
            ai,
            MAX_TURNS * 2,
            self.ttl_seconds,
            session_id,
        )

    def clear_session(self, principal_id: str, session_id: str) -> None:
        key = self._key(principal_id, session_id)
        pipe = self.client.pipeline(transaction=True)
        pipe.delete(key)
        pipe.srem(self._index_key(principal_id), session_id)
        pipe.execute()

    def clear_history(self, principal_id: str, session_id: str) -> None:
        self.client.delete(self._key(principal_id, session_id))
        self.client.sadd(self._index_key(principal_id), session_id)
        self.client.expire(self._index_key(principal_id), self.ttl_seconds)

    def create_session(self, principal_id: str) -> str:
        session_id = secrets.token_urlsafe(24)
        self.client.sadd(self._index_key(principal_id), session_id)
        self.client.expire(self._index_key(principal_id), self.ttl_seconds)
        return session_id

    def list_sessions(self, principal_id: str) -> List[str]:
        return sorted(self.client.smembers(self._index_key(principal_id)))

    @contextmanager
    def turn_lock(self, principal_id: str, session_id: str):
        lock = self.client.lock(
            f"session-lock:{principal_id}:{session_id}",
            timeout=60,
            blocking_timeout=10,
        )
        if not lock.acquire(blocking=True):
            raise TimeoutError("The session is busy.")
        try:
            yield
        finally:
            lock.release()


class InMemorySessionStore(SessionStore):
    """
    In-memory fallback session store.
    Permitted ONLY when ENV=local_no_redis is explicitly configured.
    """

    def __init__(self):
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._sessions: Dict[str, set[str]] = {}
        self._locks: Dict[str, RLock] = {}
        self._guard = RLock()
        logger.warning(
            "InMemorySessionStore active (ENV=local_no_redis). "
            "Note: In-memory session store does NOT support multi-worker synchronization."
        )

    def _key(self, principal_id: str, session_id: str) -> str:
        if not principal_id.strip() or not session_id.strip():
            raise ValueError("principal_id and session_id are required.")
        return f"{principal_id}:{session_id}"

    def get_history(self, principal_id: str, session_id: str) -> List[Dict[str, str]]:
        key = self._key(principal_id, session_id)
        with self._guard:
            return list(self._store.get(key, [])[- (MAX_TURNS * 2):])

    def save_turn(
        self, principal_id: str, session_id: str, human: str, ai: str
    ) -> None:
        key = self._key(principal_id, session_id)
        with self._guard:
            history = self._store.setdefault(key, [])
            history.extend([
                {"role": "human", "content": human},
                {"role": "ai", "content": ai},
            ])
            self._store[key] = history[- (MAX_TURNS * 2):]
            self._sessions.setdefault(principal_id, set()).add(session_id)

    def clear_session(self, principal_id: str, session_id: str) -> None:
        key = self._key(principal_id, session_id)
        with self._guard:
            self._store.pop(key, None)
            self._sessions.setdefault(principal_id, set()).discard(session_id)

    def clear_history(self, principal_id: str, session_id: str) -> None:
        key = self._key(principal_id, session_id)
        with self._guard:
            self._store.pop(key, None)
            self._sessions.setdefault(principal_id, set()).add(session_id)

    def create_session(self, principal_id: str) -> str:
        session_id = secrets.token_urlsafe(24)
        with self._guard:
            self._sessions.setdefault(principal_id, set()).add(session_id)
        return session_id

    def list_sessions(self, principal_id: str) -> List[str]:
        with self._guard:
            return sorted(self._sessions.get(principal_id, set()))

    @contextmanager
    def turn_lock(self, principal_id: str, session_id: str):
        key = self._key(principal_id, session_id)
        with self._guard:
            lock = self._locks.setdefault(key, RLock())
        with lock:
            yield


def get_session_store() -> SessionStore:
    """
    Factory creating the configured SessionStore instance.
    Defaults to RedisSessionStore. Falls back to InMemorySessionStore ONLY if ENV=local_no_redis.
    """
    app_env = os.getenv("APP_ENV", os.getenv("ENV", "development")).lower()
    env = os.getenv("ENV", "").lower()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    if env == "local_no_redis":
        return InMemorySessionStore()

    try:
        return RedisSessionStore(redis_url=redis_url)
    except Exception as exc:
        if app_env in {"production", "staging"}:
            raise RuntimeError(
                "Redis is required for staging and production session storage."
            ) from exc
        logger.warning(
            "Could not connect to Redis; using local in-memory storage. "
            f"exception_type={type(exc).__name__}. "
            "Set REDIS_URL or run redis container for multi-worker support."
        )
        return InMemorySessionStore()
