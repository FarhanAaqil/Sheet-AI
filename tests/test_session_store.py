# =============================================================================
# SheetSense AI — Session Store Unit Tests
# =============================================================================

import os
import json
import pytest
from unittest.mock import MagicMock, patch

from session_store import (
    SessionStore,
    RedisSessionStore,
    InMemorySessionStore,
    get_session_store,
    SESSION_TTL_SECONDS,
    MAX_TURNS,
)


def test_in_memory_session_store_lifecycle():
    """Verify turn capping and clear functionality in in-memory session store."""
    store = InMemorySessionStore()
    sid = "test-session-123"

    # Save 12 turns
    for i in range(12):
        store.save_turn(sid, f"Question {i}", f"Answer {i}")

    history = store.get_history(sid)
    # Should cap at MAX_TURNS * 2 = 20 messages (turns 2 through 11)
    assert len(history) == MAX_TURNS * 2
    assert history[0]["content"] == "Question 2"
    assert history[-1]["content"] == "Answer 11"

    # Clear session
    store.clear_session(sid)
    assert store.get_history(sid) == []


def test_redis_session_store_ttl_and_truncation():
    """Verify Redis session store sets 24h TTL and rolls expiration on access."""
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    # In-memory dict backing for mock redis
    storage = {}

    def mock_set(key, val, ex=None):
        storage[key] = val
        assert ex == SESSION_TTL_SECONDS

    def mock_get(key):
        return storage.get(key)

    mock_redis.set.side_effect = mock_set
    mock_redis.get.side_effect = mock_get

    with patch("redis.from_url", return_value=mock_redis):
        store = RedisSessionStore("redis://localhost:6379/0")

        sid = "redis-test-session"
        store.save_turn(sid, "Hello", "Hi there!")

        # Verify Redis set was called with 24h TTL
        assert mock_redis.set.call_count >= 1
        history = store.get_history(sid)
        assert len(history) == 2
        assert history[0]["role"] == "human"
        assert history[1]["role"] == "ai"

        # Verify rolling TTL was refreshed on get
        mock_redis.expire.assert_called_with(f"session:{sid}", SESSION_TTL_SECONDS)


def test_session_store_factory_local_env(monkeypatch):
    """Verify factory returns InMemorySessionStore when ENV=local_no_redis."""
    monkeypatch.setenv("ENV", "local_no_redis")
    store = get_session_store()
    assert isinstance(store, InMemorySessionStore)
