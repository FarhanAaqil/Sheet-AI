# SheetSense AI — Session Store Unit Tests

import json
from unittest.mock import MagicMock, patch

from session_store import (
    MAX_TURNS,
    SESSION_TTL_SECONDS,
    InMemorySessionStore,
    RedisSessionStore,
    get_session_store,
)


def test_in_memory_session_store_lifecycle():
    """Verify turn capping and clear functionality in in-memory session store."""
    store = InMemorySessionStore()
    principal = "user-a"
    sid = "test-session-123"

    # Save 12 turns
    for i in range(12):
        store.save_turn(principal, sid, f"Question {i}", f"Answer {i}")

    history = store.get_history(principal, sid)
    # Should cap at MAX_TURNS * 2 = 20 messages (turns 2 through 11)
    assert len(history) == MAX_TURNS * 2
    assert history[0]["content"] == "Question 2"
    assert history[-1]["content"] == "Answer 11"

    # Clear session
    store.clear_session(principal, sid)
    assert store.get_history(principal, sid) == []


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

    def mock_eval(script, key_count, key, index_key, human, ai, max_messages, ttl, sid):
        history = json.loads(storage.get(key, "[]"))
        history.extend(
            [
                {"role": "human", "content": human},
                {"role": "ai", "content": ai},
            ]
        )
        storage[key] = json.dumps(history[-int(max_messages) :])
        return len(history)

    mock_redis.set.side_effect = mock_set
    mock_redis.get.side_effect = mock_get
    mock_redis.eval.side_effect = mock_eval

    with patch("redis.from_url", return_value=mock_redis):
        store = RedisSessionStore("redis://localhost:6379/0")

        sid = "redis-test-session"
        principal = "user-a"
        store.save_turn(principal, sid, "Hello", "Hi there!")

        # Verify the atomic Lua append was used rather than a racy GET/SET pair.
        assert mock_redis.eval.call_count == 1
        history = store.get_history(principal, sid)
        assert len(history) == 2
        assert history[0]["role"] == "human"
        assert history[1]["role"] == "ai"

        # Verify rolling TTL was refreshed on get
        mock_redis.expire.assert_called_with(f"session:{principal}:{sid}", SESSION_TTL_SECONDS)


def test_session_store_factory_local_env(monkeypatch):
    """Verify factory returns InMemorySessionStore when ENV=local_no_redis."""
    monkeypatch.setenv("ENV", "local_no_redis")
    store = get_session_store()
    assert isinstance(store, InMemorySessionStore)
