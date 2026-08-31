# =============================================================================
# SheetSense AI — Multi-Worker Session Synchronization Test
# =============================================================================
# Verifies that two distinct worker/agent instances sharing a RedisSessionStore
# correctly synchronize multi-turn conversation memory across workers.
# =============================================================================

import os
import json
import pytest
from unittest.mock import MagicMock, patch

from session_store import RedisSessionStore
from agent import SheetTools, SheetSenseAgent


def test_multi_worker_session_memory_synchronization():
    """
    Simulate Worker A (process 1) and Worker B (process 2) receiving consecutive
    turns of the same session_id.
    """
    mock_redis = MagicMock()
    mock_redis.ping.return_value = True

    # Shared storage between the two worker processes
    shared_redis_storage = {}

    def mock_set(key, val, ex=None):
        shared_redis_storage[key] = val

    def mock_get(key):
        return shared_redis_storage.get(key)

    mock_redis.set.side_effect = mock_set
    mock_redis.get.side_effect = mock_get

    with patch("redis.from_url", return_value=mock_redis):
        # Worker A initializes its session store
        store_worker_a = RedisSessionStore("redis://localhost:6379/0")
        
        # Worker B initializes its session store
        store_worker_b = RedisSessionStore("redis://localhost:6379/0")

        session_id = "cross-worker-session-999"

        # Turn 1 handled by Worker A
        store_worker_a.save_turn(
            session_id=session_id,
            human="How many orders were placed by Alice?",
            ai="Alice placed 3 orders totaling $450.",
        )

        # Turn 2 received by Worker B (different process/instance)
        worker_b_history = store_worker_b.get_history(session_id)
        assert len(worker_b_history) == 2
        assert worker_b_history[0]["content"] == "How many orders were placed by Alice?"
        assert worker_b_history[1]["content"] == "Alice placed 3 orders totaling $450."

        # Turn 2 completed by Worker B
        store_worker_b.save_turn(
            session_id=session_id,
            human="And what was her largest order?",
            ai="Her largest order was ORD-1004 for $399.",
        )

        # Turn 3 received by Worker A
        worker_a_history = store_worker_a.get_history(session_id)
        assert len(worker_a_history) == 4
        assert worker_a_history[2]["content"] == "And what was her largest order?"
        assert worker_a_history[3]["content"] == "Her largest order was ORD-1004 for $399."
