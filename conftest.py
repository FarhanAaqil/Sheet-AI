import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pytest
import database


@pytest.fixture(autouse=True, scope="session")
def _ensure_db_schema():
    """
    Guarantee tool_calls / pending_actions / eval_runs tables exist before any
    test runs. database.init_db() previously only ran via FastAPI's `startup`
    event, which several tests never trigger (direct DB calls, or a
    TestClient(app) instantiated without a `with` block/lifespan). Without
    this fixture those tests hit 'no such table' errors, and — critically —
    the eval-endpoint tests silently report Confirmation Gate Adherence as
    0.0 instead of failing loudly, masking the real result of the one metric
    that must always read 100%.
    """
    database.init_db()
