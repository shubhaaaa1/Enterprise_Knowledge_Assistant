"""Unit tests for ConversationManager.

Covers:
- Req 7.5: session clear deletes history and starts a new session
- Req 7.3: session expiry after 60 min inactivity
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from enterprise_rag.conversation_manager import ConversationManager
from enterprise_rag.models import Turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(index: int = 0) -> Turn:
    return Turn(
        role="user",
        original_query=f"query-{index}",
        rewritten_query=f"rewritten-{index}",
        answer=f"answer-{index}",
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _make_cm() -> ConversationManager:
    """Return a ConversationManager using in-memory storage only."""
    return ConversationManager(redis_client=None, db_conn=None)


# ---------------------------------------------------------------------------
# Req 7.5 — Session clear
# ---------------------------------------------------------------------------

class TestClearSession:
    def test_clear_removes_all_turns(self):
        cm = _make_cm()
        session_id = "sess-clear-1"

        for i in range(5):
            cm.append_turn(session_id, _make_turn(i))

        assert len(cm.get_history(session_id, last=10)) == 5

        cm.clear_session(session_id)

        assert cm.get_history(session_id, last=10) == []

    def test_clear_allows_new_session_to_start(self):
        cm = _make_cm()
        session_id = "sess-clear-2"

        cm.append_turn(session_id, _make_turn(0))
        cm.clear_session(session_id)

        # Append a new turn after clearing — should start fresh
        new_turn = _make_turn(99)
        cm.append_turn(session_id, new_turn)

        history = cm.get_history(session_id, last=10)
        assert len(history) == 1
        assert history[0].original_query == "query-99"

    def test_clear_nonexistent_session_is_noop(self):
        cm = _make_cm()
        # Should not raise
        cm.clear_session("nonexistent-session")
        assert cm.get_history("nonexistent-session", last=10) == []

    def test_clear_removes_last_active_tracking(self):
        cm = _make_cm()
        session_id = "sess-clear-3"

        cm.append_turn(session_id, _make_turn(0))
        assert session_id in cm._last_active

        cm.clear_session(session_id)
        assert session_id not in cm._last_active

    def test_clear_calls_postgres_delete(self):
        """Verify clear_session issues a DELETE to Postgres when db_conn is set."""
        stored_rows = [("sess-pg-clear", 0, "user", "q", "rq", "a", "2024-01-01", "2024-01-01")]

        def mock_execute(sql: str, params=None):
            sql_stripped = sql.strip().upper()
            if sql_stripped.startswith("DELETE"):
                stored_rows.clear()

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = mock_execute
        mock_db = MagicMock()
        mock_db.cursor.return_value = mock_cursor

        cm = ConversationManager(redis_client=None, db_conn=mock_db)
        cm.clear_session("sess-pg-clear")

        # Verify DELETE was called
        calls = [str(c) for c in mock_cursor.execute.call_args_list]
        assert any("DELETE" in c.upper() for c in calls)

    def test_clear_calls_redis_delete(self):
        """Verify clear_session calls redis.delete when redis_client is set."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        cm = ConversationManager(redis_client=mock_redis, db_conn=None)
        cm.clear_session("sess-redis-clear")

        mock_redis.delete.assert_called()


# ---------------------------------------------------------------------------
# Req 7.3 — Session expiry after 60 min inactivity
# ---------------------------------------------------------------------------

class TestExpireInactiveSessions:
    def test_active_session_not_expired(self):
        cm = _make_cm()
        session_id = "sess-active"

        cm.append_turn(session_id, _make_turn(0))
        # last_active is just set — well within 60 min
        cm.expire_inactive_sessions(ttl_minutes=60)

        assert len(cm.get_history(session_id, last=10)) == 1

    def test_inactive_session_expired(self):
        cm = _make_cm()
        session_id = "sess-inactive"

        cm.append_turn(session_id, _make_turn(0))

        # Backdate last_active_at to 61 minutes ago
        past = datetime.now(tz=timezone.utc) - timedelta(minutes=61)
        cm._last_active[session_id] = past

        cm.expire_inactive_sessions(ttl_minutes=60)

        assert cm.get_history(session_id, last=10) == []

    def test_session_inactive_for_less_than_ttl_not_expired(self):
        """A session inactive for less than ttl_minutes should NOT be expired."""
        cm = _make_cm()
        session_id = "sess-boundary"

        cm.append_turn(session_id, _make_turn(0))

        # Backdate by only 30 minutes — well within 60-min TTL
        recent = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        cm._last_active[session_id] = recent

        cm.expire_inactive_sessions(ttl_minutes=60)

        # Should NOT be expired
        assert len(cm.get_history(session_id, last=10)) == 1

    def test_multiple_sessions_only_inactive_expired(self):
        cm = _make_cm()
        active_id = "sess-still-active"
        inactive_id = "sess-gone"

        cm.append_turn(active_id, _make_turn(0))
        cm.append_turn(inactive_id, _make_turn(1))

        # Only backdate the inactive session
        past = datetime.now(tz=timezone.utc) - timedelta(minutes=90)
        cm._last_active[inactive_id] = past

        cm.expire_inactive_sessions(ttl_minutes=60)

        assert len(cm.get_history(active_id, last=10)) == 1
        assert cm.get_history(inactive_id, last=10) == []

    def test_expire_queries_postgres_for_inactive_sessions(self):
        """expire_inactive_sessions should query Postgres for sessions with
        stale last_active_at and clear them."""
        stale_session = "sess-stale-pg"
        cutoff_str = None

        def mock_execute(sql: str, params=None):
            nonlocal cutoff_str
            sql_stripped = sql.strip().upper()
            if "LAST_ACTIVE_AT" in sql_stripped and sql_stripped.startswith("SELECT"):
                cutoff_str = params[0] if params else None

        def mock_fetchall():
            return [(stale_session,)]

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = mock_execute
        mock_cursor.fetchall.side_effect = mock_fetchall
        mock_db = MagicMock()
        mock_db.cursor.return_value = mock_cursor

        cm = ConversationManager(redis_client=None, db_conn=mock_db)
        cm.expire_inactive_sessions(ttl_minutes=60)

        # Verify a SELECT with last_active_at was issued
        assert cutoff_str is not None

    def test_custom_ttl_respected(self):
        cm = _make_cm()
        session_id = "sess-custom-ttl"

        cm.append_turn(session_id, _make_turn(0))

        # Backdate by 31 minutes — should expire with ttl=30 but not ttl=60
        past = datetime.now(tz=timezone.utc) - timedelta(minutes=31)
        cm._last_active[session_id] = past

        # With 60-min TTL: should NOT expire
        cm.expire_inactive_sessions(ttl_minutes=60)
        assert len(cm.get_history(session_id, last=10)) == 1

        # With 30-min TTL: should expire
        cm.expire_inactive_sessions(ttl_minutes=30)
        assert cm.get_history(session_id, last=10) == []
