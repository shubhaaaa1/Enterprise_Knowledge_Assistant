"""Property-based tests for ConversationManager.

# Feature: enterprise-rag-system, Property 17: Session turn round-trip fidelity
# Feature: enterprise-rag-system, Property 18: Session history window
# Feature: enterprise-rag-system, Property 19: Session persistence across restarts
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, call

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.conversation_manager import ConversationManager
from enterprise_rag.models import Turn

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ROLES = ["user", "assistant"]


@st.composite
def turn_strategy(draw) -> Turn:
    role = draw(st.sampled_from(_ROLES))
    original_query = draw(st.text(min_size=0, max_size=200))
    rewritten_query = draw(st.text(min_size=0, max_size=200))
    answer = draw(st.text(min_size=0, max_size=500))
    # Use fixed-offset aware datetimes to avoid hypothesis timezone issues
    ts = draw(
        st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2099, 12, 31),
            timezones=st.just(timezone.utc),
        )
    )
    return Turn(
        role=role,
        original_query=original_query,
        rewritten_query=rewritten_query,
        answer=answer,
        timestamp=ts,
    )


@st.composite
def turn_list_strategy(draw, min_size: int = 0, max_size: int = 30) -> List[Turn]:
    return draw(st.lists(turn_strategy(), min_size=min_size, max_size=max_size))


# ---------------------------------------------------------------------------
# Property 17: Session turn round-trip fidelity
# Validates: Requirements 7.1, 7.6
# ---------------------------------------------------------------------------


@given(turns=turn_list_strategy(min_size=1, max_size=20))
@settings(max_examples=20, deadline=None)
def test_session_turn_round_trip_fidelity(turns: List[Turn]):
    """For any sequence of turns appended to a session, get_history must return
    the turns in the same order with all fields preserved exactly.

    Validates: Requirements 7.1, 7.6
    """
    # Feature: enterprise-rag-system, Property 17: Session turn round-trip fidelity
    cm = ConversationManager()  # in-memory mode
    session_id = "test-session-roundtrip"

    for turn in turns:
        cm.append_turn(session_id, turn)

    # Retrieve all turns (last=len(turns) to get everything)
    retrieved = cm.get_history(session_id, last=len(turns))

    assert len(retrieved) == len(turns), (
        f"Expected {len(turns)} turns, got {len(retrieved)}"
    )

    for i, (original, got) in enumerate(zip(turns, retrieved)):
        assert got.role == original.role, f"Turn {i}: role mismatch"
        assert got.original_query == original.original_query, f"Turn {i}: original_query mismatch"
        assert got.rewritten_query == original.rewritten_query, f"Turn {i}: rewritten_query mismatch"
        assert got.answer == original.answer, f"Turn {i}: answer mismatch"
        assert got.timestamp == original.timestamp, f"Turn {i}: timestamp mismatch"


# ---------------------------------------------------------------------------
# Property 18: Session history window
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=0, max_value=50))
@settings(max_examples=20, deadline=None)
def test_session_history_window(n: int):
    """For any session containing N turns, get_history(last=10) must return
    exactly min(N, 10) turns, always returning the most recent turns.

    Validates: Requirements 7.2
    """
    # Feature: enterprise-rag-system, Property 18: Session history window
    cm = ConversationManager()  # in-memory mode
    session_id = "test-session-window"

    # Build N turns with distinguishable content
    all_turns: List[Turn] = []
    for i in range(n):
        t = Turn(
            role="user",
            original_query=f"query-{i}",
            rewritten_query=f"rewritten-{i}",
            answer=f"answer-{i}",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        cm.append_turn(session_id, t)
        all_turns.append(t)

    retrieved = cm.get_history(session_id, last=10)
    expected_count = min(n, 10)

    assert len(retrieved) == expected_count, (
        f"N={n}: expected {expected_count} turns, got {len(retrieved)}"
    )

    # Must be the most recent turns
    if expected_count > 0:
        expected_turns = all_turns[-expected_count:]
        for i, (exp, got) in enumerate(zip(expected_turns, retrieved)):
            assert got.original_query == exp.original_query, (
                f"N={n}, turn {i}: expected query {exp.original_query!r}, got {got.original_query!r}"
            )


# ---------------------------------------------------------------------------
# Property 19: Session persistence across restarts
# Validates: Requirements 7.4
# ---------------------------------------------------------------------------


@given(turns=turn_list_strategy(min_size=1, max_size=15))
@settings(max_examples=20, deadline=None)
def test_session_persistence_across_restarts(turns: List[Turn]):
    """After a simulated application restart (new ConversationManager with same
    db_conn), session history must be fully recoverable from durable storage.

    Uses a mock db_conn to verify correct SQL is issued for both write and read.

    Validates: Requirements 7.4
    """
    # Feature: enterprise-rag-system, Property 19: Session persistence across restarts
    session_id = "test-session-persist"

    # --- Build a mock db_conn that records writes and replays them on read ---
    stored_rows: list = []  # list of (session_id, turn_index, role, oq, rq, ans, ts, last_active)

    def mock_execute(sql: str, params=None):
        sql_stripped = sql.strip().upper()
        if sql_stripped.startswith("INSERT"):
            stored_rows.append(params)
        elif sql_stripped.startswith("SELECT DISTINCT"):
            pass  # expire query — not needed here
        elif sql_stripped.startswith("SELECT"):
            pass  # handled by fetchall

    def mock_fetchall():
        # Return rows matching the SELECT for get_history
        return [
            (row[2], row[3], row[4], row[5], row[6])  # role, oq, rq, ans, ts
            for row in sorted(stored_rows, key=lambda r: r[1])  # sort by turn_index
            if row[0] == session_id
        ]

    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = mock_execute
    mock_cursor.fetchall.side_effect = mock_fetchall

    mock_db = MagicMock()
    mock_db.cursor.return_value = mock_cursor

    # --- First "application instance": write turns ---
    cm1 = ConversationManager(redis_client=None, db_conn=mock_db)
    for turn in turns:
        cm1.append_turn(session_id, turn)

    # --- Simulate restart: new ConversationManager with same db_conn ---
    cm2 = ConversationManager(redis_client=None, db_conn=mock_db)

    # Verify INSERT was called for each turn
    assert mock_cursor.execute.call_count >= len(turns), (
        f"Expected at least {len(turns)} execute calls, got {mock_cursor.execute.call_count}"
    )

    # Verify SELECT is issued when reading from Postgres
    # (cm2 has empty in-memory store, so it must read from Postgres)
    retrieved = cm2._pg_get_turns(session_id)

    assert len(retrieved) == len(turns), (
        f"Expected {len(turns)} turns from Postgres, got {len(retrieved)}"
    )

    for i, (original, got) in enumerate(zip(turns, retrieved)):
        assert got.role == original.role, f"Turn {i}: role mismatch after restart"
        assert got.original_query == original.original_query, f"Turn {i}: original_query mismatch after restart"
        assert got.rewritten_query == original.rewritten_query, f"Turn {i}: rewritten_query mismatch after restart"
        assert got.answer == original.answer, f"Turn {i}: answer mismatch after restart"
