"""Conversation Manager for session history persistence.

Manages session lifecycle with Redis (primary) + Postgres (durable) storage.
Falls back to in-memory dict when Redis is unavailable.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from enterprise_rag.models import Turn

logger = logging.getLogger(__name__)

# Redis key prefix for session turns
_SESSION_KEY = "session:{session_id}"
# Redis key prefix for last_active_at tracking
_ACTIVE_KEY = "session_active:{session_id}"


def _turn_to_dict(turn: Turn) -> dict:
    """Serialize a Turn to a JSON-serialisable dict."""
    return {
        "role": turn.role,
        "original_query": turn.original_query,
        "rewritten_query": turn.rewritten_query,
        "answer": turn.answer,
        "timestamp": turn.timestamp.isoformat(),
    }


def _dict_to_turn(d: dict) -> Turn:
    """Deserialize a dict back to a Turn."""
    return Turn(
        role=d["role"],
        original_query=d["original_query"],
        rewritten_query=d["rewritten_query"],
        answer=d["answer"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
    )


class ConversationManager:
    """Manages conversational session history.

    Storage strategy:
    - Redis (primary): fast read/write; key ``session:{session_id}`` holds a
      JSON-encoded list of turn dicts.
    - Postgres (durable): survives application restarts; table ``sessions``
      with columns (session_id, turn_index, role, original_query,
      rewritten_query, answer, timestamp, last_active_at).
    - In-memory dict: fallback when ``redis_client`` is None.

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
    """

    def __init__(self, redis_client=None, db_conn=None) -> None:
        self._redis = redis_client
        self._db_conn = db_conn
        # In-memory fallback: session_id -> List[Turn]
        self._memory: Dict[str, List[Turn]] = {}
        # In-memory last_active_at: session_id -> datetime
        self._last_active: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redis_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def _active_key(self, session_id: str) -> str:
        return f"session_active:{session_id}"

    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def _touch_active(self, session_id: str) -> None:
        """Update last_active_at in Redis and in-memory tracker."""
        now = self._now()
        self._last_active[session_id] = now
        if self._redis is not None:
            try:
                self._redis.set(self._active_key(session_id), now.isoformat())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis set active key failed for %s: %s", session_id, exc)

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    def _redis_get_turns(self, session_id: str) -> Optional[List[Turn]]:
        """Read all turns from Redis. Returns None on failure."""
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._redis_key(session_id))
            if raw is None:
                return []
            data = json.loads(raw)
            return [_dict_to_turn(d) for d in data]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis get failed for session %s: %s", session_id, exc)
            return None

    def _redis_set_turns(self, session_id: str, turns: List[Turn]) -> bool:
        """Write all turns to Redis. Returns True on success."""
        if self._redis is None:
            return False
        try:
            self._redis.set(
                self._redis_key(session_id),
                json.dumps([_turn_to_dict(t) for t in turns]),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis set failed for session %s: %s", session_id, exc)
            return False

    def _redis_delete(self, session_id: str) -> None:
        """Delete session keys from Redis."""
        if self._redis is None:
            return
        try:
            self._redis.delete(self._redis_key(session_id))
            self._redis.delete(self._active_key(session_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis delete failed for session %s: %s", session_id, exc)

    # ------------------------------------------------------------------
    # Postgres helpers
    # ------------------------------------------------------------------

    def _pg_get_turns(self, session_id: str) -> List[Turn]:
        """Read all turns from Postgres ordered by turn_index."""
        if self._db_conn is None:
            return []
        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                SELECT role, original_query, rewritten_query, answer, timestamp
                FROM sessions
                WHERE session_id = %s
                ORDER BY turn_index ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            return [
                Turn(
                    role=row[0],
                    original_query=row[1],
                    rewritten_query=row[2],
                    answer=row[3],
                    timestamp=row[4] if isinstance(row[4], datetime) else datetime.fromisoformat(str(row[4])),
                )
                for row in rows
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Postgres get_turns failed for session %s: %s", session_id, exc)
            return []

    def _pg_append_turn(self, session_id: str, turn_index: int, turn: Turn) -> None:
        """Insert a single turn into Postgres and update last_active_at."""
        if self._db_conn is None:
            return
        now = self._now()
        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO sessions
                    (session_id, turn_index, role, original_query, rewritten_query, answer, timestamp, last_active_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id, turn_index) DO UPDATE
                    SET role = EXCLUDED.role,
                        original_query = EXCLUDED.original_query,
                        rewritten_query = EXCLUDED.rewritten_query,
                        answer = EXCLUDED.answer,
                        timestamp = EXCLUDED.timestamp,
                        last_active_at = EXCLUDED.last_active_at
                """,
                (
                    session_id,
                    turn_index,
                    turn.role,
                    turn.original_query,
                    turn.rewritten_query,
                    turn.answer,
                    turn.timestamp.isoformat(),
                    now.isoformat(),
                ),
            )
            self._db_conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Postgres append_turn failed for session %s: %s", session_id, exc)

    def _pg_delete(self, session_id: str) -> None:
        """Delete all rows for a session from Postgres."""
        if self._db_conn is None:
            return
        try:
            cursor = self._db_conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
            self._db_conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Postgres delete failed for session %s: %s", session_id, exc)

    def _pg_get_inactive_sessions(self, cutoff: datetime) -> List[str]:
        """Return session_ids with last_active_at older than cutoff."""
        if self._db_conn is None:
            return []
        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT session_id FROM sessions
                WHERE last_active_at < %s
                """,
                (cutoff.isoformat(),),
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Postgres get_inactive_sessions failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self, session_id: str, last: int = 10) -> List[Turn]:
        """Return the last ``last`` turns for the session.

        Reads from Redis first; falls back to Postgres if Redis is
        unavailable or returns None.

        Requirements: 7.1, 7.2, 7.6
        """
        turns = self._redis_get_turns(session_id)
        if turns is None:
            # Redis unavailable — fall back to Postgres
            turns = self._pg_get_turns(session_id)
        elif not turns and self._redis is None:
            # No Redis configured — use in-memory
            turns = self._memory.get(session_id, [])

        if self._redis is None:
            turns = self._memory.get(session_id, [])

        return turns[-last:] if last > 0 else []

    def append_turn(self, session_id: str, turn: Turn) -> None:
        """Append a turn to the session history.

        Writes to Redis (primary) and Postgres (durable). Updates
        last_active_at. Falls back to in-memory when Redis is None.

        Requirements: 7.1, 7.6
        """
        if self._redis is None:
            # In-memory path
            if session_id not in self._memory:
                self._memory[session_id] = []
            self._memory[session_id].append(turn)
            self._touch_active(session_id)
            # Still write to Postgres if available
            turn_index = len(self._memory[session_id]) - 1
            self._pg_append_turn(session_id, turn_index, turn)
            return

        # Redis path: read current list, append, write back
        existing = self._redis_get_turns(session_id) or []
        existing.append(turn)
        self._redis_set_turns(session_id, existing)
        self._touch_active(session_id)

        # Durable write to Postgres
        turn_index = len(existing) - 1
        self._pg_append_turn(session_id, turn_index, turn)

    def clear_session(self, session_id: str) -> None:
        """Delete session history from Redis, Postgres, and in-memory store.

        Requirements: 7.5
        """
        # In-memory
        self._memory.pop(session_id, None)
        self._last_active.pop(session_id, None)
        # Redis
        self._redis_delete(session_id)
        # Postgres
        self._pg_delete(session_id)

    def expire_inactive_sessions(self, ttl_minutes: int = 60) -> None:
        """Expire sessions that have been inactive for ``ttl_minutes`` minutes.

        Checks both in-memory last_active_at and Postgres last_active_at.

        Requirements: 7.3
        """
        cutoff = self._now() - timedelta(minutes=ttl_minutes)

        # Expire in-memory sessions
        expired_ids = [
            sid
            for sid, last_active in list(self._last_active.items())
            if last_active < cutoff
        ]
        for sid in expired_ids:
            logger.info("Expiring inactive session %s", sid)
            self.clear_session(sid)

        # Expire Postgres sessions not already cleared
        pg_expired = self._pg_get_inactive_sessions(cutoff)
        for sid in pg_expired:
            if sid not in expired_ids:
                logger.info("Expiring inactive Postgres session %s", sid)
                self.clear_session(sid)
