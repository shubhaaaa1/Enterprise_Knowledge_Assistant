"""Access controller for role-based access control (RBAC)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from enterprise_rag.models import AccessFilter, Role

logger = logging.getLogger(__name__)


@dataclass
class Permissions:
    permitted_source_ids: List[str]
    permitted_tags: List[str]


@dataclass
class AccessDecisionLog:
    timestamp: datetime
    user_id: str
    roles_evaluated: List[str]
    sources_filtered: List[str]
    chunks_excluded: int


class AccessController:
    """Enforces RBAC rules for the enterprise RAG system.

    Requirements: 2.1, 2.2, 2.4, 2.5
    """

    def __init__(
        self,
        role_permissions: Optional[Dict[str, Role]] = None,
        db_conn=None,
        cache_ttl_seconds: int = 60,
    ) -> None:
        # In-memory cache: role_name -> (Role, cached_at)
        self._cache: Dict[str, Tuple[Role, datetime]] = {}
        self._cache_ttl = cache_ttl_seconds
        self._db_conn = db_conn
        self._revoked_tokens: Set[str] = set()

        # Seed cache from initial role_permissions dict
        if role_permissions:
            now = datetime.now(tz=timezone.utc)
            for name, role in role_permissions.items():
                self._cache[name] = (role, now)

    # ------------------------------------------------------------------
    # Token validation (Req 2.4)
    # ------------------------------------------------------------------

    def validate_token(self, token: str) -> bool:
        """Return True if the token is valid.

        A token is invalid if it is:
        - empty
        - prefixed with "expired_" or "invalid_"
        - present in the revoked set
        """
        if not token:
            return False
        if token.startswith("expired_") or token.startswith("invalid_"):
            return False
        if token in self._revoked_tokens:
            return False
        return True

    def revoke_token(self, token: str) -> None:
        """Add a token to the revoked set."""
        self._revoked_tokens.add(token)

    # ------------------------------------------------------------------
    # Role resolution (Req 2.2)
    # ------------------------------------------------------------------

    def resolve_roles(self, token: str) -> List[Role]:
        """Parse roles from token and look them up in the cache.

        Token format: "user_id:role1,role2"
        Unknown role names are silently skipped.
        """
        if ":" not in token:
            return []
        _, roles_part = token.split(":", 1)
        role_names = [r.strip() for r in roles_part.split(",") if r.strip()]
        roles: List[Role] = []
        for name in role_names:
            role = self._get_role(name)
            if role is not None:
                roles.append(role)
        return roles

    def _get_role(self, name: str) -> Optional[Role]:
        """Return a Role from cache, refreshing from DB if TTL expired."""
        now = datetime.now(tz=timezone.utc)
        if name in self._cache:
            role, cached_at = self._cache[name]
            age = (now - cached_at).total_seconds()
            if age <= self._cache_ttl:
                return role
            # TTL expired — try to refresh from DB
            if self._db_conn is not None:
                refreshed = self._fetch_role_from_db(name)
                if refreshed is not None:
                    self._cache[name] = (refreshed, now)
                    return refreshed
            # No DB or not found — return stale entry rather than nothing
            return role
        # Not in cache — try DB
        if self._db_conn is not None:
            role = self._fetch_role_from_db(name)
            if role is not None:
                self._cache[name] = (role, now)
                return role
        return None

    def _fetch_role_from_db(self, name: str) -> Optional[Role]:
        """Fetch a role from Postgres. Returns None if not found."""
        try:
            cursor = self._db_conn.cursor()
            cursor.execute(
                "SELECT permitted_source_ids, permitted_tags FROM roles WHERE name = %s",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                return Role(
                    name=name,
                    permitted_source_ids=row[0] or [],
                    permitted_tags=row[1] or [],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch role %r from DB: %s", name, exc)
        return None

    # ------------------------------------------------------------------
    # Access filter construction (Req 2.2)
    # ------------------------------------------------------------------

    def build_access_filter(self, roles: List[Role]) -> AccessFilter:
        """Merge permitted_source_ids and permitted_tags from all roles."""
        source_ids: List[str] = []
        tags: List[str] = []
        seen_sources: Set[str] = set()
        seen_tags: Set[str] = set()

        for role in roles:
            for sid in role.permitted_source_ids:
                if sid not in seen_sources:
                    seen_sources.add(sid)
                    source_ids.append(sid)
            for tag in role.permitted_tags:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    tags.append(tag)

        return AccessFilter(
            permitted_source_ids=source_ids,
            permitted_tags=tags,
        )

    # ------------------------------------------------------------------
    # Role permission updates (Req 2.1, 2.5)
    # ------------------------------------------------------------------

    def update_role_permissions(self, role: Role, permissions: Permissions) -> None:
        """Update role permissions in the in-memory cache (60s TTL).

        Also persists to Postgres if db_conn is provided.
        """
        updated_role = Role(
            name=role.name,
            permitted_source_ids=permissions.permitted_source_ids,
            permitted_tags=permissions.permitted_tags,
        )
        now = datetime.now(tz=timezone.utc)
        self._cache[role.name] = (updated_role, now)

        if self._db_conn is not None:
            try:
                cursor = self._db_conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO roles (name, permitted_source_ids, permitted_tags)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (name) DO UPDATE
                        SET permitted_source_ids = EXCLUDED.permitted_source_ids,
                            permitted_tags = EXCLUDED.permitted_tags
                    """,
                    (
                        role.name,
                        permissions.permitted_source_ids,
                        permissions.permitted_tags,
                    ),
                )
                self._db_conn.commit()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist role %r to DB: %s", role.name, exc)

    def get_role(self, name: str) -> Optional[Role]:
        """Return the cached Role for the given name, or None."""
        return self._get_role(name)

    # ------------------------------------------------------------------
    # Audit logging (Req 10.2)
    # ------------------------------------------------------------------

    def log_access_decision(
        self,
        user_id: str,
        roles: List[Role],
        access_filter: AccessFilter,
        chunks_excluded: int,
    ) -> AccessDecisionLog:
        """Emit a structured log entry for an access control decision.

        Returns the log entry for inspection/testing.
        """
        entry = AccessDecisionLog(
            timestamp=datetime.now(tz=timezone.utc),
            user_id=user_id,
            roles_evaluated=[r.name for r in roles],
            sources_filtered=list(access_filter.permitted_source_ids),
            chunks_excluded=chunks_excluded,
        )
        logger.info(
            "access_decision user_id=%s roles=%s sources=%s chunks_excluded=%d",
            entry.user_id,
            entry.roles_evaluated,
            entry.sources_filtered,
            entry.chunks_excluded,
        )
        return entry
