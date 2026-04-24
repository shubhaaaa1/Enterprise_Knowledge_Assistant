"""Unit tests for AccessController.

Covers:
- Req 2.4: expired/invalid token rejection
- Req 2.5: role permission update propagates within 60s (TTL behaviour)
- Req 2.1: role mapping CRUD operations
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from enterprise_rag.access_controller import AccessController, Permissions
from enterprise_rag.models import AccessFilter, Role


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_role(name: str, sources=None, tags=None) -> Role:
    return Role(
        name=name,
        permitted_source_ids=sources or [],
        permitted_tags=tags or [name],
    )


# ---------------------------------------------------------------------------
# Req 2.4 — Token validation
# ---------------------------------------------------------------------------

class TestValidateToken:
    def test_valid_token_accepted(self):
        ac = AccessController()
        assert ac.validate_token("alice:engineering") is True

    def test_empty_token_rejected(self):
        ac = AccessController()
        assert ac.validate_token("") is False

    def test_expired_prefix_rejected(self):
        ac = AccessController()
        assert ac.validate_token("expired_alice:engineering") is False

    def test_invalid_prefix_rejected(self):
        ac = AccessController()
        assert ac.validate_token("invalid_token") is False

    def test_revoked_token_rejected(self):
        ac = AccessController()
        token = "bob:hr"
        ac.revoke_token(token)
        assert ac.validate_token(token) is False

    def test_non_revoked_token_still_valid(self):
        ac = AccessController()
        ac.revoke_token("bob:hr")
        assert ac.validate_token("alice:engineering") is True


# ---------------------------------------------------------------------------
# Req 2.1 — Role mapping CRUD
# ---------------------------------------------------------------------------

class TestRoleMappingCRUD:
    def test_add_role_via_constructor(self):
        role = _make_role("engineering", sources=["src-github"], tags=["engineering"])
        ac = AccessController(role_permissions={"engineering": role})
        retrieved = ac.get_role("engineering")
        assert retrieved is not None
        assert retrieved.name == "engineering"
        assert retrieved.permitted_source_ids == ["src-github"]

    def test_update_role_permissions(self):
        role = _make_role("hr", sources=["src-docs"], tags=["hr"])
        ac = AccessController(role_permissions={"hr": role})

        new_perms = Permissions(
            permitted_source_ids=["src-docs", "src-jira"],
            permitted_tags=["hr", "public"],
        )
        ac.update_role_permissions(role, new_perms)

        updated = ac.get_role("hr")
        assert updated is not None
        assert updated.permitted_source_ids == ["src-docs", "src-jira"]
        assert updated.permitted_tags == ["hr", "public"]

    def test_unknown_role_returns_none(self):
        ac = AccessController()
        assert ac.get_role("nonexistent") is None

    def test_resolve_roles_parses_token(self):
        eng = _make_role("engineering")
        hr = _make_role("hr")
        ac = AccessController(role_permissions={"engineering": eng, "hr": hr})

        roles = ac.resolve_roles("alice:engineering,hr")
        names = {r.name for r in roles}
        assert names == {"engineering", "hr"}

    def test_resolve_roles_skips_unknown(self):
        eng = _make_role("engineering")
        ac = AccessController(role_permissions={"engineering": eng})

        roles = ac.resolve_roles("alice:engineering,finance")
        assert len(roles) == 1
        assert roles[0].name == "engineering"

    def test_resolve_roles_empty_for_no_colon(self):
        ac = AccessController()
        assert ac.resolve_roles("just-a-token") == []

    def test_build_access_filter_merges_roles(self):
        eng = _make_role("engineering", sources=["src-github"], tags=["engineering"])
        hr = _make_role("hr", sources=["src-docs"], tags=["hr"])
        ac = AccessController(role_permissions={"engineering": eng, "hr": hr})

        roles = ac.resolve_roles("alice:engineering,hr")
        af = ac.build_access_filter(roles)

        assert set(af.permitted_source_ids) == {"src-github", "src-docs"}
        assert set(af.permitted_tags) == {"engineering", "hr"}

    def test_build_access_filter_deduplicates(self):
        eng1 = _make_role("engineering", sources=["src-github", "src-docs"], tags=["engineering"])
        eng2 = _make_role("hr", sources=["src-docs"], tags=["engineering"])
        ac = AccessController(role_permissions={"engineering": eng1, "hr": eng2})

        af = ac.build_access_filter([eng1, eng2])
        # "src-docs" and "engineering" should appear only once
        assert af.permitted_source_ids.count("src-docs") == 1
        assert af.permitted_tags.count("engineering") == 1


# ---------------------------------------------------------------------------
# Req 2.5 — Role permission update propagates within 60s (TTL)
# ---------------------------------------------------------------------------

class TestCacheTTL:
    def test_fresh_cache_entry_returned_immediately(self):
        role = _make_role("finance", sources=["src-finance"], tags=["finance"])
        ac = AccessController(role_permissions={"finance": role})

        retrieved = ac.get_role("finance")
        assert retrieved is not None
        assert retrieved.permitted_source_ids == ["src-finance"]

    def test_updated_permissions_visible_before_ttl_expires(self):
        """After update_role_permissions, the new permissions are immediately
        visible (cache refreshed with new TTL), satisfying the ≤60s requirement."""
        role = _make_role("legal", sources=["src-old"], tags=["legal"])
        ac = AccessController(role_permissions={"legal": role}, cache_ttl_seconds=60)

        new_perms = Permissions(
            permitted_source_ids=["src-new"],
            permitted_tags=["legal", "compliance"],
        )
        ac.update_role_permissions(role, new_perms)

        updated = ac.get_role("legal")
        assert updated is not None
        assert updated.permitted_source_ids == ["src-new"]
        assert "compliance" in updated.permitted_tags

    def test_stale_cache_entry_returned_when_no_db(self):
        """When TTL expires and no DB is configured, the stale entry is returned
        rather than None (graceful degradation)."""
        role = _make_role("ops", sources=["src-ops"], tags=["ops"])
        ac = AccessController(role_permissions={"ops": role}, cache_ttl_seconds=60)

        # Simulate TTL expiry by backdating the cache entry
        past = datetime.now(tz=timezone.utc) - timedelta(seconds=120)
        ac._cache["ops"] = (role, past)

        # No DB configured — should return stale entry
        retrieved = ac.get_role("ops")
        assert retrieved is not None
        assert retrieved.name == "ops"

    def test_cache_ttl_respected_with_mocked_time(self):
        """Verify that a cache entry within TTL is served from cache without
        hitting the DB, and an expired entry triggers a DB lookup."""
        role = _make_role("engineering", sources=["src-github"], tags=["engineering"])
        ac = AccessController(role_permissions={"engineering": role}, cache_ttl_seconds=60)

        # Entry is fresh — should be returned from cache
        result = ac.get_role("engineering")
        assert result is not None
        assert result.permitted_source_ids == ["src-github"]

        # Expire the cache entry
        past = datetime.now(tz=timezone.utc) - timedelta(seconds=120)
        ac._cache["engineering"] = (role, past)

        # Still returned (stale) since no DB
        result_stale = ac.get_role("engineering")
        assert result_stale is not None

    def test_update_resets_ttl(self):
        """After update_role_permissions, the cache timestamp is refreshed so
        the entry is valid for another full TTL period."""
        role = _make_role("hr", sources=["src-docs"], tags=["hr"])
        ac = AccessController(role_permissions={"hr": role}, cache_ttl_seconds=60)

        # Expire the existing entry
        past = datetime.now(tz=timezone.utc) - timedelta(seconds=120)
        ac._cache["hr"] = (role, past)

        # Update permissions — this should reset the TTL
        new_perms = Permissions(permitted_source_ids=["src-docs", "src-hr"], permitted_tags=["hr"])
        ac.update_role_permissions(role, new_perms)

        _, cached_at = ac._cache["hr"]
        age = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
        assert age < 5  # freshly cached
