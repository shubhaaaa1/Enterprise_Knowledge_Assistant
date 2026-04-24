"""Property-based tests for AccessController — audit log completeness.

# Feature: enterprise-rag-system, Property 22: Access control audit log completeness
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.access_controller import AccessController
from enterprise_rag.models import AccessFilter, Role

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ROLE_NAMES = ["engineering", "hr", "finance", "legal", "ops"]
_SOURCE_IDS = ["src-docs", "src-github", "src-jira", "src-wiki", "src-slack"]
_TAGS = ["engineering", "hr", "finance", "legal", "ops", "public"]


@st.composite
def role_strategy(draw) -> Role:
    name = draw(st.sampled_from(_ROLE_NAMES))
    source_ids = draw(st.lists(st.sampled_from(_SOURCE_IDS), min_size=0, max_size=3, unique=True))
    tags = draw(st.lists(st.sampled_from(_TAGS), min_size=0, max_size=3, unique=True))
    return Role(name=name, permitted_source_ids=source_ids, permitted_tags=tags)


@st.composite
def access_filter_strategy(draw) -> AccessFilter:
    source_ids = draw(st.lists(st.sampled_from(_SOURCE_IDS), min_size=0, max_size=3, unique=True))
    tags = draw(st.lists(st.sampled_from(_TAGS), min_size=0, max_size=3, unique=True))
    return AccessFilter(permitted_source_ids=source_ids, permitted_tags=tags)


# ---------------------------------------------------------------------------
# Property 22: Access control audit log completeness
# Validates: Requirements 10.2
# ---------------------------------------------------------------------------


@given(
    user_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_")),
    roles=st.lists(role_strategy(), min_size=0, max_size=5),
    access_filter=access_filter_strategy(),
    chunks_excluded=st.integers(min_value=0, max_value=1000),
)
@settings(max_examples=20, deadline=None)
def test_audit_log_completeness(user_id, roles, access_filter, chunks_excluded):
    """For random access decisions, the emitted log entry must have non-null
    timestamp, user_id, roles_evaluated, sources_filtered, chunks_excluded.

    Validates: Requirements 10.2
    """
    controller = AccessController()

    entry = controller.log_access_decision(
        user_id=user_id,
        roles=roles,
        access_filter=access_filter,
        chunks_excluded=chunks_excluded,
    )

    # timestamp must be a non-null datetime
    assert entry.timestamp is not None
    assert isinstance(entry.timestamp, datetime)

    # user_id must be non-null and match input
    assert entry.user_id is not None
    assert entry.user_id == user_id

    # roles_evaluated must be a list (may be empty if no roles provided)
    assert entry.roles_evaluated is not None
    assert isinstance(entry.roles_evaluated, list)
    assert entry.roles_evaluated == [r.name for r in roles]

    # sources_filtered must be a list
    assert entry.sources_filtered is not None
    assert isinstance(entry.sources_filtered, list)
    assert entry.sources_filtered == list(access_filter.permitted_source_ids)

    # chunks_excluded must be non-negative integer matching input
    assert entry.chunks_excluded is not None
    assert isinstance(entry.chunks_excluded, int)
    assert entry.chunks_excluded == chunks_excluded
