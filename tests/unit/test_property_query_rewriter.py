"""Property-based tests for QueryRewriter.

# Feature: enterprise-rag-system, Property 7: Query rewriter always produces ≥1 variant
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.models import Turn
from enterprise_rag.query_rewriter import QueryRewriter

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_ROLES = ["user", "assistant"]


@st.composite
def turn_strategy(draw) -> Turn:
    role = draw(st.sampled_from(_ROLES))
    text = draw(st.text(min_size=0, max_size=100))
    return Turn(
        role=role,
        original_query=text,
        rewritten_query=text,
        answer="",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


@st.composite
def history_strategy(draw) -> List[Turn]:
    return draw(st.lists(turn_strategy(), min_size=0, max_size=10))


# ---------------------------------------------------------------------------
# Property 7: Query rewriter always produces ≥1 variant
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------


@given(query=st.text(min_size=1), history=history_strategy())
@settings(max_examples=20, deadline=None)
def test_query_rewriter_always_produces_at_least_one_variant(
    query: str, history: List[Turn]
):
    """For any non-empty query and any conversation history, the rewriter must
    return a list with at least one element — even when Groq API is unavailable.

    Validates: Requirements 3.1
    """
    # Feature: enterprise-rag-system, Property 7: Query rewriter always produces ≥1 variant

    # Mock the Groq API HTTP call to return a successful response with variants
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "variant one\nvariant two\nvariant three"
                }
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    rewriter = QueryRewriter(api_key="test-api-key", model="llama-3.1-8b-instant")

    with patch("requests.post", return_value=mock_response):
        result = rewriter.rewrite(query, history)

    assert isinstance(result, list), "rewrite() must return a list"
    assert len(result) >= 1, f"Expected ≥1 variant, got {len(result)}"
    assert result[0] == query, "First variant must be the original query"


@given(query=st.text(min_size=1), history=history_strategy())
@settings(max_examples=20, deadline=None)
def test_query_rewriter_fallback_on_timeout_produces_at_least_one_variant(
    query: str, history: List[Turn]
):
    """Even when Groq API times out, the rewriter must return at least the
    original query — the fallback path must never return an empty list.

    Validates: Requirements 3.1, 3.4
    """
    # Feature: enterprise-rag-system, Property 7: Query rewriter always produces ≥1 variant
    import requests as req_lib

    rewriter = QueryRewriter(api_key="test-api-key", model="llama-3.1-8b-instant")

    with patch("requests.post", side_effect=req_lib.Timeout("simulated timeout")):
        result = rewriter.rewrite(query, history)

    assert isinstance(result, list), "rewrite() must return a list on timeout"
    assert len(result) >= 1, f"Expected ≥1 variant on timeout, got {len(result)}"
    assert result[0] == query, "Fallback must return the original query"
