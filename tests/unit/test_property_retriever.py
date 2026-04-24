"""Property-based tests for the Retriever component.

# Feature: enterprise-rag-system, Property 8: Reciprocal rank fusion correctness
# Feature: enterprise-rag-system, Property 9: Retrieval result ordering and size
# Feature: enterprise-rag-system, Property 10: Hybrid score formula correctness
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from enterprise_rag.retriever import hybrid_score, rrf_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Property 8: Reciprocal rank fusion correctness
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

@settings(max_examples=20, deadline=None)
@given(
    # A list of ranked lists; each inner list is a sequence of document IDs (strings)
    ranked_lists=st.lists(
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10, unique=True),
        min_size=2,
        max_size=5,
    ),
    # The "more-listed" document appears in all lists at the same rank position
    rank_pos=st.integers(min_value=1, max_value=5),
)
def test_property8_rrf_more_lists_higher_score(ranked_lists, rank_pos):
    """Property 8: A document appearing in more ranked lists receives a higher RRF score
    than a document appearing in fewer lists, all else equal (same rank in each list).

    **Validates: Requirements 3.3**
    """
    # doc_many appears in all lists at rank_pos (1-based)
    # doc_few appears in only one list at rank_pos
    n_lists = len(ranked_lists)
    assume(n_lists >= 2)

    # Score for doc_many: appears in all n_lists at rank_pos
    score_many = rrf_score([rank_pos] * n_lists)

    # Score for doc_few: appears in exactly 1 list at rank_pos
    score_few = rrf_score([rank_pos])

    assert score_many > score_few, (
        f"Expected doc in {n_lists} lists (score={score_many:.6f}) to beat "
        f"doc in 1 list (score={score_few:.6f}) at rank {rank_pos}"
    )


@settings(max_examples=20, deadline=None)
@given(
    rank=st.integers(min_value=1, max_value=100),
    n_more=st.integers(min_value=2, max_value=10),
    n_fewer=st.integers(min_value=1, max_value=9),
)
def test_property8_rrf_monotone_in_list_count(rank, n_more, n_fewer):
    """Property 8 (monotonicity): More appearances always yields a higher RRF score.

    **Validates: Requirements 3.3**
    """
    assume(n_more > n_fewer)
    score_more = rrf_score([rank] * n_more)
    score_fewer = rrf_score([rank] * n_fewer)
    assert score_more > score_fewer


# ---------------------------------------------------------------------------
# Property 9: Retrieval result ordering and size
# Validates: Requirements 4.1
# ---------------------------------------------------------------------------

from datetime import datetime, timezone
from enterprise_rag.models import AccessFilter, ScoredChunk
from unittest.mock import MagicMock
from enterprise_rag.retriever import Retriever


def _make_scored_chunk(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        source_type="docs",
        source_id="src1",
        document_title="Doc",
        document_url="http://example.com",
        text=f"text for {chunk_id}",
        token_count=10,
        permission_tags=["engineering"],
        created_at=_NOW,
        source_modified_at=_NOW,
        relevance_score=score,
        rrf_score=0.0,
    )


@settings(max_examples=20, deadline=None)
@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=20,
    ),
    k=st.integers(min_value=1, max_value=15),
)
def test_property9_retrieval_ordering_and_size(scores, k):
    """Property 9: Retriever returns at most K chunks sorted by rrf_score descending.

    **Validates: Requirements 4.1**
    """
    # Build mock chunks with unique IDs
    chunks = [_make_scored_chunk(f"chunk_{i}", s) for i, s in enumerate(scores)]

    # Mock vector_store.query to return our chunks
    mock_vs = MagicMock()
    mock_vs.query.return_value = chunks

    retriever = Retriever(vector_store=mock_vs, embed_fn=lambda t: [0.0] * 4)
    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["engineering"])

    results = retriever.retrieve(
        variants=["test query"],
        access_filter=access_filter,
        k=k,
        semantic_weight=0.7,
        threshold=0.0,  # accept all
    )

    # Result count must be ≤ k
    assert len(results) <= k, f"Expected ≤ {k} results, got {len(results)}"

    # Results must be sorted by rrf_score descending
    for i in range(len(results) - 1):
        assert results[i].rrf_score >= results[i + 1].rrf_score, (
            f"Results not sorted: index {i} score={results[i].rrf_score} "
            f"< index {i+1} score={results[i+1].rrf_score}"
        )


# ---------------------------------------------------------------------------
# Property 10: Hybrid score formula correctness
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------

@settings(max_examples=20, deadline=None)
@given(
    sem=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    bm25=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_property10_hybrid_score_formula(sem, bm25, weight):
    """Property 10: combined == semantic_weight * sem + (1 - semantic_weight) * bm25.

    **Validates: Requirements 4.3**
    """
    combined = hybrid_score(sem, bm25, weight)
    expected = weight * sem + (1.0 - weight) * bm25
    assert math.isclose(combined, expected, rel_tol=1e-9, abs_tol=1e-12), (
        f"hybrid_score({sem}, {bm25}, {weight}) = {combined}, expected {expected}"
    )
