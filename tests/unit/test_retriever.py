"""Unit tests for the Retriever component."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from enterprise_rag.models import AccessFilter, ScoredChunk
from enterprise_rag.retriever import Retriever

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_chunk(chunk_id: str, score: float, tags: list[str] | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        source_type="docs",
        source_id="src1",
        document_title="Doc",
        document_url="http://example.com",
        text=f"content of {chunk_id}",
        token_count=10,
        permission_tags=tags or ["engineering"],
        created_at=_NOW,
        source_modified_at=_NOW,
        relevance_score=score,
        rrf_score=0.0,
    )


# ---------------------------------------------------------------------------
# Req 4.4 — Empty result when no chunks above threshold
# ---------------------------------------------------------------------------

def test_empty_result_when_no_chunks_above_threshold():
    """Retriever returns empty list when vector store returns no chunks (Req 4.4)."""
    mock_vs = MagicMock()
    mock_vs.query.return_value = []

    retriever = Retriever(vector_store=mock_vs, embed_fn=lambda t: [0.0] * 4)
    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["engineering"])

    results = retriever.retrieve(
        variants=["what is the deployment process?"],
        access_filter=access_filter,
        k=10,
        threshold=0.5,
    )

    assert results == []


def test_empty_result_when_all_chunks_below_threshold():
    """Retriever returns empty list when all RRF-normalised scores are below threshold."""
    # Return chunks with very low semantic scores so normalised RRF < threshold
    chunks = [_make_chunk(f"c{i}", 0.01) for i in range(3)]
    mock_vs = MagicMock()
    mock_vs.query.return_value = chunks

    retriever = Retriever(vector_store=mock_vs, embed_fn=lambda t: [0.0] * 4)
    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["engineering"])

    # threshold=1.0 means only the top-normalised chunk (score=1.0) would pass,
    # but with a single variant all chunks get the same normalised score of 1.0
    # so use threshold > 1.0 to force empty
    results = retriever.retrieve(
        variants=["query"],
        access_filter=access_filter,
        k=10,
        threshold=1.1,  # impossible threshold
    )

    assert results == []


def test_empty_variants_returns_empty():
    """Retriever returns empty list when no variants are provided."""
    mock_vs = MagicMock()
    retriever = Retriever(vector_store=mock_vs, embed_fn=lambda t: [0.0] * 4)
    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["engineering"])

    results = retriever.retrieve(variants=[], access_filter=access_filter, k=10)
    assert results == []
    mock_vs.query.assert_not_called()


# ---------------------------------------------------------------------------
# Req 4.2 — Access filter applied before scoring
# ---------------------------------------------------------------------------

def test_access_filter_passed_to_vector_store():
    """Access filter is forwarded to vector_store.query (Req 4.2)."""
    mock_vs = MagicMock()
    mock_vs.query.return_value = []

    access_filter = AccessFilter(
        permitted_source_ids=["repo-42"],
        permitted_tags=["engineering", "platform"],
    )
    retriever = Retriever(vector_store=mock_vs, embed_fn=lambda t: [0.0] * 4)

    retriever.retrieve(
        variants=["how does auth work?"],
        access_filter=access_filter,
        k=5,
    )

    # vector_store.query must have been called with the exact access_filter
    mock_vs.query.assert_called_once()
    _, kwargs = mock_vs.query.call_args
    # access_filter may be positional or keyword
    call_args = mock_vs.query.call_args
    passed_filter = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("access_filter")
    assert passed_filter is access_filter


def test_access_filter_passed_for_multiple_variants():
    """Access filter is forwarded for every variant (Req 4.2)."""
    mock_vs = MagicMock()
    mock_vs.query.return_value = []

    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["hr"])
    retriever = Retriever(vector_store=mock_vs, embed_fn=lambda t: [0.0] * 4)

    retriever.retrieve(
        variants=["query A", "query B", "query C"],
        access_filter=access_filter,
        k=5,
    )

    # Should be called once per variant
    assert mock_vs.query.call_count == 3
    for c in mock_vs.query.call_args_list:
        passed_filter = c.args[1] if len(c.args) > 1 else c.kwargs.get("access_filter")
        assert passed_filter is access_filter


# ---------------------------------------------------------------------------
# Result ordering and K limit
# ---------------------------------------------------------------------------

def test_results_sorted_by_rrf_score_descending():
    """Returned chunks are sorted by rrf_score descending."""
    chunks = [
        _make_chunk("c1", 0.9),
        _make_chunk("c2", 0.5),
        _make_chunk("c3", 0.1),
    ]
    mock_vs = MagicMock()
    mock_vs.query.return_value = chunks

    retriever = Retriever(vector_store=mock_vs, embed_fn=lambda t: [0.0] * 4)
    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["engineering"])

    results = retriever.retrieve(
        variants=["query"], access_filter=access_filter, k=10, threshold=0.0
    )

    for i in range(len(results) - 1):
        assert results[i].rrf_score >= results[i + 1].rrf_score


def test_results_capped_at_k():
    """Retriever returns at most K chunks."""
    chunks = [_make_chunk(f"c{i}", float(i) / 20) for i in range(20)]
    mock_vs = MagicMock()
    mock_vs.query.return_value = chunks

    retriever = Retriever(vector_store=mock_vs, embed_fn=lambda t: [0.0] * 4)
    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["engineering"])

    results = retriever.retrieve(
        variants=["query"], access_filter=access_filter, k=5, threshold=0.0
    )

    assert len(results) <= 5


# ---------------------------------------------------------------------------
# Neo4j degradation
# ---------------------------------------------------------------------------

def test_neo4j_unavailability_sets_flag():
    """When Neo4j raises an exception, last_dependency_graph_unavailable is True."""
    mock_vs = MagicMock()
    mock_vs.query.return_value = [_make_chunk("c1", 0.8)]

    mock_gs = MagicMock()
    mock_gs.query_dependencies.side_effect = Exception("Neo4j down")

    retriever = Retriever(
        vector_store=mock_vs,
        graph_store=mock_gs,
        embed_fn=lambda t: [0.0] * 4,
    )
    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["engineering"])

    # Code-related query to trigger graph lookup
    results = retriever.retrieve(
        variants=["show me the function calls for class MyClass"],
        access_filter=access_filter,
        k=10,
        threshold=0.0,
    )

    assert retriever.last_dependency_graph_unavailable is True
    # Should still return vector results
    assert len(results) > 0


def test_neo4j_unavailability_flag_reset_on_success():
    """last_dependency_graph_unavailable is False when Neo4j succeeds."""
    from enterprise_rag.models import DependencyGraph

    mock_vs = MagicMock()
    mock_vs.query.return_value = [_make_chunk("c1", 0.8)]

    mock_gs = MagicMock()
    mock_gs.query_dependencies.return_value = DependencyGraph(
        nodes=[], edges=[], root_symbol="foo", direction="both", depth=2
    )

    retriever = Retriever(
        vector_store=mock_vs,
        graph_store=mock_gs,
        embed_fn=lambda t: [0.0] * 4,
    )
    access_filter = AccessFilter(permitted_source_ids=["src1"], permitted_tags=["engineering"])

    retriever.retrieve(
        variants=["show me the function calls for class MyClass"],
        access_filter=access_filter,
        k=10,
        threshold=0.0,
    )

    assert retriever.last_dependency_graph_unavailable is False
