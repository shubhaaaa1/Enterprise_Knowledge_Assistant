"""Integration tests for the Enterprise RAG System API layer.

Uses FastAPI's TestClient with mocked external dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from enterprise_rag.api import app, get_conversation_manager, get_generator, get_retriever
from enterprise_rag.models import (
    AccessFilter,
    Citation,
    CitedAnswer,
    ComponentStatus,
    GraphCitation,
    Role,
    ScoredChunk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scored_chunk(chunk_id: str = "chunk-1") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        source_type="docs",
        source_id="src-1",
        document_title="Test Doc",
        document_url="https://example.com/doc",
        text="Some relevant content [1].",
        token_count=10,
        permission_tags=["engineering"],
        created_at=datetime.now(tz=timezone.utc),
        source_modified_at=datetime.now(tz=timezone.utc),
        relevance_score=0.9,
        rrf_score=0.9,
    )


def _make_cited_answer(dependency_graph_unavailable: bool = False) -> CitedAnswer:
    return CitedAnswer(
        answer_text="Answer [1].",
        citations=[
            Citation(
                number=1,
                source_type="docs",
                document_title="Test Doc",
                document_url="https://example.com/doc",
                excerpt="Some relevant content",
                chunk_ids=["chunk-1"],
            )
        ],
        graph_citations=[],
        grounding_score=1.0,
        unverified_claims=[],
        low_confidence_warning=False,
        dependency_graph_unavailable=dependency_graph_unavailable,
    )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Test 1: valid query → cited answer with grounding score (Req 2.4)
# ---------------------------------------------------------------------------

def test_query_returns_cited_answer():
    mock_cm = MagicMock()
    mock_cm.get_history.return_value = []

    chunk = _make_scored_chunk()
    mock_ret = MagicMock()
    mock_ret.retrieve.return_value = [chunk]
    mock_ret.last_dependency_graph_unavailable = False

    mock_gen = MagicMock()
    mock_gen.generate.return_value = "Answer [1]."  # plain string → non-streaming

    mock_ce = MagicMock()
    mock_ce.cite.return_value = _make_cited_answer()

    mock_sl = MagicMock()
    mock_sl.new_correlation_id.return_value = "corr-123"

    app.dependency_overrides[get_conversation_manager] = lambda: mock_cm
    app.dependency_overrides[get_retriever] = lambda: mock_ret
    app.dependency_overrides[get_generator] = lambda: mock_gen

    # Also override citation_engine, structured_logger
    from enterprise_rag.api import get_citation_engine, get_structured_logger
    app.dependency_overrides[get_citation_engine] = lambda: mock_ce
    app.dependency_overrides[get_structured_logger] = lambda: mock_sl

    try:
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"session_id": "sess-1", "query": "What is X?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "grounding_score" in data
        assert data["grounding_score"] == 1.0
        assert len(data["citations"]) == 1
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test 2: GET /health returns all component statuses (Req 10.3)
# ---------------------------------------------------------------------------

def test_health_returns_all_component_statuses():
    now = datetime.now(tz=timezone.utc)

    mock_vs = MagicMock()
    mock_vs.health_check.return_value = ComponentStatus(
        name="chromadb", status="ok", last_checked=now, detail=None
    )

    mock_ret = MagicMock()
    mock_ret._vector_store = mock_vs

    from enterprise_rag.api import get_structured_logger
    mock_sl = MagicMock()
    mock_sl.new_correlation_id.return_value = "corr-health"
    app.dependency_overrides[get_retriever] = lambda: mock_ret
    app.dependency_overrides[get_structured_logger] = lambda: mock_sl

    try:
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        component_names = [c["name"] for c in data["components"]]
        assert "chromadb" in component_names
        assert "session_store" in component_names
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test 3: DELETE /session/{id} clears session history (Req 7.5)
# ---------------------------------------------------------------------------

def test_delete_session_clears_history():
    mock_cm = MagicMock()

    app.dependency_overrides[get_conversation_manager] = lambda: mock_cm

    try:
        with TestClient(app) as client:
            response = client.delete("/session/my-session-id")

        assert response.status_code == 204
        mock_cm.clear_session.assert_called_once_with("my-session-id")
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test 4: Groq API unavailable → 503 (Req 8.2)
# ---------------------------------------------------------------------------

def test_groq_unavailable_returns_503():
    mock_cm = MagicMock()
    mock_cm.get_history.return_value = []

    mock_ret = MagicMock()
    mock_ret.retrieve.return_value = [_make_scored_chunk()]
    mock_ret.last_dependency_graph_unavailable = False

    mock_gen = MagicMock()
    mock_gen.generate.side_effect = ConnectionError("Groq API unreachable")

    mock_sl = MagicMock()
    mock_sl.new_correlation_id.return_value = "corr-503"

    from enterprise_rag.api import get_citation_engine, get_structured_logger
    app.dependency_overrides[get_conversation_manager] = lambda: mock_cm
    app.dependency_overrides[get_retriever] = lambda: mock_ret
    app.dependency_overrides[get_generator] = lambda: mock_gen
    app.dependency_overrides[get_structured_logger] = lambda: mock_sl

    try:
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"session_id": "sess-503", "query": "What is X?"},
            )
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test 5: generator timeout → 504 within 30s (Req 5.4)
# ---------------------------------------------------------------------------

def test_generator_timeout_returns_504():
    mock_cm = MagicMock()
    mock_cm.get_history.return_value = []

    mock_ret = MagicMock()
    mock_ret.retrieve.return_value = [_make_scored_chunk()]
    mock_ret.last_dependency_graph_unavailable = False

    mock_gen = MagicMock()
    mock_gen.generate.side_effect = TimeoutError("Generation timed out")

    mock_sl = MagicMock()
    mock_sl.new_correlation_id.return_value = "corr-504"

    from enterprise_rag.api import get_citation_engine, get_structured_logger
    app.dependency_overrides[get_conversation_manager] = lambda: mock_cm
    app.dependency_overrides[get_retriever] = lambda: mock_ret
    app.dependency_overrides[get_generator] = lambda: mock_gen
    app.dependency_overrides[get_structured_logger] = lambda: mock_sl

    try:
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"session_id": "sess-504", "query": "What is X?"},
            )
        assert response.status_code == 504
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test 6: Neo4j unavailable → vector-only fallback, dependency_graph_unavailable: true (Req 11.6)
# ---------------------------------------------------------------------------

def test_neo4j_unavailable_returns_vector_only_fallback():
    mock_cm = MagicMock()
    mock_cm.get_history.return_value = []

    chunk = _make_scored_chunk()
    mock_ret = MagicMock()
    mock_ret.retrieve.return_value = [chunk]
    # Simulate Neo4j was unavailable during retrieval
    mock_ret.last_dependency_graph_unavailable = True

    mock_gen = MagicMock()
    mock_gen.generate.return_value = "Answer [1]."

    mock_ce = MagicMock()
    mock_ce.cite.return_value = _make_cited_answer(dependency_graph_unavailable=False)

    mock_sl = MagicMock()
    mock_sl.new_correlation_id.return_value = "corr-neo4j"

    from enterprise_rag.api import get_citation_engine, get_structured_logger
    app.dependency_overrides[get_conversation_manager] = lambda: mock_cm
    app.dependency_overrides[get_retriever] = lambda: mock_ret
    app.dependency_overrides[get_generator] = lambda: mock_gen
    app.dependency_overrides[get_citation_engine] = lambda: mock_ce
    app.dependency_overrides[get_structured_logger] = lambda: mock_sl

    try:
        with TestClient(app) as client:
            response = client.post(
                "/query",
                json={"session_id": "sess-neo4j", "query": "What is X?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["dependency_graph_unavailable"] is True
    finally:
        app.dependency_overrides.clear()
