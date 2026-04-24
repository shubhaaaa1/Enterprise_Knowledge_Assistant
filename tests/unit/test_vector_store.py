"""Unit tests for VectorStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from enterprise_rag.models import AccessFilter, Chunk, ComponentStatus, EmbeddedChunk, ScoredChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedded_chunk(
    chunk_id: str = "c1",
    source_type: str = "docs",
    source_id: str = "src1",
    document_title: str = "Doc",
    document_url: str = "https://example.com/doc",
    text: str = "hello world",
    token_count: int = 2,
    permission_tags: List[str] = None,
    embedding: List[float] = None,
) -> EmbeddedChunk:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return EmbeddedChunk(
        chunk_id=chunk_id,
        source_type=source_type,
        source_id=source_id,
        document_title=document_title,
        document_url=document_url,
        text=text,
        token_count=token_count,
        permission_tags=permission_tags or ["engineering"],
        created_at=now,
        source_modified_at=now,
        embedding=embedding or [0.1, 0.2, 0.3],
    )


# ---------------------------------------------------------------------------
# Import guard — skip all tests if chromadb is not installed
# ---------------------------------------------------------------------------

chromadb = pytest.importorskip("chromadb", reason="chromadb not installed")

from enterprise_rag.vector_store import VectorStore, COLLECTION_NAME  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path):
    """VectorStore backed by a temporary ChromaDB directory."""
    return VectorStore(persist_path=str(tmp_path / "chroma"))


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_upsert_empty_list_is_noop(self, store):
        store.upsert([])  # should not raise

    def test_upsert_stores_chunk(self, store):
        chunk = _make_embedded_chunk()
        store.upsert([chunk])
        result = store.get_by_id(chunk.chunk_id)
        assert result is not None
        assert result.chunk_id == chunk.chunk_id

    def test_upsert_stores_permission_tags(self, store):
        chunk = _make_embedded_chunk(permission_tags=["hr", "finance"])
        store.upsert([chunk])
        result = store.get_by_id(chunk.chunk_id)
        assert result is not None
        assert set(result.permission_tags) == {"hr", "finance"}

    def test_upsert_multiple_chunks(self, store):
        chunks = [
            _make_embedded_chunk(chunk_id=f"c{i}", text=f"text {i}")
            for i in range(5)
        ]
        store.upsert(chunks)
        for chunk in chunks:
            assert store.get_by_id(chunk.chunk_id) is not None

    def test_upsert_overwrites_existing_chunk(self, store):
        chunk = _make_embedded_chunk(text="original")
        store.upsert([chunk])
        updated = _make_embedded_chunk(text="updated")
        store.upsert([updated])
        result = store.get_by_id(chunk.chunk_id)
        assert result.text == "updated"


# ---------------------------------------------------------------------------
# get_by_id
# ---------------------------------------------------------------------------

class TestGetById:
    def test_get_nonexistent_returns_none(self, store):
        assert store.get_by_id("does-not-exist") is None

    def test_get_returns_correct_metadata(self, store):
        chunk = _make_embedded_chunk(
            source_type="github",
            source_id="repo1",
            document_title="README",
            document_url="https://github.com/org/repo",
        )
        store.upsert([chunk])
        result = store.get_by_id(chunk.chunk_id)
        assert result.source_type == "github"
        assert result.source_id == "repo1"
        assert result.document_title == "README"
        assert result.document_url == "https://github.com/org/repo"


# ---------------------------------------------------------------------------
# query — RBAC pre-retrieval filtering
# ---------------------------------------------------------------------------

class TestQuery:
    def _insert_chunks_with_tags(self, store, tag_sets: List[List[str]]) -> List[EmbeddedChunk]:
        """Insert chunks with distinct embeddings and given permission_tags."""
        chunks = []
        for i, tags in enumerate(tag_sets):
            # Use orthogonal-ish embeddings so cosine similarity is meaningful
            emb = [0.0] * 10
            emb[i % 10] = 1.0
            chunks.append(
                _make_embedded_chunk(
                    chunk_id=f"chunk-{i}",
                    text=f"content {i}",
                    permission_tags=tags,
                    embedding=emb,
                )
            )
        store.upsert(chunks)
        return chunks

    def test_query_returns_only_permitted_chunks(self, store):
        """Chunks whose permission_tags don't intersect with user tags must not appear."""
        self._insert_chunks_with_tags(
            store,
            [["engineering"], ["hr"], ["finance"], ["engineering", "hr"]],
        )
        access_filter = AccessFilter(
            permitted_source_ids=[], permitted_tags=["engineering"]
        )
        query_emb = [1.0] + [0.0] * 9
        results = store.query(query_emb, access_filter, k=10, threshold=0.0)
        for chunk in results:
            assert any(tag in chunk.permission_tags for tag in ["engineering"]), (
                f"Chunk {chunk.chunk_id} with tags {chunk.permission_tags} "
                "should not be returned for 'engineering' filter"
            )

    def test_query_empty_permitted_tags_returns_nothing(self, store):
        self._insert_chunks_with_tags(store, [["engineering"], ["hr"]])
        access_filter = AccessFilter(permitted_source_ids=[], permitted_tags=[])
        results = store.query([1.0] + [0.0] * 9, access_filter, k=10, threshold=0.0)
        assert results == []

    def test_query_respects_threshold(self, store):
        """Chunks with relevance_score below threshold must be excluded."""
        chunk = _make_embedded_chunk(
            chunk_id="far-chunk",
            permission_tags=["engineering"],
            # Embedding orthogonal to query → low similarity
            embedding=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
        store.upsert([chunk])
        access_filter = AccessFilter(permitted_source_ids=[], permitted_tags=["engineering"])
        query_emb = [1.0] + [0.0] * 9
        # High threshold — orthogonal vectors should not pass
        results = store.query(query_emb, access_filter, k=10, threshold=0.99)
        assert all(r.chunk_id != "far-chunk" for r in results)

    def test_query_returns_scored_chunks(self, store):
        chunk = _make_embedded_chunk(
            permission_tags=["engineering"],
            embedding=[1.0] + [0.0] * 9,
        )
        store.upsert([chunk])
        access_filter = AccessFilter(permitted_source_ids=[], permitted_tags=["engineering"])
        results = store.query([1.0] + [0.0] * 9, access_filter, k=5, threshold=0.0)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, ScoredChunk)
            assert 0.0 <= r.relevance_score <= 1.0

    def test_query_respects_k_limit(self, store):
        self._insert_chunks_with_tags(store, [["eng"]] * 8)
        access_filter = AccessFilter(permitted_source_ids=[], permitted_tags=["eng"])
        results = store.query([1.0] + [0.0] * 9, access_filter, k=3, threshold=0.0)
        assert len(results) <= 3

    def test_query_multi_tag_filter(self, store):
        """User with multiple permitted tags should see chunks matching any of them."""
        self._insert_chunks_with_tags(
            store,
            [["engineering"], ["hr"], ["finance"]],
        )
        access_filter = AccessFilter(
            permitted_source_ids=[], permitted_tags=["engineering", "hr"]
        )
        results = store.query([1.0] + [0.0] * 9, access_filter, k=10, threshold=0.0)
        for chunk in results:
            assert any(
                tag in chunk.permission_tags for tag in ["engineering", "hr"]
            ), f"Unexpected chunk with tags {chunk.permission_tags}"


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_returns_ok(self, store):
        status = store.health_check()
        assert isinstance(status, ComponentStatus)
        assert status.name == "chromadb"
        assert status.status == "ok"
        assert status.last_checked is not None

    def test_health_check_returns_down_on_error(self, tmp_path):
        store = VectorStore(persist_path=str(tmp_path / "chroma"))
        # Simulate a broken client
        store._client.heartbeat = MagicMock(side_effect=Exception("connection refused"))
        status = store.health_check()
        assert status.status == "down"
        assert "connection refused" in (status.detail or "")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestMetadataRoundtrip:
    def test_chunk_metadata_roundtrip(self, store):
        """Metadata serialisation/deserialisation preserves all fields."""
        now = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        chunk = EmbeddedChunk(
            chunk_id="rt-1",
            source_type="jira",
            source_id="PROJ-42",
            document_title="Bug report",
            document_url="https://jira.example.com/PROJ-42",
            text="Some ticket text",
            token_count=10,
            permission_tags=["engineering", "qa"],
            created_at=now,
            source_modified_at=now,
            embedding=[0.5, 0.5],
        )
        store.upsert([chunk])
        result = store.get_by_id("rt-1")
        assert result is not None
        assert result.source_type == "jira"
        assert result.source_id == "PROJ-42"
        assert result.document_title == "Bug report"
        assert result.document_url == "https://jira.example.com/PROJ-42"
        assert result.text == "Some ticket text"
        assert result.token_count == 10
        assert set(result.permission_tags) == {"engineering", "qa"}
        assert result.created_at == now
        assert result.source_modified_at == now
