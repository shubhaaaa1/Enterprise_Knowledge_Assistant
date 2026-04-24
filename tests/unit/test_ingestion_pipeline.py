"""Unit tests for IngestionPipeline."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import pytest

from enterprise_rag.ast_parser import ASTParser
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.models import (
    Chunk,
    CodeSymbol,
    Document,
    EmbeddedChunk,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_doc(content: str, source_type: str = "docs") -> Document:
    return Document(
        doc_id="doc-1",
        source_type=source_type,
        source_id="src-1",
        title="Test Doc",
        url="https://example.com/doc",
        content=content,
        permission_tags=["engineering"],
        modified_at=datetime(2024, 1, 1),
    )


def _identity_embed(chunks: List[Chunk]) -> List[EmbeddedChunk]:
    """Embed function that returns EmbeddedChunks with a dummy vector."""
    return [
        EmbeddedChunk(
            chunk_id=c.chunk_id,
            source_type=c.source_type,
            source_id=c.source_id,
            document_title=c.document_title,
            document_url=c.document_url,
            text=c.text,
            token_count=c.token_count,
            permission_tags=c.permission_tags,
            created_at=c.created_at,
            source_modified_at=c.source_modified_at,
            embedding=[0.1, 0.2, 0.3],
        )
        for c in chunks
    ]


def _make_pipeline(
    connector=None,
    embed_fn=None,
    vector_store=None,
    graph_store=None,
    db_conn=None,
) -> IngestionPipeline:
    vs = vector_store or MagicMock()
    gs = graph_store or MagicMock()
    conn = connector or MagicMock()
    ef = embed_fn or _identity_embed
    return IngestionPipeline(
        vector_store=vs,
        graph_store=gs,
        connectors={"src-1": conn},
        embed_fn=ef,
        ast_parser=ASTParser(),
        db_conn=db_conn,
    )


# ---------------------------------------------------------------------------
# chunk() tests
# ---------------------------------------------------------------------------

class TestChunk:
    def test_token_count_within_size(self):
        """Every chunk's token_count must be in [1, size]."""
        doc = _make_doc(" ".join(str(i) for i in range(1000)))
        pipeline = _make_pipeline()
        chunks = pipeline.chunk(doc, size=100, overlap=20)
        assert chunks, "Expected at least one chunk"
        for c in chunks:
            assert 1 <= c.token_count <= 100, (
                f"token_count={c.token_count} out of [1, 100]"
            )

    def test_consecutive_chunks_share_overlap_tokens(self):
        """Consecutive chunks must share exactly `overlap` tokens."""
        words = [str(i) for i in range(200)]
        doc = _make_doc(" ".join(words))
        pipeline = _make_pipeline()
        size, overlap = 50, 10
        chunks = pipeline.chunk(doc, size=size, overlap=overlap)
        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"
        for i in range(len(chunks) - 1):
            tokens_a = chunks[i].text.split(" ")
            tokens_b = chunks[i + 1].text.split(" ")
            shared = tokens_a[-overlap:]
            leading = tokens_b[:overlap]
            assert shared == leading, (
                f"Chunk {i} and {i+1} do not share {overlap} tokens: "
                f"{shared!r} != {leading!r}"
            )

    def test_metadata_propagated_to_chunks(self):
        """All metadata from the Document must appear on each Chunk."""
        doc = _make_doc("hello world foo bar")
        pipeline = _make_pipeline()
        chunks = pipeline.chunk(doc)
        for c in chunks:
            assert c.source_type == doc.source_type
            assert c.source_id == doc.source_id
            assert c.document_title == doc.title
            assert c.document_url == doc.url
            assert c.permission_tags == doc.permission_tags
            assert c.source_modified_at == doc.modified_at

    def test_empty_content_returns_no_chunks(self):
        doc = _make_doc("")
        pipeline = _make_pipeline()
        assert pipeline.chunk(doc) == []

    def test_content_shorter_than_size_produces_one_chunk(self):
        doc = _make_doc("one two three")
        pipeline = _make_pipeline()
        chunks = pipeline.chunk(doc, size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0].token_count == 3

    def test_exact_size_boundary(self):
        """Content with exactly `size` tokens → one chunk."""
        words = ["w"] * 50
        doc = _make_doc(" ".join(words))
        pipeline = _make_pipeline()
        chunks = pipeline.chunk(doc, size=50, overlap=10)
        assert len(chunks) == 1
        assert chunks[0].token_count == 50


# ---------------------------------------------------------------------------
# run() retry tests
# ---------------------------------------------------------------------------

class TestRunRetry:
    def test_retries_3x_on_connection_failure_with_exponential_backoff(self):
        """run() retries up to 3 times with exponential backoff (1s, 2s, 4s)."""
        connector = MagicMock()
        connector.fetch_incremental.side_effect = ConnectionError("timeout")

        pipeline = _make_pipeline(connector=connector)

        sleep_calls: List[float] = []
        with patch("enterprise_rag.ingestion.pipeline.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            result = pipeline.run("src-1", incremental=True)

        # fetch_incremental called 3 times (attempts 0, 1, 2)
        assert connector.fetch_incremental.call_count == 3
        # Exponential backoff: sleep(1), sleep(2), sleep(4) — but only between retries
        # Attempt 0 → sleep(2^0=1), attempt 1 → sleep(2^1=2), attempt 2 → no sleep after last
        assert sleep_calls == [1, 2, 4]
        assert result.status == "failed"
        assert result.chunks_indexed == 0

    def test_retries_3x_on_fetch_all_failure(self):
        """run() retries fetch_all up to 3 times when incremental=False."""
        connector = MagicMock()
        connector.fetch_all.side_effect = OSError("network error")

        pipeline = _make_pipeline(connector=connector)

        sleep_calls: List[float] = []
        with patch("enterprise_rag.ingestion.pipeline.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            result = pipeline.run("src-1", incremental=False)

        assert connector.fetch_all.call_count == 3
        assert sleep_calls == [1, 2, 4]
        assert result.status == "failed"

    def test_succeeds_after_transient_failure(self):
        """run() succeeds if connection recovers before 3 attempts."""
        connector = MagicMock()
        doc = _make_doc("hello world")
        # Fail once, then succeed
        connector.fetch_incremental.side_effect = [ConnectionError("fail"), iter([doc])]

        pipeline = _make_pipeline(connector=connector)

        with patch("enterprise_rag.ingestion.pipeline.time.sleep"):
            result = pipeline.run("src-1", incremental=True)

        assert connector.fetch_incremental.call_count == 2
        assert result.status == "success"
        assert result.chunks_indexed == 1


# ---------------------------------------------------------------------------
# run() per-document error tests
# ---------------------------------------------------------------------------

class TestRunPerDocumentErrors:
    def test_continues_on_per_document_error(self):
        """Per-document errors are logged and remaining docs continue."""
        doc_ok = _make_doc("good content here")
        doc_bad = _make_doc("bad content")

        connector = MagicMock()
        connector.fetch_incremental.return_value = iter([doc_bad, doc_ok])

        call_count = 0

        def flaky_embed(chunks: List[Chunk]) -> List[EmbeddedChunk]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("embedding failed")
            return _identity_embed(chunks)

        pipeline = _make_pipeline(connector=connector, embed_fn=flaky_embed)
        result = pipeline.run("src-1", incremental=True)

        # Should have indexed the good doc despite the bad one failing
        assert result.chunks_indexed >= 1
        assert result.status == "success"

    def test_partial_ingestion_counts_only_successful_docs(self):
        """chunks_indexed reflects only successfully processed documents."""
        docs = [_make_doc(f"word{i} content here") for i in range(5)]
        connector = MagicMock()
        connector.fetch_incremental.return_value = iter(docs)

        embed_call = 0

        def selective_embed(chunks: List[Chunk]) -> List[EmbeddedChunk]:
            nonlocal embed_call
            embed_call += 1
            # Fail on the 3rd document
            if embed_call == 3:
                raise RuntimeError("embed error")
            return _identity_embed(chunks)

        pipeline = _make_pipeline(connector=connector, embed_fn=selective_embed)
        result = pipeline.run("src-1", incremental=True)

        # 4 out of 5 docs succeed → chunks_indexed == 4
        assert result.chunks_indexed == 4


# ---------------------------------------------------------------------------
# run() Neo4j unavailability tests
# ---------------------------------------------------------------------------

class TestRunNeo4jUnavailable:
    def test_continues_chromadb_when_neo4j_unavailable(self):
        """When Neo4j is unavailable, ChromaDB indexing continues."""
        from neo4j.exceptions import ServiceUnavailable

        # GitHub doc with a .py extension so AST parsing is triggered
        doc = Document(
            doc_id="doc-gh",
            source_type="github",
            source_id="src-1",
            title="main.py",
            url="https://github.com/org/repo/blob/HEAD/main.py",
            content="def hello():\n    pass\n",
            permission_tags=["engineering"],
            modified_at=datetime(2024, 1, 1),
        )

        connector = MagicMock()
        connector.fetch_incremental.return_value = iter([doc])

        graph_store = MagicMock()
        graph_store.upsert_symbols.side_effect = ServiceUnavailable("Neo4j down")

        vector_store = MagicMock()

        pipeline = IngestionPipeline(
            vector_store=vector_store,
            graph_store=graph_store,
            connectors={"src-1": connector},
            embed_fn=_identity_embed,
            ast_parser=ASTParser(),
        )

        result = pipeline.run("src-1", incremental=True)

        # ChromaDB upsert must have been called
        assert vector_store.upsert.called, "VectorStore.upsert should have been called"
        # chunks_indexed > 0 (text chunks were indexed)
        assert result.chunks_indexed > 0
        # symbols_indexed == 0 because Neo4j was unavailable
        assert result.symbols_indexed == 0

    def test_graph_indexing_failure_does_not_affect_status_as_failed(self):
        """Graph indexing failure results in partial status, not full failure."""
        from neo4j.exceptions import ServiceUnavailable

        doc = Document(
            doc_id="doc-gh",
            source_type="github",
            source_id="src-1",
            title="utils.py",
            url="https://github.com/org/repo/blob/HEAD/utils.py",
            content="def add(a, b):\n    return a + b\n",
            permission_tags=["engineering"],
            modified_at=datetime(2024, 1, 1),
        )

        connector = MagicMock()
        connector.fetch_incremental.return_value = iter([doc])

        graph_store = MagicMock()
        graph_store.upsert_symbols.side_effect = ServiceUnavailable("Neo4j down")

        pipeline = IngestionPipeline(
            vector_store=MagicMock(),
            graph_store=graph_store,
            connectors={"src-1": connector},
            embed_fn=_identity_embed,
            ast_parser=ASTParser(),
        )

        result = pipeline.run("src-1", incremental=True)
        # Status should be "partial" not "failed"
        assert result.status in ("partial", "success")


# ---------------------------------------------------------------------------
# run() job log tests
# ---------------------------------------------------------------------------

class TestRunJobLog:
    def test_writes_job_log_when_db_conn_provided(self):
        """JobResult is written to Postgres when db_conn is provided."""
        doc = _make_doc("some content here")
        connector = MagicMock()
        connector.fetch_incremental.return_value = iter([doc])

        mock_cursor = MagicMock()
        mock_db = MagicMock()
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)

        pipeline = _make_pipeline(connector=connector, db_conn=mock_db)
        result = pipeline.run("src-1", incremental=True)

        assert mock_cursor.execute.called
        assert mock_db.commit.called
        assert result.status == "success"

    def test_no_db_write_when_db_conn_is_none(self):
        """No DB write attempted when db_conn is None."""
        doc = _make_doc("some content here")
        connector = MagicMock()
        connector.fetch_incremental.return_value = iter([doc])

        pipeline = _make_pipeline(connector=connector, db_conn=None)
        # Should not raise
        result = pipeline.run("src-1", incremental=True)
        assert result.status == "success"
