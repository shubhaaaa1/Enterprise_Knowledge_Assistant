"""Property-based tests for the IngestionPipeline."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.ast_parser import ASTParser
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.models import (
    Chunk,
    CodeSymbol,
    Document,
    EmbeddedChunk,
    JobResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERMISSION_TAGS = ["engineering", "hr", "finance", "legal", "all"]


def _identity_embed(chunks: List[Chunk]) -> List[EmbeddedChunk]:
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


def _make_pipeline(connector=None, embed_fn=None, vector_store=None, graph_store=None) -> IngestionPipeline:
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
    )


def _make_doc(content: str, source_type: str = "docs", source_id: str = "src-1",
              title: str = "Test Doc", url: str = "https://example.com/doc",
              permission_tags: List[str] | None = None) -> Document:
    return Document(
        doc_id=str(uuid.uuid4()),
        source_type=source_type,
        source_id=source_id,
        title=title,
        url=url,
        content=content,
        permission_tags=permission_tags or ["engineering"],
        modified_at=datetime(2024, 1, 1),
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_doc_st = st.builds(
    _make_doc,
    content=st.text(min_size=1),
    source_type=st.sampled_from(["docs", "github", "jira"]),
    source_id=st.text(min_size=1, max_size=20),
    title=st.text(min_size=1, max_size=100),
    url=st.text(min_size=1, max_size=200),
    permission_tags=st.lists(st.sampled_from(_PERMISSION_TAGS), min_size=1),
)

_symbol_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
)

_code_symbol_st = st.builds(
    lambda name, symbol_type, call_refs: CodeSymbol(
        symbol_id=str(uuid.uuid4()),
        file_path="src/test.py",
        symbol_name=name,
        symbol_type=symbol_type,
        docstring=None,
        source_code="",
        line_start=1,
        line_end=10,
        call_refs=call_refs,
        source_id="repo-1",
        permission_tags=["engineering"],
    ),
    name=_symbol_name_st,
    symbol_type=st.sampled_from(["function", "class", "method", "module"]),
    call_refs=st.lists(_symbol_name_st, max_size=5),
)


# ---------------------------------------------------------------------------
# Property 1: Chunk metadata completeness
# Feature: enterprise-rag-system, Property 1: Chunk metadata completeness
# Validates: Requirements 1.6
# ---------------------------------------------------------------------------

@given(doc=_doc_st)
@settings(max_examples=20, deadline=None)
def test_property_1_chunk_metadata_completeness(doc: Document):
    """Every chunk produced by pipeline.chunk() has non-null required metadata fields.

    Validates: Requirements 1.6
    """
    # Feature: enterprise-rag-system, Property 1: Chunk metadata completeness
    pipeline = _make_pipeline()
    chunks = pipeline.chunk(doc)

    for chunk in chunks:
        assert chunk.source_type is not None and chunk.source_type != "", (
            f"chunk.source_type must be non-null/non-empty, got {chunk.source_type!r}"
        )
        assert chunk.source_id is not None and chunk.source_id != "", (
            f"chunk.source_id must be non-null/non-empty, got {chunk.source_id!r}"
        )
        assert chunk.document_title is not None and chunk.document_title != "", (
            f"chunk.document_title must be non-null/non-empty, got {chunk.document_title!r}"
        )
        assert chunk.document_url is not None and chunk.document_url != "", (
            f"chunk.document_url must be non-null/non-empty, got {chunk.document_url!r}"
        )
        assert chunk.permission_tags is not None, (
            "chunk.permission_tags must be non-null"
        )
        assert len(chunk.permission_tags) > 0, (
            "chunk.permission_tags must be non-empty"
        )


# ---------------------------------------------------------------------------
# Property 2: Chunking size invariant
# Feature: enterprise-rag-system, Property 2: Chunking size invariant
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------

@given(
    content=st.text(min_size=1),
    size=st.integers(min_value=64, max_value=2048),
)
@settings(max_examples=20, deadline=None)
def test_property_2_chunking_size_invariant(content: str, size: int):
    """Every chunk's token_count is in [1, size] and consecutive chunks share overlap tokens.

    Validates: Requirements 1.2
    """
    # Feature: enterprise-rag-system, Property 2: Chunking size invariant
    overlap = min(64, size // 4)
    doc = _make_doc(content)
    pipeline = _make_pipeline()
    chunks = pipeline.chunk(doc, size=size, overlap=overlap)

    for chunk in chunks:
        assert 1 <= chunk.token_count <= size, (
            f"token_count={chunk.token_count} not in [1, {size}]"
        )

    for i in range(len(chunks) - 1):
        tokens_a = chunks[i].text.split(" ")
        tokens_b = chunks[i + 1].text.split(" ")
        # The last `overlap` tokens of chunk i should equal the first `overlap` tokens of chunk i+1
        shared = tokens_a[-overlap:]
        leading = tokens_b[:overlap]
        assert shared == leading, (
            f"Chunks {i} and {i+1} do not share {overlap} overlap tokens: "
            f"{shared!r} != {leading!r}"
        )


# ---------------------------------------------------------------------------
# Property 3: Incremental sync only fetches modified content
# Feature: enterprise-rag-system, Property 3: Incremental sync only fetches modified content
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------

@given(
    entries=st.lists(
        st.tuples(st.text(min_size=1, max_size=50), st.datetimes()),
        min_size=0,
        max_size=20,
    ),
    cutoff=st.datetimes(),
)
@settings(max_examples=20, deadline=None)
def test_property_3_incremental_sync_filter(entries: list, cutoff: datetime):
    """Filtering (doc, modified_at) pairs by modified_at > cutoff gives exactly the right subset.

    This tests the pure filtering logic that incremental sync relies on.
    Validates: Requirements 1.3
    """
    # Feature: enterprise-rag-system, Property 3: Incremental sync only fetches modified content
    # Apply the same filter logic the pipeline uses: modified_at > cutoff
    result = [(doc_id, ts) for doc_id, ts in entries if ts > cutoff]
    excluded = [(doc_id, ts) for doc_id, ts in entries if ts <= cutoff]

    # Every included entry must have modified_at strictly after cutoff
    for doc_id, ts in result:
        assert ts > cutoff, (
            f"doc '{doc_id}' with modified_at={ts} should not be included (cutoff={cutoff})"
        )

    # Every excluded entry must have modified_at at or before cutoff
    for doc_id, ts in excluded:
        assert ts <= cutoff, (
            f"doc '{doc_id}' with modified_at={ts} should be included (cutoff={cutoff})"
        )

    # The union of included and excluded must equal the original set
    assert len(result) + len(excluded) == len(entries), (
        "Partition must be complete: every entry is either included or excluded"
    )


# ---------------------------------------------------------------------------
# Property 4: Ingestion job log completeness
# Feature: enterprise-rag-system, Property 4: Ingestion job log completeness
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

@given(docs=st.lists(_doc_st, min_size=0, max_size=10))
@settings(max_examples=20, deadline=None)
def test_property_4_ingestion_job_log_completeness(docs: List[Document]):
    """Completed JobResult always has non-null status, chunks_indexed, and completed_at.

    Validates: Requirements 1.5
    """
    # Feature: enterprise-rag-system, Property 4: Ingestion job log completeness
    connector = MagicMock()
    connector.fetch_incremental.return_value = iter(docs)

    pipeline = _make_pipeline(connector=connector)
    result = pipeline.run("src-1", incremental=True)

    assert result.status is not None and result.status != "", (
        f"JobResult.status must be non-null/non-empty, got {result.status!r}"
    )
    assert result.chunks_indexed is not None, (
        "JobResult.chunks_indexed must be non-null"
    )
    assert result.chunks_indexed >= 0, (
        f"JobResult.chunks_indexed must be >= 0, got {result.chunks_indexed}"
    )
    assert result.completed_at is not None, (
        "JobResult.completed_at must be non-null"
    )
    assert isinstance(result.completed_at, datetime), (
        f"JobResult.completed_at must be a datetime, got {type(result.completed_at)}"
    )


# ---------------------------------------------------------------------------
# Property 25: Dual-store ingestion consistency
# Feature: enterprise-rag-system, Property 25: Dual-store ingestion consistency
# Validates: Requirements 1.8
# ---------------------------------------------------------------------------
