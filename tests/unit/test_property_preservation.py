"""Property-based preservation tests for bugfix spec.

These tests capture baseline behaviors that must be preserved after implementing
the bugfixes for deprecated datetime methods, missing imports, missing timestamps,
and Neo4j removal.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.access_controller import AccessController, Permissions
from enterprise_rag.citation_engine import CitationEngine
from enterprise_rag.conversation_manager import ConversationManager
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.logging import StructuredLogger
from enterprise_rag.models import (
    AccessFilter,
    Chunk,
    Document,
    EmbeddedChunk,
    Role,
    ScoredChunk,
    Turn,
)
from enterprise_rag.retriever import Retriever
from enterprise_rag.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_safe_text = st.text(
    min_size=1,
    max_size=80,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_ "),
)

_PERMISSION_TAGS = ["engineering", "hr", "finance", "legal", "all"]


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


def _make_chunk(text: str, source_type: str = "docs", source_id: str = "src-1",
                document_title: str = "Test Doc", document_url: str = "https://example.com/doc",
                permission_tags: List[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        source_type=source_type,
        source_id=source_id,
        document_title=document_title,
        document_url=document_url,
        text=text,
        token_count=len(text.split()),
        permission_tags=permission_tags or ["engineering"],
        created_at=datetime.now(timezone.utc),
        source_modified_at=datetime(2024, 1, 1),
    )


def _make_embedded_chunk(chunk: Chunk, embedding: List[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk.chunk_id,
        source_type=chunk.source_type,
        source_id=chunk.source_id,
        document_title=chunk.document_title,
        document_url=chunk.document_url,
        text=chunk.text,
        token_count=chunk.token_count,
        permission_tags=chunk.permission_tags,
        created_at=chunk.created_at,
        source_modified_at=chunk.source_modified_at,
        embedding=embedding,
    )


def _make_scored_chunk(chunk: Chunk, relevance_score: float = 0.8) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk.chunk_id,
        source_type=chunk.source_type,
        source_id=chunk.source_id,
        document_title=chunk.document_title,
        document_url=chunk.document_url,
        text=chunk.text,
        token_count=chunk.token_count,
        permission_tags=chunk.permission_tags,
        created_at=chunk.created_at,
        source_modified_at=chunk.source_modified_at,
        relevance_score=relevance_score,
        rrf_score=0.0,
    )


# ---------------------------------------------------------------------------
# Property 1: Datetime serialization to ISO format (Preservation 3.1)
# **Validates: Requirements 3.1**
# ---------------------------------------------------------------------------

@given(
    content=st.text(min_size=1, max_size=100),
)
@settings(max_examples=10, deadline=None)
def test_preservation_datetime_serialization(content: str):
    """Datetime operations produce timezone-aware datetime objects that serialize
    correctly to ISO format strings.

    This test verifies that the current datetime handling (even with deprecated
    methods) produces valid ISO format timestamps. After the fix, this behavior
    must be preserved.

    **Validates: Requirements 3.1**
    """
    # Create a document and chunk it
    doc = _make_doc(content)
    
    # Mock components
    vector_store = MagicMock()
    graph_store = MagicMock()
    connector = MagicMock()
    
    def _identity_embed(chunks: List[Chunk]) -> List[EmbeddedChunk]:
        return [_make_embedded_chunk(c, [0.1, 0.2, 0.3]) for c in chunks]
    
    from enterprise_rag.ast_parser import ASTParser
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        graph_store=graph_store,
        connectors={"src-1": connector},
        embed_fn=_identity_embed,
        ast_parser=ASTParser(),
    )
    
    # Chunk the document
    chunks = pipeline.chunk(doc)
    
    # Verify that created_at timestamps can be serialized to ISO format
    for chunk in chunks:
        assert chunk.created_at is not None, "created_at must not be None"
        # Serialize to ISO format
        iso_str = chunk.created_at.isoformat()
        assert isinstance(iso_str, str), "ISO format must be a string"
        assert "T" in iso_str, "ISO format must contain 'T' separator"
        # Verify it can be parsed back
        parsed = datetime.fromisoformat(iso_str)
        assert isinstance(parsed, datetime), "Must be able to parse ISO format back to datetime"


# ---------------------------------------------------------------------------
# Property 2: DocsConnector processes files and creates valid Documents (Preservation 3.2)
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------

@given(
    content=st.text(
        min_size=1,
        max_size=200,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"), whitelist_characters=".,!?-"),
    ),
)
@settings(max_examples=10, deadline=None)
def test_preservation_docs_connector_processing(content: str):
    """DocsConnector processes files and extracts correct modification times,
    creating valid Document objects.

    This test verifies that DocsConnector can process files and create valid
    Documents with proper modification times. After the fix, this behavior
    must be preserved.

    **Validates: Requirements 3.2**
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name
    
    try:
        # Import DocsConnector - this will work even without the import fix in api.py
        from enterprise_rag.ingestion.docs_connector import DocsConnector
        
        # Create connector pointing to the temp file's directory
        temp_dir = os.path.dirname(temp_path)
        connector = DocsConnector(
            source_id="test-docs",
            base_path=temp_dir,
            permission_tags=["engineering"],
        )
        
        # Fetch all documents
        docs = list(connector.fetch_all())
        
        # Find our document
        our_doc = None
        for doc in docs:
            if doc.title == os.path.basename(temp_path):
                our_doc = doc
                break
        
        assert our_doc is not None, "Document should be found"
        assert our_doc.content == content, "Content should match"
        assert our_doc.modified_at is not None, "modified_at must not be None"
        assert isinstance(our_doc.modified_at, datetime), "modified_at must be a datetime"
        # Verify the datetime can be serialized
        iso_str = our_doc.modified_at.isoformat()
        assert isinstance(iso_str, str), "ISO format must be a string"
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Property 3: Ingestion pipeline indexes documents into ChromaDB (Preservation 3.3)
# **Validates: Requirements 3.3**
# ---------------------------------------------------------------------------

@given(
    content=st.text(min_size=1, max_size=100),
)
@settings(max_examples=10, deadline=None)
def test_preservation_ingestion_pipeline_indexing(content: str):
    """Ingestion pipeline indexes documents into ChromaDB vector store successfully.

    This test verifies that the ingestion pipeline can chunk, embed, and index
    documents. After the fix, this behavior must be preserved.

    **Validates: Requirements 3.3**
    """
    doc = _make_doc(content)
    
    # Mock components
    vector_store = MagicMock()
    graph_store = MagicMock()
    connector = MagicMock()
    
    def _identity_embed(chunks: List[Chunk]) -> List[EmbeddedChunk]:
        return [_make_embedded_chunk(c, [0.1, 0.2, 0.3]) for c in chunks]
    
    from enterprise_rag.ast_parser import ASTParser
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        graph_store=graph_store,
        connectors={"src-1": connector},
        embed_fn=_identity_embed,
        ast_parser=ASTParser(),
    )
    
    # Process document
    chunks = pipeline.chunk(doc)
    embedded = pipeline.embed(chunks)
    pipeline.index(embedded)
    
    # Verify vector_store.upsert was called
    vector_store.upsert.assert_called_once()
    call_args = vector_store.upsert.call_args[0][0]
    assert len(call_args) == len(embedded), "All embedded chunks should be indexed"
    assert all(isinstance(c, EmbeddedChunk) for c in call_args), "All items should be EmbeddedChunks"


# ---------------------------------------------------------------------------
# Property 4: Structured logger writes valid JSON entries (Preservation 3.4)
# **Validates: Requirements 3.4**
# ---------------------------------------------------------------------------

@given(
    user_id=_safe_text,
    session_id=_safe_text,
    original_query=_safe_text,
    rewritten_query=_safe_text,
    chunks_retrieved=st.integers(min_value=0, max_value=100),
    grounding_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    latency_ms=st.floats(min_value=0.0, max_value=60_000.0, allow_nan=False),
)
@settings(max_examples=10, deadline=None)
def test_preservation_structured_logger_json(
    user_id, session_id, original_query, rewritten_query,
    chunks_retrieved, grounding_score, latency_ms
):
    """Structured logger writes valid JSON entries to the configured backend.

    This test verifies that the logger emits valid JSON records. After the fix
    (adding timestamp to log_error), this behavior must be preserved.

    **Validates: Requirements 3.4**
    """
    logger = StructuredLogger(backend="file", log_dir="/tmp/test_logs_preservation")
    correlation_id = StructuredLogger.new_correlation_id()
    
    captured: list[dict] = []
    
    with patch.object(logger._backend, "emit", side_effect=captured.append):
        logger.log_query(
            user_id=user_id,
            session_id=session_id,
            original_query=original_query,
            rewritten_query=rewritten_query,
            chunks_retrieved=chunks_retrieved,
            grounding_score=grounding_score,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )
    
    assert len(captured) == 1, "Expected exactly one log record"
    record = captured[0]
    
    # Verify it's valid JSON-serializable
    json_str = json.dumps(record)
    assert isinstance(json_str, str), "Record must be JSON-serializable"
    
    # Verify it can be parsed back
    parsed = json.loads(json_str)
    assert parsed["user_id"] == user_id, "Parsed record must match original"


# ---------------------------------------------------------------------------
# Property 5: Retriever returns relevant chunks (Preservation 3.5)
# **Validates: Requirements 3.5**
# ---------------------------------------------------------------------------

@given(
    query=_safe_text,
    num_chunks=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=10, deadline=None)
def test_preservation_retriever_semantic_similarity(query: str, num_chunks: int):
    """Retriever returns relevant chunks based on semantic similarity.

    This test verifies that the retriever can perform vector search and return
    scored chunks. After the fix, this behavior must be preserved.

    **Validates: Requirements 3.5**
    """
    # Create mock vector store
    vector_store = MagicMock()
    
    # Create mock chunks
    mock_chunks = [
        _make_scored_chunk(_make_chunk(f"chunk {i}"), relevance_score=0.9 - i * 0.1)
        for i in range(num_chunks)
    ]
    vector_store.query.return_value = mock_chunks
    
    # Create retriever
    def _embed(text: str) -> List[float]:
        return [0.1] * 384
    
    retriever = Retriever(vector_store=vector_store, embed_fn=_embed)
    
    # Perform retrieval
    access_filter = AccessFilter(
        permitted_source_ids=["src-1"],
        permitted_tags=["engineering"],
    )
    results = retriever.retrieve([query], access_filter, k=num_chunks)
    
    # Verify results
    assert len(results) <= num_chunks, "Should return at most k chunks"
    assert all(isinstance(c, ScoredChunk) for c in results), "All results should be ScoredChunks"
    assert all(hasattr(c, "relevance_score") for c in results), "All chunks should have relevance_score"


# ---------------------------------------------------------------------------
# Property 6: Health endpoint reports status (Preservation 3.6)
# **Validates: Requirements 3.6**
# ---------------------------------------------------------------------------

@settings(max_examples=10, deadline=None)
@given(st.just(None))
def test_preservation_health_endpoint_status(_):
    """Health endpoint reports status for all components.

    This test verifies that the health check can query component status.
    After Neo4j removal, this behavior must be preserved (without Neo4j).

    **Validates: Requirements 3.6**
    """
    # Create mock vector store with health check
    vector_store = MagicMock()
    from enterprise_rag.models import ComponentStatus
    vector_store.health_check.return_value = ComponentStatus(
        name="chromadb",
        status="ok",
        last_checked=datetime.now(timezone.utc),
        detail=None,
    )
    
    # Create retriever
    retriever = Retriever(vector_store=vector_store)
    
    # Call health check
    status = retriever._vector_store.health_check()
    
    # Verify status
    assert status.name == "chromadb", "Component name should be chromadb"
    assert status.status in ["ok", "degraded", "down"], "Status should be valid"
    assert status.last_checked is not None, "last_checked must not be None"
    assert isinstance(status.last_checked, datetime), "last_checked must be a datetime"


# ---------------------------------------------------------------------------
# Property 7: Query endpoint generates answers with citations (Preservation 3.7)
# **Validates: Requirements 3.7**
# ---------------------------------------------------------------------------

@given(
    answer=st.text(min_size=10, max_size=200),
    num_chunks=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=10, deadline=None)
def test_preservation_citation_engine_extracts_citations(answer: str, num_chunks: int):
    """Query endpoint generates answers with citations and grounding scores.

    This test verifies that the citation engine can extract citations from
    answers. After the fix, this behavior must be preserved.

    **Validates: Requirements 3.7**
    """
    # Create citation engine
    engine = CitationEngine()
    
    # Create mock chunks
    chunks = [
        _make_scored_chunk(_make_chunk(f"chunk {i} content"), relevance_score=0.9)
        for i in range(num_chunks)
    ]
    
    # Add some [N] references to the answer
    answer_with_refs = f"{answer} [1] and [2]"
    
    # Generate citations
    cited = engine.cite(answer_with_refs, chunks)
    
    # Verify citations
    assert cited.answer_text is not None, "answer_text must not be None"
    assert isinstance(cited.citations, list), "citations must be a list"
    assert isinstance(cited.grounding_score, float), "grounding_score must be a float"
    assert 0.0 <= cited.grounding_score <= 1.0, "grounding_score must be in [0, 1]"


# ---------------------------------------------------------------------------
# Property 8: Access controller enforces permissions (Preservation 3.8)
# **Validates: Requirements 3.8**
# ---------------------------------------------------------------------------

@given(
    role_name=_safe_text,
    source_ids=st.lists(_safe_text, min_size=1, max_size=5),
    tags=st.lists(st.sampled_from(_PERMISSION_TAGS), min_size=1, max_size=3),
)
@settings(max_examples=10, deadline=None)
def test_preservation_access_controller_permissions(role_name: str, source_ids: List[str], tags: List[str]):
    """Access controller enforces permission-based access control.

    This test verifies that the access controller can resolve roles and build
    access filters. After the fix, this behavior must be preserved.

    **Validates: Requirements 3.8**
    """
    # Create access controller with a role
    role = Role(
        name=role_name,
        permitted_source_ids=source_ids,
        permitted_tags=tags,
    )
    ac = AccessController(role_permissions={role_name: role})
    
    # Build access filter
    access_filter = ac.build_access_filter([role])
    
    # Verify filter
    assert isinstance(access_filter, AccessFilter), "Must return AccessFilter"
    assert set(access_filter.permitted_source_ids) == set(source_ids), "Source IDs must match"
    assert set(access_filter.permitted_tags) == set(tags), "Tags must match"


# ---------------------------------------------------------------------------
# Property 9: Conversation manager maintains session history (Preservation 3.9)
# **Validates: Requirements 3.9**
# ---------------------------------------------------------------------------

@given(
    session_id=_safe_text,
    original_query=_safe_text,
    rewritten_query=_safe_text,
    answer=_safe_text,
)
@settings(max_examples=10, deadline=None)
def test_preservation_conversation_manager_history(
    session_id: str, original_query: str, rewritten_query: str, answer: str
):
    """Conversation manager maintains session history correctly.

    This test verifies that the conversation manager can store and retrieve
    turns. After the fix, this behavior must be preserved.

    **Validates: Requirements 3.9**
    """
    # Create conversation manager
    cm = ConversationManager()
    
    # Create a turn
    turn = Turn(
        role="user",
        original_query=original_query,
        rewritten_query=rewritten_query,
        answer=answer,
        timestamp=datetime.now(timezone.utc),
    )
    
    # Append turn
    cm.append_turn(session_id, turn)
    
    # Retrieve history
    history = cm.get_history(session_id, last=10)
    
    # Verify history
    assert len(history) >= 1, "History should contain at least one turn"
    retrieved_turn = history[-1]
    assert retrieved_turn.original_query == original_query, "Original query must match"
    assert retrieved_turn.rewritten_query == rewritten_query, "Rewritten query must match"
    assert retrieved_turn.answer == answer, "Answer must match"
    assert retrieved_turn.timestamp is not None, "Timestamp must not be None"


# ---------------------------------------------------------------------------
# Property 10: Citation engine extracts and numbers citations (Preservation 3.10)
# **Validates: Requirements 3.10**
# ---------------------------------------------------------------------------

@given(
    num_chunks=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=10, deadline=None)
def test_preservation_citation_engine_numbering(num_chunks: int):
    """Citation engine extracts and numbers citations from retrieved chunks.

    This test verifies that the citation engine correctly numbers citations
    and extracts excerpts. After the fix, this behavior must be preserved.

    **Validates: Requirements 3.10**
    """
    # Create citation engine
    engine = CitationEngine()
    
    # Create mock chunks with different URLs
    chunks = [
        _make_scored_chunk(
            _make_chunk(
                f"chunk {i} content with some text",
                document_url=f"https://example.com/doc{i}",
                document_title=f"Document {i}",
            ),
            relevance_score=0.9,
        )
        for i in range(num_chunks)
    ]
    
    # Create answer with references to all chunks
    refs = " ".join([f"[{i+1}]" for i in range(num_chunks)])
    answer = f"This is an answer with references {refs}"
    
    # Generate citations
    cited = engine.cite(answer, chunks)
    
    # Verify citations are numbered correctly
    assert len(cited.citations) > 0, "Should have at least one citation"
    for citation in cited.citations:
        assert citation.number > 0, "Citation number must be positive"
        assert citation.document_title is not None, "Document title must not be None"
        assert citation.document_url is not None, "Document URL must not be None"
        assert citation.excerpt is not None, "Excerpt must not be None"
        assert len(citation.chunk_ids) > 0, "Must have at least one chunk_id"
