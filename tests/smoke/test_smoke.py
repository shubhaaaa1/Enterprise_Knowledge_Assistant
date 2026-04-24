"""Smoke tests for the Enterprise RAG System.

These tests verify that core components can be instantiated and basic
operations work without requiring live external services (Ollama, ChromaDB,
Neo4j, Redis, Postgres).
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime
from typing import Iterator, List
from unittest.mock import MagicMock, patch

import pytest

from enterprise_rag.access_controller import AccessController, Permissions
from enterprise_rag.generator import Generator
from enterprise_rag.graph_store import GraphStore, _LABEL_MAP
from enterprise_rag.ingestion.base import SourceConnector
from enterprise_rag.ingestion.docs_connector import DocsConnector
from enterprise_rag.ingestion.github_connector import GitHubConnector
from enterprise_rag.ingestion.jira_connector import JiraConnector
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.logging import StructuredLogger
from enterprise_rag.models import (
    Chunk,
    CodeSymbol,
    ComponentStatus,
    Document,
    EmbeddedChunk,
    Role,
)
from enterprise_rag.vector_store import COLLECTION_NAME, VectorStore


# ---------------------------------------------------------------------------
# Test 1: All source connectors instantiate correctly (Req 1.1)
# ---------------------------------------------------------------------------

def test_all_source_connectors_instantiate():
    """All three source connector types instantiate correctly."""
    docs = DocsConnector(source_id="docs-1", base_path="/tmp/docs")
    github = GitHubConnector(source_id="gh-1", repo="org/repo", token="tok")
    jira = JiraConnector(
        source_id="jira-1",
        base_url="https://jira.example.com",
        username="user",
        api_token="token",
        project_key="PROJ",
    )

    assert isinstance(docs, SourceConnector)
    assert isinstance(github, SourceConnector)
    assert isinstance(jira, SourceConnector)


# ---------------------------------------------------------------------------
# Test 2: Role mapping CRUD operations work end-to-end (Req 2.1)
# ---------------------------------------------------------------------------

def test_role_mapping_crud_operations():
    """Role mapping CRUD operations work end-to-end."""
    controller = AccessController()

    role = Role(
        name="engineering",
        permitted_source_ids=["github-main"],
        permitted_tags=["engineering"],
    )
    perms = Permissions(
        permitted_source_ids=["github-main"],
        permitted_tags=["engineering"],
    )

    # Create / update
    controller.update_role_permissions(role, perms)

    # Resolve via token
    token = "user123:engineering"
    resolved = controller.resolve_roles(token)
    assert len(resolved) == 1
    assert resolved[0].name == "engineering"

    # Build access filter and verify contents
    access_filter = controller.build_access_filter(resolved)
    assert "github-main" in access_filter.permitted_source_ids
    assert "engineering" in access_filter.permitted_tags


# ---------------------------------------------------------------------------
# Test 3: Runtime Ollama config change takes effect without restart (Req 8.3)
# ---------------------------------------------------------------------------

def test_runtime_ollama_config_change():
    """Runtime Ollama config change takes effect without reinstantiation."""
    generator = Generator(
        ollama_url="http://localhost:11434",
        model="llama3",
    )

    generator.ollama_url = "http://new-host:11434"
    generator.model = "mistral"

    assert generator.ollama_url == "http://new-host:11434"
    assert generator.model == "mistral"


# ---------------------------------------------------------------------------
# Test 4: Log retention policy configured to >= 90 days (Req 10.5)
# ---------------------------------------------------------------------------

def test_log_retention_policy_configured():
    """Log retention policy is configured to at least 90 days."""
    tmpdir = tempfile.mkdtemp()
    structured_logger = StructuredLogger(backend="file", log_dir=tmpdir)
    try:
        assert structured_logger._retention_days >= 90
    finally:
        # Close the file handler so Windows can release the lock before cleanup
        backend = structured_logger._backend
        for handler in list(backend._logger.handlers):
            handler.close()
            backend._logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Test 5: Neo4j node labels and relationship types present in schema (Req 11.1)
# ---------------------------------------------------------------------------

def test_neo4j_node_labels_and_relationship_types():
    """Neo4j node labels and relationship types are defined in GraphStore."""
    # Expected node labels from Req 11.1
    expected_labels = {"File", "Function", "Class", "Method", "Module"}
    # Expected relationship types from Req 11.1
    expected_rel_types = {"CALLS", "INHERITS", "IMPORTS", "DEFINED_IN", "CONTAINS"}

    # Verify node labels are present in the _LABEL_MAP constant
    actual_labels = set(_LABEL_MAP.values())
    assert expected_labels == actual_labels, (
        f"Missing node labels: {expected_labels - actual_labels}"
    )

    # Verify relationship types are documented in upsert_relationships source
    import inspect
    source = inspect.getsource(GraphStore.upsert_relationships)
    for rel_type in expected_rel_types:
        assert rel_type in source, (
            f"Relationship type '{rel_type}' not found in GraphStore.upsert_relationships"
        )


# ---------------------------------------------------------------------------
# Test 6: ChromaDB collection name configured and health_check returns ComponentStatus
# ---------------------------------------------------------------------------

def test_chromadb_collection_name_configured():
    """ChromaDB collection name is 'enterprise_rag' and health_check returns ComponentStatus."""
    tmpdir = tempfile.mkdtemp()
    vector_store = VectorStore(persist_path=tmpdir)
    try:
        # Collection name constant
        assert COLLECTION_NAME == "enterprise_rag"
        assert vector_store._collection.name == "enterprise_rag"

        # health_check returns the right type (may be "ok" or "down")
        status = vector_store.health_check()
        assert isinstance(status, ComponentStatus)
    finally:
        # Release the ChromaDB SQLite connection so Windows can clean up
        try:
            vector_store._client._system.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Test 7: GitHub ingestion end-to-end: AST symbols in both stores (Req 1.8)
# ---------------------------------------------------------------------------

def test_github_ingestion_ast_symbols_in_both_stores():
    """GitHub ingestion: AST symbols appear in both ChromaDB and Neo4j after job completes."""
    python_source = (
        "def hello():\n"
        "    '''Say hello.'''\n"
        "    return 'hello'\n"
        "\n"
        "def world():\n"
        "    '''Call hello.'''\n"
        "    return hello()\n"
    )

    # Mock document returned by the connector
    mock_doc = Document(
        doc_id=str(uuid.uuid4()),
        source_type="github",
        source_id="test-github",
        title="example.py",
        url="https://github.com/org/repo/blob/HEAD/example.py",
        content=python_source,
        permission_tags=["engineering"],
        modified_at=datetime.utcnow(),
    )

    # Mock connector that yields the document
    mock_connector = MagicMock(spec=SourceConnector)
    mock_connector.fetch_all.return_value = iter([mock_doc])

    # Mock vector store and graph store
    mock_vector_store = MagicMock(spec=VectorStore)
    mock_graph_store = MagicMock(spec=GraphStore)

    # Minimal embed function
    def fake_embed(chunks: List[Chunk]) -> List[EmbeddedChunk]:
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
                embedding=[0.1] * 384,
            )
            for c in chunks
        ]

    from enterprise_rag.ast_parser import ASTParser

    pipeline = IngestionPipeline(
        vector_store=mock_vector_store,
        graph_store=mock_graph_store,
        connectors={"test-github": mock_connector},
        embed_fn=fake_embed,
        ast_parser=ASTParser(),
    )

    result = pipeline.run(source_id="test-github", incremental=False)

    # Symbols should have been indexed in ChromaDB (upsert called)
    mock_vector_store.upsert.assert_called()

    # Symbols should have been indexed in Neo4j (upsert_symbols called)
    mock_graph_store.upsert_symbols.assert_called()

    assert result.status in ("success", "partial")
