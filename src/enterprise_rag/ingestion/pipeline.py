"""Ingestion pipeline for the Enterprise RAG System."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, Iterator, List, Optional

from enterprise_rag.ast_parser import ASTParser, _ext, _NON_PARSEABLE_EXTENSIONS
from enterprise_rag.ingestion.base import SourceConnector
from enterprise_rag.models import (
    Chunk,
    CodeSymbol,
    Document,
    EmbeddedChunk,
    JobResult,
)
from enterprise_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Extensions that the AST parser can handle (parseable source code)
_PARSEABLE_EXTENSIONS = {
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx",
    ".java", ".go", ".rs", ".cpp", ".cc", ".cxx",
    ".hpp", ".h", ".c", ".rb", ".php",
}

# Language mapping by extension
_EXT_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".h": "c",
    ".c": "c",
    ".rb": "ruby",
    ".php": "php",
}


class IngestionPipeline:
    """Orchestrates fetch → chunk → embed → index → index_graph."""

    def __init__(
        self,
        vector_store: VectorStore,
        connectors: Dict[str, SourceConnector],
        embed_fn: Callable[[List[Chunk]], List[EmbeddedChunk]],
        ast_parser: ASTParser,
        graph_store=None,
        db_conn=None,
    ) -> None:
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._connectors = connectors
        self._embed_fn = embed_fn
        self._ast_parser = ast_parser
        self._db_conn = db_conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: Document, size: int = 512, overlap: int = 64) -> List[Chunk]:
        """Token-based chunking with overlap using whitespace tokenization."""
        tokens = doc.content.split(" ")
        if not tokens or (len(tokens) == 1 and tokens[0] == ""):
            return []

        chunks: List[Chunk] = []
        start = 0
        total = len(tokens)

        while start < total:
            end = min(start + size, total)
            window = tokens[start:end]
            text = " ".join(window)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    source_type=doc.source_type,
                    source_id=doc.source_id,
                    document_title=doc.title,
                    document_url=doc.url,
                    text=text,
                    token_count=len(window),
                    permission_tags=list(doc.permission_tags),
                    created_at=datetime.now(timezone.utc),
                    source_modified_at=doc.modified_at,
                )
            )
            if end == total:
                break
            start = end - overlap

        return chunks

    def embed(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """Calls the injected embedding function."""
        return self._embed_fn(chunks)

    def index(self, chunks: List[EmbeddedChunk]) -> None:
        """Writes embedded chunks to the VectorStore."""
        self._vector_store.upsert(chunks)

    def run(self, source_id: str, incremental: bool = True) -> JobResult:
        """Orchestrate fetch → chunk → embed → index."""
        job_id = str(uuid.uuid4())
        chunks_indexed = 0
        symbols_indexed = 0
        error_message: Optional[str] = None

        connector = self._connectors.get(source_id)
        if connector is None:
            error_message = f"No connector registered for source_id='{source_id}'"
            logger.error(error_message)
            result = JobResult(
                job_id=job_id,
                source_id=source_id,
                status="failed",
                chunks_indexed=0,
                symbols_indexed=0,
                completed_at=datetime.now(timezone.utc),
                error_message=error_message,
            )
            self._write_job_log(result)
            return result

        # Fetch documents with retry + exponential backoff
        docs: List[Document] = []
        fetch_error: Optional[Exception] = None
        for attempt in range(3):
            try:
                if incremental:
                    # Use epoch as default "since" when no prior sync timestamp
                    since = datetime(1970, 1, 1)
                    doc_iter: Iterator[Document] = connector.fetch_incremental(since)
                else:
                    doc_iter = connector.fetch_all()
                docs = list(doc_iter)
                fetch_error = None
                break
            except Exception as exc:
                fetch_error = exc
                logger.warning(
                    "Source connection failed for '%s' (attempt %d/3): %s",
                    source_id,
                    attempt + 1,
                    exc,
                )
                time.sleep(2 ** attempt)

        if fetch_error is not None:
            error_message = (
                f"Source connection failed after 3 attempts: {fetch_error}"
            )
            logger.error(error_message)
            result = JobResult(
                job_id=job_id,
                source_id=source_id,
                status="failed",
                chunks_indexed=0,
                symbols_indexed=0,
                completed_at=datetime.now(timezone.utc),
                error_message=error_message,
            )
            self._write_job_log(result)
            return result

        # Process each document — per-document errors are logged and skipped
        for doc in docs:
            try:
                doc_chunks = self.chunk(doc)
                embedded = self.embed(doc_chunks)
                self.index(embedded)
                chunks_indexed += len(embedded)
            except Exception as doc_exc:
                logger.error(
                    "Error processing document '%s' from source '%s': %s",
                    getattr(doc, "doc_id", "unknown"),
                    source_id,
                    doc_exc,
                )
                continue

        status = "success"
        if fetch_error is not None:
            status = "failed"

        result = JobResult(
            job_id=job_id,
            source_id=source_id,
            status=status,
            chunks_indexed=chunks_indexed,
            symbols_indexed=symbols_indexed,
            completed_at=datetime.now(timezone.utc),
            error_message=error_message,
        )
        self._write_job_log(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_job_log(self, result: JobResult) -> None:
        """Insert a JobResult into the Postgres job_log table if db_conn is set."""
        if self._db_conn is None:
            return
        try:
            with self._db_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO job_log
                        (job_id, source_id, status, chunks_indexed,
                         symbols_indexed, completed_at, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result.job_id,
                        result.source_id,
                        result.status,
                        result.chunks_indexed,
                        result.symbols_indexed,
                        result.completed_at,
                        result.error_message,
                    ),
                )
            self._db_conn.commit()
        except Exception as exc:
            logger.error("Failed to write job log: %s", exc)
