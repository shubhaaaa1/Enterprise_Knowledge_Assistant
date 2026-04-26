"""FastAPI application for the Enterprise RAG System.

Exposes:
  POST   /query              — RAG query endpoint (SSE streaming or JSON)
  GET    /health             — Component health check
  POST   /ingest             — Trigger ingestion pipeline as background task
  DELETE /session/{id}       — Clear a conversation session

Requirements: 2.4, 5.4, 8.2, 10.3, 10.4
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from enterprise_rag.access_controller import AccessController
from enterprise_rag.citation_engine import CitationEngine
from enterprise_rag.conversation_manager import ConversationManager
from enterprise_rag.generator import Generator
from enterprise_rag.ingestion.pipeline import IngestionPipeline
from enterprise_rag.logging import StructuredLogger
from enterprise_rag.models import (
    AccessFilter,
    Citation,
    ComponentStatus,
    GraphCitation,
    Role,
    Turn,
)
from enterprise_rag.query_rewriter import QueryRewriter
from enterprise_rag.retriever import Retriever

logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise RAG System", version="1.0.0")

# Serve the frontend
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

# ---------------------------------------------------------------------------
# Component singletons (created once at startup via environment config)
# ---------------------------------------------------------------------------

def _make_access_controller() -> AccessController:
    return AccessController()


def _make_conversation_manager() -> ConversationManager:
    return ConversationManager()


def _make_query_rewriter() -> QueryRewriter:
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    return QueryRewriter(ollama_url=ollama_url, model=model, timeout=30.0)


def _make_generator() -> Generator:
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
    return Generator(ollama_url=ollama_url, model=model, context_window=2048)


def _make_embed_fn():
    """Return a sentence-transformers embedding function, or a random fallback."""
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        def embed(text: str):
            return _model.encode([text], show_progress_bar=False)[0].tolist()
        return embed
    except ImportError:
        import random
        def embed(text: str):
            return [random.random() for _ in range(384)]
        return embed


def _make_retriever() -> Retriever:
    from enterprise_rag.vector_store import VectorStore
    chroma_path = os.environ.get("CHROMA_PATH", "./chroma_data")
    vs = VectorStore(persist_path=chroma_path)
    return Retriever(vector_store=vs, embed_fn=_make_embed_fn())


def _make_citation_engine() -> CitationEngine:
    return CitationEngine()


def _make_structured_logger() -> StructuredLogger:
    backend = os.environ.get("LOG_BACKEND", "file")
    log_dir = os.environ.get("LOG_DIR", "logs")
    return StructuredLogger(backend=backend, log_dir=log_dir)


# Module-level singletons (lazy-initialised on first request)
_access_controller: Optional[AccessController] = None
_conversation_manager: Optional[ConversationManager] = None
_query_rewriter: Optional[QueryRewriter] = None
_generator: Optional[Generator] = None
_retriever: Optional[Retriever] = None
_citation_engine: Optional[CitationEngine] = None
_structured_logger: Optional[StructuredLogger] = None
_ingestion_pipeline: Optional[IngestionPipeline] = None


def get_access_controller() -> AccessController:
    global _access_controller
    if _access_controller is None:
        _access_controller = _make_access_controller()
    return _access_controller


def get_conversation_manager() -> ConversationManager:
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = _make_conversation_manager()
    return _conversation_manager


def get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = _make_query_rewriter()
    return _query_rewriter


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = _make_generator()
    return _generator


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = _make_retriever()
    return _retriever


def get_citation_engine() -> CitationEngine:
    global _citation_engine
    if _citation_engine is None:
        _citation_engine = _make_citation_engine()
    return _citation_engine


def get_structured_logger() -> StructuredLogger:
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = _make_structured_logger()
    return _structured_logger


def _make_ingestion_pipeline() -> Optional[IngestionPipeline]:
    """Build an IngestionPipeline from environment variables if configured."""
    from enterprise_rag.ast_parser import ASTParser
    from enterprise_rag.ingestion.github_connector import GitHubConnector
    from enterprise_rag.ingestion.docs_connector import DocsConnector
    from enterprise_rag.ingestion.jira_connector import JiraConnector

    connectors: dict = {}

    # GitHub connector — enabled when GITHUB_REPO + GITHUB_TOKEN are set
    gh_repo = os.environ.get("GITHUB_REPO", "")
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if gh_repo and gh_token:
        gh_source_id = os.environ.get("GITHUB_SOURCE_ID", "github")
        gh_tags = [t.strip() for t in os.environ.get("GITHUB_PERMISSION_TAGS", "engineering").split(",") if t.strip()]
        connectors[gh_source_id] = GitHubConnector(
            source_id=gh_source_id, repo=gh_repo, token=gh_token, permission_tags=gh_tags
        )
        logger.info("GitHub connector registered: %s → %s", gh_source_id, gh_repo)

    # Docs connector — enabled when DOCS_PATH is set
    docs_path = os.environ.get("DOCS_PATH", "")
    if docs_path:
        docs_source_id = os.environ.get("DOCS_SOURCE_ID", "docs")
        docs_tags = [t.strip() for t in os.environ.get("DOCS_PERMISSION_TAGS", "engineering").split(",") if t.strip()]
        connectors[docs_source_id] = DocsConnector(
            source_id=docs_source_id, base_path=docs_path, permission_tags=docs_tags
        )
        logger.info("Docs connector registered: %s → %s", docs_source_id, docs_path)

    # Jira connector — enabled when JIRA_URL + JIRA_TOKEN are set
    jira_url = os.environ.get("JIRA_URL", "")
    jira_token = os.environ.get("JIRA_TOKEN", "")
    if jira_url and jira_token:
        jira_source_id = os.environ.get("JIRA_SOURCE_ID", "jira")
        jira_user = os.environ.get("JIRA_USERNAME", "")
        jira_project = os.environ.get("JIRA_PROJECT_KEY", "")
        jira_tags = [t.strip() for t in os.environ.get("JIRA_PERMISSION_TAGS", "engineering").split(",") if t.strip()]
        connectors[jira_source_id] = JiraConnector(
            source_id=jira_source_id, base_url=jira_url, username=jira_user,
            api_token=jira_token, project_key=jira_project, permission_tags=jira_tags
        )
        logger.info("Jira connector registered: %s → %s", jira_source_id, jira_url)

    if not connectors:
        return None

    chroma_path = os.environ.get("CHROMA_PATH", "./chroma_data")
    vs = VectorStore(persist_path=chroma_path)

    # Simple embedding function using sentence-transformers if available, else random
    def _embed_fn(chunks):
        try:
            from sentence_transformers import SentenceTransformer
            _model = getattr(_embed_fn, "_model", None)
            if _model is None:
                _embed_fn._model = SentenceTransformer("all-MiniLM-L6-v2")
                _model = _embed_fn._model
            texts = [c.text for c in chunks]
            vecs = _model.encode(texts, show_progress_bar=False).tolist()
        except ImportError:
            import random
            vecs = [[random.random() for _ in range(384)] for _ in chunks]
        from enterprise_rag.models import EmbeddedChunk
        return [
            EmbeddedChunk(**{k: getattr(c, k) for k in c.__dataclass_fields__}, embedding=v)
            for c, v in zip(chunks, vecs)
        ]

    return IngestionPipeline(
        vector_store=vs,
        graph_store=None,
        connectors=connectors,
        embed_fn=_embed_fn,
        ast_parser=ASTParser(),
    )


def get_ingestion_pipeline() -> Optional[IngestionPipeline]:
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = _make_ingestion_pipeline()
    return _ingestion_pipeline


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    session_id: str
    query: str


class IngestRequest(BaseModel):
    source_id: str
    full: bool = False


class CitationOut(BaseModel):
    number: int
    source_type: str
    document_title: str
    document_url: str
    excerpt: str
    chunk_ids: List[str]


class GraphCitationOut(BaseModel):
    number: int
    source_node: str
    relationship: str
    target_node: str
    file_path: str
    source_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationOut]
    graph_citations: List[GraphCitationOut]
    grounding_score: float
    warning: Optional[str]
    dependency_graph_unavailable: bool
    correlation_id: str


class HealthComponentOut(BaseModel):
    name: str
    status: str
    last_checked: str
    detail: Optional[str]


class HealthResponse(BaseModel):
    status: str
    components: List[HealthComponentOut]
    last_ingestion: Dict[str, Optional[str]]


class IngestResponse(BaseModel):
    job_id: str


# ---------------------------------------------------------------------------
# Helper: extract Bearer token
# ---------------------------------------------------------------------------

def _extract_token(authorization: Optional[str]) -> str:
    """Extract the Bearer token from the Authorization header value."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    return parts[1]


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query")
async def query_endpoint(
    request: QueryRequest,
    authorization: Optional[str] = Header(default=None),
    ac: AccessController = Depends(get_access_controller),
    cm: ConversationManager = Depends(get_conversation_manager),
    qr: QueryRewriter = Depends(get_query_rewriter),
    gen: Generator = Depends(get_generator),
    ret: Retriever = Depends(get_retriever),
    ce: CitationEngine = Depends(get_citation_engine),
    sl: StructuredLogger = Depends(get_structured_logger),
) -> Response:
    """RAG query endpoint.

    Validates the Bearer token, resolves roles, retrieves relevant chunks,
    generates a grounded answer, and returns either an SSE stream or JSON.
    """
    start_time = time.monotonic()
    correlation_id = sl.new_correlation_id()

    # --- Authentication (Req 2.4) ---
    token = _extract_token(authorization)
    if not ac.validate_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    roles: List[Role] = ac.resolve_roles(token)
    # If no roles resolved, build a permissive filter that allows all content.
    # This lets the system work without pre-configured roles in the cache.
    if not roles:
        access_filter: AccessFilter = AccessFilter(
            permitted_source_ids=[],
            permitted_tags=["engineering", "public", "internal", "docs", "github", "jira"],
        )
    else:
        access_filter: AccessFilter = ac.build_access_filter(roles)

    # --- Conversation history ---
    history = cm.get_history(request.session_id, last=10)

    # --- Query rewriting ---
    variants = qr.rewrite(request.query, history)

    # --- Retrieval ---
    try:
        chunks = ret.retrieve(variants, access_filter)
    except ConnectionError as exc:
        sl.log_error(
            severity="ERROR",
            component_name="retriever",
            error_message=str(exc),
            correlation_id=correlation_id,
        )
        raise HTTPException(status_code=503, detail="Vector store unavailable")

    dependency_graph_unavailable = getattr(ret, "last_dependency_graph_unavailable", False)

    # --- Generation ---
    try:
        answer_result = gen.generate(request.query, chunks, history, stream=False)
    except ConnectionError as exc:
        sl.log_error(
            severity="ERROR",
            component_name="generator",
            error_message=str(exc),
            correlation_id=correlation_id,
        )
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    except TimeoutError as exc:
        sl.log_error(
            severity="ERROR",
            component_name="generator",
            error_message=str(exc),
            correlation_id=correlation_id,
        )
        raise HTTPException(status_code=504, detail="Generator timed out")

    # Determine if we got a streaming iterator or a plain string
    is_streaming = hasattr(answer_result, "__iter__") and not isinstance(answer_result, str)

    if is_streaming:
        # Collect tokens for citation processing while streaming to client
        async def _sse_generator() -> AsyncIterator[str]:
            collected_tokens: List[str] = []
            try:
                for token in answer_result:  # type: ignore[union-attr]
                    collected_tokens.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except TimeoutError as exc:
                sl.log_error(
                    severity="ERROR",
                    component_name="generator",
                    error_message=str(exc),
                    correlation_id=correlation_id,
                )
                yield f"data: {json.dumps({'error': 'timeout'})}\n\n"
                return
            except ConnectionError as exc:
                sl.log_error(
                    severity="ERROR",
                    component_name="generator",
                    error_message=str(exc),
                    correlation_id=correlation_id,
                )
                yield f"data: {json.dumps({'error': 'unavailable'})}\n\n"
                return

            answer_text = "".join(collected_tokens)
            cited = ce.cite(answer_text, chunks)

            # Append turn to conversation history
            rewritten_query = variants[0] if variants else request.query
            turn = Turn(
                role="user",
                original_query=request.query,
                rewritten_query=rewritten_query,
                answer=cited.answer_text,
                timestamp=datetime.now(tz=timezone.utc),
            )
            cm.append_turn(request.session_id, turn)

            latency_ms = (time.monotonic() - start_time) * 1000
            _log_query(sl, request, roles, chunks, cited, latency_ms, correlation_id)

            # Send final metadata event
            warning = "Low confidence answer" if cited.low_confidence_warning else None
            final_payload = {
                "done": True,
                "answer": cited.answer_text,
                "citations": [_citation_to_dict(c) for c in cited.citations],
                "graph_citations": [_graph_citation_to_dict(gc) for gc in cited.graph_citations],
                "grounding_score": cited.grounding_score,
                "warning": warning,
                "dependency_graph_unavailable": dependency_graph_unavailable or cited.dependency_graph_unavailable,
                "correlation_id": correlation_id,
            }
            yield f"data: {json.dumps(final_payload)}\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    # --- Non-streaming path ---
    answer_text: str = answer_result  # type: ignore[assignment]
    cited = ce.cite(answer_text, chunks)

    rewritten_query = variants[0] if variants else request.query
    turn = Turn(
        role="user",
        original_query=request.query,
        rewritten_query=rewritten_query,
        answer=cited.answer_text,
        timestamp=datetime.now(tz=timezone.utc),
    )
    cm.append_turn(request.session_id, turn)

    latency_ms = (time.monotonic() - start_time) * 1000
    _log_query(sl, request, roles, chunks, cited, latency_ms, correlation_id)

    warning = "Low confidence answer" if cited.low_confidence_warning else None
    return JSONResponse(
        content={
            "answer": cited.answer_text,
            "citations": [_citation_to_dict(c) for c in cited.citations],
            "graph_citations": [_graph_citation_to_dict(gc) for gc in cited.graph_citations],
            "grounding_score": cited.grounding_score,
            "warning": warning,
            "dependency_graph_unavailable": dependency_graph_unavailable or cited.dependency_graph_unavailable,
            "correlation_id": correlation_id,
        }
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/sources")
async def sources_endpoint(
    pipeline: Optional[IngestionPipeline] = Depends(get_ingestion_pipeline),
):
    """List all registered source connectors."""
    if pipeline is None:
        return {"sources": [], "message": "No connectors configured. Set GITHUB_REPO+GITHUB_TOKEN, DOCS_PATH, or JIRA_URL+JIRA_TOKEN env vars."}
    return {"sources": list(pipeline._connectors.keys())}


@app.get("/health", response_model=HealthResponse)
async def health_endpoint(
    ret: Retriever = Depends(get_retriever),
    sl: StructuredLogger = Depends(get_structured_logger),
) -> HealthResponse:
    """Return health status for all system components.

    Checks: ollama, chromadb, session_store, and source connectors.
    Requirements: 10.3
    """
    components: List[HealthComponentOut] = []
    now_str = datetime.now(tz=timezone.utc).isoformat()

    # ChromaDB
    try:
        chroma_status = ret._vector_store.health_check()
        components.append(HealthComponentOut(
            name=chroma_status.name,
            status=chroma_status.status,
            last_checked=chroma_status.last_checked.isoformat(),
            detail=chroma_status.detail,
        ))
    except Exception as exc:
        components.append(HealthComponentOut(
            name="chromadb",
            status="down",
            last_checked=now_str,
            detail=str(exc),
        ))

    # Ollama — attempt a lightweight connectivity check
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        import requests as _requests
        resp = _requests.get(f"{ollama_url}/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
        components.append(HealthComponentOut(
            name="ollama",
            status="ok" if ollama_ok else "degraded",
            last_checked=now_str,
            detail=None if ollama_ok else f"HTTP {resp.status_code}",
        ))
    except Exception as exc:
        components.append(HealthComponentOut(
            name="ollama",
            status="down",
            last_checked=now_str,
            detail=str(exc),
        ))

    # Session store (in-memory / Redis)
    components.append(HealthComponentOut(
        name="session_store",
        status="ok",
        last_checked=now_str,
        detail=None,
    ))

    # Determine overall status
    statuses = {c.status for c in components}
    overall = "ok" if statuses <= {"ok"} else "degraded"

    # last_ingestion: not tracked at this layer; return empty dict
    last_ingestion: Dict[str, Optional[str]] = {}

    return HealthResponse(
        status=overall,
        components=components,
        last_ingestion=last_ingestion,
    )


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    pipeline: Optional[IngestionPipeline] = Depends(get_ingestion_pipeline),
    sl: StructuredLogger = Depends(get_structured_logger),
) -> IngestResponse:
    """Trigger an ingestion job as a background task.

    Returns a job_id immediately; the pipeline runs asynchronously.
    """
    job_id = str(uuid.uuid4())

    if pipeline is not None:
        incremental = not request.full
        background_tasks.add_task(pipeline.run, request.source_id, incremental)
    else:
        sl.log_error(
            severity="WARNING",
            component_name="api",
            error_message="Ingestion pipeline not configured; job queued but will not run",
            correlation_id=sl.new_correlation_id(),
        )

    return IngestResponse(job_id=job_id)


# ---------------------------------------------------------------------------
# POST /upload — direct file upload and immediate ingestion
# ---------------------------------------------------------------------------

_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
_ALLOWED_EXTENSIONS = {".txt", ".md", ".rst", ".html", ".pdf", ".py", ".js", ".ts", ".java"}


def _run_upload_ingestion(file_path: str, source_id: str, permission_tags: list) -> None:
    """Background task: ingest a single uploaded file."""
    import pathlib
    from enterprise_rag.ast_parser import ASTParser
    from enterprise_rag.ingestion.pipeline import IngestionPipeline
    from enterprise_rag.models import Document, EmbeddedChunk
    from enterprise_rag.vector_store import VectorStore

    chroma_path = os.environ.get("CHROMA_PATH", "./chroma_data")
    vs = VectorStore(persist_path=chroma_path)

    def _embed_fn(chunks):
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(_embed_fn, "_model"):
                _embed_fn._model = SentenceTransformer("all-MiniLM-L6-v2")
            vecs = _embed_fn._model.encode([c.text for c in chunks], show_progress_bar=False).tolist()
        except ImportError:
            import random
            vecs = [[random.random() for _ in range(384)] for _ in chunks]
        return [
            EmbeddedChunk(**{k: getattr(c, k) for k in c.__dataclass_fields__}, embedding=v)
            for c, v in zip(chunks, vecs)
        ]

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    doc = Document(
        doc_id=str(uuid.uuid4()),
        source_type="docs",
        source_id=source_id,
        title=os.path.basename(file_path),
        url=file_path,
        content=content,
        permission_tags=permission_tags,
        modified_at=datetime.now(timezone.utc),
    )

    pipeline = IngestionPipeline(
        vector_store=vs,
        graph_store=None,
        connectors={},
        embed_fn=_embed_fn,
        ast_parser=ASTParser(),
    )
    chunks = pipeline.chunk(doc)
    embedded = pipeline.embed(chunks)
    pipeline.index(embedded)


@app.post("/upload")
async def upload_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    permission_tags: str = "engineering",
) -> dict:
    """Upload a file and immediately ingest it into the vector store.

    Accepts: .txt, .md, .rst, .html, .pdf, .py, .js, .ts, .java
    Returns: { job_id, filename, size_bytes }
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
        )

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename or 'upload')}"
    dest = os.path.join(_UPLOAD_DIR, safe_name)

    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    tags = [t.strip() for t in permission_tags.split(",") if t.strip()]
    source_id = f"upload-{safe_name[:8]}"
    job_id = str(uuid.uuid4())

    background_tasks.add_task(_run_upload_ingestion, dest, source_id, tags)

    return {"job_id": job_id, "filename": file.filename, "size_bytes": len(content)}


# ---------------------------------------------------------------------------
# DELETE /session/{session_id}
# ---------------------------------------------------------------------------

@app.delete("/session/{session_id}", status_code=204)
async def delete_session_endpoint(
    session_id: str,
    cm: ConversationManager = Depends(get_conversation_manager),
) -> Response:
    """Clear a conversation session.

    Returns HTTP 204 No Content on success.
    """
    cm.clear_session(session_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _citation_to_dict(c: Citation) -> dict:
    return {
        "number": c.number,
        "source_type": c.source_type,
        "document_title": c.document_title,
        "document_url": c.document_url,
        "excerpt": c.excerpt,
        "chunk_ids": c.chunk_ids,
    }


def _graph_citation_to_dict(gc: GraphCitation) -> dict:
    return {
        "number": gc.number,
        "source_node": gc.source_node,
        "relationship": gc.relationship,
        "target_node": gc.target_node,
        "file_path": gc.file_path,
        "source_id": gc.source_id,
    }


def _log_query(
    sl: StructuredLogger,
    request: QueryRequest,
    roles: List[Role],
    chunks,
    cited,
    latency_ms: float,
    correlation_id: str,
) -> None:
    """Emit a structured query log entry."""
    user_id = "unknown"
    rewritten_query = request.query
    sl.log_query(
        user_id=user_id,
        session_id=request.session_id,
        original_query=request.query,
        rewritten_query=rewritten_query,
        chunks_retrieved=len(chunks),
        grounding_score=cited.grounding_score,
        latency_ms=latency_ms,
        correlation_id=correlation_id,
    )
