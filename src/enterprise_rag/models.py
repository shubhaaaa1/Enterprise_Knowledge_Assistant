"""Data models for the Enterprise RAG System."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Role:
    name: str                          # e.g. "engineering", "hr"
    permitted_source_ids: List[str]
    permitted_tags: List[str]


@dataclass
class AccessFilter:
    permitted_source_ids: List[str]
    permitted_tags: List[str]


@dataclass
class Chunk:
    chunk_id: str                      # uuid
    source_type: str                   # "docs" | "github" | "jira"
    source_id: str
    document_title: str
    document_url: str
    text: str
    token_count: int
    permission_tags: List[str]         # roles that can access this chunk
    created_at: datetime
    source_modified_at: datetime


@dataclass
class EmbeddedChunk(Chunk):
    embedding: List[float] = field(default_factory=list)   # dense vector


@dataclass
class ScoredChunk(Chunk):
    relevance_score: float = 0.0
    rrf_score: float = 0.0


@dataclass
class CodeSymbol:
    symbol_id: str                     # uuid
    file_path: str
    symbol_name: str
    symbol_type: str                   # "function" | "class" | "method" | "module"
    docstring: Optional[str]
    source_code: str
    line_start: int
    line_end: int
    call_refs: List[str]               # names of symbols called by this symbol
    source_id: str
    permission_tags: List[str]


@dataclass
class Citation:
    number: int
    source_type: str
    document_title: str
    document_url: str
    excerpt: str                       # verbatim, up to 300 chars
    chunk_ids: List[str]               # merged if same URL


@dataclass
class GraphCitation:
    number: int
    source_node: str                   # symbol name
    relationship: str                  # e.g. "CALLS"
    target_node: str                   # symbol name
    file_path: str
    source_id: str


@dataclass
class CitedAnswer:
    answer_text: str                   # with inline [N] references
    citations: List[Citation]
    graph_citations: List[GraphCitation]
    grounding_score: float             # cited_claims / total_claims
    unverified_claims: List[str]       # claims with no chunk mapping
    low_confidence_warning: bool       # grounding_score < 0.7
    dependency_graph_unavailable: bool # True if graph store was unreachable


@dataclass
class Turn:
    role: str                          # "user" | "assistant"
    original_query: str
    rewritten_query: str
    answer: str
    timestamp: datetime


@dataclass
class Session:
    session_id: str
    user_id: str
    turns: List[Turn]
    created_at: datetime
    last_active_at: datetime           # used for 60-min expiry


@dataclass
class JobResult:
    job_id: str
    source_id: str
    status: str                        # "success" | "failed"
    chunks_indexed: int
    symbols_indexed: int               # CodeSymbols indexed
    completed_at: datetime
    error_message: Optional[str]


@dataclass
class ComponentStatus:
    name: str
    status: str                        # "ok" | "degraded" | "down"
    last_checked: datetime
    detail: Optional[str]


@dataclass
class Document:
    doc_id: str
    source_type: str
    source_id: str
    title: str
    url: str
    content: str
    permission_tags: List[str]
    modified_at: datetime
