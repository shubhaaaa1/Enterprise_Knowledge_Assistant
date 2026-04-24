# Implementation Plan: Enterprise RAG System

## Overview

Implement the enterprise RAG system incrementally, starting with project scaffolding and data models, then building each component in dependency order: stores → parsers → ingestion → access control → conversation → query/retrieval → generation → citation → API layer. Property-based tests (Hypothesis) are placed immediately after the component they validate.

## Tasks

- [x] 1. Project scaffolding and data models
  - Create Python project structure: `src/enterprise_rag/`, `tests/unit/`, `tests/integration/`, `tests/smoke/`
  - Create `pyproject.toml` with dependencies: `chromadb`, `neo4j`, `ollama`, `tree-sitter`, `hypothesis`, `fastapi`, `uvicorn`, `redis`, `psycopg2-binary`, `pytest`, `pytest-asyncio`, `httpx`
  - Implement all dataclasses in `src/enterprise_rag/models.py`: `Role`, `AccessFilter`, `Chunk`, `EmbeddedChunk`, `ScoredChunk`, `CodeSymbol`, `DependencyGraph`, `Citation`, `GraphCitation`, `CitedAnswer`, `Turn`, `Session`, `JobResult`, `ComponentStatus`
  - _Requirements: 1.2, 1.6, 1.7, 2.1, 6.2, 7.6, 9.3, 11.1_

- [x] 2. ChromaDB vector store
  - [x] 2.1 Implement `VectorStore` class in `src/enterprise_rag/vector_store.py`
    - Persistent ChromaDB client, single collection `enterprise_rag`
    - `upsert(chunks: List[EmbeddedChunk]) -> None` — stores vectors with full metadata including `permission_tags`
    - `query(embedding: List[float], access_filter: AccessFilter, k: int, threshold: float) -> List[ScoredChunk]` — applies metadata filter for RBAC pre-retrieval
    - `get_by_id(chunk_id: str) -> Optional[Chunk]`
    - `health_check() -> ComponentStatus`
    - _Requirements: 1.2, 1.6, 2.2, 2.3, 4.1, 4.2_

  - [x] 2.2 Write property test for access filter soundness (Property 5)
    - **Property 5: Access filter soundness**
    - **Validates: Requirements 2.2, 2.3, 4.2**
    - Use `st.lists(st.sampled_from(roles))` for user roles and random chunk permission tags; assert no returned chunk has `permission_tags` disjoint from user roles

- [x] 3. Neo4j graph store
  - [x] 3.1 Implement `GraphStore` class in `src/enterprise_rag/graph_store.py`
    - Bolt-protocol connection via `neo4j` Python driver
    - `upsert_symbols(symbols: List[CodeSymbol]) -> None` — creates/merges nodes with labels `File`, `Function`, `Class`, `Method`, `Module`
    - `upsert_relationships(symbols: List[CodeSymbol]) -> None` — creates `CALLS`, `INHERITS`, `IMPORTS`, `DEFINED_IN`, `CONTAINS` edges; only between nodes that exist in the graph
    - `query_dependencies(symbol: str, direction: Literal["callers","callees","both"], depth: int = 2) -> DependencyGraph`
    - `health_check() -> ComponentStatus`
    - _Requirements: 11.1, 11.2, 11.4, 11.7_

  - [x] 3.2 Write property test for Neo4j graph edge validity (Property 26)
    - **Property 26: No orphaned CALLS edges**
    - **Validates: Requirements 11.2, 11.7**
    - Generate random `CodeSymbol` lists; after `upsert_symbols` + `upsert_relationships`, assert every CALLS edge source and target exists as a node

  - [x] 3.3 Write property test for dependency query direction correctness (Property 27)
    - **Property 27: Dependency query direction correctness**
    - **Validates: Requirements 11.4**
    - Build known graph structures with generated symbol names; assert `callers` returns only inbound, `callees` only outbound, `both` returns union

- [x] 4. AST parser
  - [x] 4.1 Implement `ASTParser` class in `src/enterprise_rag/ast_parser.py`
    - `parse(file_path: str, content: str, language: str) -> List[CodeSymbol]`
    - Python files: use `ast` module to extract functions, classes, methods, docstrings, call references, line ranges
    - Other languages (JS, TS, Java, etc.): use `tree-sitter` bindings
    - `supported_languages() -> List[str]`
    - Non-parseable files (markdown, YAML, config, binary): return empty list (caller falls back to text chunking)
    - _Requirements: 1.7_

  - [x] 4.2 Write property test for AST symbol extraction completeness (Property 24)
    - **Property 24: AST symbol extraction completeness**
    - **Validates: Requirements 1.7**
    - Generate Python/JS source snippets with functions, classes, methods; assert every returned `CodeSymbol` has non-null `symbol_name`, `symbol_type`, `file_path`, `line_start`, `line_end`, and `call_refs` (may be empty list)

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Ingestion pipeline
  - [x] 6.1 Implement `SourceConnector` base class and three connectors in `src/enterprise_rag/ingestion/`
    - `SourceConnector` abstract base: `fetch_all() -> Iterator[Document]`, `fetch_incremental(since: datetime) -> Iterator[Document]`
    - `DocsConnector` — file-based or URL documentation source
    - `GitHubConnector` — GitHub API; runs `ASTParser` on each source file; parseable → `CodeSymbol` list; non-parseable → text chunks
    - `JiraConnector` — Jira REST API
    - _Requirements: 1.1, 1.7_

  - [x] 6.2 Implement `IngestionPipeline` in `src/enterprise_rag/ingestion/pipeline.py`
    - `chunk(doc: Document, size: int = 512, overlap: int = 64) -> List[Chunk]` — token-based chunking with overlap
    - `embed(chunks: List[Chunk]) -> List[EmbeddedChunk]` — calls configured embedding model
    - `index(chunks: List[EmbeddedChunk]) -> None` — writes to `VectorStore`
    - `index_graph(symbols: List[CodeSymbol]) -> None` — writes to `GraphStore` (upsert nodes then relationships)
    - `run(source_id: str, incremental: bool = True) -> JobResult` — orchestrates fetch → chunk → embed → index → index_graph; retry up to 3× with exponential backoff (1s, 2s, 4s) on source connection failure; writes `JobResult` to Postgres job log
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.8_

  - [x] 6.3 Write property test for chunk metadata completeness (Property 1)
    - **Property 1: Chunk metadata completeness**
    - **Validates: Requirements 1.6**
    - Use `st.text()` and `st.lists()` for document content; assert every resulting chunk has non-null `source_type`, `source_id`, `document_title`, `document_url`, `permission_tags`

  - [x] 6.4 Write property test for chunking size invariant (Property 2)
    - **Property 2: Chunking size invariant**
    - **Validates: Requirements 1.2**
    - Use `st.text()` and `st.integers(min_value=64, max_value=2048)` for size; assert every chunk token count is in `[1, size]` and consecutive chunks share `overlap` tokens

  - [x] 6.5 Write property test for incremental sync filter (Property 3)
    - **Property 3: Incremental sync only fetches modified content**
    - **Validates: Requirements 1.3**
    - Use `st.lists(st.datetimes())` for timestamps; assert incremental sync processes exactly documents with `modified_at > cutoff`

  - [x] 6.6 Write property test for ingestion job log completeness (Property 4)
    - **Property 4: Ingestion job log completeness**
    - **Validates: Requirements 1.5**
    - For random document sets, assert completed `JobResult` has non-null `status`, `chunks_indexed`, `completed_at`

  - [x] 6.7 Write property test for dual-store ingestion consistency (Property 25)
    - **Property 25: Dual-store ingestion consistency**
    - **Validates: Requirements 1.8**
    - Generate random `CodeSymbol` lists; after pipeline run, assert every symbol is retrievable from ChromaDB by `symbol_id` AND exists as a node in Neo4j by `symbol_name` + `file_path`

  - [x] 6.8 Write unit tests for ingestion pipeline
    - Test retry behavior: source failure triggers 3 retries with exponential backoff (Req 1.4)
    - Test partial ingestion: per-document errors logged, remaining docs continue (Req 1.4)
    - Test Neo4j unavailability during ingestion: graph indexing marked failed, ChromaDB indexing continues (Req 11.6)

- [x] 7. Access controller
  - [x] 7.1 Implement `AccessController` in `src/enterprise_rag/access_controller.py`
    - `validate_token(token: str) -> bool`
    - `resolve_roles(token: str) -> List[Role]`
    - `build_access_filter(roles: List[Role]) -> AccessFilter`
    - `update_role_permissions(role: Role, permissions: Permissions) -> None` — in-memory cache with 60s TTL backed by Postgres
    - _Requirements: 2.1, 2.2, 2.4, 2.5_

  - [x] 7.2 Write property test for access control audit log completeness (Property 22)
    - **Property 22: Access control audit log completeness**
    - **Validates: Requirements 10.2**
    - For random access decisions, assert emitted log entry has non-null `timestamp`, `user_id`, `roles_evaluated`, `sources_filtered`, `chunks_excluded`

  - [x] 7.3 Write unit tests for access controller
    - Test expired/invalid token → rejected with auth error, no chunks returned (Req 2.4)
    - Test role permission update propagates within 60s (Req 2.5)
    - Test role mapping CRUD operations (Req 2.1)

- [x] 8. Conversation manager
  - [x] 8.1 Implement `ConversationManager` in `src/enterprise_rag/conversation_manager.py`
    - Redis primary + Postgres durable storage
    - `get_history(session_id: str, last: int = 10) -> List[Turn]`
    - `append_turn(session_id: str, turn: Turn) -> None`
    - `clear_session(session_id: str) -> None`
    - `expire_inactive_sessions(ttl_minutes: int = 60) -> None` — background task
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x] 8.2 Write property test for session turn round-trip fidelity (Property 17)
    - **Property 17: Session turn round-trip fidelity**
    - **Validates: Requirements 7.1, 7.6**
    - Generate random turn sequences; assert `get_history` returns turns in same order with all fields preserved exactly

  - [x] 8.3 Write property test for session history window (Property 18)
    - **Property 18: Session history window**
    - **Validates: Requirements 7.2**
    - Use `st.integers(min_value=0, max_value=50)` for N; assert `get_history(last=10)` returns exactly `min(N, 10)` most recent turns

  - [x] 8.4 Write property test for session persistence across restarts (Property 19)
    - **Property 19: Session persistence across restarts**
    - **Validates: Requirements 7.4**
    - Simulate application restart; assert session history fully recoverable from durable storage with all turns intact

  - [x] 8.5 Write unit tests for conversation manager
    - Test session clear deletes history and starts new session (Req 7.5)
    - Test session expiry after 60 min inactivity (Req 7.3)

- [x] 9. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Query rewriter
  - [x] 10.1 Implement `QueryRewriter` in `src/enterprise_rag/query_rewriter.py`
    - `rewrite(query: str, history: List[Turn], max_variants: int = 3) -> List[str]`
    - Calls Ollama with a lightweight prompt incorporating last 5 turns of history
    - 5s timeout: on timeout, return `[original_query]` and log timeout event with session and query identifiers
    - _Requirements: 3.1, 3.2, 3.4, 3.5_

  - [x] 10.2 Write property test for query rewriter always produces at least one variant (Property 7)
    - **Property 7: Query rewriter always produces ≥1 variant**
    - **Validates: Requirements 3.1**
    - Use `st.text(min_size=1)` for queries; assert returned list is non-empty (including fallback case)

  - [x] 10.3 Write unit tests for query rewriter
    - Test timeout fallback: rewriter exceeds 5s → returns original query, logs timeout (Req 3.4)

- [x] 11. Retriever
  - [x] 11.1 Implement `Retriever` in `src/enterprise_rag/retriever.py`
    - `retrieve(variants: List[str], access_filter: AccessFilter, k: int = 10, semantic_weight: float = 0.7, threshold: float = 0.5) -> List[ScoredChunk]`
    - Per variant: semantic search + BM25 against ChromaDB with access filter applied pre-scoring
    - Per-variant score: `semantic_weight * sem_score + (1 - semantic_weight) * bm25_score`
    - Merge across variants via RRF: `rrf_score(d) = Σ 1 / (k + rank_i(d))`
    - Filter chunks below threshold; return top-K by RRF score in descending order
    - Code-related query detection: if query contains symbol names or dependency keywords, call `GraphStore.query_dependencies` and append graph-derived chunks
    - Neo4j unavailability: fall back to vector-only, log degradation, set `dependency_graph_unavailable=True`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 11.3, 11.4, 11.6_

  - [x] 11.2 Write property test for RRF correctness (Property 8)
    - **Property 8: Reciprocal rank fusion correctness**
    - **Validates: Requirements 3.3**
    - Use `st.lists(st.lists(...))` for ranked lists; assert a document appearing in more lists receives higher RRF score than one appearing in fewer, all else equal

  - [x] 11.3 Write property test for retrieval result ordering and size (Property 9)
    - **Property 9: Retrieval result ordering and size**
    - **Validates: Requirements 4.1**
    - For random scored chunk sets and K values, assert result count ≤ K and chunks sorted by final score descending

  - [x] 11.4 Write property test for hybrid score formula correctness (Property 10)
    - **Property 10: Hybrid score formula correctness**
    - **Validates: Requirements 4.3**
    - Use `st.floats(0, 1)` for scores and weights; assert `combined == semantic_weight * sem + (1 - semantic_weight) * bm25`

  - [x] 11.5 Write unit tests for retriever
    - Test empty result set when no chunks above threshold → generator receives empty set (Req 4.4)
    - Test access filter applied before scoring (Req 4.2)

- [x] 12. Generator
  - [x] 12.1 Implement `Generator` in `src/enterprise_rag/generator.py`
    - `generate(query: str, chunks: List[ScoredChunk], history: List[Turn], stream: bool = True) -> AsyncIterator[str] | str`
    - Wraps Ollama `/api/generate`; configurable endpoint URL, model name, temperature, top-p, max tokens
    - Prompt template: grounding instruction + numbered chunks in descending relevance order + last N turns + question
    - Context window enforcement: truncate lowest-ranked chunks until total tokens fit within model limit; log truncation event with count removed
    - Empty chunk set → return standardized "insufficient information" response, do NOT call Ollama
    - 30s timeout → raise timeout, caller returns HTTP 504
    - Streaming: SSE token delivery when Ollama supports it
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 12.2 Write property test for generator prompt chunk ordering (Property 11)
    - **Property 11: Generator prompt contains chunks in descending relevance order**
    - **Validates: Requirements 5.1, 5.5**
    - Generate random scored chunk lists; assert constructed prompt presents chunks in strictly descending `relevance_score` order

  - [x] 12.3 Write property test for context window truncation (Property 12)
    - **Property 12: Context window truncation preserves highest-ranked chunks**
    - **Validates: Requirements 8.4**
    - Generate random chunk lists with token counts exceeding context limit; assert resulting prompt fits within limit and lowest-ranked chunks were removed first

  - [x] 12.4 Write unit tests for generator
    - Test empty chunk set → no Ollama call, returns "insufficient information" (Req 5.3)
    - Test Ollama unavailability → error within 10s, no external fallback (Req 8.2)
    - Test prompt template contains grounding instruction (Req 5.2, 9.1)

- [x] 13. Citation engine
  - [x] 13.1 Implement `CitationEngine` in `src/enterprise_rag/citation_engine.py`
    - `cite(answer: str, chunks: List[ScoredChunk]) -> CitedAnswer`
    - `compute_grounding_score(answer: str, chunks: List[ScoredChunk]) -> float` — `cited_claims / total_claims`
    - `flag_unverified_claims(answer: str, chunks: List[ScoredChunk]) -> str` — removes claims with no chunk mapping, adds to `unverified_claims`
    - Deduplication: one `Citation` per unique `document_url`, merging excerpts from multiple chunks
    - `GraphCitation` support: for graph-derived chunks, produce `GraphCitation` with `source_node`, `relationship`, `target_node`, `file_path`
    - `low_confidence_warning = True` when `grounding_score < 0.7`; emit log entry for administrator review
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 9.2, 9.3, 9.4, 9.5_

  - [x] 13.2 Write property test for citation-answer correspondence (Property 13)
    - **Property 13: Citation-answer correspondence**
    - **Validates: Requirements 6.1, 6.3**
    - Generate random answer strings with `[N]` refs; assert every `[N]` has a matching citation and every citation is referenced at least once

  - [x] 13.3 Write property test for citation field completeness and excerpt length (Property 14)
    - **Property 14: Citation field completeness and excerpt length**
    - **Validates: Requirements 6.2**
    - Generate random chunk metadata; assert every citation has non-null `source_type`, `document_title`, `document_url`, `excerpt` and `len(excerpt) <= 300`

  - [x] 13.4 Write property test for citation deduplication by URL (Property 15)
    - **Property 15: Citation deduplication by URL**
    - **Validates: Requirements 6.5**
    - Generate random chunks with overlapping `document_url` values; assert exactly one citation per unique URL

  - [x] 13.5 Write property test for invalid chunk references removed and flagged (Property 16)
    - **Property 16: Invalid chunk references removed and flagged**
    - **Validates: Requirements 6.4, 9.2**
    - Generate answers with out-of-range `[N]` refs; assert unsupported claims removed from answer and present in `unverified_claims`

  - [x] 13.6 Write property test for grounding score formula correctness (Property 20)
    - **Property 20: Grounding score formula correctness**
    - **Validates: Requirements 9.3**
    - Use `st.integers(min_value=0)` for cited and total claim counts; assert `grounding_score == cited / total`

  - [x] 13.7 Write property test for low-confidence behavior consistency (Property 21)
    - **Property 21: Low-confidence behavior is consistent**
    - **Validates: Requirements 9.4, 9.5**
    - Use `st.floats(0, 1)` for grounding scores; assert `grounding_score < 0.7` → `low_confidence_warning=True` AND log entry emitted

  - [x] 13.8 Write property test for graph citation field completeness (Property 28)
    - **Property 28: Graph citation field completeness**
    - **Validates: Requirements 11.5**
    - Generate random `DependencyGraph` objects; assert every `GraphCitation` has non-null `source_node`, `relationship`, `target_node`, `file_path`

- [x] 14. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Observability and structured logging
  - [x] 15.1 Implement `StructuredLogger` in `src/enterprise_rag/logging.py`
    - Emit query log entries: `timestamp`, `user_id`, `session_id`, `original_query`, `rewritten_query`, `chunks_retrieved`, `grounding_score`, `latency_ms`
    - Emit access control log entries: `timestamp`, `user_id`, `roles_evaluated`, `sources_filtered`, `chunks_excluded`
    - Emit error log entries: `severity`, `component_name`, `error_message`, `correlation_id`
    - Configurable backend (file, Elasticsearch, S3); 90-day retention policy
    - Assign `correlation_id` per query; propagate to all downstream log entries
    - _Requirements: 2.6, 10.1, 10.2, 10.4, 10.5_

  - [x] 15.2 Write property test for query audit log completeness (Property 6)
    - **Property 6: Query audit log completeness**
    - **Validates: Requirements 2.6, 10.1**
    - Generate random query/user/session combos; assert emitted log entry has all required fields non-null

  - [x] 15.3 Write property test for error log correlation (Property 23)
    - **Property 23: Error log correlation**
    - **Validates: Requirements 10.4**
    - Generate random error scenarios; assert error log entry `correlation_id` matches the originating query log entry's `correlation_id`

- [x] 16. API layer
  - [x] 16.1 Implement FastAPI application in `src/enterprise_rag/api.py`
    - `POST /query` — validate token (401 on failure), load history, rewrite, retrieve, generate, cite, append turn; SSE streaming response; assign `correlation_id`
    - `GET /health` — returns `ComponentStatus` for ollama, chromadb, neo4j, session_store, and each source connector; last ingestion timestamp per source
    - `POST /ingest` — triggers `IngestionPipeline.run()` as background task; returns `job_id`
    - `DELETE /session/{session_id}` — calls `ConversationManager.clear_session()`; returns 204
    - Wire all components together with dependency injection
    - _Requirements: 2.4, 5.4, 8.2, 10.3, 10.4_

  - [x] 16.2 Write integration tests for API layer
    - Test `POST /query` end-to-end with mocked Ollama: valid token → cited answer with grounding score
    - Test `POST /query` with expired token → 401, no chunks returned (Req 2.4)
    - Test `GET /health` returns all component statuses including ChromaDB and Neo4j (Req 10.3)
    - Test `DELETE /session/{id}` clears session history (Req 7.5)
    - Test Ollama unavailability → 503 within 10s, no external fallback (Req 8.2)
    - Test generator timeout → 504 within 30s (Req 5.4)
    - Test Neo4j unavailability → vector-only fallback, `dependency_graph_unavailable: true` in response (Req 11.6)
    - Test role permission update propagates within 60s (Req 2.5)
    - Test session expiry after 60 min inactivity (Req 7.3)

- [x] 17. Smoke tests
  - [x] 17.1 Implement smoke tests in `tests/smoke/test_smoke.py`
    - All three source connector types instantiate correctly (Req 1.1)
    - Role mapping CRUD operations work end-to-end (Req 2.1)
    - Runtime Ollama config change takes effect without restart (Req 8.3)
    - Log retention policy configured to ≥ 90 days (Req 10.5)
    - Neo4j node labels and relationship types present in schema (Req 11.1)
    - ChromaDB collection accessible and metadata filtering functional
    - GitHub ingestion end-to-end: AST symbols appear in both ChromaDB and Neo4j after job completes (Req 1.8)
    - _Requirements: 1.1, 1.8, 2.1, 8.3, 10.5, 11.1_

- [x] 18. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis with `@settings(max_examples=100)` and are tagged: `# Feature: enterprise-rag-system, Property N: <property_text>`
- All 28 correctness properties from the design document are covered by property test sub-tasks
- Checkpoints at tasks 5, 9, 14, and 18 ensure incremental validation
