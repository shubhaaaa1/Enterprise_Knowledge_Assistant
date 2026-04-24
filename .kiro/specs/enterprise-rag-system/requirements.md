# Requirements Document

## Introduction

An enterprise Retrieval-Augmented Generation (RAG) system that allows employees to ask natural language questions against internal knowledge sources — including documentation, GitHub repositories, and Jira tickets — and receive grounded, cited answers. The system uses a local Ollama LLM backend to avoid data leaving the enterprise perimeter. It supports conversational memory for follow-up questions, query rewriting to improve retrieval quality, role-based access control (RBAC) to enforce data permissions, and citation highlighting so users can verify every answer against its source.

## Glossary

- **RAG_System**: The end-to-end enterprise retrieval-augmented generation application described in this document.
- **Ingestion_Pipeline**: The component responsible for fetching, chunking, embedding, and indexing content from external sources.
- **Query_Rewriter**: The component that rewrites or expands a user query before retrieval to improve recall.
- **Retriever**: The component that performs semantic and/or keyword search against the vector store to find relevant chunks.
- **Generator**: The Ollama-backed LLM component that produces a natural language answer conditioned on retrieved context.
- **Conversation_Manager**: The component that stores and retrieves per-session conversational history.
- **Citation_Engine**: The component that maps answer spans back to their source chunks and metadata.
- **Access_Controller**: The component that enforces RBAC rules, filtering retrievable content to what the requesting user is permitted to see.
- **Vector_Store**: The persistent store of embedded document chunks used for semantic retrieval.
- **User**: An authenticated enterprise employee interacting with the RAG_System.
- **Role**: A named permission group (e.g., `engineering`, `hr`, `finance`) assigned to a User that determines which data sources and documents are accessible.
- **Source**: A connected data origin — one of: internal documentation, GitHub repository, or Jira project.
- **Chunk**: A fixed-size, overlapping segment of a document produced by the Ingestion_Pipeline.
- **Citation**: A reference attached to an answer that identifies the Source, document title, URL, and character range of the supporting content.
- **Session**: A stateful conversation context scoped to a single User interaction sequence.
- **Hallucination**: An answer claim that is not supported by any retrieved Chunk.
- **AST_Parser**: The component that parses source code files using an Abstract Syntax Tree to extract structured code symbols rather than performing naive text chunking.
- **CodeSymbol**: A structured unit of code extracted by the AST_Parser, representing a function, class, method, or module, along with its docstring, source range, and call references.
- **Dependency_Graph**: A directed graph stored in Neo4j that captures relationships between CodeSymbols — including function calls, class inheritance, and module imports.
- **Neo4j_Store**: The persistent graph database used to store and query the Dependency_Graph.

---

## Requirements

### Requirement 1: Multi-Source Ingestion

**User Story:** As a knowledge administrator, I want to ingest content from internal docs, GitHub repositories, and Jira tickets, so that the RAG_System has a unified, up-to-date knowledge base.

#### Acceptance Criteria

1. THE Ingestion_Pipeline SHALL support three Source types: documentation (file-based or URL), GitHub repository (via GitHub API), and Jira project (via Jira REST API).
2. WHEN a new ingestion job is triggered for a Source, THE Ingestion_Pipeline SHALL fetch all accessible content, split it into Chunks of configurable size (default 512 tokens) with configurable overlap (default 64 tokens), embed each Chunk using a configured embedding model, and store the resulting vectors in the Vector_Store.
3. WHEN an incremental sync is triggered for a Source, THE Ingestion_Pipeline SHALL fetch only content modified since the last successful sync timestamp and update the Vector_Store accordingly.
4. IF the connection to a Source fails during ingestion, THEN THE Ingestion_Pipeline SHALL log the error with the Source identifier and retry up to 3 times with exponential backoff before marking the job as failed.
5. WHEN an ingestion job completes, THE Ingestion_Pipeline SHALL record the job status (success or failure), the number of Chunks indexed, and the completion timestamp in a persistent job log.
6. THE Ingestion_Pipeline SHALL associate each Chunk with metadata including: Source type, Source identifier, document title, document URL, and the Role permissions required to access it.
7. WHEN ingesting a GitHub repository, THE AST_Parser SHALL parse source code files to extract CodeSymbols (functions, classes, methods) including their name, type, docstring, file path, line range, and call references (list of symbols called by that symbol); non-parseable files (markdown, configuration, binary) SHALL fall back to standard text chunking.
8. WHEN the AST_Parser extracts CodeSymbols from a GitHub repository, THE Ingestion_Pipeline SHALL store each CodeSymbol as a Chunk in the Vector_Store with its structured metadata, and SHALL simultaneously populate the Dependency_Graph in the Neo4j_Store with the corresponding nodes and relationships.

---

### Requirement 2: Role-Based Access Control (RBAC)

**User Story:** As a security administrator, I want retrieval to be filtered by the user's assigned roles, so that employees can only receive answers derived from content they are authorized to access.

#### Acceptance Criteria

1. THE Access_Controller SHALL maintain a mapping of Roles to permitted Source identifiers and document-level permission tags.
2. WHEN a User submits a query, THE Access_Controller SHALL resolve the User's assigned Roles and produce an access filter that restricts retrieval to Chunks whose permission metadata matches at least one of the User's Roles.
3. THE Retriever SHALL apply the access filter produced by the Access_Controller before returning any Chunks to the Generator.
4. IF a User's session token is expired or invalid, THEN THE RAG_System SHALL reject the query with an authentication error and SHALL NOT return any Chunks or generated answer.
5. WHEN a Role's permissions are updated, THE Access_Controller SHALL apply the updated permissions to all subsequent queries within 60 seconds without requiring a system restart.
6. THE RAG_System SHALL log each query with the User identifier, resolved Roles, and the number of Chunks returned after access filtering, for audit purposes.

---

### Requirement 3: Query Rewriting and Expansion

**User Story:** As a user, I want my questions to be automatically improved before retrieval, so that I get more relevant results even when my phrasing is imprecise.

#### Acceptance Criteria

1. WHEN a User submits a query, THE Query_Rewriter SHALL produce at least one rewritten or expanded version of the query before passing it to the Retriever.
2. THE Query_Rewriter SHALL incorporate the last 5 turns of the current Session's conversational history when rewriting a query, so that pronouns and references are resolved to their antecedents.
3. WHEN the Query_Rewriter produces multiple query variants, THE Retriever SHALL execute retrieval for each variant and merge the resulting Chunk sets using reciprocal rank fusion before returning the final ranked list.
4. IF the Query_Rewriter fails to produce a rewritten query within 5 seconds, THEN THE RAG_System SHALL fall back to using the original user query for retrieval and SHALL log the timeout event.
5. THE Query_Rewriter SHALL not alter the semantic intent of the original query; the rewritten query SHALL be a clarification or expansion, not a substitution of meaning.

---

### Requirement 4: Semantic Retrieval

**User Story:** As a user, I want the system to find the most relevant content across all my permitted sources, so that answers are grounded in the best available evidence.

#### Acceptance Criteria

1. WHEN a (rewritten) query is submitted to the Retriever, THE Retriever SHALL perform a semantic similarity search against the Vector_Store and return the top-K Chunks (default K=10, configurable) ranked by relevance score.
2. THE Retriever SHALL apply the access filter from the Access_Controller before scoring, ensuring no unauthorized Chunks appear in results regardless of relevance score.
3. WHERE hybrid search is enabled, THE Retriever SHALL combine semantic similarity scores with BM25 keyword scores using a configurable weighting factor (default: 0.7 semantic, 0.3 BM25).
4. WHEN no Chunks with a relevance score above the configured threshold (default: 0.5) are found, THE Retriever SHALL return an empty result set and THE Generator SHALL respond with a message indicating insufficient information rather than generating an unsupported answer.
5. THE Retriever SHALL return each Chunk with its associated metadata (Source type, document title, URL, relevance score) alongside the Chunk text.

---

### Requirement 5: Grounded Answer Generation

**User Story:** As a user, I want answers that are strictly based on retrieved content, so that I can trust the information and avoid acting on hallucinations.

#### Acceptance Criteria

1. WHEN the Retriever returns a non-empty Chunk set, THE Generator SHALL produce an answer using only the content of those Chunks as context, with the Ollama LLM as the inference backend.
2. THE Generator SHALL include an explicit instruction in the LLM prompt directing the model to answer only from the provided context and to state when the context does not contain sufficient information.
3. WHEN the Retriever returns an empty result set, THE Generator SHALL return a standardized "insufficient information" response and SHALL NOT invoke the Ollama LLM with a speculative prompt.
4. THE Generator SHALL complete answer generation within 30 seconds of receiving the Chunk set; IF generation exceeds 30 seconds, THEN THE RAG_System SHALL return a timeout error to the User.
5. THE Generator SHALL pass the top-K Chunks to the Ollama model in descending relevance order, with the most relevant Chunk appearing first in the context window.

---

### Requirement 6: Citation Highlighting

**User Story:** As a user, I want every answer to include citations linking back to the source documents, so that I can verify claims and read further.

#### Acceptance Criteria

1. WHEN the Generator produces an answer, THE Citation_Engine SHALL identify which Chunks from the retrieved set were used to support the answer and attach a Citation for each.
2. THE Citation_Engine SHALL include in each Citation: the Source type, document title, document URL, and the verbatim Chunk excerpt (up to 300 characters) that supports the answer.
3. THE RAG_System SHALL present Citations to the User in a structured format alongside the answer, with each Citation numbered and referenced inline within the answer text (e.g., [1], [2]).
4. IF the Generator produces a claim that cannot be mapped to any retrieved Chunk, THEN THE Citation_Engine SHALL flag that claim as unverified in the response.
5. THE Citation_Engine SHALL deduplicate Citations that reference the same document URL, merging multiple Chunk excerpts from the same document into a single Citation entry.

---

### Requirement 7: Conversational Memory

**User Story:** As a user, I want to ask follow-up questions without repeating context, so that I can have a natural multi-turn conversation with the system.

#### Acceptance Criteria

1. THE Conversation_Manager SHALL maintain a Session for each authenticated User, storing the ordered sequence of query-answer pairs for the duration of the session.
2. WHEN a User submits a follow-up query, THE Conversation_Manager SHALL provide the last 10 query-answer pairs from the current Session to the Query_Rewriter and Generator as conversational context.
3. WHEN a Session has been inactive for 60 minutes, THE Conversation_Manager SHALL expire the Session and release its stored history.
4. THE Conversation_Manager SHALL persist Session history to durable storage so that an application restart does not cause loss of active Sessions with less than 60 minutes of inactivity.
5. WHEN a User explicitly requests to clear the conversation, THE Conversation_Manager SHALL delete the current Session history and start a new Session for that User.
6. THE Conversation_Manager SHALL store Session history in a format that preserves the role (user or assistant), the original query text, the rewritten query text, and the generated answer for each turn.

---

### Requirement 8: Ollama LLM Backend Integration

**User Story:** As an infrastructure administrator, I want the system to use a locally hosted Ollama instance for all LLM inference, so that no enterprise data is sent to external services.

#### Acceptance Criteria

1. THE Generator SHALL communicate with the Ollama API exclusively for all LLM inference operations, using the configured Ollama endpoint URL and model name.
2. WHEN the Ollama service is unavailable, THE RAG_System SHALL return a service unavailability error to the User within 10 seconds and SHALL NOT fall back to any external LLM provider.
3. THE RAG_System SHALL support runtime configuration of the Ollama endpoint URL, model name, and inference parameters (temperature, top-p, max tokens) without requiring a code change or restart.
4. THE Generator SHALL enforce a maximum context window size consistent with the configured Ollama model's limits; IF the combined Chunk context exceeds the limit, THEN THE Generator SHALL truncate lower-ranked Chunks to fit within the limit.
5. WHERE streaming responses are supported by the configured Ollama model, THE Generator SHALL stream tokens to the User interface incrementally rather than waiting for the full response.

---

### Requirement 9: Hallucination Mitigation

**User Story:** As a user, I want the system to minimize unsupported claims, so that I can rely on answers for business decisions.

#### Acceptance Criteria

1. THE Generator SHALL use a prompt template that instructs the Ollama model to cite the specific Chunk number for every factual claim in the answer.
2. WHEN the Generator produces an answer, THE Citation_Engine SHALL verify that each cited Chunk number corresponds to a Chunk in the retrieved set; IF a cited Chunk number does not exist, THEN THE Citation_Engine SHALL remove the unsupported claim from the answer and log the discrepancy.
3. THE RAG_System SHALL compute a grounding score for each answer, defined as the ratio of cited claims to total claims; THE RAG_System SHALL include this score in the response metadata.
4. WHERE a grounding score falls below 0.7, THE RAG_System SHALL append a low-confidence warning to the answer visible to the User.
5. THE RAG_System SHALL log all answers with a grounding score below 0.7 for administrator review.

---

### Requirement 10: Observability and Audit Logging

**User Story:** As a system administrator, I want comprehensive logs of all queries, retrievals, and access decisions, so that I can audit usage, debug issues, and monitor system health.

#### Acceptance Criteria

1. THE RAG_System SHALL emit a structured log entry for every query containing: timestamp, User identifier, Session identifier, original query, rewritten query, number of Chunks retrieved, grounding score, and response latency in milliseconds.
2. THE RAG_System SHALL emit a structured log entry for every access control decision containing: timestamp, User identifier, Roles evaluated, Sources filtered, and number of Chunks excluded by access control.
3. THE RAG_System SHALL expose a health check endpoint that returns the status of each connected component (Ollama, Vector_Store, each configured Source connector) and the timestamp of the last successful ingestion per Source.
4. WHEN any component returns an error, THE RAG_System SHALL log the error with severity level, component name, error message, and a correlation ID that links the error to the originating query log entry.
5. THE RAG_System SHALL retain structured logs for a minimum of 90 days in a configurable log storage backend.

---

### Requirement 11: Code Dependency Graph

**User Story:** As a developer, I want to query code dependency relationships — such as what calls a function or what a module imports — so that I can understand code structure and receive answers that include dependency context with citations.

#### Acceptance Criteria

1. THE Neo4j_Store SHALL persist a Dependency_Graph containing nodes of types `File`, `Function`, `Class`, `Method`, and `Module`, and relationships of types `CALLS`, `INHERITS`, `IMPORTS`, `DEFINED_IN`, and `CONTAINS`.
2. WHEN a GitHub repository is ingested, THE Ingestion_Pipeline SHALL populate the Dependency_Graph with nodes and relationships derived from the AST_Parser output, such that every CodeSymbol extracted becomes a node and every call reference becomes a `CALLS` edge.
3. WHEN a user submits a query that is identified as code-related (e.g., contains symbol names or dependency keywords), THE Retriever SHALL query the Neo4j_Store for dependency relationships in addition to performing vector search, and SHALL merge the graph results with the vector results before passing context to the Generator.
4. WHEN the Retriever queries the Dependency_Graph, it SHALL support querying by direction: `callers` (symbols that call the target), `callees` (symbols called by the target), or `both`, and SHALL support a configurable traversal depth (default: 2 hops).
5. WHEN the Generator produces an answer that incorporates Dependency_Graph results, THE Citation_Engine SHALL include graph-derived citations that identify the source node, relationship type, and target node for each dependency referenced in the answer.
6. IF the Neo4j_Store is unavailable during a query, THE RAG_System SHALL fall back to vector-only retrieval, log the degradation, and include a notice in the response metadata that dependency graph results are unavailable.
7. THE Neo4j_Store SHALL enforce that every `CALLS` edge references source and target nodes that exist in the graph; orphaned edges SHALL NOT be created during ingestion.
