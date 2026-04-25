# Enterprise RAG System

A locally-hosted Retrieval-Augmented Generation (RAG) platform that lets employees ask natural language questions against internal knowledge sources — documentation, GitHub repositories, and Jira tickets — and receive grounded, cited answers.

All LLM inference runs through a local [Ollama](https://ollama.com) instance, so no enterprise data ever leaves your network.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)
![Ollama](https://img.shields.io/badge/Ollama-local-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Features

- **Multi-source ingestion** — ingest from local docs, GitHub repos, and Jira projects
- **File upload** — drag-and-drop files directly from the browser UI
- **Hybrid search** — semantic similarity + BM25 keyword matching with Reciprocal Rank Fusion
- **Grounded answers** — every answer is backed by retrieved chunks with inline citations
- **Citation highlighting** — source title, URL, and excerpt for every claim
- **Grounding score** — ratio of cited claims to total claims, shown per answer
- **Role-based access control (RBAC)** — filter retrievable content by user roles
- **Conversational memory** — multi-turn sessions with 60-minute inactivity expiry
- **Query rewriting** — automatic query expansion using conversation history
- **Code dependency graph** — AST parsing + Neo4j graph for code structure queries
- **Local-only inference** — Ollama backend, no external API calls
- **Web UI** — built-in chat interface served at `/`

---

## Architecture

```
User → FastAPI (api.py)
         ├── AccessController   — token validation, RBAC filter
         ├── ConversationManager — session history (Redis + Postgres)
         ├── QueryRewriter      — query expansion via Ollama
         ├── Retriever          — hybrid search (ChromaDB + Neo4j)
         ├── Generator          — answer generation via Ollama
         └── CitationEngine     — citation mapping + grounding score

Background:
  IngestionPipeline → DocsConnector / GitHubConnector / JiraConnector
                    → ASTParser → VectorStore (ChromaDB) + GraphStore (Neo4j)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| [Ollama](https://ollama.com) | latest | Must be running locally |
| ChromaDB | auto-installed | Persistent local storage |
| Neo4j | optional | Only needed for code dependency graph |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/enterprise-rag.git
cd enterprise-rag
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
pip install python-multipart sentence-transformers
```

### 3. Install and start Ollama

```bash
# Windows (PowerShell)
irm https://ollama.com/install.ps1 | iex

# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama and pull a model:

```bash
ollama serve          # keep this running
ollama pull llama3.2:1b   # lightweight model (~1.3 GB)
# or for better quality (needs 5+ GB RAM):
# ollama pull llama3
```

### 4. Start the server

```bash
uvicorn enterprise_rag.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## Configuration

All settings are via environment variables — no code changes needed.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Model name |
| `CHROMA_PATH` | `./chroma_data` | ChromaDB storage path |
| `LOG_BACKEND` | `file` | Logging backend (`file`, `elasticsearch`, `s3`) |
| `LOG_DIR` | `logs` | Log file directory |
| `GITHUB_REPO` | — | GitHub repo to ingest (`owner/repo`) |
| `GITHUB_TOKEN` | — | GitHub personal access token |
| `GITHUB_SOURCE_ID` | `github` | Source identifier |
| `GITHUB_PERMISSION_TAGS` | `engineering` | Comma-separated permission tags |
| `DOCS_PATH` | — | Local docs folder to ingest |
| `JIRA_URL` | — | Jira base URL |
| `JIRA_TOKEN` | — | Jira API token |
| `JIRA_USERNAME` | — | Jira username/email |
| `JIRA_PROJECT_KEY` | — | Jira project key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |

Example with GitHub ingestion:

```bash
GITHUB_REPO=myorg/myrepo \
GITHUB_TOKEN=ghp_xxxx \
OLLAMA_MODEL=llama3.2:1b \
uvicorn enterprise_rag.api:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/query` | Ask a question |
| `POST` | `/upload` | Upload a file for ingestion |
| `POST` | `/ingest` | Trigger source ingestion |
| `GET` | `/health` | Component health check |
| `GET` | `/sources` | List registered connectors |
| `DELETE` | `/session/{id}` | Clear a conversation session |
| `GET` | `/docs` | Interactive API docs (Swagger UI) |

### POST /query

```json
// Request
{
  "session_id": "my-session",
  "query": "What does the authenticate_user function do?"
}

// Headers
Authorization: Bearer alice:engineering

// Response
{
  "answer": "The authenticate_user function validates credentials [1] and returns a JWT token [2].",
  "citations": [
    {
      "number": 1,
      "source_type": "github",
      "document_title": "auth.py",
      "document_url": "https://github.com/org/repo/blob/HEAD/auth.py",
      "excerpt": "def authenticate_user(username, password): ..."
    }
  ],
  "grounding_score": 0.95,
  "correlation_id": "uuid"
}
```

### POST /upload

```
POST /upload
Content-Type: multipart/form-data

file: <file>
permission_tags: engineering,internal
```

Supported formats: `.txt`, `.md`, `.rst`, `.html`, `.py`, `.js`, `.ts`, `.java`

---

## Authentication

The system uses Bearer token authentication. Token format:

```
Authorization: Bearer <username>:<role1>,<role2>
```

Examples:
- `Bearer alice:engineering` — user alice with engineering role
- `Bearer bob:hr,finance` — user bob with hr and finance roles

Roles control which documents are retrievable. Without pre-configured roles, the system defaults to allowing access to all content.

---

## Project Structure

```
enterprise-rag/
├── src/enterprise_rag/
│   ├── api.py                  # FastAPI application
│   ├── models.py               # Data models (dataclasses)
│   ├── vector_store.py         # ChromaDB vector store
│   ├── graph_store.py          # Neo4j graph store
│   ├── retriever.py            # Hybrid search + RRF
│   ├── generator.py            # Ollama LLM wrapper
│   ├── citation_engine.py      # Citation mapping + grounding score
│   ├── access_controller.py    # RBAC
│   ├── conversation_manager.py # Session management
│   ├── query_rewriter.py       # Query expansion
│   ├── ast_parser.py           # AST-based code parsing
│   ├── logging.py              # Structured logging
│   ├── ingestion/
│   │   ├── pipeline.py         # Ingestion orchestrator
│   │   ├── docs_connector.py   # File/URL connector
│   │   ├── github_connector.py # GitHub API connector
│   │   └── jira_connector.py   # Jira REST API connector
│   └── static/
│       └── index.html          # Web UI
├── tests/
│   ├── unit/                   # Unit + property-based tests
│   ├── integration/            # API integration tests
│   └── smoke/                  # Smoke tests
├── pyproject.toml
└── README.md
```

---

## Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest tests/ --cov=enterprise_rag
```

---

## Ingesting Content

### Upload a file (UI)

1. Open http://localhost:8000
2. Drag a file into the left sidebar upload area
3. Wait for "✓ indexed!" confirmation
4. Ask questions about it

### Connect a GitHub repo

Set env vars and restart, or use the sidebar form in the UI:

```bash
GITHUB_REPO=owner/repo GITHUB_TOKEN=ghp_xxx uvicorn enterprise_rag.api:app ...
```

Then trigger ingestion:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_id": "github", "full": true}'
```

### Connect Jira

```bash
JIRA_URL=https://yourorg.atlassian.net \
JIRA_USERNAME=you@company.com \
JIRA_TOKEN=your_api_token \
JIRA_PROJECT_KEY=ENG \
uvicorn enterprise_rag.api:app ...
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Vector store | ChromaDB |
| Graph store | Neo4j |
| LLM inference | Ollama |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Session store | In-memory (Redis + Postgres optional) |
| Code parsing | Python `ast` + tree-sitter |
| Testing | pytest + Hypothesis (property-based) |

---

## License

MIT
