# Design Document: Groq Migration and Cleanup

## Overview

This design document specifies the technical approach for migrating the Enterprise RAG system from a dual-provider architecture (Ollama + Groq) with RBAC to a simplified single-provider system using exclusively Groq API with open access.

### Goals

- Remove all Ollama dependencies and code
- Simplify to single LLM provider (Groq only)
- Remove authentication and authorization systems
- Clean up workspace by removing unused files
- Fix frontend UI layout issues
- Streamline configuration and error handling

### Non-Goals

- Adding new features beyond migration
- Maintaining backward compatibility with Ollama
- Preserving authentication capabilities
- Supporting multiple LLM providers

## Architecture

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (index.html)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Token Input  │  │ Ollama Btn   │  │  Groq Btn    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Authentication Layer (Bearer Token Validation)      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Access Controller (RBAC)                            │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Generator Factory (Provider Selection)              │   │
│  │    ├─ Generator (Ollama)                             │   │
│  │    └─ GroqGenerator (Groq API)                       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Query Rewriter (Ollama-based)                       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Retriever (with Access Filtering)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌──────────────┐    ┌──────────────┐
            │   Ollama     │    │  Groq API    │
            │  (Local)     │    │  (Cloud)     │
            └──────────────┘    └──────────────┘
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (index.html)                 │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ File Upload  │  │  Groq Model  │                         │
│  │   (Fixed)    │  │   Selector   │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GroqGenerator (Direct Instantiation)                │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Query Rewriter (Groq-based)                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Retriever (No Access Filtering)                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │  Groq API    │
                      │  (Cloud)     │
                      └──────────────┘
```

### Key Architectural Changes

1. **Single Provider**: Only GroqGenerator, no provider selection logic
2. **No Authentication**: Remove Bearer token validation and AccessController
3. **Open Access**: All users can access all indexed content
4. **Simplified Dependencies**: Fewer components in dependency injection
5. **Direct API Calls**: No fallback logic or provider switching

## Components and Interfaces

### 1. GroqGenerator

**Location**: `src/enterprise_rag/groq_generator.py`

**Status**: Keep (already exists, no changes needed)

**Interface**:
```python
class GroqGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context_window: int = 4096,
    ) -> None: ...
    
    def generate(
        self,
        query: str,
        chunks: List[ScoredChunk],
        history: List[Turn],
        stream: bool = True,
    ) -> Iterator[str] | str: ...
```

**Responsibilities**:
- Generate answers using Groq API
- Handle streaming and non-streaming responses
- Manage context window truncation
- Raise appropriate errors for API failures

### 2. Generator (Ollama)

**Location**: `src/enterprise_rag/generator.py`

**Status**: DELETE

**Rationale**: Ollama support is being completely removed

### 3. QueryRewriter

**Location**: `src/enterprise_rag/query_rewriter.py`

**Status**: Modify (migrate to Groq API)

**Current Interface**:
```python
class QueryRewriter:
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: float = 5.0,
    ) -> None: ...
```

**New Interface**:
```python
class QueryRewriter:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-8b-instant",
        timeout: float = 5.0,
    ) -> None: ...
    
    def rewrite(
        self,
        query: str,
        history: List[Turn],
        max_variants: int = 3,
    ) -> List[str]: ...
```

**Changes**:
- Replace `ollama_url` parameter with `api_key`
- Update API endpoint to `https://api.groq.com/openai/v1/chat/completions`
- Use OpenAI-compatible chat format instead of Ollama format
- Maintain same fallback behavior (return original query on error)

### 4. AccessController

**Location**: `src/enterprise_rag/access_controller.py`

**Status**: DELETE

**Rationale**: Authentication and authorization are being removed

### 5. API Layer

**Location**: `src/enterprise_rag/api.py`

**Status**: Modify (extensive changes)

**Changes**:

1. **Remove Imports**:
   - `from enterprise_rag.generator import Generator`
   - `from enterprise_rag.access_controller import AccessController`

2. **Remove Functions**:
   - `_make_access_controller()`
   - `get_access_controller()`
   - `_extract_token()`
   - `_make_generator()` (replace with simpler version)

3. **Remove Endpoints**:
   - `GET /llm-config`
   - `POST /llm-config`

4. **Modify Endpoints**:
   - `POST /query`: Remove `authorization` parameter, remove `ac` dependency
   - `GET /health`: Remove Ollama health check

5. **Simplify Generator Creation**:
```python
def get_generator() -> GroqGenerator:
    global _generator
    if _generator is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="Groq API key not configured"
            )
        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        _generator = GroqGenerator(
            api_key=api_key,
            model=model,
            context_window=4096
        )
    return _generator
```

6. **Update Access Filter**:
```python
# In query_endpoint, replace token validation with:
access_filter = AccessFilter(
    permitted_source_ids=[],
    permitted_tags=["engineering", "public", "internal", "docs", "github", "jira", "all"],
)
```

### 6. Retriever

**Location**: `src/enterprise_rag/retriever.py`

**Status**: Keep (no changes needed)

**Note**: The retriever already accepts an AccessFilter parameter. We'll just pass a permissive filter from the API layer.

### 7. Frontend

**Location**: `src/enterprise_rag/static/index.html`

**Status**: Modify (UI fixes and cleanup)

**Changes**:

1. **Remove Elements**:
   - `#token-bar` (entire section)
   - `#token-input`
   - `#session-display`
   - Ollama provider button
   - Provider switching logic

2. **Fix Layout**:
```css
#sidebar {
    width: 360px;
    min-width: 360px;
    max-width: 360px;
    /* ... */
}

#chat-panel {
    flex: 1;
    min-width: 0;
    /* ... */
}

.bubble {
    max-width: calc(100% - 60px);
    /* ... */
}
```

3. **Update LLM Settings**:
   - Show only Groq provider (no toggle)
   - Display model dropdown with Groq models only
   - Remove `loadLLMConfig()` and `selectProvider()` functions

4. **Remove Authorization Headers**:
```javascript
// Remove authHeaders() function
// Update all fetch calls to remove authorization headers
```

## Data Models

### AccessFilter

**Location**: `src/enterprise_rag/models.py`

**Status**: Keep (no changes)

**Usage Change**: Always instantiate with all permission tags enabled:

```python
AccessFilter(
    permitted_source_ids=[],
    permitted_tags=["engineering", "public", "internal", "docs", "github", "jira", "all"],
)
```

### Configuration Models (Remove)

The following Pydantic models will be removed from `api.py`:

- `LLMConfigRequest`
- `LLMConfigResponse`

## Error Handling

### Groq API Errors

**Error Mapping**:

| HTTP Status | Error Type | Message |
|-------------|------------|---------|
| 401 | ConnectionError | "Invalid Groq API key" |
| 429 | ConnectionError | "Groq rate limit exceeded" |
| Timeout | TimeoutError | "Groq API timed out" |
| Connection | ConnectionError | "Groq API unavailable" |

**Implementation**:

```python
# In GroqGenerator._stream() and _complete()
try:
    response = requests.post(...)
    response.raise_for_status()
except requests.exceptions.Timeout as exc:
    raise TimeoutError("Groq API timed out") from exc
except requests.exceptions.ConnectionError as exc:
    raise ConnectionError("Groq API unavailable") from exc
except requests.exceptions.HTTPError as exc:
    if exc.response.status_code == 401:
        raise ConnectionError("Invalid Groq API key") from exc
    elif exc.response.status_code == 429:
        raise ConnectionError("Groq rate limit exceeded") from exc
    raise ConnectionError(f"Groq API error: {exc}") from exc
```

### API Endpoint Errors

**Error Responses**:

```python
# Missing API key
if not os.environ.get("GROQ_API_KEY"):
    raise HTTPException(
        status_code=503,
        detail="Groq API key not configured"
    )

# Generator errors
except ConnectionError as exc:
    raise HTTPException(
        status_code=503,
        detail="Groq API unavailable"
    )

except TimeoutError as exc:
    raise HTTPException(
        status_code=504,
        detail="Groq API timed out"
    )
```

### Frontend Error Display

**Error Messages**:

```javascript
// Connection errors
if (resp.status === 503) {
    appendErrorBubble('Groq API is unavailable. Please try again later.');
}

// Timeout errors
if (resp.status === 504) {
    appendErrorBubble('Groq API timed out. Please try again.');
}

// Streaming errors
if (evt.error === 'unavailable') {
    appendErrorBubble('Groq API connection lost during streaming.');
}
```

## Testing Strategy

### Unit Tests

**Test Files to Update**:
- `tests/test_api.py`
- `tests/test_generator.py` (delete or rename to test_groq_generator.py)
- `tests/test_query_rewriter.py`
- `tests/test_access_controller.py` (delete)

**Key Test Cases**:

1. **GroqGenerator Tests**:
   - Test successful generation (streaming and non-streaming)
   - Test API key validation
   - Test rate limit handling
   - Test timeout handling
   - Test context window truncation

2. **QueryRewriter Tests**:
   - Test successful query rewriting with Groq
   - Test fallback to original query on error
   - Test timeout handling
   - Test history incorporation

3. **API Tests**:
   - Test query endpoint without authentication
   - Test health endpoint without Ollama
   - Test error responses for missing API key
   - Test streaming response handling

4. **Integration Tests**:
   - Test end-to-end query flow with Groq
   - Test file upload and retrieval
   - Test session management

### Manual Testing Checklist

- [ ] Start application with GROQ_API_KEY set
- [ ] Verify health check shows only ChromaDB status
- [ ] Upload a test file
- [ ] Send a query and verify streaming response
- [ ] Verify citations are displayed correctly
- [ ] Test UI layout (sidebar + chat panel)
- [ ] Verify no Ollama references in UI
- [ ] Test error handling (invalid API key, timeout)
- [ ] Verify no authentication required

## Migration Strategy

### Phase 1: Remove Authentication System

**Files to Modify**:
- `src/enterprise_rag/api.py`

**Steps**:
1. Remove `_extract_token()` function
2. Remove `get_access_controller()` dependency
3. Remove `authorization` parameter from `query_endpoint`
4. Update `access_filter` to use permissive tags
5. Remove authentication-related imports

**Validation**:
- API accepts requests without Authorization header
- All content is accessible to all users

### Phase 2: Remove Ollama Completely

**Files to Delete**:
- `src/enterprise_rag/generator.py`
- `src/enterprise_rag/access_controller.py`

**Files to Modify**:
- `src/enterprise_rag/api.py`

**Steps**:
1. Delete `generator.py`
2. Delete `access_controller.py`
3. Remove Generator import from `api.py`
4. Remove `_make_generator()` provider selection logic
5. Simplify `get_generator()` to only create GroqGenerator
6. Remove LLM config endpoints
7. Update health check to remove Ollama

**Validation**:
- No imports of Generator class
- No references to "ollama" in Python files
- Health check doesn't check Ollama

### Phase 3: Update Query Rewriter

**Files to Modify**:
- `src/enterprise_rag/query_rewriter.py`

**Steps**:
1. Replace `ollama_url` with `api_key` parameter
2. Update API endpoint to Groq
3. Change request format to OpenAI-compatible
4. Update error handling for Groq API
5. Maintain fallback behavior

**Validation**:
- Query rewriting works with Groq API
- Fallback to original query on error
- No Ollama references

### Phase 4: Clean Up Workspace

**Files to Remove** (20+ files):
- Unused test files
- Outdated documentation
- Redundant configuration files
- `.hypothesis` directory (if not actively used)

**Steps**:
1. Identify unused files in root directory
2. Create backup of files to be removed
3. Delete unused files
4. Update documentation to reflect changes

**Validation**:
- All actively used files preserved
- No broken imports or references
- Documentation is current

### Phase 5: Fix UI Layout

**Files to Modify**:
- `src/enterprise_rag/static/index.html`

**Steps**:
1. Remove token-bar element
2. Fix sidebar width (360px fixed)
3. Fix chat-panel flex layout
4. Remove Ollama provider button
5. Update LLM settings to show only Groq
6. Remove provider switching JavaScript
7. Remove authorization headers from fetch calls

**Validation**:
- Sidebar and chat panel don't overlap
- File upload area visible and functional
- Only Groq provider shown
- No authentication UI elements

## Configuration Changes

### Environment Variables

**Remove**:
- `OLLAMA_URL`
- `OLLAMA_MODEL`

**Keep/Add**:
- `GROQ_API_KEY` (required)
- `GROQ_MODEL` (optional, default: "llama-3.1-8b-instant")

**Example `.env`**:
```bash
# Groq API Configuration
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.1-8b-instant

# Vector Store
CHROMA_PATH=./chroma_data

# Logging
LOG_BACKEND=file
LOG_DIR=logs

# Optional: Data Source Connectors
GITHUB_REPO=owner/repo
GITHUB_TOKEN=ghp_...
DOCS_PATH=./docs
JIRA_URL=https://yourorg.atlassian.net
JIRA_TOKEN=...
```

### Application Configuration

**Remove from `api.py`**:
```python
# Remove this entire section
_llm_config = {
    "provider": "groq" if os.environ.get("GROQ_API_KEY") else "ollama",
    "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant") if os.environ.get("GROQ_API_KEY") else os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b"),
}
```

**Replace with**:
```python
# No runtime configuration needed - read from env vars directly
```

## Deployment Considerations

### Prerequisites

1. **Groq API Key**: Must be set in environment
2. **Remove Ollama**: Uninstall Ollama service if running
3. **Update Documentation**: README, deployment guides

### Startup Validation

**Add to application startup**:
```python
@app.on_event("startup")
async def validate_config():
    if not os.environ.get("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not set - application will not function")
        # Don't raise error, let health check report it
```

### Health Check Updates

**Remove Ollama check**:
```python
# DELETE this section from health_endpoint
ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
try:
    import requests as _requests
    resp = _requests.get(f"{ollama_url}/api/tags", timeout=5)
    # ...
except Exception as exc:
    # ...
```

**Keep only**:
- ChromaDB health check
- Session store health check

### Monitoring

**Key Metrics**:
- Groq API response time
- Groq API error rate
- Query success rate
- Token usage (if tracked by Groq)

**Alerts**:
- Groq API unavailable
- High error rate (>5%)
- Slow response time (>5s)

## Rollback Plan

### If Migration Fails

1. **Restore from Git**: Revert to pre-migration commit
2. **Restore Environment**: Set OLLAMA_URL and OLLAMA_MODEL
3. **Restart Services**: Restart Ollama and application

### Backup Strategy

1. **Code Backup**: Create git branch before migration
2. **Data Backup**: Backup ChromaDB data directory
3. **Config Backup**: Save current .env file

## Success Criteria

### Functional Requirements

- [ ] Application starts with only GROQ_API_KEY set
- [ ] Queries return answers using Groq API
- [ ] Streaming responses work correctly
- [ ] File upload and indexing work
- [ ] No authentication required
- [ ] All content accessible to all users

### Code Quality

- [ ] No references to "ollama" in Python files
- [ ] No references to "Generator" class (only GroqGenerator)
- [ ] No unused imports
- [ ] All tests pass
- [ ] No linting errors

### UI/UX

- [ ] Sidebar and chat panel don't overlap
- [ ] Only Groq provider shown
- [ ] No token input field
- [ ] Error messages reference Groq (not Ollama)
- [ ] Layout works on mobile (responsive)

### Documentation

- [ ] README updated
- [ ] Environment variables documented
- [ ] Deployment guide updated
- [ ] Architecture diagrams updated

## Timeline

**Estimated Duration**: 2-3 days

**Phase Breakdown**:
- Phase 1 (Auth Removal): 4 hours
- Phase 2 (Ollama Removal): 6 hours
- Phase 3 (Query Rewriter): 3 hours
- Phase 4 (Workspace Cleanup): 2 hours
- Phase 5 (UI Fixes): 4 hours
- Testing & Validation: 4 hours

**Total**: ~23 hours of development + testing

## Risks and Mitigation

### Risk 1: Groq API Availability

**Impact**: High - Application won't function without Groq

**Mitigation**:
- Implement robust error handling
- Add retry logic with exponential backoff
- Monitor Groq API status
- Have fallback plan (restore Ollama if needed)

### Risk 2: Breaking Changes

**Impact**: Medium - Existing users may be affected

**Mitigation**:
- Thorough testing before deployment
- Clear communication about changes
- Provide migration guide
- Keep rollback plan ready

### Risk 3: Data Loss

**Impact**: Low - No data model changes

**Mitigation**:
- Backup ChromaDB before migration
- Test with copy of production data
- Verify data integrity after migration

### Risk 4: UI Regressions

**Impact**: Medium - Layout issues may affect usability

**Mitigation**:
- Test on multiple screen sizes
- Test on different browsers
- Get user feedback before full rollout
- Keep old UI code in git history

## Appendix

### Files to Delete

**Root Directory** (20+ files to identify and remove):
- Unused test files
- Outdated documentation
- Redundant configuration files
- `.hypothesis` directory (if not actively used)

**Source Files**:
- `src/enterprise_rag/generator.py`
- `src/enterprise_rag/access_controller.py`

### Files to Modify

**Backend**:
- `src/enterprise_rag/api.py` (extensive changes)
- `src/enterprise_rag/query_rewriter.py` (migrate to Groq)

**Frontend**:
- `src/enterprise_rag/static/index.html` (UI fixes and cleanup)

**Configuration**:
- `.env` (remove Ollama vars, document Groq vars)
- `README.md` (update documentation)

### Files to Keep Unchanged

- `src/enterprise_rag/groq_generator.py`
- `src/enterprise_rag/retriever.py`
- `src/enterprise_rag/vector_store.py`
- `src/enterprise_rag/citation_engine.py`
- `src/enterprise_rag/conversation_manager.py`
- `src/enterprise_rag/models.py`
- All ingestion pipeline files
- All connector files

### Reference Links

- [Groq API Documentation](https://console.groq.com/docs)
- [OpenAI API Compatibility](https://console.groq.com/docs/openai)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
