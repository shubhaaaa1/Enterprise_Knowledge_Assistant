# Implementation Plan: Groq Migration and Cleanup

## Overview

This plan implements a complete migration from dual-provider (Ollama + Groq) architecture with RBAC to a simplified single-provider (Groq-only) system with open access. The implementation follows a 5-phase approach: remove authentication, delete Ollama code, migrate query rewriter, clean workspace, and fix UI layout.

## Tasks

- [x] 1. Phase 1: Remove Authentication System
  - [x] 1.1 Remove authentication from API endpoints
    - Remove `_extract_token()` helper function from `src/enterprise_rag/api.py`
    - Remove `authorization` parameter from `query_endpoint` function signature
    - Remove `ac: AccessController` dependency from `query_endpoint`
    - Remove `Depends(get_access_controller)` from endpoint
    - Update `access_filter` to use permissive tags: `["engineering", "public", "internal", "docs", "github", "jira", "all"]`
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 17.10_
  
  - [x] 1.2 Remove AccessController dependency injection
    - Remove `get_access_controller()` function from `src/enterprise_rag/api.py`
    - Remove `_make_access_controller()` function from `src/enterprise_rag/api.py`
    - Remove `_access_controller` module-level variable from `src/enterprise_rag/api.py`
    - Remove `from enterprise_rag.access_controller import AccessController` import
    - _Requirements: 4.2, 12.1, 12.2, 17.1, 17.2, 17.3, 17.4, 17.9, 17.10_
  
  - [x] 1.3 Delete access_controller.py file
    - Delete `src/enterprise_rag/access_controller.py` file completely
    - _Requirements: 4.1, 4.2_

- [x] 2. Phase 2: Remove Ollama Completely
  - [x] 2.1 Delete Ollama generator file
    - Delete `src/enterprise_rag/generator.py` file completely
    - _Requirements: 1.1, 1.2_
  
  - [x] 2.2 Remove Ollama imports from api.py
    - Remove `from enterprise_rag.generator import Generator` import from `src/enterprise_rag/api.py`
    - Update type hints to remove `Generator` from union types (change `Generator | GroqGenerator` to just `GroqGenerator`)
    - _Requirements: 1.2, 1.3, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_
  
  - [x] 2.3 Simplify generator creation logic
    - Replace `_make_generator()` function in `src/enterprise_rag/api.py` with simplified version that only creates GroqGenerator
    - Remove provider selection logic and Ollama fallback
    - Update function to read `GROQ_API_KEY` and `GROQ_MODEL` from environment
    - Raise HTTPException(503) if `GROQ_API_KEY` is not set
    - Update `get_generator()` return type to `GroqGenerator` only
    - _Requirements: 1.8, 1.9, 2.1, 2.2, 2.3, 2.4, 2.12, 2.13, 10.2, 10.3, 12.5_
  
  - [x] 2.4 Remove LLM configuration endpoints
    - Remove `GET /llm-config` endpoint from `src/enterprise_rag/api.py`
    - Remove `POST /llm-config` endpoint from `src/enterprise_rag/api.py`
    - Remove `LLMConfigRequest` Pydantic model
    - Remove `LLMConfigResponse` Pydantic model
    - Remove `_llm_config` module-level dictionary
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_
  
  - [x] 2.5 Update health check to remove Ollama
    - Remove Ollama health check code from `health_endpoint` in `src/enterprise_rag/api.py`
    - Remove HTTP request to Ollama `/api/tags` endpoint
    - Remove Ollama component from health check response
    - Keep only ChromaDB and session_store health checks
    - _Requirements: 1.10, 13.1, 13.2, 13.3, 13.4, 13.5, 13.9, 13.10_
  
  - [x] 2.6 Update error messages to reference Groq
    - Update ConnectionError messages in `query_endpoint` to say "Groq API unavailable" instead of "Ollama service unavailable"
    - Update TimeoutError messages to say "Groq API timed out" instead of "Generator timed out"
    - Remove any remaining error messages mentioning "Ollama"
    - _Requirements: 1.11, 9.1, 9.2, 9.5, 9.6, 9.9, 9.10_

- [x] 3. Phase 3: Update Query Rewriter to Use Groq
  - [x] 3.1 Migrate QueryRewriter to Groq API
    - Update `src/enterprise_rag/query_rewriter.py` constructor to accept `api_key` parameter instead of `ollama_url`
    - Change API endpoint to `https://api.groq.com/openai/v1/chat/completions`
    - Update request format to OpenAI-compatible chat format
    - Add Authorization header with Bearer token from `api_key`
    - Update error handling to catch Groq-specific errors (401, 429)
    - Maintain fallback behavior (return original query on error)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_
  
  - [x] 3.2 Update QueryRewriter instantiation in api.py
    - Update code that creates QueryRewriter instance to pass `api_key` from `GROQ_API_KEY` environment variable
    - Remove `ollama_url` parameter
    - Update to use `GROQ_MODEL` environment variable
    - _Requirements: 3.2, 3.3_

- [x] 4. Phase 4: Clean Up Workspace
  - [x] 4.1 Identify and document unused files
    - List all files in root directory
    - Identify at least 20 unused files (test files, outdated docs, redundant configs)
    - Create list of files to be removed
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.10_
  
  - [x] 4.2 Remove unused files from workspace
    - Delete identified unused files
    - Preserve all actively used source files in `src/enterprise_rag/`
    - Preserve `requirements.txt`, `pyproject.toml`, `.env`, `README.md`
    - Remove `.hypothesis` directory if not actively used
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9_
  
  - [x] 4.3 Update environment variable documentation
    - Remove `OLLAMA_URL` and `OLLAMA_MODEL` from `.env` file
    - Document `GROQ_API_KEY` as required
    - Document `GROQ_MODEL` as optional with default "llama-3.1-8b-instant"
    - Update README.md to remove Ollama setup instructions
    - Update README.md to add Groq API key setup instructions
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10, 19.7, 19.8_

- [x] 5. Phase 5: Fix UI Layout and Remove Ollama References
  - [x] 5.1 Remove authentication UI elements
    - Remove `#token-bar` element from `src/enterprise_rag/static/index.html`
    - Remove `#token-input` element
    - Remove `#session-display` element
    - Remove `getToken()` JavaScript function
    - Remove `authHeaders()` JavaScript function
    - Remove Authorization headers from all fetch calls
    - _Requirements: 4.11, 4.12, 4.13, 4.14, 4.15, 16.5_
  
  - [x] 5.2 Fix sidebar and chat panel layout
    - Set `#sidebar` width to fixed 360px in CSS
    - Set `#sidebar` min-width to 360px
    - Set `#sidebar` max-width to 360px
    - Set `#chat-panel` to `flex: 1` and `min-width: 0`
    - Ensure `.bubble` has `max-width: calc(100% - 60px)` to prevent overflow
    - Test responsive layout on mobile (sidebar stacks above chat panel)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10_
  
  - [x] 5.3 Remove Ollama provider UI elements
    - Remove Ollama provider button from LLM Settings section
    - Remove `selectProvider()` JavaScript function
    - Remove provider switching event handlers
    - Remove `data-provider="ollama"` button element
    - Update provider buttons to show only Groq
    - _Requirements: 1.12, 1.13, 1.14, 8.1, 8.2, 8.3, 8.7, 8.10, 16.1, 16.2, 16.3, 16.4, 16.6_
  
  - [x] 5.4 Update LLM settings to Groq-only
    - Remove `loadLLMConfig()` JavaScript function
    - Remove `updateModelDropdown()` logic for Ollama models
    - Hardcode Groq models in model dropdown: ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
    - Remove provider parameter from API requests
    - Update status message to show "✓ Using Groq (model-name)"
    - Remove conditional logic based on provider selection
    - _Requirements: 8.4, 8.5, 8.6, 8.8, 8.9, 10.8, 10.9, 10.10, 16.5, 16.7, 16.8, 16.9, 16.10_
  
  - [x] 5.5 Update frontend error messages
    - Update error handling to display "Groq API unavailable" instead of Ollama errors
    - Update timeout errors to say "Groq API timed out"
    - Update streaming error messages to reference Groq
    - Remove any remaining "Ollama" references from JavaScript
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.9, 18.7, 18.8, 18.9_
  
  - [x] 5.6 Remove Ollama from health modal
    - Remove Ollama component row from health modal display
    - Update `renderHealthModal()` to not show Ollama status
    - Ensure health modal only shows ChromaDB and session_store
    - _Requirements: 13.6, 13.7, 13.8_

- [x] 6. Checkpoint - Verify migration completeness
  - Ensure all tests pass
  - Verify no "ollama" references remain in Python files (case-insensitive grep)
  - Verify no "Generator" class imports (only GroqGenerator)
  - Verify application starts with only GROQ_API_KEY set
  - Test query endpoint without authentication
  - Test streaming responses with Groq
  - Test UI layout (sidebar + chat panel don't overlap)
  - Verify only Groq provider shown in UI
  - Ask the user if questions arise

## Notes

- All tasks reference specific requirements for traceability
- The migration is designed to be executed sequentially by phase
- Each phase builds on the previous phase
- No backward compatibility with Ollama is maintained
- The system will fail fast with clear errors if GROQ_API_KEY is missing
- UI changes ensure proper layout on desktop and mobile devices
