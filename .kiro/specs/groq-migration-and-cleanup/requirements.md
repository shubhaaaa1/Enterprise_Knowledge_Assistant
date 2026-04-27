# Requirements Document

## Introduction

This document specifies requirements for refactoring the Enterprise RAG system to use exclusively Groq API for LLM operations, removing all authentication/authorization systems, and cleaning up the workspace. The system will transition from a dual-provider architecture (Ollama + Groq) with RBAC to a simplified single-provider system with open access.

## Glossary

- **System**: The Enterprise RAG application (backend + frontend)
- **Ollama**: Local LLM provider being completely removed
- **Groq**: Cloud-based LLM API provider (sole provider after migration)
- **Generator**: Component responsible for answer generation
- **Query_Rewriter**: Component that expands user queries for better retrieval
- **Access_Controller**: RBAC component being removed
- **RBAC**: Role-Based Access Control system being removed
- **API**: FastAPI backend application
- **Frontend**: Single-page HTML application (index.html)
- **Workspace**: Root directory containing all project files
- **Bearer_Token**: Authentication token mechanism being removed
- **LLM_Config**: Runtime configuration for LLM provider selection

## Requirements

### Requirement 1: Complete Ollama Removal

**User Story:** As a system administrator, I want all traces of Ollama completely removed from the codebase, so that the system has zero dependencies on local LLM infrastructure.

#### Acceptance Criteria

1. THE System SHALL NOT contain the file src/enterprise_rag/generator.py
2. THE System SHALL NOT contain any import statements referencing "generator" module
3. THE System SHALL NOT contain any import statements referencing "Generator" class from generator module
4. THE System SHALL NOT contain the string "ollama" in any Python source file (case-insensitive)
5. THE System SHALL NOT contain the string "OLLAMA_URL" in any source file
6. THE System SHALL NOT contain the string "OLLAMA_MODEL" in any source file
7. THE System SHALL NOT reference environment variable OLLAMA_URL in any configuration
8. THE System SHALL NOT reference environment variable OLLAMA_MODEL in any configuration
9. THE API SHALL NOT contain any fallback logic to Ollama provider
10. THE API SHALL NOT contain any health check code for Ollama service
11. THE API SHALL NOT contain any error messages mentioning "Ollama"
12. THE Frontend SHALL NOT contain any UI elements for Ollama provider selection
13. THE Frontend SHALL NOT contain the string "ollama" in any HTML, CSS, or JavaScript code (case-insensitive)
14. THE Frontend SHALL NOT display "Ollama" as a provider option
15. THE System SHALL NOT contain any comments referencing Ollama
16. THE System SHALL NOT contain any function parameters named with "ollama" prefix
17. THE System SHALL NOT contain any variable names containing "ollama" substring
18. THE LLM_Config SHALL NOT include "ollama" as a valid provider option
19. THE System SHALL NOT contain any timeout or connection error messages specific to Ollama
20. THE System SHALL use ONLY Groq API for all LLM operations

### Requirement 2: Groq API Migration

**User Story:** As a developer, I want the system to use exclusively Groq API for all LLM operations, so that I can leverage fast cloud-based inference.

#### Acceptance Criteria

1. THE API SHALL use GroqGenerator class for all answer generation operations
2. THE API SHALL read GROQ_API_KEY from environment variables
3. THE API SHALL read GROQ_MODEL from environment variables with default value "llama-3.1-8b-instant"
4. WHEN GROQ_API_KEY is not set, THEN THE System SHALL return HTTP 503 error with message "Groq API key not configured"
5. THE Generator SHALL make HTTP requests to https://api.groq.com/openai/v1/chat/completions
6. THE Generator SHALL include Authorization header with Bearer token from GROQ_API_KEY
7. THE Generator SHALL support streaming mode for token-by-token responses
8. THE Generator SHALL support non-streaming mode for complete responses
9. WHEN Groq API returns 401 error, THEN THE Generator SHALL raise ConnectionError with message "Invalid Groq API key"
10. WHEN Groq API returns 429 error, THEN THE Generator SHALL raise ConnectionError with message "Groq rate limit exceeded"
11. THE Generator SHALL use timeout of 30 seconds for Groq API requests
12. THE API SHALL NOT create Generator instances (Ollama-based)
13. THE API SHALL remove _make_generator function logic for Ollama fallback

### Requirement 3: Query Rewriter Groq Migration

**User Story:** As a developer, I want the query rewriter to use Groq API instead of Ollama, so that query expansion uses the same infrastructure as answer generation.

#### Acceptance Criteria

1. THE Query_Rewriter SHALL use Groq API for query rewriting operations
2. THE Query_Rewriter SHALL read GROQ_API_KEY from environment variables
3. THE Query_Rewriter SHALL read GROQ_MODEL from environment variables
4. THE Query_Rewriter SHALL make HTTP requests to https://api.groq.com/openai/v1/chat/completions
5. THE Query_Rewriter SHALL include Authorization header with Bearer token
6. THE Query_Rewriter SHALL NOT reference ollama_url parameter
7. THE Query_Rewriter SHALL NOT make requests to Ollama endpoints
8. WHEN Groq API request fails, THEN THE Query_Rewriter SHALL return original query as fallback
9. THE Query_Rewriter SHALL maintain same interface (rewrite method signature)
10. THE Query_Rewriter SHALL maintain same timeout behavior (5 seconds)

### Requirement 4: Authentication System Removal

**User Story:** As a system administrator, I want all authentication and authorization removed, so that the system allows open access to all content.

#### Acceptance Criteria

1. THE System SHALL NOT contain the file src/enterprise_rag/access_controller.py
2. THE API SHALL NOT import AccessController class
3. THE API SHALL NOT validate Bearer tokens on any endpoint
4. THE API SHALL NOT extract Authorization header from requests
5. THE API SHALL NOT call validate_token method
6. THE API SHALL NOT call resolve_roles method
7. THE API SHALL NOT call build_access_filter method
8. THE API SHALL remove _extract_token helper function
9. THE API SHALL remove get_access_controller dependency injection
10. THE API SHALL remove authorization parameter from query_endpoint
11. THE Frontend SHALL NOT display token input field
12. THE Frontend SHALL NOT send Authorization header in API requests
13. THE Frontend SHALL remove token-bar element completely
14. THE Frontend SHALL remove token-input element completely
15. THE Frontend SHALL remove session-display element completely

### Requirement 5: Access Filtering Removal

**User Story:** As a developer, I want all permission-based filtering removed from retrieval, so that all users can access all indexed content.

#### Acceptance Criteria

1. THE API SHALL create AccessFilter with all permission tags enabled
2. THE AccessFilter SHALL include tags: ["engineering", "public", "internal", "docs", "github", "jira", "all"]
3. THE API SHALL NOT filter chunks based on user roles
4. THE API SHALL NOT filter chunks based on source permissions
5. THE Retriever SHALL return all matching chunks regardless of permission tags
6. THE API SHALL NOT log access control decisions
7. THE API SHALL remove all RBAC-related logging statements
8. THE API SHALL NOT call log_access_decision method

### Requirement 6: Workspace Cleanup

**User Story:** As a developer, I want unnecessary files removed from the workspace, so that the codebase is minimal and maintainable.

#### Acceptance Criteria

1. THE System SHALL remove at least 20 unused files from root directory
2. THE System SHALL remove unused test files not actively maintained
3. THE System SHALL remove outdated documentation files
4. THE System SHALL remove redundant configuration files
5. THE System SHALL preserve all actively used source files in src/enterprise_rag/
6. THE System SHALL preserve requirements.txt and pyproject.toml
7. THE System SHALL preserve .env file
8. THE System SHALL preserve README.md if it contains current information
9. THE System SHALL remove .hypothesis directory if not actively used
10. THE System SHALL document removed files in cleanup summary

### Requirement 7: Frontend UI Layout Fix

**User Story:** As a user, I want the frontend UI elements properly separated, so that file upload and chat interface don't overlap.

#### Acceptance Criteria

1. THE Frontend SHALL display sidebar and chat panel without overlap
2. THE Frontend SHALL ensure drop-zone element does not overlap with messages element
3. THE Frontend SHALL ensure input-bar element remains visible at bottom of chat panel
4. THE Frontend SHALL use flexbox layout to separate sidebar from chat panel
5. THE Frontend SHALL set sidebar width to fixed 360px
6. THE Frontend SHALL set chat-panel to flex: 1 to fill remaining space
7. THE Frontend SHALL ensure messages container has proper overflow-y: auto
8. THE Frontend SHALL ensure bubbles have max-width that prevents overflow
9. WHEN window width is less than 768px, THEN THE Frontend SHALL stack sidebar above chat panel
10. THE Frontend SHALL remove token-bar element to reduce UI clutter

### Requirement 8: Frontend Provider UI Cleanup

**User Story:** As a user, I want the frontend to show only Groq as the LLM provider, so that the UI reflects the simplified architecture.

#### Acceptance Criteria

1. THE Frontend SHALL display only Groq provider button in LLM Settings
2. THE Frontend SHALL remove Ollama provider button completely
3. THE Frontend SHALL remove provider selection toggle functionality
4. THE Frontend SHALL display model dropdown with only Groq models
5. THE Frontend SHALL include models: ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
6. THE Frontend SHALL remove selectProvider JavaScript function for Ollama
7. THE Frontend SHALL remove provider-btn elements for Ollama
8. THE Frontend SHALL update loadLLMConfig function to expect only Groq provider
9. THE Frontend SHALL display status message "✓ Using Groq (model-name)"
10. THE Frontend SHALL NOT send provider parameter in LLM config requests

### Requirement 9: Error Message Updates

**User Story:** As a developer, I want all error messages to reference Groq instead of Ollama, so that error messages are accurate and helpful.

#### Acceptance Criteria

1. WHEN Generator connection fails, THEN THE System SHALL return error message "Groq API unavailable"
2. WHEN Generator times out, THEN THE System SHALL return error message "Groq API timed out"
3. WHEN API key is invalid, THEN THE System SHALL return error message "Invalid Groq API key"
4. WHEN rate limit is exceeded, THEN THE System SHALL return error message "Groq rate limit exceeded"
5. THE System SHALL NOT contain any error messages mentioning "Ollama service"
6. THE System SHALL NOT contain any error messages mentioning "Cannot connect to Ollama"
7. THE Health endpoint SHALL NOT check Ollama service status
8. THE Health endpoint SHALL remove Ollama component from health check results
9. THE Frontend SHALL display Groq-specific error messages to users
10. THE API SHALL return HTTP 503 with detail "Groq API unavailable" when Groq is unreachable

### Requirement 10: LLM Configuration Simplification

**User Story:** As a developer, I want LLM configuration simplified to support only Groq, so that runtime configuration is straightforward.

#### Acceptance Criteria

1. THE API SHALL remove _llm_config dictionary with provider field
2. THE API SHALL remove provider selection logic from _make_generator function
3. THE API SHALL always instantiate GroqGenerator in get_generator function
4. THE API SHALL remove /llm-config GET endpoint
5. THE API SHALL remove /llm-config POST endpoint
6. THE API SHALL remove LLMConfigRequest model
7. THE API SHALL remove LLMConfigResponse model
8. THE Frontend SHALL remove LLM provider switching functionality
9. THE Frontend SHALL remove loadLLMConfig JavaScript function
10. THE Frontend SHALL remove selectProvider JavaScript function

### Requirement 11: Environment Variable Cleanup

**User Story:** As a system administrator, I want environment variables cleaned up to remove Ollama references, so that configuration is clear and minimal.

#### Acceptance Criteria

1. THE .env file SHALL NOT contain OLLAMA_URL variable
2. THE .env file SHALL NOT contain OLLAMA_MODEL variable
3. THE .env file SHALL contain GROQ_API_KEY variable
4. THE .env file SHALL contain GROQ_MODEL variable with default "llama-3.1-8b-instant"
5. THE System SHALL NOT read OLLAMA_URL from environment
6. THE System SHALL NOT read OLLAMA_MODEL from environment
7. THE Documentation SHALL NOT reference OLLAMA_URL configuration
8. THE Documentation SHALL NOT reference OLLAMA_MODEL configuration
9. THE System SHALL document GROQ_API_KEY as required environment variable
10. THE System SHALL document GROQ_MODEL as optional environment variable with default

### Requirement 12: Import Statement Cleanup

**User Story:** As a developer, I want all import statements cleaned up to remove Ollama references, so that the codebase has no dead imports.

#### Acceptance Criteria

1. THE API SHALL NOT import Generator class from enterprise_rag.generator
2. THE API SHALL import only GroqGenerator from enterprise_rag.groq_generator
3. THE API SHALL remove "from enterprise_rag.generator import Generator" statement
4. THE API SHALL update type hints to use only GroqGenerator type
5. THE API SHALL remove Generator from union types (Generator | GroqGenerator)
6. THE System SHALL NOT have any unused imports related to Ollama
7. THE Query_Rewriter SHALL NOT import any Ollama-specific modules
8. THE System SHALL pass import validation checks
9. THE System SHALL NOT have any ImportError exceptions related to generator module
10. THE API SHALL update get_generator return type to GroqGenerator only

### Requirement 13: Health Check Updates

**User Story:** As a system administrator, I want health checks updated to remove Ollama monitoring, so that health status reflects actual system dependencies.

#### Acceptance Criteria

1. THE Health endpoint SHALL NOT check Ollama service connectivity
2. THE Health endpoint SHALL NOT return Ollama component status
3. THE Health endpoint SHALL remove Ollama HTTP request to /api/tags
4. THE Health endpoint SHALL NOT import requests for Ollama health checks
5. THE Health endpoint SHALL return only ChromaDB and session_store components
6. THE Frontend SHALL NOT display Ollama status in health modal
7. THE Frontend SHALL remove Ollama component row from health display
8. WHEN health endpoint is called, THEN THE Response SHALL NOT include "ollama" in components list
9. THE Health endpoint SHALL return HTTP 200 with status "ok" when ChromaDB is healthy
10. THE Health endpoint SHALL return HTTP 200 with status "degraded" when ChromaDB is unhealthy

### Requirement 14: Code Comment Cleanup

**User Story:** As a developer, I want all code comments updated to remove Ollama references, so that documentation is accurate.

#### Acceptance Criteria

1. THE System SHALL NOT contain comments mentioning "Ollama"
2. THE System SHALL NOT contain comments mentioning "ollama_url"
3. THE System SHALL NOT contain comments mentioning "local LLM"
4. THE System SHALL update docstrings to reference Groq API
5. THE Generator docstring SHALL state "Generates grounded answers via Groq API"
6. THE Query_Rewriter docstring SHALL state "Rewrites queries using Groq API"
7. THE System SHALL NOT contain TODO comments about Ollama migration
8. THE System SHALL NOT contain comments about Ollama fallback logic
9. THE API docstring SHALL NOT mention Ollama service availability
10. THE System SHALL update all module-level docstrings to reflect Groq-only architecture

### Requirement 15: Test File Updates

**User Story:** As a developer, I want test files updated to remove Ollama mocking, so that tests reflect the new architecture.

#### Acceptance Criteria

1. THE Test files SHALL NOT mock Ollama HTTP endpoints
2. THE Test files SHALL mock Groq API endpoints instead
3. THE Test files SHALL NOT import Generator class from generator module
4. THE Test files SHALL import only GroqGenerator for testing
5. THE Test files SHALL NOT test Ollama fallback behavior
6. THE Test files SHALL NOT test provider switching logic
7. THE Test files SHALL remove fixtures for Ollama URL configuration
8. THE Test files SHALL update fixtures to provide GROQ_API_KEY
9. THE Test files SHALL NOT assert on Ollama-specific error messages
10. THE Test files SHALL assert on Groq-specific error messages

### Requirement 16: Frontend JavaScript Cleanup

**User Story:** As a developer, I want frontend JavaScript cleaned up to remove Ollama logic, so that client-side code is simplified.

#### Acceptance Criteria

1. THE Frontend SHALL remove selectProvider function completely
2. THE Frontend SHALL remove provider switching event handlers
3. THE Frontend SHALL remove provider-btn click handlers for Ollama
4. THE Frontend SHALL remove currentLLMConfig.provider field
5. THE Frontend SHALL remove provider parameter from API requests
6. THE Frontend SHALL remove updateModelDropdown logic for Ollama models
7. THE Frontend SHALL hardcode Groq models in model dropdown
8. THE Frontend SHALL remove conditional logic based on provider selection
9. THE Frontend SHALL remove provider validation in sendQuery function
10. THE Frontend SHALL simplify LLM settings UI to show only model selection

### Requirement 17: Dependency Injection Cleanup

**User Story:** As a developer, I want dependency injection simplified to remove AccessController, so that API endpoints have fewer dependencies.

#### Acceptance Criteria

1. THE API SHALL remove get_access_controller function
2. THE API SHALL remove _make_access_controller function
3. THE API SHALL remove _access_controller module-level variable
4. THE API SHALL remove ac parameter from query_endpoint
5. THE API SHALL remove Depends(get_access_controller) from endpoint signatures
6. THE API SHALL simplify query_endpoint to have only 5 dependencies
7. THE API SHALL maintain dependencies: ConversationManager, GroqGenerator, Retriever, CitationEngine, StructuredLogger
8. THE API SHALL remove all AccessController type hints
9. THE API SHALL remove AccessController from imports
10. THE API SHALL update endpoint docstrings to remove authentication mentions

### Requirement 18: Streaming Response Updates

**User Story:** As a developer, I want streaming responses to use only Groq, so that real-time token delivery works correctly.

#### Acceptance Criteria

1. THE API SHALL use GroqGenerator._stream method for streaming responses
2. THE API SHALL NOT call Generator._stream method
3. THE Streaming logic SHALL handle Groq API SSE format correctly
4. THE Streaming logic SHALL parse "data: " prefixed lines from Groq
5. THE Streaming logic SHALL handle "[DONE]" message from Groq
6. THE Streaming logic SHALL yield tokens from delta.content field
7. WHEN Groq streaming fails, THEN THE API SHALL send error event with "unavailable" status
8. THE Frontend SHALL display streamed tokens in real-time
9. THE Frontend SHALL handle Groq-specific streaming errors
10. THE API SHALL maintain SSE format for frontend compatibility

### Requirement 19: Configuration File Cleanup

**User Story:** As a developer, I want configuration files cleaned up to remove Ollama settings, so that deployment configuration is accurate.

#### Acceptance Criteria

1. THE System SHALL remove Ollama configuration from docker-compose.yml if present
2. THE System SHALL remove Ollama service definitions from deployment configs
3. THE System SHALL update deployment documentation to remove Ollama setup steps
4. THE System SHALL document GROQ_API_KEY as required environment variable in deployment guides
5. THE System SHALL remove Ollama port mappings from container configurations
6. THE System SHALL remove Ollama volume mounts from container configurations
7. THE System SHALL update README.md to remove Ollama installation instructions
8. THE System SHALL update README.md to add Groq API key setup instructions
9. THE System SHALL remove Ollama from system architecture diagrams
10. THE System SHALL update system requirements to remove Ollama dependency

### Requirement 20: Backward Compatibility Removal

**User Story:** As a developer, I want all backward compatibility code for Ollama removed, so that the codebase is clean and maintainable.

#### Acceptance Criteria

1. THE System SHALL remove all try-except blocks catching Ollama-specific exceptions
2. THE System SHALL remove all conditional logic checking for Ollama availability
3. THE System SHALL remove all fallback paths to Ollama provider
4. THE System SHALL remove all migration code for Ollama-to-Groq transition
5. THE System SHALL remove all feature flags for provider selection
6. THE System SHALL remove all environment variable checks for OLLAMA_URL
7. THE System SHALL remove all default values falling back to Ollama
8. THE System SHALL assume Groq API is always the provider
9. THE System SHALL fail fast with clear error if GROQ_API_KEY is missing
10. THE System SHALL NOT attempt to detect or use Ollama as fallback under any circumstances
