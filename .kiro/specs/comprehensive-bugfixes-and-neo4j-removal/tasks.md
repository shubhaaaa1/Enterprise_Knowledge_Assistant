# Implementation Plan

## Phase 1: Bug Condition Exploration Tests

- [x] 1. Write bug condition exploration test for deprecated datetime methods
  - **Property 1: Bug Condition** - Deprecated datetime.utcnow() and utcfromtimestamp() Usage
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bugs exist
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate deprecated methods are used
  - **Scoped PBT Approach**: Scan all Python files for deprecated datetime patterns
  - Test that codebase uses `datetime.utcnow()` in 8+ locations (from Bug Condition 1.1)
  - Test that DocsConnector uses `datetime.utcfromtimestamp()` in 2 locations (from Bug Condition 1.2)
  - The test assertions should verify timezone-aware replacements are used (from Expected Behavior 2.1, 2.2)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves the deprecated methods exist)
  - Document counterexamples found (file paths and line numbers with deprecated usage)
  - Mark task complete when test is written, run, and failures are documented
  - _Requirements: 1.1, 1.2_

- [x] 2. Write bug condition exploration test for missing DocsConnector import
  - **Property 1: Bug Condition** - Missing DocsConnector Import in api.py
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface the NameError that occurs when DOCS_PATH is configured
  - **Scoped PBT Approach**: Test with DOCS_PATH environment variable set
  - Test that api.py line 219 attempts to instantiate DocsConnector (from Bug Condition 1.3)
  - Test that DocsConnector import is present in api.py (from Expected Behavior 2.3)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS with NameError (this is correct - it proves the import is missing)
  - Document counterexample found (NameError: name 'DocsConnector' is not defined)
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.3_

- [x] 3. Write bug condition exploration test for missing timestamp in log_error()
  - **Property 1: Bug Condition** - Missing Timestamp Field in log_error()
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface the inconsistency in log record structure
  - **Scoped PBT Approach**: Compare log_error() output with log_query() and log_access_control()
  - Test that log_error() does not include timestamp field (from Bug Condition 1.4)
  - Test that log_error() should include timestamp field like other log methods (from Expected Behavior 2.4)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves timestamp is missing)
  - Document counterexample found (log_error record missing 'timestamp' key)
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.4_

- [x] 4. Write bug condition exploration test for Neo4j code presence
  - **Property 1: Bug Condition** - Neo4j Code Still Present in Codebase
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface all Neo4j-related code that should be removed
  - **Scoped PBT Approach**: Scan codebase for Neo4j imports, classes, and references
  - Test that Neo4j-related code exists in graph_store.py, models.py, health checks (from Bug Condition 1.5)
  - Test that no Neo4j-related code should exist (from Expected Behavior 2.5)
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct - it proves Neo4j code exists)
  - Document counterexamples found (files and locations with Neo4j references)
  - Mark task complete when test is written, run, and failures are documented
  - _Requirements: 1.5_

## Phase 2: Preservation Property Tests

- [x] 5. Write preservation property tests (BEFORE implementing fixes)
  - **Property 2: Preservation** - Core RAG Functionality Preserved
  - **IMPORTANT**: Follow observation-first methodology
  - Observe: Datetime operations serialize correctly to ISO format on unfixed code
  - Observe: DocsConnector processes files and creates valid Documents on unfixed code (when import is manually added)
  - Observe: Ingestion pipeline indexes documents into ChromaDB successfully on unfixed code
  - Observe: Structured logger writes valid JSON entries on unfixed code
  - Observe: Retriever returns relevant chunks based on semantic similarity on unfixed code
  - Observe: Health endpoint reports status for all components on unfixed code
  - Observe: Query endpoint generates answers with citations on unfixed code
  - Observe: Access controller enforces permission-based access control on unfixed code
  - Observe: Conversation manager maintains session history on unfixed code
  - Observe: Citation engine extracts and numbers citations on unfixed code
  - Write property-based tests capturing these observed behaviors (from Preservation Requirements 3.1-3.10)
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

## Phase 3: Implementation

- [x] 6. Fix deprecated datetime.utcnow() usage

  - [x] 6.1 Replace datetime.utcnow() with datetime.now(timezone.utc) in all locations
    - Update ingestion pipeline files
    - Update connector files
    - Update base classes
    - Ensure all replacements use timezone-aware datetime objects
    - _Bug_Condition: Uses datetime.utcnow() in 8+ locations (1.1)_
    - _Expected_Behavior: Uses datetime.now(timezone.utc) everywhere (2.1)_
    - _Preservation: Datetime serialization to ISO format continues to work (3.1)_
    - _Requirements: 1.1, 2.1, 3.1_

  - [x] 6.2 Verify deprecated datetime exploration test now passes
    - **Property 1: Expected Behavior** - Timezone-Aware Datetime Usage
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms datetime.utcnow() is replaced
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms deprecated datetime.utcnow() is fixed)
    - _Requirements: 2.1_

- [x] 7. Fix deprecated datetime.utcfromtimestamp() usage in DocsConnector

  - [x] 7.1 Replace datetime.utcfromtimestamp() with datetime.fromtimestamp(ts, tz=timezone.utc)
    - Update DocsConnector in src/enterprise_rag/ingestion/connectors/docs_connector.py
    - Replace both occurrences with timezone-aware version
    - Ensure modification times are correctly extracted
    - _Bug_Condition: Uses datetime.utcfromtimestamp() in 2 locations (1.2)_
    - _Expected_Behavior: Uses datetime.fromtimestamp(ts, tz=timezone.utc) (2.2)_
    - _Preservation: DocsConnector processes files and creates valid Documents (3.2)_
    - _Requirements: 1.2, 2.2, 3.2_

  - [x] 7.2 Verify deprecated datetime exploration test now passes
    - **Property 1: Expected Behavior** - Timezone-Aware Timestamp Conversion
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms datetime.utcfromtimestamp() is replaced
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms deprecated datetime.utcfromtimestamp() is fixed)
    - _Requirements: 2.2_

- [x] 8. Fix missing DocsConnector import in api.py

  - [x] 8.1 Add DocsConnector import to api.py
    - Add import statement: from enterprise_rag.ingestion.connectors.docs_connector import DocsConnector
    - Place import with other connector imports
    - Verify DocsConnector can be instantiated at line 219
    - _Bug_Condition: DocsConnector not imported, causes NameError (1.3)_
    - _Expected_Behavior: DocsConnector successfully imported and instantiated (2.3)_
    - _Preservation: Ingestion pipeline indexes documents successfully (3.3)_
    - _Requirements: 1.3, 2.3, 3.3_

  - [x] 8.2 Verify missing import exploration test now passes
    - **Property 1: Expected Behavior** - DocsConnector Import Present
    - **IMPORTANT**: Re-run the SAME test from task 2 - do NOT write a new test
    - The test from task 2 encodes the expected behavior
    - When this test passes, it confirms DocsConnector import is present
    - Run bug condition exploration test from step 2
    - **EXPECTED OUTCOME**: Test PASSES (confirms import is fixed)
    - _Requirements: 2.3_

- [x] 9. Fix missing timestamp in log_error() method

  - [x] 9.1 Add timestamp field to log_error() method
    - Update log_error() in src/enterprise_rag/logging.py
    - Add timestamp field using datetime.now(timezone.utc).isoformat()
    - Match format used in log_query() and log_access_control()
    - Ensure log record structure is consistent across all log methods
    - _Bug_Condition: log_error() missing timestamp field (1.4)_
    - _Expected_Behavior: log_error() includes timestamp field (2.4)_
    - _Preservation: Structured logger writes valid JSON entries (3.4)_
    - _Requirements: 1.4, 2.4, 3.4_

  - [x] 9.2 Verify missing timestamp exploration test now passes
    - **Property 1: Expected Behavior** - Timestamp Field Present in log_error()
    - **IMPORTANT**: Re-run the SAME test from task 3 - do NOT write a new test
    - The test from task 3 encodes the expected behavior
    - When this test passes, it confirms timestamp field is present
    - Run bug condition exploration test from step 3
    - **EXPECTED OUTCOME**: Test PASSES (confirms timestamp is added)
    - _Requirements: 2.4_

- [x] 10. Remove Neo4j/graph store code and dependencies

  - [x] 10.1 Remove graph_store.py file
    - Delete src/enterprise_rag/graph_store.py
    - _Bug_Condition: Neo4j code still present (1.5)_
    - _Expected_Behavior: No Neo4j-related code exists (2.5)_
    - _Requirements: 1.5, 2.5_

  - [x] 10.2 Remove Neo4j imports from api.py
    - Remove GraphStore import
    - Remove graph_store initialization
    - Remove graph_store from health check
    - _Bug_Condition: Neo4j code still present (1.5)_
    - _Expected_Behavior: No Neo4j-related code exists (2.5)_
    - _Preservation: Health endpoint reports status for remaining components (3.6)_
    - _Requirements: 1.5, 2.5, 3.6_

  - [x] 10.3 Remove Neo4j models from models.py
    - Remove EntityNode class
    - Remove RelationshipEdge class
    - Remove any Neo4j-specific model definitions
    - _Bug_Condition: Neo4j code still present (1.5)_
    - _Expected_Behavior: No Neo4j-related code exists (2.5)_
    - _Requirements: 1.5, 2.5_

  - [x] 10.4 Remove Neo4j dependencies from pyproject.toml
    - Remove neo4j package from dependencies
    - Remove any Neo4j-related optional dependencies
    - _Bug_Condition: Neo4j code still present (1.5)_
    - _Expected_Behavior: No Neo4j-related code exists (2.5)_
    - _Requirements: 1.5, 2.5_

  - [x] 10.5 Verify Neo4j removal exploration test now passes
    - **Property 1: Expected Behavior** - No Neo4j Code Present
    - **IMPORTANT**: Re-run the SAME test from task 4 - do NOT write a new test
    - The test from task 4 encodes the expected behavior
    - When this test passes, it confirms all Neo4j code is removed
    - Run bug condition exploration test from step 4
    - **EXPECTED OUTCOME**: Test PASSES (confirms Neo4j code is removed)
    - _Requirements: 2.5_

- [~] 11. Improve exception handling specificity (optional enhancement)

  - [ ] 11.1 Review broad exception handlers in codebase
    - Identify 20+ locations with "except Exception" handlers
    - Determine which can be made more specific
    - _Bug_Condition: Broad exception handlers hide specific errors (1.6)_
    - _Expected_Behavior: Use specific exception types where appropriate (2.6)_
    - _Requirements: 1.6, 2.6_

  - [ ] 11.2 Replace broad handlers with specific exception types where appropriate
    - Update exception handlers in critical paths
    - Keep broad handlers only where truly necessary
    - Add comments explaining why broad handlers are needed
    - _Bug_Condition: Broad exception handlers hide specific errors (1.6)_
    - _Expected_Behavior: Use specific exception types where appropriate (2.6)_
    - _Preservation: All core functionality continues to work (3.1-3.10)_
    - _Requirements: 1.6, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

  - [ ] 11.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Core RAG Functionality Preserved
    - **IMPORTANT**: Re-run the SAME tests from task 5 - do NOT write new tests
    - Run preservation property tests from step 5
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all tests still pass after exception handling improvements

## Phase 4: Final Validation

- [x] 12. Verify all preservation tests still pass
  - **Property 2: Preservation** - Core RAG Functionality Preserved
  - **IMPORTANT**: Re-run the SAME tests from task 5 - do NOT write new tests
  - Run preservation property tests from step 5
  - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions from all fixes)
  - Confirm datetime serialization works correctly (3.1)
  - Confirm DocsConnector processes files correctly (3.2)
  - Confirm ingestion pipeline indexes documents (3.3)
  - Confirm structured logger writes valid JSON (3.4)
  - Confirm retriever returns relevant chunks (3.5)
  - Confirm health endpoint reports status (3.6)
  - Confirm query endpoint generates answers (3.7)
  - Confirm access controller enforces permissions (3.8)
  - Confirm conversation manager maintains history (3.9)
  - Confirm citation engine extracts citations (3.10)

- [x] 13. Checkpoint - Ensure all tests pass
  - Ensure all exploration tests pass (deprecated datetime, missing import, missing timestamp, Neo4j removal)
  - Ensure all preservation tests pass (core RAG functionality)
  - Ask the user if questions arise
