"""Bug condition exploration test for missing timestamp in log_error().

**Validates: Requirements 1.4, 2.4**

This test encodes the EXPECTED BEHAVIOR and will FAIL on unfixed code.
When it fails, it proves the bug exists (timestamp missing from log_error).
When it passes after the fix, it confirms the bug is resolved.

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.logging import StructuredLogger

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_safe_text = st.text(
    min_size=1,
    max_size=80,
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_ "),
)

_SEVERITIES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
_COMPONENTS = ["retriever", "generator", "citation_engine", "access_controller", "ingestion"]


# ---------------------------------------------------------------------------
# Property 1: Bug Condition - Missing Timestamp Field in log_error()
# Validates: Requirements 1.4, 2.4
# ---------------------------------------------------------------------------


@given(
    severity=st.sampled_from(_SEVERITIES),
    component_name=st.sampled_from(_COMPONENTS),
    error_message=_safe_text,
    user_id=_safe_text,
    session_id=_safe_text,
    original_query=_safe_text,
    rewritten_query=_safe_text,
    chunks_retrieved=st.integers(min_value=0, max_value=100),
    grounding_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    latency_ms=st.floats(min_value=0.0, max_value=60_000.0, allow_nan=False),
    roles_evaluated=st.lists(_safe_text, min_size=1, max_size=5),
    sources_filtered=st.lists(_safe_text, min_size=0, max_size=10),
    chunks_excluded=st.integers(min_value=0, max_value=50),
)
@settings(max_examples=50, deadline=None)
def test_log_error_missing_timestamp_field(
    severity,
    component_name,
    error_message,
    user_id,
    session_id,
    original_query,
    rewritten_query,
    chunks_retrieved,
    grounding_score,
    latency_ms,
    roles_evaluated,
    sources_filtered,
    chunks_excluded,
):
    """Bug Condition Exploration: log_error() should include timestamp field
    consistent with log_query() and log_access_control().

    This test compares the structure of log records from all three logging methods.
    
    EXPECTED BEHAVIOR (from Requirements 2.4):
    - log_error() SHOULD include a 'timestamp' field
    - The timestamp field should be consistent with other log methods
    
    BUG CONDITION (from Requirements 1.4):
    - log_error() does NOT include a timestamp field
    
    EXPECTED OUTCOME ON UNFIXED CODE:
    - This test will FAIL because log_error() is missing the timestamp field
    - The failure proves the bug exists
    
    EXPECTED OUTCOME AFTER FIX:
    - This test will PASS because log_error() includes the timestamp field
    - The pass confirms the bug is resolved
    
    **Validates: Requirements 1.4, 2.4**
    """
    logger = StructuredLogger(backend="file", log_dir="/tmp/test_logs_bugfix_timestamp")
    correlation_id = StructuredLogger.new_correlation_id()

    captured: list[dict] = []

    with patch.object(logger._backend, "emit", side_effect=captured.append):
        # Log a query
        logger.log_query(
            user_id=user_id,
            session_id=session_id,
            original_query=original_query,
            rewritten_query=rewritten_query,
            chunks_retrieved=chunks_retrieved,
            grounding_score=grounding_score,
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )
        
        # Log access control
        logger.log_access_control(
            user_id=user_id,
            roles_evaluated=roles_evaluated,
            sources_filtered=sources_filtered,
            chunks_excluded=chunks_excluded,
            correlation_id=correlation_id,
        )
        
        # Log an error
        logger.log_error(
            severity=severity,
            component_name=component_name,
            error_message=error_message,
            correlation_id=correlation_id,
        )

    assert len(captured) == 3, "Expected exactly three log records (query + access_control + error)"

    query_record = captured[0]
    access_control_record = captured[1]
    error_record = captured[2]

    # Verify log_query() includes timestamp field
    assert "timestamp" in query_record, (
        "log_query() record must include 'timestamp' field"
    )
    assert query_record["timestamp"] is not None, (
        "log_query() timestamp must not be None"
    )

    # Verify log_access_control() includes timestamp field
    assert "timestamp" in access_control_record, (
        "log_access_control() record must include 'timestamp' field"
    )
    assert access_control_record["timestamp"] is not None, (
        "log_access_control() timestamp must not be None"
    )

    # CRITICAL ASSERTION: log_error() should include timestamp field
    # This assertion will FAIL on unfixed code, proving the bug exists
    assert "timestamp" in error_record, (
        "log_error() record must include 'timestamp' field consistent with "
        "log_query() and log_access_control() (Requirements 1.4, 2.4). "
        f"Found fields: {list(error_record.keys())}"
    )
    
    # Verify timestamp is not None
    assert error_record["timestamp"] is not None, (
        "log_error() timestamp must not be None"
    )
    
    # Verify timestamp format is consistent (ISO format string)
    assert isinstance(error_record["timestamp"], str), (
        "log_error() timestamp must be a string in ISO format"
    )
    assert "T" in error_record["timestamp"], (
        "log_error() timestamp must be in ISO format (contains 'T' separator)"
    )
