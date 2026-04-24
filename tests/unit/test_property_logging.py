"""Property-based tests for StructuredLogger.

# Feature: enterprise-rag-system, Property 6: Query audit log completeness
# Feature: enterprise-rag-system, Property 23: Error log correlation
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
# Property 6: Query audit log completeness
# Validates: Requirements 2.6, 10.1
# ---------------------------------------------------------------------------


@given(
    user_id=_safe_text,
    session_id=_safe_text,
    original_query=_safe_text,
    rewritten_query=_safe_text,
    chunks_retrieved=st.integers(min_value=0, max_value=100),
    grounding_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    latency_ms=st.floats(min_value=0.0, max_value=60_000.0, allow_nan=False),
)
@settings(max_examples=100, deadline=None)
def test_query_audit_log_completeness(
    user_id,
    session_id,
    original_query,
    rewritten_query,
    chunks_retrieved,
    grounding_score,
    latency_ms,
):
    """For any query processed by the system, the emitted log entry must contain
    non-null values for timestamp, user_id, session_id, original_query,
    rewritten_query, chunks_retrieved, grounding_score, and latency_ms.

    Validates: Requirements 2.6, 10.1
    """
    logger = StructuredLogger(backend="file", log_dir="/tmp/test_logs_prop6")
    correlation_id = StructuredLogger.new_correlation_id()

    captured: list[dict] = []

    with patch.object(logger._backend, "emit", side_effect=captured.append):
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

    assert len(captured) == 1, "Expected exactly one log record to be emitted"
    record = captured[0]

    required_fields = [
        "timestamp",
        "user_id",
        "session_id",
        "original_query",
        "rewritten_query",
        "chunks_retrieved",
        "grounding_score",
        "latency_ms",
    ]
    for field in required_fields:
        assert field in record, f"Missing field: {field!r}"
        assert record[field] is not None, f"Field {field!r} must not be None"

    # Verify values match what was passed in
    assert record["user_id"] == user_id
    assert record["session_id"] == session_id
    assert record["original_query"] == original_query
    assert record["rewritten_query"] == rewritten_query
    assert record["chunks_retrieved"] == chunks_retrieved
    assert record["grounding_score"] == grounding_score
    assert record["latency_ms"] == latency_ms


# ---------------------------------------------------------------------------
# Property 23: Error log correlation
# Validates: Requirements 10.4
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
)
@settings(max_examples=100, deadline=None)
def test_error_log_correlation(
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
):
    """For any component error during query processing, the emitted error log
    entry must contain severity, component_name, error_message, and a
    correlation_id that matches the originating query log entry's correlation_id.

    Validates: Requirements 10.4
    """
    logger = StructuredLogger(backend="file", log_dir="/tmp/test_logs_prop23")
    correlation_id = StructuredLogger.new_correlation_id()

    captured: list[dict] = []

    with patch.object(logger._backend, "emit", side_effect=captured.append):
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
        logger.log_error(
            severity=severity,
            component_name=component_name,
            error_message=error_message,
            correlation_id=correlation_id,
        )

    assert len(captured) == 2, "Expected exactly two log records (query + error)"

    query_record = captured[0]
    error_record = captured[1]

    # Error record must have all required fields non-null
    for field in ("severity", "component_name", "error_message", "correlation_id"):
        assert field in error_record, f"Missing field in error record: {field!r}"
        assert error_record[field] is not None, f"Field {field!r} must not be None"

    # Values must match what was passed in
    assert error_record["severity"] == severity
    assert error_record["component_name"] == component_name
    assert error_record["error_message"] == error_message

    # The correlation_id in the error record must match the query record's correlation_id
    assert "correlation_id" in query_record, "Query record must have correlation_id"
    assert error_record["correlation_id"] == query_record["correlation_id"], (
        f"Error correlation_id {error_record['correlation_id']!r} does not match "
        f"query correlation_id {query_record['correlation_id']!r}"
    )
    assert error_record["correlation_id"] == correlation_id
