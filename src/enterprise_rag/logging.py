"""Structured logging for the Enterprise RAG System."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _FileBackend:
    """Write JSON log entries to a file."""

    def __init__(self, log_dir: str, retention_days: int) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "enterprise_rag.log")
        self._retention_days = retention_days
        # Configure an underlying Python logger that writes to the file
        self._logger = logging.getLogger(f"enterprise_rag.file_backend.{log_dir}")
        if not self._logger.handlers:
            handler = logging.FileHandler(self._log_path)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)
            self._logger.propagate = False

    def emit(self, record: dict) -> None:
        self._logger.info(json.dumps(record))


class _ElasticsearchBackend:
    """Stub Elasticsearch backend."""

    def __init__(self, retention_days: int) -> None:
        self._retention_days = retention_days

    def emit(self, record: dict) -> None:  # pragma: no cover
        # Stub: in production, POST to Elasticsearch index
        pass


class _S3Backend:
    """Stub S3 backend."""

    def __init__(self, retention_days: int) -> None:
        self._retention_days = retention_days

    def emit(self, record: dict) -> None:  # pragma: no cover
        # Stub: in production, write to S3 bucket
        pass


class StructuredLogger:
    """Emit structured JSON log entries for queries, access control, and errors.

    Supports file (default), Elasticsearch, and S3 backends.
    Retention policy defaults to 90 days (informational; enforcement is
    backend-specific).
    """

    def __init__(
        self,
        backend: str = "file",
        log_dir: str = "logs",
        retention_days: int = 90,
    ) -> None:
        self._retention_days = retention_days
        if backend == "file":
            self._backend = _FileBackend(log_dir, retention_days)
        elif backend == "elasticsearch":
            self._backend = _ElasticsearchBackend(retention_days)
        elif backend == "s3":
            self._backend = _S3Backend(retention_days)
        else:
            raise ValueError(f"Unsupported logging backend: {backend!r}")

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_query(
        self,
        *,
        user_id: str,
        session_id: str,
        original_query: str,
        rewritten_query: str,
        chunks_retrieved: int,
        grounding_score: float,
        latency_ms: float,
        correlation_id: str,
    ) -> None:
        """Emit a structured query log entry."""
        record = {
            "event": "query",
            "timestamp": _utc_now(),
            "user_id": user_id,
            "session_id": session_id,
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "chunks_retrieved": chunks_retrieved,
            "grounding_score": grounding_score,
            "latency_ms": latency_ms,
            "correlation_id": correlation_id,
            "retention_days": self._retention_days,
        }
        self._backend.emit(record)

    def log_access_control(
        self,
        *,
        user_id: str,
        roles_evaluated: List[str],
        sources_filtered: List[str],
        chunks_excluded: int,
        correlation_id: str,
    ) -> None:
        """Emit a structured access control log entry."""
        record = {
            "event": "access_control",
            "timestamp": _utc_now(),
            "user_id": user_id,
            "roles_evaluated": roles_evaluated,
            "sources_filtered": sources_filtered,
            "chunks_excluded": chunks_excluded,
            "correlation_id": correlation_id,
            "retention_days": self._retention_days,
        }
        self._backend.emit(record)

    def log_error(
        self,
        *,
        severity: str,
        component_name: str,
        error_message: str,
        correlation_id: str,
    ) -> None:
        """Emit a structured error log entry."""
        record = {
            "event": "error",
            "timestamp": _utc_now(),
            "severity": severity,
            "component_name": component_name,
            "error_message": error_message,
            "correlation_id": correlation_id,
            "retention_days": self._retention_days,
        }
        self._backend.emit(record)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def new_correlation_id() -> str:
        """Generate a new UUID-based correlation ID."""
        return str(uuid.uuid4())
