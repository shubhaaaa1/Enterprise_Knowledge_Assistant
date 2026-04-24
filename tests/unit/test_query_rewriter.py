"""Unit tests for QueryRewriter.

Covers:
- Req 3.4: timeout fallback — returns original query and logs timeout
- Req 3.1: successful rewrite returns all variants
- Req 3.1: original query always present in result
- Req 3.2: last 5 turns of history incorporated in prompt
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from enterprise_rag.models import Turn
from enterprise_rag.query_rewriter import QueryRewriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(index: int, role: str = "user") -> Turn:
    return Turn(
        role=role,
        original_query=f"question-{index}",
        rewritten_query=f"rewritten-{index}",
        answer=f"answer-{index}",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _mock_ollama_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"response": text}
    resp.raise_for_status.return_value = None
    return resp


def _make_rewriter(**kwargs) -> QueryRewriter:
    return QueryRewriter(
        ollama_url="http://mock-ollama:11434",
        model="llama3",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Req 3.4 — Timeout fallback
# ---------------------------------------------------------------------------

class TestTimeoutFallback:
    def test_timeout_returns_original_query(self):
        """When Ollama times out, rewrite() must return [original_query]."""
        rewriter = _make_rewriter(timeout=5.0)
        query = "what is the deployment process?"

        with patch("requests.post", side_effect=requests.Timeout("timed out")):
            result = rewriter.rewrite(query, history=[])

        assert result == [query]

    def test_timeout_logs_warning(self, caplog):
        """Timeout must emit a WARNING log containing the query text."""
        rewriter = _make_rewriter(timeout=5.0)
        query = "how do I reset my password?"

        with caplog.at_level(logging.WARNING, logger="enterprise_rag.query_rewriter"):
            with patch("requests.post", side_effect=requests.Timeout("timed out")):
                rewriter.rewrite(query, history=[])

        assert any(query in record.message for record in caplog.records), (
            "Expected log message containing the query text"
        )
        assert any(record.levelno == logging.WARNING for record in caplog.records), (
            "Expected a WARNING-level log entry"
        )

    def test_timeout_log_contains_session_id_unknown(self, caplog):
        """Timeout log must include session_id=unknown as per Req 3.4."""
        rewriter = _make_rewriter()
        query = "show me the architecture diagram"

        with caplog.at_level(logging.WARNING, logger="enterprise_rag.query_rewriter"):
            with patch("requests.post", side_effect=requests.Timeout("timed out")):
                rewriter.rewrite(query, history=[])

        assert any("unknown" in record.message for record in caplog.records), (
            "Expected 'unknown' session_id in timeout log"
        )

    def test_timeout_result_is_list_with_one_element(self):
        """Fallback result must be a list of exactly one element."""
        rewriter = _make_rewriter()
        query = "list all open Jira tickets"

        with patch("requests.post", side_effect=requests.Timeout()):
            result = rewriter.rewrite(query, history=[])

        assert isinstance(result, list)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Req 3.1 — Successful rewrite
# ---------------------------------------------------------------------------

class TestSuccessfulRewrite:
    def test_returns_original_plus_variants(self):
        """Successful Ollama response: original query prepended to variants."""
        rewriter = _make_rewriter()
        query = "how does authentication work?"
        ollama_text = "how is auth implemented?\nhow does login work?\nwhat is the auth flow?"

        with patch("requests.post", return_value=_mock_ollama_response(ollama_text)):
            result = rewriter.rewrite(query, history=[], max_variants=3)

        assert result[0] == query, "First element must be the original query"
        assert len(result) == 4, "Expected original + 3 variants"
        assert "how is auth implemented?" in result
        assert "how does login work?" in result
        assert "what is the auth flow?" in result

    def test_respects_max_variants_limit(self):
        """Returned variants must not exceed max_variants (plus original)."""
        rewriter = _make_rewriter()
        query = "explain the ingestion pipeline"
        # Ollama returns 5 lines but max_variants=2
        ollama_text = "line1\nline2\nline3\nline4\nline5"

        with patch("requests.post", return_value=_mock_ollama_response(ollama_text)):
            result = rewriter.rewrite(query, history=[], max_variants=2)

        # original + up to 2 variants
        assert len(result) <= 3

    def test_empty_ollama_response_returns_original_only(self):
        """If Ollama returns empty text, result is just [original_query]."""
        rewriter = _make_rewriter()
        query = "what is the SLA?"

        with patch("requests.post", return_value=_mock_ollama_response("")):
            result = rewriter.rewrite(query, history=[])

        assert result == [query]

    def test_http_error_returns_original_query(self):
        """Non-timeout HTTP errors also fall back to [original_query]."""
        rewriter = _make_rewriter()
        query = "list all services"

        with patch("requests.post", side_effect=requests.ConnectionError("refused")):
            result = rewriter.rewrite(query, history=[])

        assert result == [query]


# ---------------------------------------------------------------------------
# Req 3.1 — Original query always in result
# ---------------------------------------------------------------------------

class TestOriginalQueryAlwaysPresent:
    def test_original_query_is_first_element_on_success(self):
        rewriter = _make_rewriter()
        query = "what are the retry policies?"

        with patch("requests.post", return_value=_mock_ollama_response("variant A\nvariant B")):
            result = rewriter.rewrite(query, history=[])

        assert result[0] == query

    def test_original_query_present_on_timeout(self):
        rewriter = _make_rewriter()
        query = "describe the vector store schema"

        with patch("requests.post", side_effect=requests.Timeout()):
            result = rewriter.rewrite(query, history=[])

        assert query in result

    def test_original_query_present_on_error(self):
        rewriter = _make_rewriter()
        query = "how many chunks are indexed?"

        with patch("requests.post", side_effect=RuntimeError("unexpected")):
            result = rewriter.rewrite(query, history=[])

        assert query in result


# ---------------------------------------------------------------------------
# Req 3.2 — History incorporated in prompt (last 5 turns)
# ---------------------------------------------------------------------------

class TestHistoryInPrompt:
    def test_last_5_turns_included_in_prompt(self):
        """Prompt sent to Ollama must contain the last 5 turns of history."""
        rewriter = _make_rewriter()
        query = "what about the caching layer?"

        # Build 8 turns; only the last 5 should appear in the prompt
        history = [_make_turn(i) for i in range(8)]

        captured_payload = {}

        def capture_post(url, json=None, timeout=None):
            captured_payload.update(json or {})
            return _mock_ollama_response("variant 1")

        with patch("requests.post", side_effect=capture_post):
            rewriter.rewrite(query, history)

        prompt: str = captured_payload.get("prompt", "")

        # Turns 3-7 (indices 3..7) should be in the prompt
        for i in range(3, 8):
            assert f"question-{i}" in prompt, (
                f"Expected turn {i} in prompt, but it was missing"
            )

        # Turns 0-2 should NOT be in the prompt (older than last 5)
        for i in range(3):
            assert f"question-{i}" not in prompt, (
                f"Turn {i} should not appear in prompt (older than last 5)"
            )

    def test_empty_history_produces_valid_prompt(self):
        """Empty history must not cause errors; prompt is still well-formed."""
        rewriter = _make_rewriter()
        query = "what is the system architecture?"

        captured_payload = {}

        def capture_post(url, json=None, timeout=None):
            captured_payload.update(json or {})
            return _mock_ollama_response("variant 1")

        with patch("requests.post", side_effect=capture_post):
            result = rewriter.rewrite(query, history=[])

        assert result[0] == query
        assert "Question:" in captured_payload.get("prompt", "")

    def test_exactly_5_turns_all_included(self):
        """When history has exactly 5 turns, all 5 must appear in the prompt."""
        rewriter = _make_rewriter()
        query = "explain the access control model"
        history = [_make_turn(i) for i in range(5)]

        captured_payload = {}

        def capture_post(url, json=None, timeout=None):
            captured_payload.update(json or {})
            return _mock_ollama_response("variant 1")

        with patch("requests.post", side_effect=capture_post):
            rewriter.rewrite(query, history)

        prompt: str = captured_payload.get("prompt", "")
        for i in range(5):
            assert f"question-{i}" in prompt
