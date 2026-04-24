"""Unit tests for the Generator component."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from enterprise_rag.generator import Generator, INSUFFICIENT_INFO_RESPONSE
from enterprise_rag.models import ScoredChunk, Turn

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_chunk(chunk_id: str, score: float, text: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        source_type="docs",
        source_id="src1",
        document_title="Doc",
        document_url="http://example.com",
        text=text or f"content of {chunk_id}",
        token_count=10,
        permission_tags=["engineering"],
        created_at=_NOW,
        source_modified_at=_NOW,
        relevance_score=score,
        rrf_score=0.0,
    )


# ---------------------------------------------------------------------------
# Req 5.3 — Empty chunk set → no Ollama call, returns INSUFFICIENT_INFO_RESPONSE
# ---------------------------------------------------------------------------

def test_empty_chunks_returns_insufficient_info_no_stream():
    """Empty chunk set → INSUFFICIENT_INFO_RESPONSE, Ollama not called (Req 5.3)."""
    gen = Generator()
    with patch("enterprise_rag.generator.requests.post") as mock_post:
        result = gen.generate(query="What is X?", chunks=[], history=[], stream=False)
    assert result == INSUFFICIENT_INFO_RESPONSE
    mock_post.assert_not_called()


def test_empty_chunks_returns_insufficient_info_stream():
    """Empty chunk set → iterator yielding INSUFFICIENT_INFO_RESPONSE, Ollama not called (Req 5.3)."""
    gen = Generator()
    with patch("enterprise_rag.generator.requests.post") as mock_post:
        result = gen.generate(query="What is X?", chunks=[], history=[], stream=True)
    tokens = list(result)
    assert tokens == [INSUFFICIENT_INFO_RESPONSE]
    mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Req 8.2 — Ollama unavailability → ConnectionError raised within 10s
# ---------------------------------------------------------------------------

def test_ollama_unavailable_raises_connection_error():
    """ConnectionError raised when Ollama is unreachable (Req 8.2)."""
    import requests as req_lib

    gen = Generator(ollama_url="http://localhost:11434")
    chunks = [_make_chunk("c1", 0.9)]

    with patch("enterprise_rag.generator.requests.post") as mock_post:
        mock_post.side_effect = req_lib.exceptions.ConnectionError("refused")
        with pytest.raises(ConnectionError):
            result = gen.generate(query="What is X?", chunks=chunks, history=[], stream=False)


def test_ollama_unavailable_stream_raises_connection_error():
    """ConnectionError raised when Ollama is unreachable in streaming mode (Req 8.2)."""
    import requests as req_lib

    gen = Generator(ollama_url="http://localhost:11434")
    chunks = [_make_chunk("c1", 0.9)]

    with patch("enterprise_rag.generator.requests.post") as mock_post:
        mock_post.side_effect = req_lib.exceptions.ConnectionError("refused")
        result = gen.generate(query="What is X?", chunks=chunks, history=[], stream=True)
        with pytest.raises(ConnectionError):
            list(result)  # consume the iterator to trigger the error


# ---------------------------------------------------------------------------
# Req 5.2, 9.1 — Prompt template contains grounding instruction
# ---------------------------------------------------------------------------

def test_prompt_contains_grounding_instruction():
    """Prompt includes the grounding instruction directing model to use ONLY context (Req 5.2, 9.1)."""
    gen = Generator()
    chunks = [_make_chunk("c1", 0.9, text="The sky is blue.")]
    prompt = gen._build_prompt("What color is the sky?", chunks, [])

    assert "ONLY the context" in prompt, "Grounding instruction missing from prompt"
    assert "cite the chunk number" in prompt, "Citation instruction missing from prompt"
    assert "does not contain sufficient information" in prompt, (
        "Insufficient-info instruction missing from prompt"
    )


def test_prompt_contains_question():
    """Prompt includes the user question."""
    gen = Generator()
    chunks = [_make_chunk("c1", 0.8)]
    query = "How does authentication work?"
    prompt = gen._build_prompt(query, chunks, [])
    assert query in prompt


# ---------------------------------------------------------------------------
# Chunks sorted by relevance_score descending in prompt
# ---------------------------------------------------------------------------

def test_chunks_sorted_descending_in_prompt():
    """Chunks appear in the prompt in descending relevance_score order (Req 5.5)."""
    gen = Generator()
    chunks = [
        _make_chunk("low", 0.2, text="low relevance text"),
        _make_chunk("high", 0.9, text="high relevance text"),
        _make_chunk("mid", 0.5, text="mid relevance text"),
    ]
    sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
    prompt = gen._build_prompt("query", sorted_chunks, [])

    pos_high = prompt.find("high relevance text")
    pos_mid = prompt.find("mid relevance text")
    pos_low = prompt.find("low relevance text")

    assert pos_high < pos_mid < pos_low, (
        "Chunks are not in descending relevance order in the prompt"
    )


def test_generate_sorts_chunks_before_building_prompt():
    """generate() sorts chunks by relevance_score descending before building prompt (Req 5.5)."""
    gen = Generator()
    chunks = [
        _make_chunk("low", 0.1, text="low text"),
        _make_chunk("high", 0.95, text="high text"),
    ]

    captured_prompts: list[str] = []

    import requests as req_lib

    def fake_post(url, json=None, **kwargs):
        captured_prompts.append(json.get("prompt", ""))
        raise req_lib.exceptions.ConnectionError("stop after capture")

    with patch("enterprise_rag.generator.requests.post", side_effect=fake_post):
        try:
            gen.generate(query="q", chunks=chunks, history=[], stream=False)
        except ConnectionError:
            pass

    assert captured_prompts, "No prompt was captured"
    prompt = captured_prompts[0]
    pos_high = prompt.find("high text")
    pos_low = prompt.find("low text")
    assert pos_high < pos_low, "High-score chunk should appear before low-score chunk"


# ---------------------------------------------------------------------------
# Timeout → TimeoutError raised
# ---------------------------------------------------------------------------

def test_timeout_raises_timeout_error():
    """TimeoutError raised when Ollama exceeds 30s (Req 5.4)."""
    import requests as req_lib

    gen = Generator()
    chunks = [_make_chunk("c1", 0.9)]

    with patch("enterprise_rag.generator.requests.post") as mock_post:
        mock_post.side_effect = req_lib.exceptions.Timeout("timed out")
        with pytest.raises(TimeoutError):
            gen.generate(query="q", chunks=chunks, history=[], stream=False)
