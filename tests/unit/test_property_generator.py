"""Property-based tests for the Generator component.

# Feature: enterprise-rag-system, Property 11: Generator prompt contains chunks in descending relevance order
# Feature: enterprise-rag-system, Property 12: Context window truncation preserves highest-ranked chunks
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.generator import Generator
from enterprise_rag.models import ScoredChunk, Turn

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, score: float, token_count: int = 10, text: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        source_type="docs",
        source_id="src1",
        document_title="Doc",
        document_url="http://example.com",
        text=text if text is not None else f"content of chunk {chunk_id}",
        token_count=token_count,
        permission_tags=["engineering"],
        created_at=_NOW,
        source_modified_at=_NOW,
        relevance_score=score,
        rrf_score=0.0,
    )


@st.composite
def scored_chunk_list(draw, min_size: int = 1, max_size: int = 10) -> List[ScoredChunk]:
    """Strategy: list of ScoredChunks with distinct relevance scores."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    return [_make_chunk(f"chunk_{i}", score) for i, score in enumerate(scores)]


# ---------------------------------------------------------------------------
# Property 11: Generator prompt contains chunks in descending relevance order
# Validates: Requirements 5.1, 5.5
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(chunks=scored_chunk_list(min_size=1, max_size=10))
def test_property11_prompt_chunks_descending_order(chunks: List[ScoredChunk]) -> None:
    """Property 11: The constructed prompt presents chunks in strictly descending
    relevance_score order, with the highest-scored chunk appearing first.

    **Validates: Requirements 5.1, 5.5**
    """
    gen = Generator()
    sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
    prompt = gen._build_prompt("test query", sorted_chunks, [])

    # Extract the order in which chunk texts appear in the prompt.
    # Each chunk is rendered as "[N] content of chunk X"
    positions = []
    for chunk in sorted_chunks:
        pos = prompt.find(chunk.text)
        assert pos != -1, f"Chunk text '{chunk.text}' not found in prompt"
        positions.append((pos, chunk.relevance_score))

    # Positions should be in ascending order (earlier in string = higher rank)
    # and scores should be in descending order at those positions.
    for i in range(len(positions) - 1):
        assert positions[i][0] < positions[i + 1][0], (
            f"Chunk with score {positions[i][1]} appears after chunk with score "
            f"{positions[i + 1][1]} in the prompt"
        )
        assert positions[i][1] >= positions[i + 1][1], (
            f"Score ordering violated: {positions[i][1]} < {positions[i + 1][1]}"
        )


# ---------------------------------------------------------------------------
# Property 12: Context window truncation preserves highest-ranked chunks
# Validates: Requirements 8.4
# ---------------------------------------------------------------------------

@st.composite
def chunks_exceeding_context(draw) -> tuple[List[ScoredChunk], int]:
    """Strategy: chunk list whose total token count exceeds a small context window."""
    context_window = draw(st.integers(min_value=50, max_value=200))
    # Each chunk has a text of ~20 words so token_count ≈ 20
    n = draw(st.integers(min_value=3, max_value=8))
    scores = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    # Make each chunk text large enough that n chunks together exceed context_window
    words_per_chunk = (context_window // n) + 10
    chunks = [
        _make_chunk(
            f"chunk_{i}",
            score,
            token_count=words_per_chunk,
            text=" ".join([f"word{j}" for j in range(words_per_chunk)]),
        )
        for i, score in enumerate(scores)
    ]
    return chunks, context_window


@settings(max_examples=20, deadline=None)
@given(data=chunks_exceeding_context())
def test_property12_truncation_preserves_highest_ranked(data: tuple) -> None:
    """Property 12: When total tokens exceed context_window, the Generator removes
    lowest-ranked chunks first; the resulting prompt fits within the limit and
    the highest-ranked chunks are preserved.

    **Validates: Requirements 8.4**
    """
    chunks, context_window = data
    gen = Generator(context_window=context_window)

    # Sort as the generator would
    sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
    truncated = gen._truncate_to_context(sorted_chunks, "test query", [])

    # 1. Resulting prompt must fit within context_window (in words/tokens)
    if truncated:
        prompt = gen._build_prompt("test query", truncated, [])
        token_count = gen._count_tokens(prompt)
        assert token_count <= context_window, (
            f"Prompt has {token_count} tokens but context_window is {context_window}"
        )

    # 2. Retained chunks must be the highest-ranked ones.
    #    i.e., every retained chunk has a higher relevance_score than every removed chunk.
    retained_ids = {c.chunk_id for c in truncated}
    removed_chunks = [c for c in sorted_chunks if c.chunk_id not in retained_ids]

    if truncated and removed_chunks:
        min_retained_score = min(c.relevance_score for c in truncated)
        max_removed_score = max(c.relevance_score for c in removed_chunks)
        assert min_retained_score >= max_removed_score, (
            f"A removed chunk (score={max_removed_score:.4f}) has a higher score "
            f"than a retained chunk (score={min_retained_score:.4f})"
        )
