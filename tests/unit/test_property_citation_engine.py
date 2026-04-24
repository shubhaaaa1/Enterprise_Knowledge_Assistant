"""Property-based tests for the CitationEngine component.

# Feature: enterprise-rag-system, Property 13: Citation-answer correspondence
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List

from hypothesis import given, settings
from hypothesis import strategies as st

from enterprise_rag.citation_engine import CitationEngine
from enterprise_rag.models import ScoredChunk

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

_REF_PATTERN = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------------------
# Helpers / strategies
# ---------------------------------------------------------------------------

def _make_chunk(index: int, url_suffix: str | None = None) -> ScoredChunk:
    url = f"http://example.com/doc-{url_suffix or index}"
    return ScoredChunk(
        chunk_id=f"chunk-{index}",
        source_type="docs",
        source_id="src1",
        document_title=f"Document {index}",
        document_url=url,
        text=f"Content of chunk {index}.",
        token_count=10,
        permission_tags=["engineering"],
        created_at=_NOW,
        source_modified_at=_NOW,
        relevance_score=0.9,
        rrf_score=0.0,
    )


@st.composite
def chunks_and_answer(draw) -> tuple[List[ScoredChunk], str]:
    """Strategy: generate a list of ScoredChunks (1..N) and an answer string
    that contains only valid [N] references (where N is within the chunk list size).
    """
    n = draw(st.integers(min_value=1, max_value=10))
    chunks = [_make_chunk(i) for i in range(1, n + 1)]

    # Generate a list of valid reference numbers (1-indexed, within chunk range)
    refs = draw(
        st.lists(
            st.integers(min_value=1, max_value=n),
            min_size=1,
            max_size=n * 3,
        )
    )

    # Build an answer string that interleaves text with [N] references
    words = draw(
        st.lists(
            st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu")), min_size=1, max_size=8),
            min_size=len(refs),
            max_size=len(refs) + 5,
        )
    )

    parts = []
    for i, ref in enumerate(refs):
        word = words[i] if i < len(words) else "text"
        parts.append(f"{word} [{ref}]")
    answer = " ".join(parts) + "."

    return chunks, answer


# ---------------------------------------------------------------------------
# Property 13: Citation-answer correspondence
# Validates: Requirements 6.1, 6.3
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(data=chunks_and_answer())
def test_property13_citation_answer_correspondence(data: tuple) -> None:
    """Property 13: Every [N] reference in the answer_text has a corresponding
    citation in the citations list (matched via the chunk URL), and every
    citation corresponds to at least one [N] reference in the answer_text.

    Note: CitationEngine deduplicates by URL, so citation numbers (sequential
    IDs) do not match the original [N] chunk indices. We verify correspondence
    via the chunk's document_url: each inline [N] must map to a chunk whose URL
    appears in the citations list, and every citation URL must be reachable from
    at least one inline [N].

    **Validates: Requirements 6.1, 6.3**
    """
    chunks, answer = data
    engine = CitationEngine()
    result = engine.cite(answer, chunks)

    # Collect all [N] refs that appear in the returned answer_text
    inline_refs = {int(m) for m in _REF_PATTERN.findall(result.answer_text)}

    # Build a set of URLs that are covered by the citations list
    cited_urls = {c.document_url for c in result.citations}

    # Build a mapping from chunk index (1-based) to its URL
    chunk_url_by_index = {i + 1: chunk.document_url for i, chunk in enumerate(chunks)}

    # 1. Every inline [N] in answer_text must have a citation whose URL matches
    #    the chunk at index N (i.e., the referenced chunk is cited).
    for ref in inline_refs:
        chunk_url = chunk_url_by_index.get(ref)
        assert chunk_url is not None, (
            f"Inline reference [{ref}] exceeds chunk list length {len(chunks)}"
        )
        assert chunk_url in cited_urls, (
            f"Inline reference [{ref}] points to chunk URL '{chunk_url}' "
            f"which has no matching citation. Cited URLs: {sorted(cited_urls)}"
        )

    # 2. Every citation URL must be reachable from at least one inline [N] ref.
    #    Build the set of URLs actually referenced by inline refs.
    referenced_urls = {chunk_url_by_index[ref] for ref in inline_refs if ref in chunk_url_by_index}
    for citation in result.citations:
        assert citation.document_url in referenced_urls, (
            f"Citation (number={citation.number}, url={citation.document_url}) "
            f"is not referenced by any inline [N] in answer_text. "
            f"Referenced URLs: {sorted(referenced_urls)}"
        )


# ---------------------------------------------------------------------------
# Property 14: Citation field completeness and excerpt length
# Validates: Requirements 6.2
# ---------------------------------------------------------------------------

# Feature: enterprise-rag-system, Property 14: Citation field completeness and excerpt length

_NON_EMPTY_TEXT = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters=" "),
    min_size=1,
    max_size=200,
)


@st.composite
def chunks_with_all_fields(draw) -> tuple[list[ScoredChunk], str]:
    """Strategy: generate ScoredChunks with random non-empty metadata fields
    and an answer that references all chunks."""
    n = draw(st.integers(min_value=1, max_value=10))

    chunks = []
    for i in range(n):
        source_type = draw(st.sampled_from(["docs", "github", "jira"]))
        document_title = draw(_NON_EMPTY_TEXT)
        document_url = f"http://example.com/doc-{i}-" + draw(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8)
        )
        text = draw(_NON_EMPTY_TEXT)
        chunks.append(
            ScoredChunk(
                chunk_id=f"chunk-{i}",
                source_type=source_type,
                source_id="src1",
                document_title=document_title,
                document_url=document_url,
                text=text,
                token_count=10,
                permission_tags=["engineering"],
                created_at=_NOW,
                source_modified_at=_NOW,
                relevance_score=0.9,
                rrf_score=0.0,
            )
        )

    # Build an answer that references every chunk at least once
    refs = " ".join(f"[{i + 1}]" for i in range(n))
    answer = f"Answer referencing all chunks: {refs}."

    return chunks, answer


@settings(max_examples=100, deadline=None)
@given(data=chunks_with_all_fields())
def test_property14_citation_field_completeness_and_excerpt_length(data: tuple) -> None:
    """Property 14: Every Citation produced by cite() has non-null source_type,
    document_title, document_url, and excerpt, and len(excerpt) <= 300.

    **Validates: Requirements 6.2**
    """
    chunks, answer = data
    engine = CitationEngine()
    result = engine.cite(answer, chunks)

    assert result.citations, "Expected at least one citation"

    for citation in result.citations:
        assert citation.source_type is not None, (
            f"Citation {citation.number} has null source_type"
        )
        assert citation.document_title is not None, (
            f"Citation {citation.number} has null document_title"
        )
        assert citation.document_url is not None, (
            f"Citation {citation.number} has null document_url"
        )
        assert citation.excerpt is not None, (
            f"Citation {citation.number} has null excerpt"
        )
        assert len(citation.excerpt) <= 300, (
            f"Citation {citation.number} excerpt length {len(citation.excerpt)} exceeds 300 chars"
        )


# ---------------------------------------------------------------------------
# Property 15: Citation deduplication by URL
# Validates: Requirements 6.5
# Feature: enterprise-rag-system, Property 15: Citation deduplication by URL
# ---------------------------------------------------------------------------

@st.composite
def chunks_with_overlapping_urls(draw) -> tuple[list[ScoredChunk], str]:
    """Strategy: generate N chunks where multiple chunks intentionally share
    the same document_url, and an answer that references all chunks."""
    # Number of distinct URLs (1..5)
    num_urls = draw(st.integers(min_value=1, max_value=5))
    # Total number of chunks (>= num_urls so at least one URL is shared)
    num_chunks = draw(st.integers(min_value=num_urls, max_value=num_urls + 5))

    # Assign URLs: first num_urls chunks each get a unique URL,
    # remaining chunks pick from the existing URLs (guaranteeing overlap when num_chunks > num_urls)
    urls = [f"http://example.com/doc-{u}" for u in range(num_urls)]
    chunk_urls = urls[:]  # first num_urls chunks get unique URLs
    for i in range(num_urls, num_chunks):
        # Pick an existing URL to create overlap
        shared_url = draw(st.sampled_from(urls))
        chunk_urls.append(shared_url)

    chunks = []
    for i, url in enumerate(chunk_urls):
        chunks.append(
            ScoredChunk(
                chunk_id=f"chunk-{i}",
                source_type="docs",
                source_id="src1",
                document_title=f"Document for {url}",
                document_url=url,
                text=f"Content of chunk {i} from {url}.",
                token_count=10,
                permission_tags=["engineering"],
                created_at=_NOW,
                source_modified_at=_NOW,
                relevance_score=0.9,
                rrf_score=0.0,
            )
        )

    # Build an answer that references every chunk at least once
    refs = " ".join(f"[{i + 1}]" for i in range(num_chunks))
    answer = f"Answer referencing all chunks: {refs}."

    return chunks, answer


@settings(max_examples=100, deadline=None)
@given(data=chunks_with_overlapping_urls())
def test_property15_citation_deduplication_by_url(data: tuple) -> None:
    """Property 15: When multiple chunks share the same document_url, cite()
    produces exactly one Citation per unique URL.

    **Validates: Requirements 6.5**
    """
    chunks, answer = data
    engine = CitationEngine()
    result = engine.cite(answer, chunks)

    unique_urls = {chunk.document_url for chunk in chunks}
    assert len(result.citations) == len(unique_urls), (
        f"Expected {len(unique_urls)} citations (one per unique URL), "
        f"got {len(result.citations)}. "
        f"Unique URLs: {sorted(unique_urls)}, "
        f"Citation URLs: {sorted(c.document_url for c in result.citations)}"
    )

    # Also verify no duplicate URLs appear in the citations list
    citation_urls = [c.document_url for c in result.citations]
    assert len(citation_urls) == len(set(citation_urls)), (
        f"Duplicate URLs found in citations: {citation_urls}"
    )


# ---------------------------------------------------------------------------
# Property 16: Invalid chunk references removed and flagged
# Validates: Requirements 6.4, 9.2
# Feature: enterprise-rag-system, Property 16: Invalid chunk references removed and flagged
# ---------------------------------------------------------------------------

@st.composite
def chunks_and_out_of_range_answer(draw) -> tuple[list[ScoredChunk], str]:
    """Strategy: generate N chunks (1..5), then build an answer that contains
    ONLY out-of-range [N] references (where N > len(chunks))."""
    n = draw(st.integers(min_value=1, max_value=5))
    chunks = [_make_chunk(i) for i in range(1, n + 1)]

    # Generate out-of-range reference numbers (all strictly > n)
    invalid_refs = draw(
        st.lists(
            st.integers(min_value=n + 1, max_value=n + 20),
            min_size=1,
            max_size=5,
        )
    )

    # Build an answer that contains ONLY the invalid references
    parts = [f"Claim about topic [{ref}]" for ref in invalid_refs]
    answer = ". ".join(parts) + "."

    return chunks, answer, invalid_refs


@settings(max_examples=100, deadline=None)
@given(data=chunks_and_out_of_range_answer())
def test_property16_invalid_chunk_references_removed_and_flagged(data: tuple) -> None:
    """Property 16: When the answer contains ONLY out-of-range [N] references,
    cite() must remove those references from answer_text and capture the
    corresponding claims in unverified_claims.

    **Validates: Requirements 6.4, 9.2**
    """
    chunks, answer, invalid_refs = data
    engine = CitationEngine()
    result = engine.cite(answer, chunks)

    # 1. answer_text must NOT contain any of the invalid [N] references
    for ref in invalid_refs:
        assert f"[{ref}]" not in result.answer_text, (
            f"Invalid reference [{ref}] was not removed from answer_text. "
            f"answer_text={result.answer_text!r}"
        )

    # 2. unverified_claims must be non-empty (the invalid claims were captured)
    assert len(result.unverified_claims) > 0, (
        f"Expected unverified_claims to be non-empty when answer contains only "
        f"out-of-range references {invalid_refs}, but got empty list. "
        f"answer_text={result.answer_text!r}"
    )


# ---------------------------------------------------------------------------
# Property 20: Grounding score formula correctness
# Validates: Requirements 9.3
# Feature: enterprise-rag-system, Property 20: Grounding score formula correctness
# ---------------------------------------------------------------------------

@st.composite
def chunks_and_mixed_answer(draw) -> tuple[list[ScoredChunk], str, int, int]:
    """Strategy: generate N chunks (1..10), then build an answer with a known
    mix of valid and invalid [N] refs.

    Returns (chunks, answer, valid_refs_count, total_refs_count).
    """
    n = draw(st.integers(min_value=1, max_value=10))
    chunks = [_make_chunk(i) for i in range(1, n + 1)]

    # Generate valid refs (within 1..n)
    valid_refs = draw(
        st.lists(
            st.integers(min_value=1, max_value=n),
            min_size=0,
            max_size=n * 2,
        )
    )
    # Generate invalid refs (strictly > n)
    invalid_refs = draw(
        st.lists(
            st.integers(min_value=n + 1, max_value=n + 20),
            min_size=0,
            max_size=5,
        )
    )

    all_refs = valid_refs + invalid_refs
    total_refs_count = len(all_refs)
    valid_refs_count = len(valid_refs)

    if total_refs_count == 0:
        answer = "An answer with no inline references."
    else:
        parts = [f"claim [{ref}]" for ref in all_refs]
        answer = " ".join(parts) + "."

    return chunks, answer, valid_refs_count, total_refs_count


@settings(max_examples=100, deadline=None)
@given(data=chunks_and_mixed_answer())
def test_property20_grounding_score_formula_correctness(data: tuple) -> None:
    """Property 20: grounding_score == cited_refs / total_refs, where
    cited_refs is the count of [N] refs with N in the valid chunk range,
    and total_refs is the total count of [N] refs in the answer.
    When total_refs == 0, grounding_score must be 1.0.

    **Validates: Requirements 9.3**
    """
    chunks, answer, valid_refs_count, total_refs_count = data
    engine = CitationEngine()
    score = engine.compute_grounding_score(answer, chunks)

    if total_refs_count == 0:
        assert score == 1.0, (
            f"Expected grounding_score=1.0 when no [N] refs present, got {score}"
        )
    else:
        expected = valid_refs_count / total_refs_count
        assert score == expected, (
            f"Expected grounding_score={expected} "
            f"(cited={valid_refs_count}, total={total_refs_count}), got {score}. "
            f"answer={answer!r}"
        )


# ---------------------------------------------------------------------------
# Property 21: Low-confidence behavior is consistent
# Validates: Requirements 9.4, 9.5
# Feature: enterprise-rag-system, Property 21: Low-confidence behavior is consistent
# ---------------------------------------------------------------------------

import unittest.mock


@st.composite
def chunks_and_low_confidence_answer(draw) -> tuple[list[ScoredChunk], str]:
    """Strategy: generate chunks and an answer where the grounding score will be < 0.7.

    We achieve a known low grounding score by constructing an answer with a mix
    of valid and invalid refs such that valid_count / total_count < 0.7.
    Specifically: 0 valid refs and at least 1 invalid ref → score = 0.0 < 0.7.
    """
    n = draw(st.integers(min_value=1, max_value=5))
    chunks = [_make_chunk(i) for i in range(1, n + 1)]

    # Use only invalid refs (> n) so grounding_score = 0 / total = 0.0 < 0.7
    invalid_refs = draw(
        st.lists(
            st.integers(min_value=n + 1, max_value=n + 20),
            min_size=1,
            max_size=5,
        )
    )
    parts = [f"claim [{ref}]" for ref in invalid_refs]
    answer = " ".join(parts) + "."
    return chunks, answer


@settings(max_examples=100, deadline=None)
@given(data=chunks_and_low_confidence_answer())
def test_property21_low_confidence_behavior_consistency(data: tuple) -> None:
    """Property 21: When grounding_score < 0.7, cite() must set
    low_confidence_warning=True AND emit a logger.warning call for
    administrator review.

    **Validates: Requirements 9.4, 9.5**
    """
    chunks, answer = data
    engine = CitationEngine()

    # Patch the module-level logger used by CitationEngine
    with unittest.mock.patch(
        "enterprise_rag.citation_engine.logger"
    ) as mock_logger:
        result = engine.cite(answer, chunks)

    # Verify grounding score is indeed below 0.7 (our strategy guarantees this)
    assert result.grounding_score < 0.7, (
        f"Expected grounding_score < 0.7, got {result.grounding_score}"
    )

    # 9.4: low_confidence_warning must be True
    assert result.low_confidence_warning is True, (
        f"Expected low_confidence_warning=True when grounding_score={result.grounding_score}"
    )

    # 9.5: a warning log entry must have been emitted for administrator review
    assert mock_logger.warning.called, (
        f"Expected logger.warning to be called when grounding_score={result.grounding_score}, "
        f"but it was not called."
    )


# ---------------------------------------------------------------------------
# Property 28: Graph citation field completeness
# Validates: Requirements 11.5
# Feature: enterprise-rag-system, Property 28: Graph citation field completeness
# ---------------------------------------------------------------------------

_IDENTIFIER = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"),
    min_size=1,
    max_size=20,
)

_RELATIONSHIP = st.sampled_from(["CALLS", "INHERITS", "IMPORTS", "DEFINED_IN", "CONTAINS"])


@st.composite
def graph_chunks_and_answer(draw) -> tuple[list[ScoredChunk], str]:
    """Strategy: generate ScoredChunks with source_type='graph' and
    document_title in the format '<source_node> -[<relationship>]-> <target_node>'.
    Returns chunks and an answer that references all of them.
    """
    n = draw(st.integers(min_value=1, max_value=8))
    chunks = []
    for i in range(n):
        source_node = draw(_IDENTIFIER)
        relationship = draw(_RELATIONSHIP)
        target_node = draw(_IDENTIFIER)
        document_title = f"{source_node} -[{relationship}]-> {target_node}"
        chunks.append(
            ScoredChunk(
                chunk_id=f"graph-chunk-{i}",
                source_type="graph",
                source_id="src-graph",
                document_title=document_title,
                document_url=f"http://example.com/graph/{i}",
                text=f"{source_node} {relationship} {target_node}",
                token_count=10,
                permission_tags=["engineering"],
                created_at=_NOW,
                source_modified_at=_NOW,
                relevance_score=0.9,
                rrf_score=0.0,
            )
        )

    # Answer references all chunks
    refs = " ".join(f"[{i + 1}]" for i in range(n))
    answer = f"The dependency graph shows: {refs}."
    return chunks, answer


@settings(max_examples=100, deadline=None)
@given(data=graph_chunks_and_answer())
def test_property28_graph_citation_field_completeness(data: tuple) -> None:
    """Property 28: Every GraphCitation produced by cite() for graph-derived
    chunks must have non-null source_node, relationship, target_node, and
    file_path.

    **Validates: Requirements 11.5**
    """
    chunks, answer = data
    engine = CitationEngine()
    result = engine.cite(answer, chunks)

    assert result.graph_citations, (
        "Expected at least one GraphCitation for graph-type chunks"
    )

    for gc in result.graph_citations:
        assert gc.source_node is not None and gc.source_node != "", (
            f"GraphCitation {gc.number} has null/empty source_node"
        )
        assert gc.relationship is not None and gc.relationship != "", (
            f"GraphCitation {gc.number} has null/empty relationship"
        )
        assert gc.target_node is not None and gc.target_node != "", (
            f"GraphCitation {gc.number} has null/empty target_node"
        )
        assert gc.file_path is not None and gc.file_path != "", (
            f"GraphCitation {gc.number} has null/empty file_path"
        )
