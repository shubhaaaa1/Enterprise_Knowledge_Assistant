"""Property-based tests for VectorStore — access filter soundness."""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Feature: enterprise-rag-system, Property 5: Access filter soundness

pytest.importorskip("chromadb")

from enterprise_rag.models import AccessFilter, EmbeddedChunk  # noqa: E402
from enterprise_rag.vector_store import VectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROLES = ["engineering", "hr", "finance", "legal", "ops"]

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_chunk(index: int, permission_tags: List[str]) -> EmbeddedChunk:
    """Create an EmbeddedChunk with a unique index-based embedding."""
    embedding = [float(index)] + [0.0] * 9
    return EmbeddedChunk(
        chunk_id=str(uuid.uuid4()),
        source_type="docs",
        source_id="src1",
        document_title=f"Doc {index}",
        document_url=f"https://example.com/doc/{index}",
        text=f"content for chunk {index}",
        token_count=4,
        permission_tags=permission_tags,
        created_at=_NOW,
        source_modified_at=_NOW,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Property 5: Access filter soundness
# Validates: Requirements 2.2, 2.3, 4.2
# ---------------------------------------------------------------------------

@given(
    permitted_tags=st.lists(
        st.sampled_from(_ROLES), min_size=1, unique=True
    ),
    chunk_tag_sets=st.lists(
        st.lists(st.sampled_from(_ROLES), min_size=1, unique=True),
        min_size=1,
        max_size=20,
    ),
)
@settings(max_examples=20, deadline=None)
def test_access_filter_soundness(permitted_tags, chunk_tag_sets):
    """No returned chunk should have permission_tags completely disjoint from
    the user's permitted_tags.

    Validates: Requirements 2.2, 2.3, 4.2
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        store = VectorStore(persist_path=tmp_dir)

        chunks = [_make_chunk(i, tags) for i, tags in enumerate(chunk_tag_sets)]
        store.upsert(chunks)

        access_filter = AccessFilter(
            permitted_source_ids=[],
            permitted_tags=permitted_tags,
        )

        query_embedding = [1.0] + [0.0] * 9
        results = store.query(query_embedding, access_filter, k=len(chunks), threshold=0.0)

        permitted_set = set(permitted_tags)
        for chunk in results:
            assert set(chunk.permission_tags) & permitted_set, (
                f"Chunk {chunk.chunk_id} with permission_tags={chunk.permission_tags} "
                f"was returned but has no overlap with permitted_tags={permitted_tags}"
            )
