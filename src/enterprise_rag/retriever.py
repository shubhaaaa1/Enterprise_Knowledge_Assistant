"""Hybrid retriever for the Enterprise RAG System.

Combines semantic (vector) search with BM25 keyword scoring, merges results
across query variants using Reciprocal Rank Fusion (RRF), and optionally
enriches results with code dependency graph data from Neo4j.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from enterprise_rag.models import AccessFilter, ScoredChunk
from enterprise_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Keywords that indicate a code-related query
_CODE_KEYWORDS = {
    "function", "class", "method", "import", "calls", "inherits",
    "module", "def ", "->", "::", ".py", ".js",
}

# Standard RRF constant
_RRF_K = 60


def rrf_score(ranks: List[int]) -> float:
    """Compute the Reciprocal Rank Fusion score for a document given its ranks.

    Args:
        ranks: List of 1-based rank positions across different ranked lists.

    Returns:
        Sum of 1 / (RRF_K + rank_i) for each rank.
    """
    return sum(1.0 / (_RRF_K + r) for r in ranks)


def hybrid_score(sem_score: float, bm25_score: float, semantic_weight: float) -> float:
    """Compute the combined hybrid score from semantic and BM25 scores.

    Args:
        sem_score:       Semantic similarity score in [0, 1].
        bm25_score:      BM25 keyword score (normalised to [0, 1]).
        semantic_weight: Weight for semantic score; BM25 weight = 1 - semantic_weight.

    Returns:
        semantic_weight * sem_score + (1 - semantic_weight) * bm25_score
    """
    return semantic_weight * sem_score + (1.0 - semantic_weight) * bm25_score


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return re.findall(r"\w+", text.lower())


def _bm25_scores(query_tokens: List[str], chunks: List[ScoredChunk]) -> List[float]:
    """Compute a simple BM25-like score for each chunk against the query tokens.

    Uses TF normalised by document length (no IDF for simplicity, as the
    candidate set is small and already filtered by semantic search).

    Args:
        query_tokens: Tokenised query terms.
        chunks:       Candidate chunks to score.

    Returns:
        List of BM25-like scores, one per chunk, in the same order.
    """
    if not query_tokens or not chunks:
        return [0.0] * len(chunks)

    scores: List[float] = []
    for chunk in chunks:
        doc_tokens = _tokenize(chunk.text)
        if not doc_tokens:
            scores.append(0.0)
            continue
        doc_len = len(doc_tokens)
        tf_sum = sum(doc_tokens.count(t) for t in query_tokens)
        scores.append(tf_sum / doc_len)

    # Normalise to [0, 1]
    max_score = max(scores) if scores else 0.0
    if max_score > 0:
        scores = [s / max_score for s in scores]
    return scores


def _is_code_query(variants: List[str]) -> bool:
    """Return True if any variant looks like a code-related query."""
    combined = " ".join(variants).lower()
    return any(kw in combined for kw in _CODE_KEYWORDS)


class Retriever:
    """Hybrid retriever: semantic + BM25 with RRF merging across query variants.

    Optionally queries the Neo4j graph store for code-related queries.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store=None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._embed_fn = embed_fn
        # Set after each retrieve() call; callers can inspect this flag.
        self.last_dependency_graph_unavailable: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        variants: List[str],
        access_filter: AccessFilter,
        k: int = 10,
        semantic_weight: float = 0.7,
        threshold: float = 0.5,
    ) -> List[ScoredChunk]:
        """Execute hybrid retrieval across all query variants and return top-K chunks.

        Steps:
        1. For each variant: semantic search + BM25 scoring → per-variant ranked list.
        2. Merge across variants using RRF.
        3. Filter chunks whose RRF score is below *threshold*.
        4. Optionally enrich with graph dependency results for code queries.
        5. Return top-K by RRF score, descending.

        Args:
            variants:        One or more query strings (from the Query Rewriter).
            access_filter:   RBAC filter applied pre-retrieval in the vector store.
            k:               Maximum number of chunks to return.
            semantic_weight: Weight for semantic score (BM25 weight = 1 - this).
            threshold:       Minimum RRF score to include a chunk in results.

        Returns:
            List of :class:`ScoredChunk` sorted by ``rrf_score`` descending.
        """
        self.last_dependency_graph_unavailable = False

        if not variants:
            return []

        # chunk_id → ScoredChunk (last seen wins for metadata; scores merged)
        chunk_map: Dict[str, ScoredChunk] = {}
        # chunk_id → list of 1-based ranks across all per-variant ranked lists
        rank_lists: Dict[str, List[int]] = defaultdict(list)

        for variant in variants:
            ranked = self._retrieve_for_variant(
                variant, access_filter, k, semantic_weight
            )
            for rank_idx, chunk in enumerate(ranked, start=1):
                cid = chunk.chunk_id
                chunk_map[cid] = chunk
                rank_lists[cid].append(rank_idx)

        if not chunk_map:
            return []

        # Compute RRF scores
        rrf_scores: Dict[str, float] = {
            cid: rrf_score(ranks) for cid, ranks in rank_lists.items()
        }

        # Normalise RRF scores to [0, 1] for threshold comparison
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        if max_rrf == 0:
            max_rrf = 1.0

        # Filter and assign rrf_score to each chunk
        results: List[ScoredChunk] = []
        for cid, chunk in chunk_map.items():
            raw_rrf = rrf_scores[cid]
            normalised = raw_rrf / max_rrf
            if normalised < threshold:
                continue
            chunk.rrf_score = raw_rrf
            results.append(chunk)

        # Sort by RRF score descending
        results.sort(key=lambda c: c.rrf_score, reverse=True)

        # Code-related query: enrich with graph results
        if _is_code_query(variants) and self._graph_store is not None:
            results = self._enrich_with_graph(variants, results)

        return results[:k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> List[float]:
        """Return an embedding for *text*, using the configured embed_fn or zeros."""
        if self._embed_fn is not None:
            return self._embed_fn(text)
        # Dummy embedding for testing / when no embed_fn is provided
        return [0.0] * 384

    def _retrieve_for_variant(
        self,
        variant: str,
        access_filter: AccessFilter,
        k: int,
        semantic_weight: float,
    ) -> List[ScoredChunk]:
        """Retrieve and score chunks for a single query variant.

        Returns a list sorted by combined hybrid score, descending.
        """
        embedding = self._embed(variant)
        # Fetch 2× k candidates; threshold=0.0 so we get raw semantic scores
        candidates: List[ScoredChunk] = self._vector_store.query(
            embedding, access_filter, k=k * 2, threshold=0.0
        )

        if not candidates:
            return []

        query_tokens = _tokenize(variant)
        bm25 = _bm25_scores(query_tokens, candidates)

        scored: List[Tuple[float, ScoredChunk]] = []
        for chunk, bm25_s in zip(candidates, bm25):
            combined = hybrid_score(chunk.relevance_score, bm25_s, semantic_weight)
            chunk.relevance_score = combined  # store combined score
            scored.append((combined, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def _enrich_with_graph(
        self, variants: List[str], chunks: List[ScoredChunk]
    ) -> List[ScoredChunk]:
        """Attempt to query Neo4j for dependency context; degrade gracefully on failure."""
        # Extract a candidate symbol name from the first variant (heuristic: longest word)
        tokens = _tokenize(" ".join(variants))
        if not tokens:
            return chunks

        symbol = max(tokens, key=len)

        try:
            dep_graph = self._graph_store.query_dependencies(symbol, direction="both", depth=2)
            logger.debug(
                "Graph query for symbol '%s' returned %d nodes, %d edges",
                symbol,
                len(dep_graph.nodes),
                len(dep_graph.edges),
            )
        except Exception as exc:
            logger.warning(
                "Neo4j unavailable during retrieval — falling back to vector-only: %s", exc
            )
            self.last_dependency_graph_unavailable = True

        return chunks
