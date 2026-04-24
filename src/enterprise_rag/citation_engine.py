"""Citation Engine for the Enterprise RAG System.

Maps [N] references in generated answers to source chunk metadata,
computes grounding scores, and flags unverified claims.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from enterprise_rag.models import (
    Citation,
    CitedAnswer,
    GraphCitation,
    ScoredChunk,
)

logger = logging.getLogger(__name__)

# Regex to find all [N] references in answer text
_REF_PATTERN = re.compile(r"\[(\d+)\]")


class CitationEngine:
    """Maps answer [N] references to chunk citations and computes grounding scores."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cite(self, answer: str, chunks: List[ScoredChunk]) -> CitedAnswer:
        """Parse [N] references in *answer*, build Citation / GraphCitation objects,
        compute grounding score, and flag unverified claims.

        Args:
            answer: Raw answer text produced by the Generator (may contain [N] refs).
            chunks: Ordered list of ScoredChunks (1-indexed in the answer).

        Returns:
            A fully populated CitedAnswer dataclass.
        """
        # 1. Identify all [N] references and which are valid
        all_refs = [int(m) for m in _REF_PATTERN.findall(answer)]
        valid_indices = set(range(1, len(chunks) + 1))

        valid_refs = sorted({r for r in all_refs if r in valid_indices})
        invalid_refs = sorted({r for r in all_refs if r not in valid_indices})

        # 2. Build Citations (deduplicated by document_url)
        citations = self._build_citations(valid_refs, chunks)

        # 3. Build GraphCitations for graph-derived chunks
        graph_citations = self._build_graph_citations(valid_refs, chunks)

        # 4. Compute grounding score
        grounding_score = self.compute_grounding_score(answer, chunks)

        # 5. Flag unverified claims — remove sentences containing invalid refs
        cleaned_answer, unverified_claims = self._extract_unverified(answer, chunks)

        # 6. Low-confidence warning
        low_confidence = grounding_score < 0.7
        if low_confidence:
            logger.warning(
                "low_confidence answer: grounding_score=%.2f", grounding_score
            )

        return CitedAnswer(
            answer_text=cleaned_answer,
            citations=citations,
            graph_citations=graph_citations,
            grounding_score=grounding_score,
            unverified_claims=unverified_claims,
            low_confidence_warning=low_confidence,
            dependency_graph_unavailable=False,
        )

    def compute_grounding_score(
        self, answer: str, chunks: List[ScoredChunk]
    ) -> float:
        """Compute grounding score = cited_claims / total_claims.

        - total_claims  = number of [N] references in the answer
        - cited_claims  = number of [N] references where N is a valid chunk index
        - Returns 0.0 when there are no claims.
        """
        all_refs = [int(m) for m in _REF_PATTERN.findall(answer)]
        total = len(all_refs)
        if total == 0:
            return 1.0
        valid_indices = set(range(1, len(chunks) + 1))
        cited = sum(1 for r in all_refs if r in valid_indices)
        return cited / total

    def flag_unverified_claims(
        self, answer: str, chunks: List[ScoredChunk]
    ) -> str:
        """Return the answer with sentences/claims containing invalid [N] refs removed."""
        cleaned, _ = self._extract_unverified(answer, chunks)
        return cleaned

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_citations(
        self, valid_refs: List[int], chunks: List[ScoredChunk]
    ) -> List[Citation]:
        """Build one Citation per unique document_url, merging chunk_ids."""
        # url -> (citation_number, citation_obj)
        url_map: Dict[str, Citation] = {}
        citation_number = 1

        for ref in valid_refs:
            chunk = chunks[ref - 1]  # 1-indexed
            url = chunk.document_url
            if url in url_map:
                existing = url_map[url]
                # Merge chunk_id into existing citation
                if chunk.chunk_id not in existing.chunk_ids:
                    existing.chunk_ids.append(chunk.chunk_id)
                    # Merge excerpt: append new excerpt up to 300 chars each
                    new_excerpt = chunk.text[:300]
                    if new_excerpt not in existing.excerpt:
                        existing.excerpt = existing.excerpt + " … " + new_excerpt
            else:
                url_map[url] = Citation(
                    number=citation_number,
                    source_type=chunk.source_type,
                    document_title=chunk.document_title,
                    document_url=url,
                    excerpt=chunk.text[:300],
                    chunk_ids=[chunk.chunk_id],
                )
                citation_number += 1

        return list(url_map.values())

    def _build_graph_citations(
        self, valid_refs: List[int], chunks: List[ScoredChunk]
    ) -> List[GraphCitation]:
        """Build GraphCitation objects for graph-derived chunks."""
        graph_citations: List[GraphCitation] = []
        gc_number = 1

        for ref in valid_refs:
            chunk = chunks[ref - 1]
            if chunk.source_type == "graph":
                # Extract graph metadata from chunk text or use defaults
                source_node, relationship, target_node = self._parse_graph_metadata(
                    chunk
                )
                graph_citations.append(
                    GraphCitation(
                        number=gc_number,
                        source_node=source_node,
                        relationship=relationship,
                        target_node=target_node,
                        file_path=chunk.document_url,
                        source_id=chunk.source_id,
                    )
                )
                gc_number += 1

        return graph_citations

    def _parse_graph_metadata(
        self, chunk: ScoredChunk
    ) -> Tuple[str, str, str]:
        """Extract source_node, relationship, target_node from a graph chunk.

        Expects chunk.document_title to encode the relationship as
        "<source_node> -[RELATIONSHIP]-> <target_node>" when available,
        otherwise falls back to the chunk title / url.
        """
        title = chunk.document_title or ""
        # Try to parse "A -[REL]-> B" pattern
        m = re.match(r"^(.+?)\s*-\[(.+?)\]->\s*(.+)$", title)
        if m:
            return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        # Fallback: use title as source_node, "RELATED_TO" as relationship, url as target
        return title or chunk.chunk_id, "RELATED_TO", chunk.document_url

    def _extract_unverified(
        self, answer: str, chunks: List[ScoredChunk]
    ) -> Tuple[str, List[str]]:
        """Remove sentences containing invalid [N] refs; return (cleaned, unverified).

        A sentence is considered a claim if it contains at least one [N] reference.
        If ALL references in a sentence are invalid, the sentence is removed and
        added to unverified_claims.
        """
        valid_indices = set(range(1, len(chunks) + 1))

        # Split into sentences (simple split on '. ', '! ', '? ', or newlines)
        sentences = re.split(r"(?<=[.!?])\s+|\n", answer)

        kept: List[str] = []
        unverified: List[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            refs_in_sentence = [int(m) for m in _REF_PATTERN.findall(sentence)]
            if not refs_in_sentence:
                # No references — keep as-is
                kept.append(sentence)
            else:
                has_valid = any(r in valid_indices for r in refs_in_sentence)
                has_invalid = any(r not in valid_indices for r in refs_in_sentence)
                if has_invalid and not has_valid:
                    # Entirely unverified — remove
                    unverified.append(sentence)
                else:
                    # At least one valid ref — keep (mixed or fully valid)
                    kept.append(sentence)

        cleaned = " ".join(kept)
        return cleaned, unverified
