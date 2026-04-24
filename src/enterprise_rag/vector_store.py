"""ChromaDB-backed vector store for the Enterprise RAG System."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from enterprise_rag.models import (
    AccessFilter,
    Chunk,
    ComponentStatus,
    EmbeddedChunk,
    ScoredChunk,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "enterprise_rag"

# Prefix used for per-tag boolean metadata fields
_TAG_PREFIX = "tag_"


def _tag_key(tag: str) -> str:
    """Return the metadata key for a permission tag."""
    return f"{_TAG_PREFIX}{tag}"


class VectorStore:
    """Persistent ChromaDB vector store with RBAC pre-retrieval filtering."""

    def __init__(self, persist_path: str = "./chroma_db") -> None:
        self._persist_path = persist_path
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, chunks: List[EmbeddedChunk]) -> None:
        """Store embedded chunks with full metadata including permission_tags."""
        if not chunks:
            return

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(self._chunk_to_metadata(chunk))

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: List[float],
        access_filter: AccessFilter,
        k: int = 10,
        threshold: float = 0.5,
    ) -> List[ScoredChunk]:
        """Semantic search with RBAC pre-retrieval filter applied before scoring."""
        where_filter = self._build_where_filter(access_filter)

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        scored_chunks: List[ScoredChunk] = []

        ids_list = results.get("ids", [[]])[0]
        distances_list = results.get("distances", [[]])[0]
        metadatas_list = results.get("metadatas", [[]])[0]
        documents_list = results.get("documents", [[]])[0]

        for chunk_id, distance, metadata, document in zip(
            ids_list, distances_list, metadatas_list, documents_list
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            relevance_score = max(0.0, 1.0 - distance)

            if relevance_score < threshold:
                continue

            scored_chunks.append(
                self._metadata_to_scored_chunk(
                    chunk_id=chunk_id,
                    text=document,
                    metadata=metadata,
                    relevance_score=relevance_score,
                )
            )

        return scored_chunks

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a single chunk by its ID."""
        try:
            results = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.error("ChromaDB get_by_id failed for %s: %s", chunk_id, exc)
            return None

        ids = results.get("ids", [])
        if not ids:
            return None

        metadatas = results.get("metadatas", [{}])
        documents = results.get("documents", [""])

        return self._metadata_to_chunk(
            chunk_id=ids[0],
            text=documents[0],
            metadata=metadatas[0],
        )

    def health_check(self) -> ComponentStatus:
        """Return the health status of the ChromaDB connection."""
        now = datetime.now(tz=timezone.utc)
        try:
            self._client.heartbeat()
            return ComponentStatus(
                name="chromadb",
                status="ok",
                last_checked=now,
                detail=f"collection='{COLLECTION_NAME}', path='{self._persist_path}'",
            )
        except Exception as exc:
            logger.error("ChromaDB health check failed: %s", exc)
            return ComponentStatus(
                name="chromadb",
                status="down",
                last_checked=now,
                detail=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_to_metadata(chunk: EmbeddedChunk) -> dict:
        """Flatten an EmbeddedChunk into a ChromaDB-compatible metadata dict.

        permission_tags are stored as individual boolean flags (tag_<name>=True)
        so ChromaDB's $eq filter can be used for reliable RBAC pre-filtering.
        The original list is also stored as JSON for reconstruction.
        """
        meta: dict = {
            "source_type": chunk.source_type,
            "source_id": chunk.source_id,
            "document_title": chunk.document_title,
            "document_url": chunk.document_url,
            "token_count": chunk.token_count,
            "permission_tags": json.dumps(chunk.permission_tags),
            "created_at": chunk.created_at.isoformat(),
            "source_modified_at": chunk.source_modified_at.isoformat(),
        }
        # Add one boolean flag per tag for reliable $eq filtering
        for tag in chunk.permission_tags:
            meta[_tag_key(tag)] = True
        return meta

    @staticmethod
    def _metadata_to_chunk(chunk_id: str, text: str, metadata: dict) -> Chunk:
        permission_tags = json.loads(metadata.get("permission_tags", "[]"))
        return Chunk(
            chunk_id=chunk_id,
            source_type=metadata.get("source_type", ""),
            source_id=metadata.get("source_id", ""),
            document_title=metadata.get("document_title", ""),
            document_url=metadata.get("document_url", ""),
            text=text,
            token_count=int(metadata.get("token_count", 0)),
            permission_tags=permission_tags,
            created_at=datetime.fromisoformat(metadata["created_at"]),
            source_modified_at=datetime.fromisoformat(metadata["source_modified_at"]),
        )

    @classmethod
    def _metadata_to_scored_chunk(
        cls,
        chunk_id: str,
        text: str,
        metadata: dict,
        relevance_score: float,
    ) -> ScoredChunk:
        base = cls._metadata_to_chunk(chunk_id, text, metadata)
        return ScoredChunk(
            chunk_id=base.chunk_id,
            source_type=base.source_type,
            source_id=base.source_id,
            document_title=base.document_title,
            document_url=base.document_url,
            text=base.text,
            token_count=base.token_count,
            permission_tags=base.permission_tags,
            created_at=base.created_at,
            source_modified_at=base.source_modified_at,
            relevance_score=relevance_score,
            rrf_score=0.0,
        )

    @staticmethod
    def _build_where_filter(access_filter: AccessFilter) -> Optional[dict]:
        """Build a ChromaDB $where filter for RBAC pre-retrieval.

        Each permission tag is stored as a boolean flag (tag_<name>=True).
        We filter with $eq: True for each permitted tag and OR them together.
        Empty permitted_tags → deny-all via an impossible filter.
        """
        if not access_filter.permitted_tags:
            return {"__deny_all__": {"$eq": True}}

        if len(access_filter.permitted_tags) == 1:
            return {_tag_key(access_filter.permitted_tags[0]): {"$eq": True}}

        return {
            "$or": [
                {_tag_key(tag): {"$eq": True}}
                for tag in access_filter.permitted_tags
            ]
        }
