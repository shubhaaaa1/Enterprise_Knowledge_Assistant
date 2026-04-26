"""File-based documentation source connector."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from enterprise_rag.ingestion.base import SourceConnector
from enterprise_rag.models import Document

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".html"}


class DocsConnector(SourceConnector):
    """Walks a local directory and yields Documents for text/doc files."""

    def __init__(
        self,
        source_id: str,
        base_path: str,
        permission_tags: Optional[List[str]] = None,
    ) -> None:
        self.source_id = source_id
        self.base_path = base_path
        self.permission_tags: List[str] = permission_tags or []

    def _iter_files(self) -> Iterator[str]:
        for root, _dirs, files in os.walk(self.base_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _SUPPORTED_EXTENSIONS:
                    yield os.path.join(root, fname)

    def _make_document(self, file_path: str) -> Document:
        stat = os.stat(file_path)
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        return Document(
            doc_id=str(uuid.uuid4()),
            source_type="docs",
            source_id=self.source_id,
            title=os.path.basename(file_path),
            url=file_path,
            content=content,
            permission_tags=list(self.permission_tags),
            modified_at=modified_at,
        )

    def fetch_all(self) -> Iterator[Document]:
        for file_path in self._iter_files():
            yield self._make_document(file_path)

    def fetch_incremental(self, since: datetime) -> Iterator[Document]:
        for file_path in self._iter_files():
            stat = os.stat(file_path)
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if mtime > since:
                yield self._make_document(file_path)
