"""Abstract base class for source connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator, List

from enterprise_rag.models import ComponentStatus, Document


class SourceConnector(ABC):
    """Abstract base for all source connectors."""

    @abstractmethod
    def fetch_all(self) -> Iterator[Document]:
        """Yield all documents from the source."""
        ...

    @abstractmethod
    def fetch_incremental(self, since: datetime) -> Iterator[Document]:
        """Yield only documents modified after *since*."""
        ...

    def health_check(self) -> ComponentStatus:
        """Return a ComponentStatus indicating the connector is reachable."""
        return ComponentStatus(
            name=self.__class__.__name__,
            status="ok",
            last_checked=datetime.utcnow(),
            detail=None,
        )
