"""Base Retriever — Abstract interface for all retrieval strategies.

All retrievers (vector, keyword, hybrid) implement this interface.
This ensures Module 3 (LLM Gateway) and Module 4 (API) only depend
on the abstract interface, not on specific retriever implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class RetrieverType(str, Enum):
    """Supported retriever types (maps to config.yaml `retriever.type`)."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """A single retrieval result with relevance metadata.

    This is the universal return type for all retrievers.
    Downstream modules (LLM Gateway) depend only on this dataclass.
    """

    text: str
    score: float  # Relevance score (0.0 to 1.0, higher = more relevant)
    book_title: str
    chapter: str
    page_number: int
    page_range: str
    chunk_id: str = ""
    source_file: str = ""
    retriever_type: str = ""  # Which retriever found this result
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "text": self.text,
            "score": round(self.score, 4),
            "book_title": self.book_title,
            "chapter": self.chapter,
            "page_number": self.page_number,
            "page_range": self.page_range,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "retriever_type": self.retriever_type,
        }


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies.

    Every retriever must implement:
    - retrieve(query, top_k) → List[RetrievalResult]

    Optional overrides:
    - retrieve_with_filter(query, top_k, filters) for metadata filtering
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the most relevant document chunks for a query.

        Args:
            query: The user's question or search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects sorted by relevance (highest first).
        """
        ...

    def retrieve_with_filter(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with optional metadata filters.

        Default implementation ignores filters — subclasses can override
        to support filtering by book_title, chapter, etc.

        Args:
            query: The user's question.
            top_k: Number of results.
            filters: Optional metadata filters (e.g., {"book_title": "..."}).

        Returns:
            Filtered list of RetrievalResult objects.
        """
        # Default: ignore filters, just call retrieve
        return self.retrieve(query, top_k)
