"""Hybrid Retriever — Combines vector (semantic) and keyword (BM25) search.

Technical documents benefit from both approaches:
- **Vector search**: Captures semantic meaning ("waste disposal methods" ≈ "refuse management techniques")
- **Keyword search**: Catches exact technical terms ("leachate", "BOD5", "tipping fee")

The hybrid approach merges results from both with configurable weights,
then deduplicates by chunk_id and re-ranks by combined score.
"""

from __future__ import annotations

from chatbot.config import RetrieverConfig
from chatbot.ingestion.embedder import Embedder
from chatbot.ingestion.store import VectorStore
from chatbot.retriever.base import BaseRetriever, RetrievalResult
from chatbot.retriever.keyword_retriever import KeywordRetriever
from chatbot.retriever.vector_retriever import VectorRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic + keyword search.

    Retrieves from both vector and keyword retrievers, merges results
    with weighted scoring, deduplicates, and returns top-k.

    Default weights: 70% semantic, 30% keyword.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        keyword_retriever: KeywordRetriever | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        config: RetrieverConfig | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """Initialize hybrid retriever.

        Args:
            vector_retriever: Pre-configured vector retriever (creates one if None).
            keyword_retriever: Pre-configured keyword retriever (creates one if None).
            embedder: Shared embedder instance.
            store: Shared vector store instance.
            config: Retriever configuration.
            vector_weight: Weight for semantic results (default: 0.7).
            keyword_weight: Weight for keyword results (default: 0.3).
        """
        self._config = config or RetrieverConfig()

        # Ensure weights sum to 1.0
        total = vector_weight + keyword_weight
        self.vector_weight = vector_weight / total
        self.keyword_weight = keyword_weight / total

        # Create sub-retrievers if not provided
        self._vector = vector_retriever or VectorRetriever(
            embedder=embedder, store=store, config=self._config
        )
        self._keyword = keyword_retriever or KeywordRetriever(
            store=store, config=self._config
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve using both semantic and keyword search, then merge.

        Args:
            query: User's question.
            top_k: Number of final results to return.

        Returns:
            Merged, deduplicated list of RetrievalResult sorted by combined score.
        """
        if top_k is None:
            top_k = self._config.top_k

        # Fetch more from each to have a good pool for merging
        fetch_k = top_k * 2

        # Run both retrievers
        vector_results = self._vector.retrieve(query, top_k=fetch_k)
        keyword_results = self._keyword.retrieve(query, top_k=fetch_k)

        # Merge with weighted scores, dedup by chunk_id
        merged = self._merge_results(vector_results, keyword_results)

        # Sort by combined score and return top-k
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    def _merge_results(
        self,
        vector_results: list[RetrievalResult],
        keyword_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Merge results from both retrievers with weighted scoring.

        If a chunk appears in both result sets, its scores are combined.
        If it appears in only one, only that weight contributes.
        """
        # Index by chunk_id for deduplication
        combined: dict[str, RetrievalResult] = {}

        for result in vector_results:
            key = result.chunk_id or result.text[:100]
            if key in combined:
                # Already seen — add vector score
                existing = combined[key]
                combined[key] = RetrievalResult(
                    text=existing.text,
                    score=existing.score + result.score * self.vector_weight,
                    book_title=existing.book_title,
                    chapter=existing.chapter,
                    page_number=existing.page_number,
                    page_range=existing.page_range,
                    chunk_id=existing.chunk_id,
                    source_file=existing.source_file,
                    retriever_type="hybrid",
                    metadata=existing.metadata,
                )
            else:
                combined[key] = RetrievalResult(
                    text=result.text,
                    score=result.score * self.vector_weight,
                    book_title=result.book_title,
                    chapter=result.chapter,
                    page_number=result.page_number,
                    page_range=result.page_range,
                    chunk_id=result.chunk_id,
                    source_file=result.source_file,
                    retriever_type="hybrid",
                    metadata=result.metadata,
                )

        for result in keyword_results:
            key = result.chunk_id or result.text[:100]
            if key in combined:
                existing = combined[key]
                combined[key] = RetrievalResult(
                    text=existing.text,
                    score=existing.score + result.score * self.keyword_weight,
                    book_title=existing.book_title,
                    chapter=existing.chapter,
                    page_number=existing.page_number,
                    page_range=existing.page_range,
                    chunk_id=existing.chunk_id,
                    source_file=existing.source_file,
                    retriever_type="hybrid",
                    metadata=existing.metadata,
                )
            else:
                combined[key] = RetrievalResult(
                    text=result.text,
                    score=result.score * self.keyword_weight,
                    book_title=result.book_title,
                    chapter=result.chapter,
                    page_number=result.page_number,
                    page_range=result.page_range,
                    chunk_id=result.chunk_id,
                    source_file=result.source_file,
                    retriever_type="hybrid",
                    metadata=result.metadata,
                )

        return list(combined.values())
