"""Vector Retriever — Semantic similarity search using embeddings.

Wraps Module 1's VectorStore and Embedder to provide semantic search.
This is the primary retrieval strategy — it embeds the user's query
and finds the most similar document chunks by cosine similarity.
"""

from __future__ import annotations

from chatbot.config import RetrieverConfig
from chatbot.ingestion.embedder import Embedder
from chatbot.ingestion.store import VectorStore
from chatbot.retriever.base import BaseRetriever, RetrievalResult


class VectorRetriever(BaseRetriever):
    """Semantic retriever using vector similarity search.

    Flow: query → embed → search ChromaDB → return RetrievalResults

    Uses Module 1's Embedder (same model used for ingestion) and
    VectorStore (ChromaDB) — no duplication of logic.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        config: RetrieverConfig | None = None,
    ):
        """Initialize the vector retriever.

        Args:
            embedder: Embedder instance (lazy-creates one if not provided).
            store: VectorStore instance (lazy-creates one if not provided).
            config: Retriever configuration (uses defaults if not provided).
        """
        self._config = config or RetrieverConfig()
        self._embedder = embedder
        self._store = store

    @property
    def embedder(self) -> Embedder:
        """Lazy-initialize the embedder."""
        if self._embedder is None:
            self._embedder = Embedder(model_name=self._config.embedding_model)
        return self._embedder

    @property
    def store(self) -> VectorStore:
        """Lazy-initialize the vector store."""
        if self._store is None:
            self._store = VectorStore(
                collection_name=self._config.collection_name,
                persist_directory=self._config.persist_directory,
            )
        return self._store

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve chunks by semantic similarity.

        Args:
            query: User's question.
            top_k: Number of results (defaults to config value).

        Returns:
            List of RetrievalResult sorted by similarity score.
        """
        if top_k is None:
            top_k = self._config.top_k

        # Step 1: Embed the query using the same model as ingestion
        query_embedding = self.embedder.embed_text(query)

        # Step 2: Search the vector store
        search_results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Step 3: Convert SearchResult → RetrievalResult
        return [
            RetrievalResult(
                text=sr.text,
                score=max(0.0, sr.score),  # Clamp to non-negative
                book_title=sr.book_title,
                chapter=sr.chapter,
                page_number=sr.page_number,
                page_range=sr.page_range,
                chunk_id=sr.chunk_id,
                source_file=sr.metadata.get("source_file", ""),
                retriever_type="vector",
                metadata=sr.metadata,
            )
            for sr in search_results
        ]

    def retrieve_with_filter(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve with metadata filtering (e.g., filter by book).

        Args:
            query: User's question.
            top_k: Number of results.
            filters: ChromaDB where clause (e.g., {"book_title": "..."}).

        Returns:
            Filtered RetrievalResult list.
        """
        if top_k is None:
            top_k = self._config.top_k

        query_embedding = self.embedder.embed_text(query)

        search_results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filters,
        )

        return [
            RetrievalResult(
                text=sr.text,
                score=max(0.0, sr.score),
                book_title=sr.book_title,
                chapter=sr.chapter,
                page_number=sr.page_number,
                page_range=sr.page_range,
                chunk_id=sr.chunk_id,
                source_file=sr.metadata.get("source_file", ""),
                retriever_type="vector",
                metadata=sr.metadata,
            )
            for sr in search_results
        ]
