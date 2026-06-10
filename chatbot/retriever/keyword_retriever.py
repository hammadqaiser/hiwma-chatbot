"""Keyword Retriever — BM25-based keyword search for technical terms.

Waste management books contain specific technical terms like "leachate",
"anaerobic digestion", "commingled recyclables" that benefit from exact
keyword matching rather than just semantic similarity.

BM25 complements vector search by catching exact term matches that
embedding models might miss.
"""

from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from chatbot.config import RetrieverConfig
from chatbot.ingestion.store import VectorStore
from chatbot.retriever.base import BaseRetriever, RetrievalResult


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Lowercases and removes punctuation for BM25 indexing.
    """
    text = text.lower()
    # Remove punctuation but keep hyphens (common in technical terms)
    text = re.sub(r"[^\w\s\-]", " ", text)
    tokens = text.split()
    # Filter very short tokens
    return [t for t in tokens if len(t) > 1]


class KeywordRetriever(BaseRetriever):
    """BM25-based keyword retriever for exact term matching.

    Indexes all chunks from the vector store and builds a BM25
    index for keyword-based retrieval. Useful for technical terms
    that semantic embeddings might not capture well.

    Note: The BM25 index is built lazily on first query and cached.
    To refresh after ingestion, call `rebuild_index()`.
    """

    def __init__(
        self,
        store: VectorStore | None = None,
        config: RetrieverConfig | None = None,
    ):
        self._config = config or RetrieverConfig()
        self._store = store
        self._bm25: BM25Okapi | None = None
        self._corpus_chunks: list[dict] = []  # Raw chunk data from ChromaDB

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(
                collection_name=self._config.collection_name,
                persist_directory=self._config.persist_directory,
            )
        return self._store

    def _build_index(self) -> None:
        """Build the BM25 index from all chunks in the vector store.

        Fetches all documents from ChromaDB and creates a BM25 index.
        This is called lazily on the first query.
        """
        count = self.store.count()
        if count == 0:
            self._bm25 = None
            self._corpus_chunks = []
            return

        # Fetch all documents from ChromaDB
        # ChromaDB get() returns all documents if no IDs specified
        all_data = self.store.collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        self._corpus_chunks = []
        tokenized_corpus = []

        if all_data and all_data["ids"]:
            for i, doc_id in enumerate(all_data["ids"]):
                text = all_data["documents"][i] if all_data.get("documents") else ""
                metadata = all_data["metadatas"][i] if all_data.get("metadatas") else {}

                self._corpus_chunks.append({
                    "chunk_id": doc_id,
                    "text": text,
                    "metadata": metadata,
                })
                tokenized_corpus.append(_tokenize(text))

        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None

    def rebuild_index(self) -> None:
        """Force rebuild of the BM25 index (call after new ingestion)."""
        self._bm25 = None
        self._corpus_chunks = []
        self._build_index()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve chunks by BM25 keyword relevance.

        Args:
            query: User's question.
            top_k: Number of results.

        Returns:
            List of RetrievalResult sorted by BM25 score.
        """
        if top_k is None:
            top_k = self._config.top_k

        # Lazy build index
        if self._bm25 is None:
            self._build_index()

        if self._bm25 is None or not self._corpus_chunks:
            return []

        # Tokenize query and score
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1.0

        results = []
        for idx, score in scored_indices:
            if score <= 0:
                continue

            chunk = self._corpus_chunks[idx]
            metadata = chunk["metadata"]

            results.append(
                RetrievalResult(
                    text=chunk["text"],
                    score=score / max_score,  # Normalize to 0-1
                    book_title=metadata.get("book_title", ""),
                    chapter=metadata.get("chapter", ""),
                    page_number=metadata.get("page_number", 0),
                    page_range=metadata.get("page_range", ""),
                    chunk_id=chunk["chunk_id"],
                    source_file=metadata.get("source_file", ""),
                    retriever_type="keyword",
                    metadata=metadata,
                )
            )

        return results
