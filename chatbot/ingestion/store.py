"""Vector Store — Stores and retrieves embedded chunks using ChromaDB.

ChromaDB is free, file-based, and perfect for ≤1000 users.
This module provides a clean interface that can be swapped for
Pinecone, Qdrant, or PGVector later without changing upstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from chatbot.ingestion.chunker import Chunk
from chatbot.ingestion.embedder import EmbeddedChunk


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk_id: str
    text: str
    score: float  # Similarity score (higher = more similar)
    metadata: dict

    @property
    def book_title(self) -> str:
        return self.metadata.get("book_title", "")

    @property
    def chapter(self) -> str:
        return self.metadata.get("chapter", "")

    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", 0)

    @property
    def page_range(self) -> str:
        return self.metadata.get("page_range", "")


class VectorStore:
    """ChromaDB-backed vector store for document chunks.

    Provides add, search, and management operations.
    Data persists to disk and survives restarts.
    """

    def __init__(
        self,
        collection_name: str = "waste_management_books",
        persist_directory: str = "./data/chromadb",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection: Any = None

    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
        return self._collection

    def add_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Add embedded chunks to the vector store.

        Args:
            embedded_chunks: List of chunks with embeddings.

        Returns:
            Number of chunks successfully added.
        """
        if not embedded_chunks:
            return 0

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for ec in embedded_chunks:
            chunk = ec.chunk
            ids.append(chunk.chunk_id)
            embeddings.append(ec.embedding)
            documents.append(chunk.text)
            metadatas.append({
                "book_title": chunk.book_title,
                "chapter": chunk.chapter,
                "page_number": chunk.page_number,
                "page_range": chunk.page_range,
                "source_file": chunk.metadata.get("source_file", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
            })

        # ChromaDB has a max batch size (~5461), so upsert in batches
        BATCH_SIZE = 5000
        for start in range(0, len(ids), BATCH_SIZE):
            end = start + BATCH_SIZE
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        return len(ids)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter (e.g., {"book_title": "..."}).

        Returns:
            List of SearchResult objects, sorted by similarity.
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.count()),
        }

        if filter_metadata:
            kwargs["where"] = filter_metadata

        if kwargs["n_results"] == 0:
            return []

        results = self.collection.query(**kwargs)

        search_results: list[SearchResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity scores
                distance = results["distances"][0][i] if results.get("distances") else 0.0
                score = 1.0 - distance  # cosine distance → cosine similarity

                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        text=results["documents"][0][i] if results.get("documents") else "",
                        score=score,
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                    )
                )

        return search_results

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all chunks from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = None

    def get_stats(self) -> dict:
        """Return statistics about the vector store."""
        count = self.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
        }
