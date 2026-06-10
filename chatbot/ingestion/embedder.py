"""Embedder — Generates vector embeddings from text chunks using sentence-transformers.

Uses all-MiniLM-L6-v2 by default (free, runs locally, 384-dimensional).
No API costs — the model is downloaded and runs on CPU/GPU locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chatbot.ingestion.chunker import Chunk


@dataclass
class EmbeddedChunk:
    """A chunk with its vector embedding attached."""

    chunk: Chunk
    embedding: list[float]

    @property
    def dimension(self) -> int:
        return len(self.embedding)


class Embedder:
    """Generates embeddings using sentence-transformers (free, local).

    The model is loaded lazily on first use to avoid slow imports
    when the embedder isn't needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the current model."""
        # Use the new method name (renamed from get_sentence_embedding_dimension)
        if hasattr(self.model, 'get_embedding_dimension'):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts in batch (more efficient).

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process at once.

        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> list[EmbeddedChunk]:
        """Embed a list of chunks.

        Args:
            chunks: List of Chunk objects to embed.
            batch_size: Batch size for encoding.

        Returns:
            List of EmbeddedChunk objects with embeddings attached.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        return [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
