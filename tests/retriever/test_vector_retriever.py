"""Tests for the Vector Retriever."""

import pytest

from chatbot.config import RetrieverConfig
from chatbot.ingestion.chunker import Chunk
from chatbot.ingestion.embedder import EmbeddedChunk, Embedder
from chatbot.ingestion.store import VectorStore
from chatbot.retriever.base import RetrievalResult
from chatbot.retriever.vector_retriever import VectorRetriever


def _make_embedded_chunk(text, chunk_id, embedding, book_title="Test Book"):
    """Helper to create test EmbeddedChunks."""
    chunk = Chunk(
        text=text,
        chunk_id=chunk_id,
        book_title=book_title,
        chapter="Chapter 1",
        page_number=1,
        page_range="page 1",
        metadata={"source_file": "test.pdf", "chunk_index": 1},
    )
    return EmbeddedChunk(chunk=chunk, embedding=embedding)


@pytest.fixture
def store_with_data(tmp_path):
    """Create a VectorStore with test data pre-loaded."""
    store = VectorStore(
        collection_name="test_retriever",
        persist_directory=str(tmp_path / "chromadb"),
    )
    chunks = [
        _make_embedded_chunk(
            "Solid waste collection involves curbside pickup and transfer stations.",
            "c1", [1.0, 0.0, 0.0], "Waste Management Handbook"
        ),
        _make_embedded_chunk(
            "Recycling reduces landfill usage and conserves natural resources.",
            "c2", [0.0, 1.0, 0.0], "Waste Management Handbook"
        ),
        _make_embedded_chunk(
            "Leachate treatment is critical for preventing groundwater contamination.",
            "c3", [0.0, 0.0, 1.0], "Environmental Engineering"
        ),
    ]
    store.add_chunks(chunks)
    return store


class TestVectorRetriever:
    """Test vector (semantic) retrieval."""

    def test_retrieve_returns_results(self, store_with_data):
        """Should return RetrievalResult objects."""
        # Use a mock embedder that returns a fixed vector
        class MockEmbedder:
            def embed_text(self, text):
                return [0.9, 0.1, 0.0]  # Similar to chunk c1

        retriever = VectorRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
        )
        results = retriever.retrieve("waste collection methods", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].retriever_type == "vector"

    def test_most_similar_ranked_first(self, store_with_data):
        """The most semantically similar chunk should be ranked first."""
        class MockEmbedder:
            def embed_text(self, text):
                return [1.0, 0.0, 0.0]  # Exact match with c1

        retriever = VectorRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
        )
        results = retriever.retrieve("collection", top_k=3)

        assert results[0].chunk_id == "c1"
        assert results[0].score >= results[1].score

    def test_result_has_metadata(self, store_with_data):
        """Results should carry book_title, chapter, page info."""
        class MockEmbedder:
            def embed_text(self, text):
                return [1.0, 0.0, 0.0]

        retriever = VectorRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
        )
        results = retriever.retrieve("test", top_k=1)

        assert results[0].book_title == "Waste Management Handbook"
        assert results[0].chapter == "Chapter 1"
        assert results[0].page_number == 1

    def test_retrieve_with_filter(self, store_with_data):
        """Should filter by metadata (e.g., book title)."""
        class MockEmbedder:
            def embed_text(self, text):
                return [0.5, 0.5, 0.5]

        retriever = VectorRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
        )
        results = retriever.retrieve_with_filter(
            "test", top_k=5,
            filters={"book_title": "Environmental Engineering"},
        )

        assert len(results) == 1
        assert results[0].book_title == "Environmental Engineering"

    def test_empty_store_returns_empty(self, tmp_path):
        """Empty store should return no results."""
        store = VectorStore(
            collection_name="empty_test",
            persist_directory=str(tmp_path / "empty_chromadb"),
        )

        class MockEmbedder:
            def embed_text(self, text):
                return [1.0, 0.0, 0.0]

        retriever = VectorRetriever(embedder=MockEmbedder(), store=store)
        results = retriever.retrieve("anything", top_k=5)
        assert results == []

    def test_uses_config_top_k(self, store_with_data):
        """Should use config's top_k when not explicitly provided."""
        class MockEmbedder:
            def embed_text(self, text):
                return [0.5, 0.5, 0.5]

        config = RetrieverConfig(top_k=2)
        retriever = VectorRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
            config=config,
        )
        results = retriever.retrieve("test")  # No top_k arg
        assert len(results) == 2

    def test_to_dict_serialization(self, store_with_data):
        """RetrievalResult.to_dict() should produce a clean dict."""
        class MockEmbedder:
            def embed_text(self, text):
                return [1.0, 0.0, 0.0]

        retriever = VectorRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
        )
        results = retriever.retrieve("test", top_k=1)
        d = results[0].to_dict()

        assert "text" in d
        assert "score" in d
        assert "book_title" in d
        assert "retriever_type" in d
        assert d["retriever_type"] == "vector"
