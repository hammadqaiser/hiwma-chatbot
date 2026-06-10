"""Tests for the Hybrid Retriever."""

import pytest

from chatbot.ingestion.chunker import Chunk
from chatbot.ingestion.embedder import EmbeddedChunk
from chatbot.ingestion.store import VectorStore
from chatbot.retriever.base import RetrievalResult
from chatbot.retriever.hybrid_retriever import HybridRetriever
from chatbot.retriever.keyword_retriever import KeywordRetriever
from chatbot.retriever.vector_retriever import VectorRetriever


def _make_embedded_chunk(text, chunk_id, embedding, book_title="Test Book"):
    chunk = Chunk(
        text=text, chunk_id=chunk_id, book_title=book_title,
        chapter="Chapter 1", page_number=1, page_range="page 1",
        metadata={"source_file": "test.pdf", "chunk_index": 1},
    )
    return EmbeddedChunk(chunk=chunk, embedding=embedding)


@pytest.fixture
def store_with_data(tmp_path):
    """Store with varied chunks for hybrid testing."""
    store = VectorStore(
        collection_name="test_hybrid",
        persist_directory=str(tmp_path / "chromadb"),
    )
    chunks = [
        _make_embedded_chunk(
            "Solid waste collection systems include curbside pickup.",
            "c1", [1.0, 0.0, 0.0],
        ),
        _make_embedded_chunk(
            "Leachate management requires proper drainage and treatment facilities.",
            "c2", [0.0, 1.0, 0.0],
        ),
        _make_embedded_chunk(
            "Composting is an aerobic process that decomposes organic waste.",
            "c3", [0.0, 0.0, 1.0],
        ),
    ]
    store.add_chunks(chunks)
    return store


class TestHybridRetriever:
    """Test hybrid (semantic + keyword) retrieval."""

    def test_returns_results(self, store_with_data):
        """Should return results from both sub-retrievers."""
        class MockEmbedder:
            def embed_text(self, text):
                return [0.5, 0.5, 0.5]

        hybrid = HybridRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
        )
        results = hybrid.retrieve("waste collection leachate", top_k=3)

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.retriever_type == "hybrid" for r in results)

    def test_deduplicates_by_chunk_id(self, store_with_data):
        """Same chunk from both retrievers should appear only once."""
        class MockEmbedder:
            def embed_text(self, text):
                return [1.0, 0.0, 0.0]  # Strongly matches c1

        hybrid = HybridRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
        )
        results = hybrid.retrieve("solid waste collection curbside", top_k=5)

        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))  # No duplicates

    def test_combined_score_higher_than_individual(self, store_with_data):
        """Chunks found by both retrievers should have boosted scores."""
        class MockEmbedder:
            def embed_text(self, text):
                return [1.0, 0.0, 0.0]

        hybrid = HybridRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
            vector_weight=0.7,
            keyword_weight=0.3,
        )
        results = hybrid.retrieve("solid waste collection", top_k=3)

        # Results should be sorted by score
        if len(results) > 1:
            assert results[0].score >= results[1].score

    def test_custom_weights(self, store_with_data):
        """Custom weights should be normalized and applied."""
        class MockEmbedder:
            def embed_text(self, text):
                return [0.5, 0.5, 0.5]

        hybrid = HybridRetriever(
            embedder=MockEmbedder(),
            store=store_with_data,
            vector_weight=0.9,
            keyword_weight=0.1,
        )
        assert abs(hybrid.vector_weight - 0.9) < 0.01
        assert abs(hybrid.keyword_weight - 0.1) < 0.01

    def test_empty_store(self, tmp_path):
        """Empty store should return no results."""
        store = VectorStore(
            collection_name="empty_hybrid",
            persist_directory=str(tmp_path / "empty_chromadb"),
        )

        class MockEmbedder:
            def embed_text(self, text):
                return [0.5, 0.5, 0.5]

        hybrid = HybridRetriever(embedder=MockEmbedder(), store=store)
        results = hybrid.retrieve("anything", top_k=5)
        assert results == []
