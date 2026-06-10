"""Tests for the vector store (ChromaDB)."""

import pytest

from chatbot.ingestion.chunker import Chunk
from chatbot.ingestion.embedder import EmbeddedChunk
from chatbot.ingestion.store import VectorStore


def _make_embedded_chunk(
    text: str, chunk_id: str, embedding: list[float], book_title: str = "Test Book"
) -> EmbeddedChunk:
    """Helper to create EmbeddedChunk for testing."""
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
def store(tmp_path):
    """Create a temporary vector store for testing."""
    return VectorStore(
        collection_name="test_collection",
        persist_directory=str(tmp_path / "chromadb"),
    )


class TestVectorStore:
    """Test ChromaDB vector store operations."""

    def test_empty_store(self, store):
        """New store should have zero chunks."""
        assert store.count() == 0

    def test_add_and_count(self, store):
        """Adding chunks should increase count."""
        chunks = [
            _make_embedded_chunk("Waste management intro", "chunk_1", [0.1] * 10),
            _make_embedded_chunk("Collection systems", "chunk_2", [0.2] * 10),
        ]
        added = store.add_chunks(chunks)
        assert added == 2
        assert store.count() == 2

    def test_search_returns_results(self, store):
        """Search should return relevant results."""
        chunks = [
            _make_embedded_chunk("Solid waste collection methods", "c1", [1.0, 0.0, 0.0]),
            _make_embedded_chunk("Recycling and composting", "c2", [0.0, 1.0, 0.0]),
            _make_embedded_chunk("Landfill design principles", "c3", [0.0, 0.0, 1.0]),
        ]
        store.add_chunks(chunks)

        # Search with a vector similar to the first chunk
        results = store.search(query_embedding=[0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"  # Most similar

    def test_search_result_has_metadata(self, store):
        """Search results should include all metadata."""
        chunks = [
            _make_embedded_chunk("Test content", "c1", [1.0, 0.0], book_title="My Book"),
        ]
        store.add_chunks(chunks)

        results = store.search(query_embedding=[1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].book_title == "My Book"
        assert results[0].chapter == "Chapter 1"
        assert results[0].page_number == 1

    def test_clear_removes_all(self, store):
        """Clear should remove all chunks."""
        chunks = [
            _make_embedded_chunk("Content", "c1", [1.0, 0.0]),
        ]
        store.add_chunks(chunks)
        assert store.count() == 1

        store.clear()
        # After clear, the collection is deleted; recreating it should be empty
        new_store = VectorStore(
            collection_name=store.collection_name,
            persist_directory=store.persist_directory,
        )
        assert new_store.count() == 0

    def test_get_stats(self, store):
        """Stats should return collection info."""
        stats = store.get_stats()
        assert "collection_name" in stats
        assert "total_chunks" in stats
        assert stats["total_chunks"] == 0

    def test_add_empty_list(self, store):
        """Adding empty list should return 0."""
        assert store.add_chunks([]) == 0

    def test_upsert_overwrites(self, store):
        """Adding chunks with same IDs should update, not duplicate."""
        chunk1 = _make_embedded_chunk("Original text", "c1", [1.0, 0.0])
        store.add_chunks([chunk1])
        assert store.count() == 1

        chunk2 = _make_embedded_chunk("Updated text", "c1", [0.0, 1.0])
        store.add_chunks([chunk2])
        assert store.count() == 1  # Still 1, not 2
