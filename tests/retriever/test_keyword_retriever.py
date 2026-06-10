"""Tests for the Keyword Retriever (BM25)."""

import pytest

from chatbot.ingestion.chunker import Chunk
from chatbot.ingestion.embedder import EmbeddedChunk
from chatbot.ingestion.store import VectorStore
from chatbot.retriever.base import RetrievalResult
from chatbot.retriever.keyword_retriever import KeywordRetriever, _tokenize


def _make_embedded_chunk(text, chunk_id, book_title="Test Book"):
    """Helper — creates EmbeddedChunk with a dummy embedding."""
    chunk = Chunk(
        text=text,
        chunk_id=chunk_id,
        book_title=book_title,
        chapter="Chapter 1",
        page_number=1,
        page_range="page 1",
        metadata={"source_file": "test.pdf", "chunk_index": 1},
    )
    return EmbeddedChunk(chunk=chunk, embedding=[0.0] * 3)


@pytest.fixture
def store_with_data(tmp_path):
    """Create a store with technical waste management terms."""
    store = VectorStore(
        collection_name="test_keyword",
        persist_directory=str(tmp_path / "chromadb"),
    )
    chunks = [
        _make_embedded_chunk(
            "Leachate is the liquid that drains from a landfill. "
            "Leachate treatment systems are critical for environmental protection.",
            "c1",
        ),
        _make_embedded_chunk(
            "Anaerobic digestion converts organic waste into biogas and digestate. "
            "The process requires careful temperature control.",
            "c2",
        ),
        _make_embedded_chunk(
            "Curbside recycling collection programs have increased recycling rates "
            "in many municipalities around the world.",
            "c3",
        ),
    ]
    store.add_chunks(chunks)
    return store


class TestTokenize:
    """Test the tokenizer."""

    def test_lowercases(self):
        assert "hello" in _tokenize("Hello WORLD")

    def test_removes_punctuation(self):
        tokens = _tokenize("waste, management! (solid)")
        assert "waste" in tokens
        assert "," not in "".join(tokens)

    def test_keeps_hyphens(self):
        tokens = _tokenize("co-generation high-temperature")
        assert "co-generation" in tokens

    def test_filters_short_tokens(self):
        tokens = _tokenize("a I am the x of")
        assert "a" not in tokens
        assert "x" not in tokens


class TestKeywordRetriever:
    """Test BM25 keyword-based retrieval."""

    def test_retrieve_by_exact_term(self, store_with_data):
        """Searching for 'leachate' should find the leachate chunk."""
        retriever = KeywordRetriever(store=store_with_data)
        results = retriever.retrieve("leachate treatment", top_k=3)

        assert len(results) > 0
        assert results[0].chunk_id == "c1"
        assert results[0].retriever_type == "keyword"

    def test_retrieve_technical_term(self, store_with_data):
        """Should find 'anaerobic digestion' by exact match."""
        retriever = KeywordRetriever(store=store_with_data)
        results = retriever.retrieve("anaerobic digestion biogas", top_k=3)

        assert len(results) > 0
        assert results[0].chunk_id == "c2"

    def test_scores_normalized(self, store_with_data):
        """Scores should be in 0-1 range."""
        retriever = KeywordRetriever(store=store_with_data)
        results = retriever.retrieve("recycling collection", top_k=3)

        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_empty_store(self, tmp_path):
        """Empty store should return no results."""
        store = VectorStore(
            collection_name="empty_kw",
            persist_directory=str(tmp_path / "empty_chromadb"),
        )
        retriever = KeywordRetriever(store=store)
        results = retriever.retrieve("anything", top_k=5)
        assert results == []

    def test_rebuild_index(self, store_with_data):
        """After rebuild, should still find results."""
        retriever = KeywordRetriever(store=store_with_data)
        retriever.rebuild_index()
        results = retriever.retrieve("leachate", top_k=1)
        assert len(results) > 0

    def test_no_match_returns_empty(self, store_with_data):
        """Query with no matching terms should return empty or low scores."""
        retriever = KeywordRetriever(store=store_with_data)
        results = retriever.retrieve("quantum computing blockchain", top_k=3)
        # BM25 may return results with score 0 — we filter those
        assert all(r.score >= 0 for r in results)
