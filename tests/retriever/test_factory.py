"""Tests for the Retriever Factory."""

import pytest

from chatbot.config import RetrieverConfig
from chatbot.ingestion.store import VectorStore
from chatbot.retriever.base import BaseRetriever
from chatbot.retriever.factory import create_retriever
from chatbot.retriever.hybrid_retriever import HybridRetriever
from chatbot.retriever.keyword_retriever import KeywordRetriever
from chatbot.retriever.vector_retriever import VectorRetriever


@pytest.fixture
def store(tmp_path):
    return VectorStore(
        collection_name="test_factory",
        persist_directory=str(tmp_path / "chromadb"),
    )


class TestCreateRetriever:
    """Test config-driven retriever creation."""

    def test_creates_vector_retriever(self, store):
        config = RetrieverConfig(type="vector")
        retriever = create_retriever(config=config, store=store)
        assert isinstance(retriever, VectorRetriever)
        assert isinstance(retriever, BaseRetriever)

    def test_creates_keyword_retriever(self, store):
        config = RetrieverConfig(type="keyword")
        retriever = create_retriever(config=config, store=store)
        assert isinstance(retriever, KeywordRetriever)

    def test_creates_hybrid_retriever(self, store):
        config = RetrieverConfig(type="hybrid")
        retriever = create_retriever(config=config, store=store)
        assert isinstance(retriever, HybridRetriever)

    def test_invalid_type_raises(self, store):
        config = RetrieverConfig(type="unknown_type")
        with pytest.raises(ValueError, match="Unsupported retriever type"):
            create_retriever(config=config, store=store)

    def test_case_insensitive(self, store):
        """Retriever type should be case-insensitive."""
        config = RetrieverConfig(type="VECTOR")
        retriever = create_retriever(config=config, store=store)
        assert isinstance(retriever, VectorRetriever)

    def test_default_config(self, store):
        """With no config, should use defaults (vector)."""
        config = RetrieverConfig()
        retriever = create_retriever(config=config, store=store)
        assert isinstance(retriever, VectorRetriever)
