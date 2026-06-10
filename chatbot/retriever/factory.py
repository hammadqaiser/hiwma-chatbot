"""Retriever Factory — Creates the appropriate retriever from config.

This is the entry point for upstream modules (API, orchestrator).
They call `create_retriever(config)` and get back a ready-to-use
BaseRetriever without knowing the concrete implementation.
"""

from __future__ import annotations

from chatbot.config import AppConfig, RetrieverConfig, load_config
from chatbot.ingestion.embedder import Embedder
from chatbot.ingestion.store import VectorStore
from chatbot.retriever.base import BaseRetriever, RetrieverType
from chatbot.retriever.hybrid_retriever import HybridRetriever
from chatbot.retriever.keyword_retriever import KeywordRetriever
from chatbot.retriever.vector_retriever import VectorRetriever


def create_retriever(
    config: AppConfig | RetrieverConfig | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> BaseRetriever:
    """Create a retriever instance based on configuration.

    Args:
        config: Application or retriever configuration.
            If None, loads from default config.yaml.
        embedder: Optional shared Embedder instance.
        store: Optional shared VectorStore instance.

    Returns:
        Configured BaseRetriever instance.

    Raises:
        ValueError: If the retriever type is not supported.
    """
    # Resolve config
    if config is None:
        app_config = load_config()
        retriever_config = app_config.retriever
    elif isinstance(config, AppConfig):
        retriever_config = config.retriever
    else:
        retriever_config = config

    # Create shared dependencies if not provided
    if store is None:
        store = VectorStore(
            collection_name=retriever_config.collection_name,
            persist_directory=retriever_config.persist_directory,
        )

    retriever_type = retriever_config.type.lower()

    if retriever_type == RetrieverType.VECTOR:
        return VectorRetriever(
            embedder=embedder,
            store=store,
            config=retriever_config,
        )

    if retriever_type == RetrieverType.KEYWORD:
        return KeywordRetriever(
            store=store,
            config=retriever_config,
        )

    if retriever_type == RetrieverType.HYBRID:
        return HybridRetriever(
            embedder=embedder,
            store=store,
            config=retriever_config,
        )

    raise ValueError(
        f"Unsupported retriever type: '{retriever_type}'. "
        f"Supported types: {[t.value for t in RetrieverType]}"
    )
