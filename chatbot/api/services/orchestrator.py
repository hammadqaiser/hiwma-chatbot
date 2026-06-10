"""Orchestrator — Wires Retriever + LLM Gateway into a single chat flow.

This is the core business logic of the chatbot:
  query → retrieve context → format for LLM → generate answer → return response

Upstream (routes) call orchestrator.chat() and get back a clean response.
Orchestrator owns the retriever and LLM instances (created once, reused).
"""

from __future__ import annotations

import logging
import re

from chatbot.config import AppConfig, load_config
from chatbot.llm_gateway.base import BaseLLM
from chatbot.llm_gateway.factory import create_llm
from chatbot.retriever.base import BaseRetriever
from chatbot.retriever.context_formatter import format_context, format_sources
from chatbot.retriever.factory import create_retriever

logger = logging.getLogger(__name__)

# Patterns for casual/greeting messages that don't need RAG
_CASUAL_PATTERNS = [
    r'^(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening|day)|yo|sup|what\'?s\s*up)\b',
    r'^(thanks?|thank\s*you|thx|ty|cheers|much\s*appreciated)\b',
    r'^(bye|goodbye|see\s*you|take\s*care|good\s*night)\b',
    r'^(how\s*are\s*you|how\'?s\s*it\s*going|what\s*do\s*you\s*do|who\s*are\s*you|what\s*can\s*you\s*do)',
    r'^(ok|okay|got\s*it|understood|sure|nice|great|awesome|cool)\b',
    r'^(help|menu|start)\s*$',
]


class ChatResult:
    """Result of a single chat interaction."""

    def __init__(
        self,
        answer: str,
        sources: list[dict],
        model: str,
        provider: str,
        usage: dict,
    ):
        self.answer = answer
        self.sources = sources
        self.model = model
        self.provider = provider
        self.usage = usage


class Orchestrator:
    """Orchestrates the full RAG pipeline: retrieve → format → generate.

    Created once at app startup. Reuses retriever and LLM instances
    across requests for efficiency.
    """

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        llm: BaseLLM | None = None,
        config: AppConfig | None = None,
    ):
        self._config = config or load_config()
        self._retriever = retriever
        self._llm = llm

    @property
    def retriever(self) -> BaseRetriever:
        if self._retriever is None:
            self._retriever = create_retriever(config=self._config)
        return self._retriever

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm(config=self._config)
        return self._llm

    @staticmethod
    def _is_casual_message(query: str) -> bool:
        """Check if the query is a greeting/casual message (not a knowledge question)."""
        cleaned = query.strip().lower()
        # Very short messages are usually casual
        if len(cleaned) <= 3 and cleaned.isalpha():
            return True
        for pattern in _CASUAL_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                return True
        return False

    def chat(self, query: str, top_k: int | None = None) -> ChatResult:
        """Process a chat query through the full RAG pipeline.

        Args:
            query: User's question.
            top_k: Number of chunks to retrieve (defaults to config).

        Returns:
            ChatResult with answer, sources, and usage.
        """
        logger.info(f"Processing query: {query[:100]}...")

        # Handle casual messages (greetings, thanks, etc.) without RAG
        if self._is_casual_message(query):
            logger.info("Casual message detected — skipping RAG retrieval")
            from chatbot.llm_gateway.prompt_templates import CASUAL_PROMPT_TEMPLATE
            casual_context = CASUAL_PROMPT_TEMPLATE.format(query=query)
            llm_response = self.llm.generate(query=query, context=casual_context)
            return ChatResult(
                answer=llm_response.answer,
                sources=[],
                model=llm_response.model,
                provider=llm_response.provider,
                usage=llm_response.usage.to_dict(),
            )

        # Step 1: Retrieve relevant chunks
        retrieval_results = self.retriever.retrieve(query, top_k=top_k or self._config.retriever.top_k)
        logger.info(f"Retrieved {len(retrieval_results)} chunks")

        # Step 2: Format context for LLM
        context = format_context(retrieval_results)
        sources = format_sources(retrieval_results)

        # Step 3: Generate answer
        llm_response = self.llm.generate(query=query, context=context)
        logger.info(f"Generated answer ({llm_response.usage.total_tokens} tokens)")

        return ChatResult(
            answer=llm_response.answer,
            sources=sources,
            model=llm_response.model,
            provider=llm_response.provider,
            usage=llm_response.usage.to_dict(),
        )

