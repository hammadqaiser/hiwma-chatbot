"""Base LLM — Abstract interface for all LLM providers.

All providers (Groq, Google, OpenAI, Anthropic, Ollama) implement this
interface. Upstream modules (API, Orchestrator) depend only on BaseLLM
and LLMResponse — never on specific providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class ProviderType(str, Enum):
    """Supported LLM providers (maps to config.yaml `llm.provider`)."""

    GROQ = "groq"
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class TokenUsage:
    """Token usage statistics for cost tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider.

    This is the universal return type. Upstream modules never see
    provider-specific response objects.
    """

    answer: str
    model: str
    provider: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "answer": self.answer,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.to_dict(),
        }


class BaseLLM(ABC):
    """Abstract base class for all LLM providers.

    Every provider must implement:
    - generate(query, context) → LLMResponse

    The `context` parameter is the formatted context string from
    Module 2's context_formatter.format_context().
    """

    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'groq', 'google')."""
        ...

    @abstractmethod
    def generate(self, query: str, context: str) -> LLMResponse:
        """Generate a response given a query and retrieved context.

        Args:
            query: The user's question.
            context: Formatted context string from the retriever
                (produced by context_formatter.format_context()).

        Returns:
            LLMResponse with the grounded answer, model info, and token usage.

        Raises:
            ConnectionError: If the provider API is unreachable.
            ValueError: If the API key is missing or invalid.
        """
        ...

    def health_check(self) -> bool:
        """Check if the provider is reachable and configured.

        Default implementation returns True. Providers can override
        to perform actual health checks (e.g., ping the API).
        """
        return True
