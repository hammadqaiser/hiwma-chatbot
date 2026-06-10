"""Groq LLM Provider — Default provider (user has API keys).

Uses the Groq SDK to access Llama 3.1, Mixtral, Gemma models
with fast inference. Free tier available.
"""

from __future__ import annotations

from chatbot.llm_gateway.base import BaseLLM, LLMResponse, TokenUsage
from chatbot.llm_gateway.prompt_templates import build_messages


class GroqLLM(BaseLLM):
    """Groq provider using the Groq Python SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._api_key = api_key
        self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def client(self):
        """Lazy-initialize the Groq client."""
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    def generate(self, query: str, context: str) -> LLMResponse:
        """Generate a response using Groq API.

        Args:
            query: User's question.
            context: Formatted context from retriever.

        Returns:
            LLMResponse with grounded answer and token usage.
        """
        messages = build_messages(query, context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Extract token usage
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )

        return LLMResponse(
            answer=response.choices[0].message.content or "",
            model=self.model,
            provider=self.provider_name,
            usage=usage,
        )
