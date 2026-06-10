"""OpenAI LLM Provider.

Uses the OpenAI SDK — also compatible with Azure OpenAI.
"""

from __future__ import annotations

from chatbot.llm_gateway.base import BaseLLM, LLMResponse, TokenUsage
from chatbot.llm_gateway.prompt_templates import build_messages


class OpenAILLM(BaseLLM):
    """OpenAI provider using the OpenAI Python SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str | None = None,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._api_key = api_key
        self._base_url = base_url  # For Azure OpenAI or compatible APIs
        self._client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def generate(self, query: str, context: str) -> LLMResponse:
        """Generate a response using OpenAI API."""
        messages = build_messages(query, context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

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
