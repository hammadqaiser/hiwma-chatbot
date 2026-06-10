"""Anthropic Claude LLM Provider.

Uses the Anthropic SDK to access Claude 3.5 Haiku and other Claude models.
"""

from __future__ import annotations

from chatbot.llm_gateway.base import BaseLLM, LLMResponse, TokenUsage
from chatbot.llm_gateway.prompt_templates import build_system_prompt, build_user_prompt


class AnthropicLLM(BaseLLM):
    """Anthropic Claude provider using the Anthropic Python SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-latest",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._api_key = api_key
        self._client = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, query: str, context: str) -> LLMResponse:
        """Generate a response using Anthropic API.

        Note: Anthropic uses a different message format than OpenAI —
        system prompt goes in a separate parameter, not in messages.
        """
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query, context)

        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Extract answer text from content blocks
        answer = ""
        if response.content:
            answer = "".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens or 0,
            completion_tokens=response.usage.output_tokens or 0,
        )

        return LLMResponse(
            answer=answer,
            model=self.model,
            provider=self.provider_name,
            usage=usage,
        )
