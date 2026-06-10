"""Google Gemini LLM Provider.

Uses the google-genai SDK to access Gemini 2.0 Flash and other models.
"""

from __future__ import annotations

from chatbot.llm_gateway.base import BaseLLM, LLMResponse, TokenUsage
from chatbot.llm_gateway.prompt_templates import build_system_prompt, build_user_prompt


class GoogleLLM(BaseLLM):
    """Google Gemini provider using the google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._api_key = api_key
        self._client = None

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def client(self):
        """Lazy-initialize the Google GenAI client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, query: str, context: str) -> LLMResponse:
        """Generate a response using Google Gemini API.

        Args:
            query: User's question.
            context: Formatted context from retriever.

        Returns:
            LLMResponse with grounded answer and token usage.
        """
        from google.genai import types

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query, context)

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

        # Extract token usage
        usage = TokenUsage()
        if response.usage_metadata:
            usage = TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
            )

        return LLMResponse(
            answer=response.text or "",
            model=self.model,
            provider=self.provider_name,
            usage=usage,
        )
