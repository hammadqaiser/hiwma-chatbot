"""Ollama LLM Provider — Local model inference (no API key needed).

Uses raw HTTP to call the local Ollama REST API.
No SDK dependency — just httpx for HTTP requests.
"""

from __future__ import annotations

import json

from chatbot.llm_gateway.base import BaseLLM, LLMResponse, TokenUsage
from chatbot.llm_gateway.prompt_templates import build_messages


class OllamaLLM(BaseLLM):
    """Ollama provider for local model inference.

    Requires Ollama running locally: https://ollama.ai
    Default endpoint: http://localhost:11434
    """

    def __init__(
        self,
        model: str = "llama3.1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.base_url = base_url.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    def generate(self, query: str, context: str) -> LLMResponse:
        """Generate a response using local Ollama API."""
        import httpx

        messages = build_messages(query, context)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120.0,  # Local models can be slow
        )
        response.raise_for_status()
        data = response.json()

        # Extract usage if available
        usage = TokenUsage()
        if "prompt_eval_count" in data:
            usage = TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            )

        answer = data.get("message", {}).get("content", "")

        return LLMResponse(
            answer=answer,
            model=self.model,
            provider=self.provider_name,
            usage=usage,
        )

    def health_check(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            import httpx
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
