"""Tests for Groq LLM Provider (mocked — no real API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from chatbot.llm_gateway.base import LLMResponse, TokenUsage
from chatbot.llm_gateway.groq_llm import GroqLLM


class TestGroqLLM:
    """Test Groq provider with mocked SDK responses."""

    def _make_mock_response(self, answer="Test answer", prompt_tokens=100, completion_tokens=50):
        """Create a mock Groq completion response."""
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = answer
        mock.usage = MagicMock()
        mock.usage.prompt_tokens = prompt_tokens
        mock.usage.completion_tokens = completion_tokens
        return mock

    def test_provider_name(self):
        llm = GroqLLM(api_key="test-key")
        assert llm.provider_name == "groq"

    def test_default_model(self):
        llm = GroqLLM(api_key="test-key")
        assert llm.model == "llama-3.1-70b-versatile"

    def test_generate_returns_llm_response(self):
        """generate() should return a properly structured LLMResponse."""
        llm = GroqLLM(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            answer="Waste is collected weekly.",
            prompt_tokens=150,
            completion_tokens=30,
        )
        llm._client = mock_client

        result = llm.generate("How is waste collected?", "Context about collection.")

        assert isinstance(result, LLMResponse)
        assert result.answer == "Waste is collected weekly."
        assert result.provider == "groq"
        assert result.model == "llama-3.1-70b-versatile"
        assert result.usage.prompt_tokens == 150
        assert result.usage.completion_tokens == 30
        assert result.usage.total_tokens == 180

    def test_generate_passes_correct_params(self):
        """Should pass temperature, max_tokens, model to the SDK."""
        llm = GroqLLM(api_key="test-key", model="mixtral-8x7b-32768", temperature=0.5, max_tokens=1024)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        llm._client = mock_client

        llm.generate("test query", "test context")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "mixtral-8x7b-32768"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1024

    def test_response_to_dict(self):
        """LLMResponse.to_dict() should serialize cleanly."""
        llm = GroqLLM(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response()
        llm._client = mock_client

        result = llm.generate("test", "context")
        d = result.to_dict()

        assert "answer" in d
        assert "model" in d
        assert "provider" in d
        assert "usage" in d
        assert d["usage"]["total_tokens"] == 150
