"""Tests for Google Gemini LLM Provider (mocked — no real API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from chatbot.llm_gateway.base import LLMResponse, TokenUsage
from chatbot.llm_gateway.google_llm import GoogleLLM


class TestGoogleLLM:
    """Test Google Gemini provider with mocked SDK responses."""

    def _make_mock_response(self, text="Test answer", prompt_tokens=100, completion_tokens=50):
        mock = MagicMock()
        mock.text = text
        mock.usage_metadata = MagicMock()
        mock.usage_metadata.prompt_token_count = prompt_tokens
        mock.usage_metadata.candidates_token_count = completion_tokens
        return mock

    def test_provider_name(self):
        llm = GoogleLLM(api_key="test-key")
        assert llm.provider_name == "google"

    def test_default_model(self):
        llm = GoogleLLM(api_key="test-key")
        assert llm.model == "gemini-2.0-flash"

    def test_generate_returns_llm_response(self):
        llm = GoogleLLM(api_key="test-key")
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._make_mock_response(
            text="Landfills are engineered facilities.",
            prompt_tokens=200,
            completion_tokens=40,
        )
        llm._client = mock_client

        result = llm.generate("What is a landfill?", "Context about landfills.")

        assert isinstance(result, LLMResponse)
        assert result.answer == "Landfills are engineered facilities."
        assert result.provider == "google"
        assert result.usage.prompt_tokens == 200
        assert result.usage.total_tokens == 240

    def test_generate_with_custom_params(self):
        llm = GoogleLLM(api_key="test-key", model="gemini-1.5-pro", temperature=0.3, max_tokens=4096)
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = self._make_mock_response()
        llm._client = mock_client

        result = llm.generate("test", "context")

        assert result.model == "gemini-1.5-pro"
        assert result.provider == "google"
