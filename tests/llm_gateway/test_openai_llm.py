"""Tests for OpenAI LLM Provider (mocked)."""

from unittest.mock import MagicMock

from chatbot.llm_gateway.base import LLMResponse
from chatbot.llm_gateway.openai_llm import OpenAILLM


class TestOpenAILLM:

    def _make_mock_response(self, answer="Test answer", prompt_tokens=100, completion_tokens=50):
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = answer
        mock.usage = MagicMock()
        mock.usage.prompt_tokens = prompt_tokens
        mock.usage.completion_tokens = completion_tokens
        return mock

    def test_provider_name(self):
        llm = OpenAILLM(api_key="test-key")
        assert llm.provider_name == "openai"

    def test_default_model(self):
        llm = OpenAILLM(api_key="test-key")
        assert llm.model == "gpt-4o-mini"

    def test_generate_returns_llm_response(self):
        llm = OpenAILLM(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            answer="Recycling reduces waste.",
        )
        llm._client = mock_client

        result = llm.generate("What is recycling?", "Context.")
        assert isinstance(result, LLMResponse)
        assert result.answer == "Recycling reduces waste."
        assert result.provider == "openai"

    def test_base_url_support(self):
        """Should support custom base_url for Azure OpenAI."""
        llm = OpenAILLM(api_key="test-key", base_url="https://my-azure.openai.azure.com")
        assert llm._base_url == "https://my-azure.openai.azure.com"
