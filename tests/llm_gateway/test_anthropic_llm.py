"""Tests for Anthropic Claude LLM Provider (mocked)."""

from unittest.mock import MagicMock

from chatbot.llm_gateway.base import LLMResponse
from chatbot.llm_gateway.anthropic_llm import AnthropicLLM


class TestAnthropicLLM:

    def _make_mock_response(self, text="Test answer", input_tokens=100, output_tokens=50):
        mock = MagicMock()
        block = MagicMock()
        block.text = text
        mock.content = [block]
        mock.usage = MagicMock()
        mock.usage.input_tokens = input_tokens
        mock.usage.output_tokens = output_tokens
        return mock

    def test_provider_name(self):
        llm = AnthropicLLM(api_key="test-key")
        assert llm.provider_name == "anthropic"

    def test_default_model(self):
        llm = AnthropicLLM(api_key="test-key")
        assert llm.model == "claude-3-5-haiku-latest"

    def test_generate_returns_llm_response(self):
        llm = AnthropicLLM(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response(
            text="Composting reduces organic waste.",
            input_tokens=180,
            output_tokens=35,
        )
        llm._client = mock_client

        result = llm.generate("What is composting?", "Context.")
        assert isinstance(result, LLMResponse)
        assert result.answer == "Composting reduces organic waste."
        assert result.provider == "anthropic"
        assert result.usage.prompt_tokens == 180
        assert result.usage.completion_tokens == 35

    def test_uses_separate_system_param(self):
        """Anthropic requires system prompt as separate param, not in messages."""
        llm = AnthropicLLM(api_key="test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response()
        llm._client = mock_client

        llm.generate("test", "context")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 100  # Non-trivial system prompt
        # Messages should NOT contain a system role
        messages = call_kwargs["messages"]
        assert all(m["role"] != "system" for m in messages)
