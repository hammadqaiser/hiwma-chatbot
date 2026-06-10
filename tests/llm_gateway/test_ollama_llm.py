"""Tests for Ollama LLM Provider (mocked — no local server needed)."""

from unittest.mock import MagicMock, patch

from chatbot.llm_gateway.base import LLMResponse
from chatbot.llm_gateway.ollama_llm import OllamaLLM


class TestOllamaLLM:

    def test_provider_name(self):
        llm = OllamaLLM()
        assert llm.provider_name == "ollama"

    def test_default_model(self):
        llm = OllamaLLM()
        assert llm.model == "llama3.1"

    def test_no_api_key_required(self):
        """Ollama should work without any API key."""
        llm = OllamaLLM(model="mistral")
        assert llm.model == "mistral"

    def test_generate_returns_llm_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Incineration reduces volume by 90%."},
            "prompt_eval_count": 120,
            "eval_count": 25,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            llm = OllamaLLM()
            result = llm.generate("What is incineration?", "Context.")

            assert isinstance(result, LLMResponse)
            assert result.answer == "Incineration reduces volume by 90%."
            assert result.provider == "ollama"
            assert result.usage.prompt_tokens == 120
            assert result.usage.completion_tokens == 25

    def test_health_check_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.get", return_value=mock_response):
            llm = OllamaLLM()
            assert llm.health_check() is True

    def test_health_check_failure(self):
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            llm = OllamaLLM()
            assert llm.health_check() is False

    def test_custom_base_url(self):
        llm = OllamaLLM(base_url="http://192.168.1.100:11434")
        assert llm.base_url == "http://192.168.1.100:11434"
