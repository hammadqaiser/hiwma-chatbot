"""Tests for the LLM Factory."""

import os

import pytest

from chatbot.config import LLMConfig
from chatbot.llm_gateway.base import BaseLLM
from chatbot.llm_gateway.factory import create_llm
from chatbot.llm_gateway.groq_llm import GroqLLM
from chatbot.llm_gateway.google_llm import GoogleLLM
from chatbot.llm_gateway.openai_llm import OpenAILLM
from chatbot.llm_gateway.anthropic_llm import AnthropicLLM
from chatbot.llm_gateway.ollama_llm import OllamaLLM


class TestCreateLLM:
    """Test config-driven LLM provider creation."""

    def test_creates_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        config = LLMConfig(provider="groq")
        llm = create_llm(config=config)
        assert isinstance(llm, GroqLLM)
        assert isinstance(llm, BaseLLM)

    def test_creates_google(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        config = LLMConfig(provider="google")
        llm = create_llm(config=config)
        assert isinstance(llm, GoogleLLM)

    def test_creates_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = LLMConfig(provider="openai")
        llm = create_llm(config=config)
        assert isinstance(llm, OpenAILLM)

    def test_creates_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = LLMConfig(provider="anthropic")
        llm = create_llm(config=config)
        assert isinstance(llm, AnthropicLLM)

    def test_creates_ollama_no_api_key(self):
        """Ollama should work without any API key."""
        config = LLMConfig(provider="ollama", model="llama3.1")
        llm = create_llm(config=config)
        assert isinstance(llm, OllamaLLM)

    def test_missing_api_key_raises(self, monkeypatch):
        """Cloud providers without API key should raise ValueError."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        config = LLMConfig(provider="groq")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            create_llm(config=config)

    def test_invalid_provider_raises(self):
        config = LLMConfig(provider="unknown_provider")
        with pytest.raises(ValueError, match="Unsupported"):
            create_llm(config=config)

    def test_case_insensitive(self, monkeypatch):
        """Retriever type should be case-insensitive."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        config = LLMConfig(provider="GROQ")
        llm = create_llm(config=config)
        assert isinstance(llm, GroqLLM)

    def test_passes_model_params(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        config = LLMConfig(provider="groq", model="mixtral-8x7b-32768", temperature=0.5, max_tokens=1024)
        llm = create_llm(config=config)
        assert llm.model == "mixtral-8x7b-32768"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1024
