"""Tests for the configuration loader."""

from pathlib import Path

import pytest

from chatbot.config import AppConfig, load_config


class TestLoadConfig:
    """Test loading configuration from YAML and defaults."""

    def test_load_defaults_when_no_file(self, tmp_path):
        """When no config file exists, should return defaults."""
        config = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(config, AppConfig)
        assert config.llm.provider == "groq"
        assert config.retriever.embedding_model == "all-MiniLM-L6-v2"
        assert config.chat_history.enabled is True

    def test_load_from_yaml(self, tmp_path):
        """Should load settings from a YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
project:
  name: "Test Project"

llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.5

retriever:
  top_k: 10
  collection_name: "test_collection"
""")
        config = load_config(config_file)
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o-mini"
        assert config.llm.temperature == 0.5
        assert config.retriever.top_k == 10
        assert config.retriever.collection_name == "test_collection"
        # Defaults should still work for unspecified fields
        assert config.retriever.vector_db == "chromadb"

    def test_api_key_from_env(self, monkeypatch):
        """API keys should be loaded from environment variables."""
        monkeypatch.setenv("GROQ_API_KEY", "test_groq_key_123")
        config = load_config(Path("nonexistent_config.yaml"))
        assert config.llm.api_key == "test_groq_key_123"

    def test_api_key_for_different_providers(self, monkeypatch):
        """Each provider should map to its own env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
        config = AppConfig(llm={"provider": "openai"})
        assert config.llm.api_key == "test_openai_key"

    def test_ollama_has_no_api_key(self):
        """Ollama (local) should not require an API key."""
        config = AppConfig(llm={"provider": "ollama"})
        assert config.llm.api_key is None
