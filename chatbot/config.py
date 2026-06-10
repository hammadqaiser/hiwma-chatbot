"""Configuration loader — reads config.yaml and environment variables.

This is the single source of truth for all pipeline settings.
To reuse for another project: edit config.yaml — no code changes needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


# Load .env file if present
load_dotenv()


class DocumentsConfig(BaseModel):
    source_dir: str = "./Books"
    supported_formats: list[str] = Field(default_factory=lambda: ["pdf"])
    exclude_patterns: list[str] = Field(default_factory=lambda: ["*resume*", "*Resume*"])


class IngestionConfig(BaseModel):
    parser: str = "pymupdf"
    chunk_size: int = 50000
    chunk_overlap: int = 200
    min_chunk_size: int = 100


class RetrieverConfig(BaseModel):
    type: str = "vector"
    vector_db: str = "chromadb"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    collection_name: str = "waste_management_books"
    persist_directory: str = "./data/chromadb"


class LLMConfig(BaseModel):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 2048

    @property
    def api_key(self) -> str | None:
        """Load API key from environment based on provider name."""
        key_map = {
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = key_map.get(self.provider.lower())
        if env_var:
            return os.getenv(env_var)
        return None


class ChatHistoryConfig(BaseModel):
    enabled: bool = True
    storage: str = "sqlite"
    database_path: str = "./data/chat_history.db"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit: int = 60


class AppConfig(BaseModel):
    """Complete application configuration."""

    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    chat_history: ChatHistoryConfig = Field(default_factory=ChatHistoryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to ./config.yaml in project root.

    Returns:
        Validated AppConfig instance.
    """
    if config_path is None:
        # Search for config.yaml relative to the project root
        config_path = Path(__file__).parent.parent / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        # Return defaults if no config file exists
        return AppConfig()

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    # Remove 'project' key — it's metadata, not config
    raw.pop("project", None)

    return AppConfig(**raw)
