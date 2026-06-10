"""LLM Factory — Creates the appropriate LLM provider from config.

Entry point for upstream modules. They call `create_llm(config)`
and get a ready-to-use BaseLLM without knowing the provider.
"""

from __future__ import annotations

from chatbot.config import AppConfig, LLMConfig, load_config
from chatbot.llm_gateway.base import BaseLLM, ProviderType


def create_llm(
    config: AppConfig | LLMConfig | None = None,
) -> BaseLLM:
    """Create an LLM provider instance based on configuration.

    Args:
        config: Application or LLM configuration.
            If None, loads from default config.yaml.

    Returns:
        Configured BaseLLM instance.

    Raises:
        ValueError: If the provider type is not supported or API key is missing.
    """
    # Resolve config
    if config is None:
        app_config = load_config()
        llm_config = app_config.llm
    elif isinstance(config, AppConfig):
        llm_config = config.llm
    else:
        llm_config = config

    provider = llm_config.provider.lower()
    api_key = llm_config.api_key

    if provider == ProviderType.GROQ:
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq provider")
        from chatbot.llm_gateway.groq_llm import GroqLLM
        return GroqLLM(
            api_key=api_key,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )

    if provider == ProviderType.GOOGLE:
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Google provider")
        from chatbot.llm_gateway.google_llm import GoogleLLM
        return GoogleLLM(
            api_key=api_key,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )

    if provider == ProviderType.OPENAI:
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        from chatbot.llm_gateway.openai_llm import OpenAILLM
        return OpenAILLM(
            api_key=api_key,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )

    if provider == ProviderType.ANTHROPIC:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")
        from chatbot.llm_gateway.anthropic_llm import AnthropicLLM
        return AnthropicLLM(
            api_key=api_key,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )

    if provider == ProviderType.OLLAMA:
        from chatbot.llm_gateway.ollama_llm import OllamaLLM
        return OllamaLLM(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        f"Supported providers: {[p.value for p in ProviderType]}"
    )
