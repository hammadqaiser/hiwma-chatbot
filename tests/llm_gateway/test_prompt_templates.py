"""Tests for Prompt Templates."""

from chatbot.llm_gateway.prompt_templates import (
    build_messages,
    build_system_prompt,
    build_user_prompt,
    SYSTEM_PROMPT,
)


class TestSystemPrompt:
    """Test the RAG system prompt."""

    def test_contains_no_inline_citation_rule(self):
        """System prompt must tell LLM NOT to include inline citations."""
        prompt = build_system_prompt()
        assert "do not include inline source citation" in prompt.lower()
        assert "book" in prompt.lower()

    def test_contains_hallucination_guard(self):
        """Must tell the LLM to handle missing info gracefully."""
        prompt = build_system_prompt()
        assert "don't have specific information" in prompt.lower() or "don't have enough" in prompt.lower()

    def test_contains_friendly_personality(self):
        """Must include warm/friendly personality instructions."""
        prompt = build_system_prompt()
        assert "friendly" in prompt.lower()
        assert "greet" in prompt.lower()

    def test_returns_stripped_string(self):
        prompt = build_system_prompt()
        assert prompt == prompt.strip()
        assert len(prompt) > 100  # Non-trivial prompt


class TestUserPrompt:
    """Test user prompt construction."""

    def test_injects_context_and_query(self):
        """User prompt must contain both context and query."""
        context = "Leachate is a liquid that forms in landfills."
        query = "What is leachate?"

        prompt = build_user_prompt(query, context)

        assert context in prompt
        assert query in prompt

    def test_context_before_query(self):
        """Context should appear before the query in the prompt."""
        context = "CONTEXT_MARKER"
        query = "QUERY_MARKER"

        prompt = build_user_prompt(query, context)
        assert prompt.index("CONTEXT_MARKER") < prompt.index("QUERY_MARKER")

    def test_returns_stripped_string(self):
        prompt = build_user_prompt("test?", "some context")
        assert prompt == prompt.strip()


class TestBuildMessages:
    """Test chat message list construction."""

    def test_returns_two_messages(self):
        messages = build_messages("What is MSW?", "context here")
        assert len(messages) == 2

    def test_first_message_is_system(self):
        messages = build_messages("test", "context")
        assert messages[0]["role"] == "system"
        assert len(messages[0]["content"]) > 100

    def test_second_message_is_user(self):
        messages = build_messages("test query", "test context")
        assert messages[1]["role"] == "user"
        assert "test query" in messages[1]["content"]
        assert "test context" in messages[1]["content"]

    def test_messages_have_required_keys(self):
        messages = build_messages("q", "c")
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
