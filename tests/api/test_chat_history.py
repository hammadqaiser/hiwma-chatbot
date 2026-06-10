"""Tests for Chat History (SQLite)."""

import os
import tempfile

import pytest

from chatbot.api.chat_history import ChatHistory


@pytest.fixture
def history(tmp_path):
    """Create a ChatHistory with a temp database."""
    db_path = str(tmp_path / "test_chat.db")
    return ChatHistory(database_path=db_path)


class TestChatHistory:

    def test_create_session(self, history):
        session_id = history.create_session()
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    def test_session_exists(self, history):
        session_id = history.create_session()
        assert history.session_exists(session_id) is True
        assert history.session_exists("nonexistent") is False

    def test_add_and_get_messages(self, history):
        session_id = history.create_session()

        history.add_message(session_id, "user", "What is leachate?")
        history.add_message(
            session_id,
            "assistant",
            "Leachate is a liquid...",
            sources=[{"book_title": "Test Book", "page": 10}],
        )

        messages = history.get_messages(session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is leachate?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["sources"][0]["book_title"] == "Test Book"

    def test_get_messages_order(self, history):
        """Messages should be returned in chronological order."""
        session_id = history.create_session()
        history.add_message(session_id, "user", "First")
        history.add_message(session_id, "assistant", "Second")
        history.add_message(session_id, "user", "Third")

        messages = history.get_messages(session_id)
        assert [m["content"] for m in messages] == ["First", "Second", "Third"]

    def test_get_messages_limit(self, history):
        session_id = history.create_session()
        for i in range(10):
            history.add_message(session_id, "user", f"Message {i}")

        messages = history.get_messages(session_id, limit=3)
        assert len(messages) == 3

    def test_separate_sessions(self, history):
        """Messages from different sessions should be isolated."""
        s1 = history.create_session()
        s2 = history.create_session()

        history.add_message(s1, "user", "Session 1 message")
        history.add_message(s2, "user", "Session 2 message")

        assert len(history.get_messages(s1)) == 1
        assert len(history.get_messages(s2)) == 1
        assert history.get_messages(s1)[0]["content"] == "Session 1 message"

    def test_get_session_count(self, history):
        assert history.get_session_count() == 0
        history.create_session()
        history.create_session()
        assert history.get_session_count() == 2

    def test_empty_session_messages(self, history):
        session_id = history.create_session()
        messages = history.get_messages(session_id)
        assert messages == []

    def test_creates_directory_if_missing(self, tmp_path):
        """Should auto-create parent directories for the database."""
        nested = str(tmp_path / "deep" / "nested" / "chat.db")
        history = ChatHistory(database_path=nested)
        session_id = history.create_session()
        assert history.session_exists(session_id)
