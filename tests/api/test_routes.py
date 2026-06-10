"""Tests for API Routes (using FastAPI TestClient with mocked dependencies)."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from chatbot.api.chat_history import ChatHistory
from chatbot.api.main import app, init_for_testing
from chatbot.api.services.orchestrator import ChatResult, Orchestrator
from chatbot.llm_gateway.base import TokenUsage


@pytest.fixture
def mock_orchestrator():
    """Create an orchestrator with mocked retriever + LLM."""
    orch = MagicMock(spec=Orchestrator)
    orch.chat.return_value = ChatResult(
        answer="Waste is collected weekly by municipal services. (Source: Solid Waste Engineering, Chapter 3, pages 45-47)",
        sources=[
            {
                "book_title": "Solid Waste Engineering",
                "chapter": "Chapter 3",
                "page_number": 45,
                "page_range": "45-47",
                "relevance_score": 0.93,
            },
        ],
        model="llama-3.1-70b-versatile",
        provider="groq",
        usage={"prompt_tokens": 150, "completion_tokens": 40, "total_tokens": 190},
    )
    # Mock retriever for health/documents endpoints
    mock_store = MagicMock()
    mock_store.get_stats.return_value = {"total_chunks": 500, "collection": "test"}
    mock_store.count.return_value = 100
    mock_store.collection.get.return_value = {
        "metadatas": [
            {"book_title": "Book A"},
            {"book_title": "Book A"},
            {"book_title": "Book B"},
        ]
    }
    mock_retriever = MagicMock()
    mock_retriever.store = mock_store
    orch.retriever = mock_retriever

    return orch


@pytest.fixture
def client(mock_orchestrator, tmp_path):
    """Create a test client with mocked services."""
    db_path = str(tmp_path / "test_chat.db")
    chat_history = ChatHistory(database_path=db_path)

    with TestClient(app, raise_server_exceptions=False) as c:
        # Inject mocks AFTER lifespan so they override real instances
        init_for_testing(mock_orchestrator, chat_history)
        yield c


class TestChatRoute:

    def test_chat_returns_answer(self, client):
        response = client.post("/api/chat", json={"message": "How is waste collected?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "waste" in data["answer"].lower()
        assert "session_id" in data

    def test_chat_returns_sources(self, client):
        response = client.post("/api/chat", json={"message": "test"})
        data = response.json()
        assert len(data["sources"]) > 0
        assert data["sources"][0]["book_title"] == "Solid Waste Engineering"

    def test_chat_returns_usage(self, client):
        response = client.post("/api/chat", json={"message": "test"})
        data = response.json()
        assert data["usage"]["total_tokens"] == 190

    def test_chat_creates_session(self, client):
        response = client.post("/api/chat", json={"message": "test"})
        data = response.json()
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0

    def test_chat_continues_session(self, client):
        # First message — creates a session
        r1 = client.post("/api/chat", json={"message": "First question"})
        session_id = r1.json()["session_id"]

        # Second message — continues the same session
        r2 = client.post("/api/chat", json={"message": "Follow up", "session_id": session_id})
        assert r2.json()["session_id"] == session_id

    def test_chat_empty_message_rejected(self, client):
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_chat_missing_message_rejected(self, client):
        response = client.post("/api/chat", json={})
        assert response.status_code == 422


class TestHealthRoute:

    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_includes_vector_store(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert "vector_store" in data


class TestDocumentsRoute:

    def test_documents_returns_list(self, client):
        response = client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total_chunks" in data

    def test_documents_have_titles(self, client):
        response = client.get("/api/documents")
        data = response.json()
        assert data["total_chunks"] == 100
        titles = [d["title"] for d in data["documents"]]
        assert "Book A" in titles
        assert "Book B" in titles
