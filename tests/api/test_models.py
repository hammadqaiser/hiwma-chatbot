"""Tests for Pydantic API Models."""

import pytest
from pydantic import ValidationError

from chatbot.api.models import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentsResponse,
    ErrorResponse,
    HealthResponse,
    SourceCitation,
    TokenUsageResponse,
)


class TestChatRequest:

    def test_valid_request(self):
        req = ChatRequest(message="What is leachate?")
        assert req.message == "What is leachate?"
        assert req.session_id is None

    def test_with_session_id(self):
        req = ChatRequest(message="test", session_id="abc-123")
        assert req.session_id == "abc-123"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_missing_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatResponse:

    def test_full_response(self):
        resp = ChatResponse(
            answer="Leachate is...",
            sources=[SourceCitation(
                book_title="Test Book",
                chapter="Ch 1",
                page_number=10,
                page_range="10-12",
                relevance_score=0.95,
            )],
            session_id="abc-123",
            model="llama-3.1",
            provider="groq",
            usage=TokenUsageResponse(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        )
        assert resp.answer == "Leachate is..."
        assert len(resp.sources) == 1
        assert resp.sources[0].book_title == "Test Book"
        assert resp.usage.total_tokens == 150

    def test_defaults(self):
        resp = ChatResponse(
            answer="test",
            session_id="s1",
            model="m",
            provider="p",
        )
        assert resp.sources == []
        assert resp.usage.total_tokens == 0


class TestHealthResponse:

    def test_defaults(self):
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.version == "0.1.0"

    def test_with_vector_info(self):
        resp = HealthResponse(
            vector_store={"total_chunks": 500, "collection": "waste_management_books"},
        )
        assert resp.vector_store["total_chunks"] == 500


class TestDocumentsResponse:

    def test_with_documents(self):
        resp = DocumentsResponse(
            total_chunks=100,
            documents=[
                DocumentInfo(title="Book A", chunk_count=60),
                DocumentInfo(title="Book B", chunk_count=40),
            ],
        )
        assert resp.total_chunks == 100
        assert len(resp.documents) == 2

    def test_empty(self):
        resp = DocumentsResponse(total_chunks=0)
        assert resp.documents == []


class TestErrorResponse:

    def test_error_response(self):
        resp = ErrorResponse(error="Something went wrong", detail="Stack trace here")
        assert resp.error == "Something went wrong"
