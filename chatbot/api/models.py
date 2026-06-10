"""Pydantic Models — Request/Response schemas for the API.

These are the API contract. Frontend and external clients
depend on these schemas — changes here are breaking changes.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ─── Request Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """POST /api/chat request body."""

    message: str = Field(..., min_length=1, max_length=5000, description="User's question")
    session_id: str | None = Field(
        default=None,
        description="Chat session ID. Omit to start a new session.",
    )


# ─── Response Models ────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """A single source citation from the knowledge base."""

    book_title: str
    chapter: str
    page_number: int
    page_range: str
    relevance_score: float


class TokenUsageResponse(BaseModel):
    """Token usage statistics for cost monitoring."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """POST /api/chat response body."""

    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    session_id: str
    model: str
    provider: str
    usage: TokenUsageResponse = Field(default_factory=TokenUsageResponse)


class HealthResponse(BaseModel):
    """GET /api/health response body."""

    status: str = "ok"
    version: str = "0.1.0"
    vector_store: dict = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """A single document in the knowledge base."""

    title: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """GET /api/documents response body."""

    total_chunks: int
    documents: list[DocumentInfo] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    detail: str | None = None
