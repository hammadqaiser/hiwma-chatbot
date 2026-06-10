"""Health Route — GET /api/health endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from chatbot.api.models import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    """Return service health status and vector store stats."""
    from chatbot.api.main import get_orchestrator

    try:
        orchestrator = get_orchestrator()
        store = orchestrator.retriever.store if hasattr(orchestrator.retriever, 'store') else None
        vector_info = store.get_stats() if store else {}
    except Exception:
        vector_info = {"status": "unavailable"}

    return HealthResponse(
        status="ok",
        version="0.1.0",
        vector_store=vector_info,
    )
