"""Chat Route — POST /api/chat endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from chatbot.api.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    SourceCitation,
    TokenUsageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
)
def chat(request: ChatRequest):
    """Process a chat message and return a grounded answer with citations.

    The orchestrator and chat_history are injected via app.state
    at startup (see main.py).
    """
    from fastapi import Request as FastAPIRequest

    # Access shared services from app state (set in main.py lifespan)
    from chatbot.api.main import get_orchestrator, get_chat_history

    orchestrator = get_orchestrator()
    chat_history = get_chat_history()

    try:
        # Handle session
        session_id = request.session_id
        if session_id is None or not chat_history.session_exists(session_id):
            session_id = chat_history.create_session()

        # Save user message
        chat_history.add_message(session_id, "user", request.message)

        # Process through orchestrator
        result = orchestrator.chat(query=request.message)

        # Save assistant response
        chat_history.add_message(
            session_id, "assistant", result.answer, sources=result.sources
        )

        # Build response
        sources = [
            SourceCitation(
                book_title=s.get("book_title", ""),
                chapter=s.get("chapter", ""),
                page_number=s.get("page_number", 0),
                page_range=s.get("page_range", ""),
                relevance_score=s.get("relevance_score", 0.0),
            )
            for s in result.sources
        ]

        return ChatResponse(
            answer=result.answer,
            sources=sources,
            session_id=session_id,
            model=result.model,
            provider=result.provider,
            usage=TokenUsageResponse(**result.usage),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
