"""FastAPI Application — Main entry point for the chatbot API.

Run with: uvicorn chatbot.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatbot.api.chat_history import ChatHistory
from chatbot.api.services.orchestrator import Orchestrator
from chatbot.config import load_config

logger = logging.getLogger(__name__)

# ─── Global instances (set during lifespan) ──────────────────────────
_orchestrator: Orchestrator | None = None
_chat_history: ChatHistory | None = None


def get_orchestrator() -> Orchestrator:
    """Get the shared orchestrator instance."""
    if _orchestrator is None:
        raise RuntimeError("Orchestrator not initialized. Is the app running?")
    return _orchestrator


def get_chat_history() -> ChatHistory:
    """Get the shared chat history instance."""
    if _chat_history is None:
        raise RuntimeError("Chat history not initialized. Is the app running?")
    return _chat_history


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize shared services on startup."""
    global _orchestrator, _chat_history

    logger.info("Starting Waste Management Chatbot API...")

    config = load_config()

    # Create shared instances
    _orchestrator = Orchestrator(config=config)
    _chat_history = ChatHistory(database_path=config.chat_history.database_path)

    logger.info(f"LLM Provider: {config.llm.provider} / {config.llm.model}")
    logger.info(f"Retriever: {config.retriever.type}")
    logger.info("API ready!")

    yield

    # Cleanup
    logger.info("Shutting down API...")
    _orchestrator = None
    _chat_history = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = load_config()

    app = FastAPI(
        title="Waste Management Chatbot API",
        description="RAG-powered chatbot for waste management technical books",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow frontend to call API from different origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules
    from chatbot.api.routes.chat import router as chat_router
    from chatbot.api.routes.health import router as health_router
    from chatbot.api.routes.documents import router as documents_router

    app.include_router(chat_router)
    app.include_router(health_router)
    app.include_router(documents_router)

    # Serve frontend static files
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(frontend_dir / "index.html")

        app.mount("/", StaticFiles(directory=str(frontend_dir)), name="frontend")

    return app


# Create the app instance (uvicorn looks for this)
app = create_app()


def init_for_testing(orchestrator: Orchestrator, chat_history: ChatHistory) -> None:
    """Initialize global instances for testing (bypasses lifespan)."""
    global _orchestrator, _chat_history
    _orchestrator = orchestrator
    _chat_history = chat_history
