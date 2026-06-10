"""Documents Route — GET /api/documents endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from chatbot.api.models import DocumentInfo, DocumentsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents", response_model=DocumentsResponse)
def list_documents():
    """List all documents in the knowledge base with chunk counts."""
    from chatbot.api.main import get_orchestrator

    try:
        orchestrator = get_orchestrator()
        store = orchestrator.retriever.store if hasattr(orchestrator.retriever, 'store') else None

        if store is None:
            return DocumentsResponse(total_chunks=0, documents=[])

        total = store.count()

        # Get unique book titles from the collection metadata
        if total == 0:
            return DocumentsResponse(total_chunks=0, documents=[])

        # Fetch all metadata to count per-book chunks
        all_data = store.collection.get(
            include=["metadatas"],
            limit=total,
        )

        book_chunks: dict[str, int] = {}
        if all_data and all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                title = meta.get("book_title", "Unknown")
                book_chunks[title] = book_chunks.get(title, 0) + 1

        documents = [
            DocumentInfo(title=title, chunk_count=count)
            for title, count in sorted(book_chunks.items())
        ]

        return DocumentsResponse(total_chunks=total, documents=documents)

    except Exception as e:
        logger.error(f"Documents error: {e}", exc_info=True)
        return DocumentsResponse(total_chunks=0, documents=[])
