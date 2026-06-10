"""Context Formatter — Prepares retrieved chunks for LLM consumption.

Converts a list of RetrievalResults into a structured context block
that the LLM prompt template can use. Includes source citations
(book, chapter, page) so the LLM can reference them in its answer.
"""

from __future__ import annotations

from chatbot.retriever.base import RetrievalResult


def format_context(
    results: list[RetrievalResult],
    max_context_chars: int = 8000,
    include_metadata: bool = True,
) -> str:
    """Format retrieval results into a structured context string for the LLM.

    Each chunk is wrapped with source metadata so the LLM can cite
    its sources in the answer (book title, chapter, page numbers).

    Args:
        results: List of RetrievalResults from any retriever.
        max_context_chars: Maximum total characters in the context block.
        include_metadata: Whether to include source metadata headers.

    Returns:
        Formatted context string ready for LLM prompt injection.
    """
    if not results:
        return "No relevant context found in the knowledge base."

    context_parts: list[str] = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        if include_metadata:
            # Build source citation header
            header_parts = [f"[Source {i}]"]
            if result.book_title:
                header_parts.append(f"Book: {result.book_title}")
            if result.chapter:
                header_parts.append(f"Chapter: {result.chapter}")
            if result.page_range:
                header_parts.append(f"({result.page_range})")
            elif result.page_number:
                header_parts.append(f"(page {result.page_number})")

            header = " | ".join(header_parts)
            chunk_text = f"{header}\n{result.text}"
        else:
            chunk_text = result.text

        # Check if adding this chunk would exceed the limit
        if total_chars + len(chunk_text) > max_context_chars:
            # Add truncated version if there's room
            remaining = max_context_chars - total_chars
            if remaining > 200:
                chunk_text = chunk_text[:remaining] + "\n[...truncated]"
                context_parts.append(chunk_text)
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n\n---\n\n".join(context_parts)


def format_sources(results: list[RetrievalResult]) -> list[dict]:
    """Extract source citations from retrieval results.

    Returns a compact list of sources for the API response,
    separate from the full context text.

    Args:
        results: Retrieval results.

    Returns:
        List of source citation dicts.
    """
    sources: list[dict] = []
    seen_chunks: set[str] = set()

    for result in results:
        # Dedup by chunk_id
        if result.chunk_id in seen_chunks:
            continue
        seen_chunks.add(result.chunk_id)

        source = {
            "book_title": result.book_title,
            "chapter": result.chapter,
            "page_number": result.page_number,
            "page_range": result.page_range,
            "relevance_score": round(result.score, 4),
        }
        sources.append(source)

    return sources
