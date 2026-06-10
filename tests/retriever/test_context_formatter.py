"""Tests for the Context Formatter."""

import pytest

from chatbot.retriever.base import RetrievalResult
from chatbot.retriever.context_formatter import format_context, format_sources


def _make_result(text, book_title="Test Book", chapter="Chapter 1",
                 page_number=1, page_range="page 1", score=0.9, chunk_id="c1"):
    return RetrievalResult(
        text=text, score=score, book_title=book_title, chapter=chapter,
        page_number=page_number, page_range=page_range, chunk_id=chunk_id,
        retriever_type="vector",
    )


class TestFormatContext:
    """Test context formatting for LLM prompt."""

    def test_formats_with_metadata(self):
        """Should include source citations in the formatted output."""
        results = [_make_result("Waste collection is important.", page_range="pages 1-3")]
        context = format_context(results)

        assert "[Source 1]" in context
        assert "Test Book" in context
        assert "pages 1-3" in context
        assert "Waste collection is important." in context

    def test_multiple_results_separated(self):
        """Multiple results should be separated by dividers."""
        results = [
            _make_result("First chunk.", chunk_id="c1"),
            _make_result("Second chunk.", chunk_id="c2"),
        ]
        context = format_context(results)

        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "---" in context

    def test_respects_max_chars(self):
        """Should truncate when exceeding max_context_chars."""
        long_text = "X" * 5000
        results = [
            _make_result(long_text, chunk_id="c1"),
            _make_result(long_text, chunk_id="c2"),
        ]
        context = format_context(results, max_context_chars=6000)

        assert len(context) <= 6200  # Some overhead for headers

    def test_without_metadata(self):
        """include_metadata=False should omit source headers."""
        results = [_make_result("Just the text.")]
        context = format_context(results, include_metadata=False)

        assert "[Source" not in context
        assert "Just the text." in context

    def test_empty_results(self):
        """Empty results should return a helpful message."""
        context = format_context([])
        assert "No relevant context" in context

    def test_includes_chapter_info(self):
        """Chapter information should be present in metadata."""
        results = [_make_result("Content.", chapter="Chapter 5: Landfills")]
        context = format_context(results)

        assert "Chapter 5: Landfills" in context


class TestFormatSources:
    """Test source citation extraction."""

    def test_extracts_sources(self):
        results = [
            _make_result("Text 1", book_title="Book A", chunk_id="c1", score=0.95),
            _make_result("Text 2", book_title="Book B", chunk_id="c2", score=0.80),
        ]
        sources = format_sources(results)

        assert len(sources) == 2
        assert sources[0]["book_title"] == "Book A"
        assert sources[0]["relevance_score"] == 0.95

    def test_deduplicates_by_chunk_id(self):
        """Same chunk_id should appear only once in sources."""
        results = [
            _make_result("Same text", chunk_id="c1", score=0.9),
            _make_result("Same text", chunk_id="c1", score=0.8),
        ]
        sources = format_sources(results)
        assert len(sources) == 1

    def test_empty_results(self):
        sources = format_sources([])
        assert sources == []
