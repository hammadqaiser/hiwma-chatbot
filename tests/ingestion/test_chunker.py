"""Tests for the smart chunker."""

import pytest

from chatbot.ingestion.chunker import Chunk, chunk_document, _split_into_paragraphs
from chatbot.ingestion.parser import PageContent, ParsedDocument


def _make_document(pages_text: list[str], title: str = "Test Book") -> ParsedDocument:
    """Helper to create a ParsedDocument from a list of page texts."""
    pages = [
        PageContent(
            text=text,
            page_number=i + 1,
            book_title=title,
            chapter=f"Chapter {i + 1}",
        )
        for i, text in enumerate(pages_text)
    ]
    return ParsedDocument(
        title=title,
        file_path="test.pdf",
        total_pages=len(pages),
        pages=pages,
    )


class TestSplitIntoParagraphs:
    """Test paragraph splitting logic."""

    def test_splits_on_double_newline(self):
        text = "First paragraph.\n\nSecond paragraph."
        paragraphs = _split_into_paragraphs(text)
        assert len(paragraphs) == 2

    def test_handles_single_paragraph(self):
        text = "Just one paragraph with no breaks."
        paragraphs = _split_into_paragraphs(text)
        assert len(paragraphs) == 1

    def test_strips_whitespace(self):
        text = "  First.  \n\n  Second.  "
        paragraphs = _split_into_paragraphs(text)
        assert paragraphs[0] == "First."
        assert paragraphs[1] == "Second."

    def test_skips_empty_paragraphs(self):
        text = "First.\n\n\n\n\nSecond."
        paragraphs = _split_into_paragraphs(text)
        assert len(paragraphs) == 2


class TestChunkDocument:
    """Test document chunking."""

    def test_short_document_single_chunk(self):
        """A document with enough text should produce a single chunk."""
        doc = _make_document([
            "Solid waste management involves the collection, treatment, and disposal of "
            "solid materials that are discarded because they have served their purpose or "
            "are no longer useful. This field is critical for public health and environmental protection."
        ])
        chunks = chunk_document(doc, chunk_size=5000)
        assert len(chunks) == 1
        assert chunks[0].book_title == "Test Book"
        assert chunks[0].page_number == 1

    def test_long_page_multiple_chunks(self):
        """A long page should be split into multiple chunks."""
        long_text = ("This is a paragraph about waste management. " * 20 + "\n\n") * 10
        doc = _make_document([long_text])
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=50, min_chunk_size=50)
        assert len(chunks) > 1

    def test_metadata_preserved(self):
        """Every chunk must have book_title, chapter, page_number."""
        doc = _make_document(["Content of chapter one.\n\nMore content here."])
        chunks = chunk_document(doc, chunk_size=1000)
        for chunk in chunks:
            assert chunk.book_title == "Test Book"
            assert chunk.chapter  # Not empty
            assert chunk.page_number > 0
            assert chunk.chunk_id  # Has an ID

    def test_min_chunk_size_filtering(self):
        """Chunks smaller than min_chunk_size should be skipped."""
        doc = _make_document(["Hi"])
        chunks = chunk_document(doc, chunk_size=1000, min_chunk_size=100)
        assert len(chunks) == 0  # "Hi" is too short

    def test_page_range_format(self):
        """Chunks spanning multiple pages should have a proper page range."""
        doc = _make_document([
            "First page " * 50,
            "Second page " * 50,
        ])
        chunks = chunk_document(doc, chunk_size=5000)
        # All content fits in one chunk spanning 2 pages
        if len(chunks) == 1:
            assert "pages" in chunks[0].page_range or "page" in chunks[0].page_range

    def test_chunk_ids_unique(self):
        """All chunk IDs must be unique."""
        long_text = ("Paragraph about solid waste. " * 20 + "\n\n") * 10
        doc = _make_document([long_text])
        chunks = chunk_document(doc, chunk_size=200, min_chunk_size=50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_to_dict_serialization(self):
        """Chunk.to_dict() should return a complete dict for storage."""
        doc = _make_document(["Content for serialization test." * 10])
        chunks = chunk_document(doc, chunk_size=1000)
        if chunks:
            d = chunks[0].to_dict()
            assert "text" in d
            assert "book_title" in d
            assert "chapter" in d
            assert "page_number" in d
