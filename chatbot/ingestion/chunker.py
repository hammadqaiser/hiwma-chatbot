"""Smart Chunker — Splits parsed pages into semantically coherent chunks.

Unlike naive fixed-length splitting, this chunker:
1. Respects paragraph boundaries (never splits mid-paragraph)
2. Carries forward metadata (book_title, chapter, page_number) to every chunk
3. Supports configurable chunk_size and overlap
4. Skips chunks smaller than min_chunk_size (noise, headers, footers)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chatbot.ingestion.parser import PageContent, ParsedDocument


@dataclass
class Chunk:
    """A single chunk of text with full provenance metadata."""

    text: str
    chunk_id: str
    book_title: str
    chapter: str
    page_number: int
    page_range: str  # e.g., "pages 45-47"
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        """Serialize for storage in vector DB."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "book_title": self.book_title,
            "chapter": self.chapter,
            "page_number": self.page_number,
            "page_range": self.page_range,
            **self.metadata,
        }


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, preserving paragraph boundaries.

    Uses double newlines as primary delimiter, falls back to single
    newlines followed by a capital letter (common in PDF extraction).
    """
    # Split on double newlines first
    paragraphs = re.split(r"\n\s*\n", text)

    # If that produces very few splits, try single newline + uppercase
    if len(paragraphs) <= 1 and len(text) > 500:
        paragraphs = re.split(r"\n(?=[A-Z])", text)

    # Clean up each paragraph
    cleaned = []
    for p in paragraphs:
        p = p.strip()
        if p:
            # Collapse internal whitespace but preserve intentional line breaks
            p = re.sub(r"[ \t]+", " ", p)
            cleaned.append(p)

    return cleaned


def chunk_document(
    document: ParsedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[Chunk]:
    """Chunk a parsed document into semantically coherent pieces.

    Strategy:
    1. Group pages by chapter
    2. Split each page into paragraphs
    3. Accumulate paragraphs until chunk_size is reached
    4. Create a chunk with overlap from the previous chunk's end

    Args:
        document: Parsed PDF document.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        min_chunk_size: Minimum chunk size — skip smaller fragments.

    Returns:
        List of Chunk objects with full metadata.
    """
    chunks: list[Chunk] = []
    chunk_counter = 0

    # Process all pages sequentially
    current_text = ""
    current_pages: list[int] = []
    current_chapter = ""

    for page in document.pages:
        paragraphs = _split_into_paragraphs(page.text)
        current_chapter = page.chapter or current_chapter

        for paragraph in paragraphs:
            # Would adding this paragraph exceed chunk_size?
            if current_text and len(current_text) + len(paragraph) + 1 > chunk_size:
                # Emit current chunk
                if len(current_text) >= min_chunk_size:
                    chunk_counter += 1
                    page_range = _format_page_range(current_pages)
                    chunks.append(
                        Chunk(
                            text=current_text.strip(),
                            chunk_id=f"{document.title[:50]}_{chunk_counter}",
                            book_title=document.title,
                            chapter=current_chapter,
                            page_number=current_pages[0] if current_pages else page.page_number,
                            page_range=page_range,
                            metadata={
                                "source_file": document.file_path,
                                "chunk_index": chunk_counter,
                            },
                        )
                    )

                # Start new chunk with overlap
                if chunk_overlap > 0 and len(current_text) > chunk_overlap:
                    # Take the last chunk_overlap chars as overlap
                    overlap_text = current_text[-chunk_overlap:]
                    # Try to start overlap at a word boundary
                    space_idx = overlap_text.find(" ")
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx + 1 :]
                    current_text = overlap_text + "\n\n" + paragraph
                else:
                    current_text = paragraph

                current_pages = [page.page_number]
            else:
                # Append paragraph to current chunk
                if current_text:
                    current_text += "\n\n" + paragraph
                else:
                    current_text = paragraph

                if page.page_number not in current_pages:
                    current_pages.append(page.page_number)

    # Emit final chunk
    if current_text and len(current_text) >= min_chunk_size:
        chunk_counter += 1
        page_range = _format_page_range(current_pages)
        chunks.append(
            Chunk(
                text=current_text.strip(),
                chunk_id=f"{document.title[:50]}_{chunk_counter}",
                book_title=document.title,
                chapter=current_chapter,
                page_number=current_pages[0] if current_pages else 0,
                page_range=page_range,
                metadata={
                    "source_file": document.file_path,
                    "chunk_index": chunk_counter,
                },
            )
        )

    return chunks


def _format_page_range(pages: list[int]) -> str:
    """Format a list of page numbers into a human-readable range."""
    if not pages:
        return ""
    if len(pages) == 1:
        return f"page {pages[0]}"

    pages_sorted = sorted(set(pages))
    return f"pages {pages_sorted[0]}-{pages_sorted[-1]}"


def chunk_documents(
    documents: list[ParsedDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[Chunk]:
    """Chunk multiple documents.

    Convenience function that applies chunk_document to each document
    and returns a flat list of all chunks.
    """
    all_chunks: list[Chunk] = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size, chunk_overlap, min_chunk_size)
        all_chunks.extend(doc_chunks)
    return all_chunks
