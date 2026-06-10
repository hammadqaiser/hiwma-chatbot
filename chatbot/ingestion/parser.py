"""PDF Parser — Extracts text and metadata from PDF books using PyMuPDF.

Each page is extracted with metadata: book_title, page_number.
Chapter detection uses heuristics on heading-like text patterns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""

    text: str
    page_number: int
    book_title: str
    chapter: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""

    title: str
    file_path: str
    total_pages: int
    pages: list[PageContent]

    @property
    def full_text(self) -> str:
        """Concatenate all page texts."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# Common chapter heading patterns in technical books
CHAPTER_PATTERNS = [
    re.compile(r"^(?:CHAPTER|Chapter)\s+(\d+)", re.MULTILINE),
    re.compile(r"^(\d+)\s*\.\s+[A-Z]", re.MULTILINE),
    re.compile(r"^(?:PART|Part)\s+(?:\d+|[IVXLC]+)", re.MULTILINE),
    re.compile(r"^(?:SECTION|Section)\s+(\d+)", re.MULTILINE),
    re.compile(r"^(?:APPENDIX|Appendix)\s+([A-Z])", re.MULTILINE),
]


def _clean_title(filename: str) -> str:
    """Extract a readable book title from a PDF filename.

    Removes file extensions, common suffixes like 'PDFDrive', ISBN numbers,
    and excessive whitespace.
    """
    title = Path(filename).stem

    # Remove common noise patterns
    noise_patterns = [
        r"\(\s*PDFDrive\s*\)",
        r"\(\s*pdfdrive\s*\)",
        r"isbn\d*\s*\d[\d\-]+",
        r"--\s*[\da-f]{20,}\s*--",
        r"--\s*Anna's\s*(?:\.)?",
        r"--\s*[A-Za-z\s,]+--",  # Author names between dashes
        r"\s*_\s*",  # Underscores as separators
    ]
    for pattern in noise_patterns:
        title = re.sub(pattern, " ", title, flags=re.IGNORECASE)

    # Collapse whitespace and strip
    title = re.sub(r"\s+", " ", title).strip()

    # Capitalize properly if all lowercase
    if title == title.lower():
        title = title.title()

    return title


def _detect_chapter(text: str, previous_chapter: str) -> str:
    """Detect chapter heading from page text.

    Uses regex patterns to find chapter/section headings in the first
    few lines of a page. Falls back to the previous chapter if none found.
    """
    # Only check the first 500 chars — chapter headings are at the top
    header_text = text[:500]

    for pattern in CHAPTER_PATTERNS:
        match = pattern.search(header_text)
        if match:
            # Return the full matched line as the chapter identifier
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = min(match.end() + 100, len(text))
            chapter_line = text[line_start:line_end].strip()
            # Limit chapter name length
            return chapter_line[:120]

    return previous_chapter


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """Parse a single PDF file and extract text with metadata.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ParsedDocument with all pages extracted and annotated.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the file is not a valid PDF.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {file_path}")

    book_title = _clean_title(file_path.name)
    pages: list[PageContent] = []
    current_chapter = ""

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{file_path.name}': {e}") from e

    total_pages = len(doc)

    try:
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")

            if not text or not text.strip():
                continue

            # Detect chapter from page content
            current_chapter = _detect_chapter(text, current_chapter)

            page_content = PageContent(
                text=text.strip(),
                page_number=page_num + 1,  # 1-indexed
                book_title=book_title,
                chapter=current_chapter,
                metadata={
                    "source_file": file_path.name,
                    "total_pages": total_pages,
                },
            )
            pages.append(page_content)
    finally:
        doc.close()

    return ParsedDocument(
        title=book_title,
        file_path=str(file_path),
        total_pages=total_pages,
        pages=pages,
    )


def parse_directory(
    directory: str | Path,
    supported_formats: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[ParsedDocument]:
    """Parse all PDF files in a directory.

    Args:
        directory: Path to directory containing PDF files.
        supported_formats: List of file extensions to process (default: ["pdf"]).
        exclude_patterns: Glob patterns to exclude (e.g., ["*resume*"]).

    Returns:
        List of ParsedDocument objects.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if supported_formats is None:
        supported_formats = ["pdf"]

    if exclude_patterns is None:
        exclude_patterns = []

    documents: list[ParsedDocument] = []

    for fmt in supported_formats:
        for pdf_file in sorted(directory.glob(f"*.{fmt}")):
            # Check exclude patterns
            should_exclude = False
            for pattern in exclude_patterns:
                if pdf_file.match(pattern):
                    should_exclude = True
                    break

            if should_exclude:
                continue

            try:
                doc = parse_pdf(pdf_file)
                documents.append(doc)
            except (ValueError, Exception) as e:
                # Log but don't crash — skip unreadable files
                print(f"Warning: Skipping '{pdf_file.name}': {e}")

    return documents
