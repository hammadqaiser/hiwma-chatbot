"""Tests for the PDF parser."""

from pathlib import Path

import pytest

from chatbot.ingestion.parser import (
    ParsedDocument,
    _clean_title,
    _detect_chapter,
    parse_pdf,
    parse_directory,
)


class TestCleanTitle:
    """Test filename-to-title cleaning."""

    def test_removes_pdfdrive(self):
        assert "Some Book" in _clean_title("Some Book (PDFDrive).pdf")

    def test_handles_underscores(self):
        title = _clean_title("Solid_Waste_Management.pdf")
        assert "Solid" in title
        assert "Waste" in title
        assert "_" not in title

    def test_preserves_title_case(self):
        title = _clean_title("Handbook of Solid Waste Management.pdf")
        assert "Handbook" in title

    def test_handles_long_messy_names(self):
        """Real-world messy filenames should produce something readable."""
        title = _clean_title(
            "Handbook-of-Solid-Waste-Management-and-Waste-Minimization-Technologies.pdf"
        )
        assert len(title) > 10  # Should produce a reasonable title


class TestDetectChapter:
    """Test chapter heading detection."""

    def test_detects_chapter_heading(self):
        text = "CHAPTER 5\nSolid Waste Collection\nThis chapter covers..."
        chapter = _detect_chapter(text, "")
        assert "5" in chapter or "CHAPTER" in chapter

    def test_detects_numbered_section(self):
        text = "3. Waste Treatment Methods\nVarious methods exist..."
        chapter = _detect_chapter(text, "")
        assert chapter  # Should detect the section heading

    def test_falls_back_to_previous(self):
        """When no heading found, should return the previous chapter."""
        text = "This is just regular paragraph text with no headings."
        chapter = _detect_chapter(text, "Chapter 2")
        assert chapter == "Chapter 2"


class TestParsePdf:
    """Test PDF parsing with real or mock PDFs."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/path/to/file.pdf")

    def test_not_a_pdf(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Not a PDF"):
            parse_pdf(txt_file)

    def test_parse_returns_document(self, tmp_path):
        """Create a minimal PDF and verify parsing works."""
        import fitz

        # Create a simple test PDF
        pdf_path = tmp_path / "test_book.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "CHAPTER 1\nIntroduction to Waste Management\n\n"
                                    "Solid waste management is the collection, treatment, "
                                    "and disposal of solid material that is discarded.")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "CHAPTER 2\nCollection Systems\n\n"
                                     "Various methods of waste collection exist including "
                                     "curbside pickup and drop-off centers.")
        doc.save(str(pdf_path))
        doc.close()

        # Parse it
        result = parse_pdf(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert result.title == "Test Book"  # Cleaned from filename
        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2
        assert "waste" in result.pages[0].text.lower()


class TestParseDirectory:
    """Test directory-level parsing."""

    def test_empty_directory(self, tmp_path):
        docs = parse_directory(tmp_path)
        assert docs == []

    def test_directory_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_directory("/nonexistent/directory")

    def test_excludes_resume(self, tmp_path):
        """Files matching exclude patterns should be skipped."""
        import fitz

        # Create a resume PDF and a book PDF
        for name in ["SomeResume.pdf", "WasteBook.pdf"]:
            pdf_path = tmp_path / name
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), f"Content of {name}")
            doc.save(str(pdf_path))
            doc.close()

        docs = parse_directory(tmp_path, exclude_patterns=["*Resume*"])
        assert len(docs) == 1
        assert "Waste" in docs[0].title
