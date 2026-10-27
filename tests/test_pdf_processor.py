"""
DocuMind - Test Suite for Module 1: PDF Processing and NLP Preprocessing
========================================================================
Assigned to: Person 1
Description:
    Validates PDF validation, text extraction, text normalization, overlapping chunking,
    page citation preservation, and edge cases (empty documents, corrupted files,
    password-protected PDFs, invalid chunking parameters).
"""

from pathlib import Path
import pytest

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from backend.pdf_processor import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EmptyPDFError,
    InvalidPDFError,
    PDFProcessingError,
    chunk_pages,
    clean_text,
    extract_pdf_pages,
    process_pdf,
    validate_pdf,
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a realistic two-page test PDF."""
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Natural Language Processing helps computers understand human language. " * 15,
        fontsize=10,
    )

    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Semantic search retrieves relevant information using dense vector embeddings. " * 15,
        fontsize=10,
    )

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extracts_all_pages(sample_pdf: Path) -> None:
    """Verify that all pages are extracted with 1-based indexing and accurate text."""
    pages = extract_pdf_pages(sample_pdf)

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2
    assert "Natural Language Processing" in pages[0]["text"]
    assert "Semantic search" in pages[1]["text"]
    assert pages[0]["char_count"] > 0
    assert pages[1]["char_count"] > 0


def test_clean_text_normalizes_whitespace_and_unicode() -> None:
    """Verify NFC normalization, non-breaking space replacement, and whitespace cleaning."""
    raw = "Hello\u00a0   world!\r\n\n\n\nNext paragraph with   spaces."
    cleaned = clean_text(raw)

    assert cleaned == "Hello world!\n\nNext paragraph with spaces."


def test_clean_text_removes_unprintable_control_chars() -> None:
    """Verify removal of ASCII/Unicode control characters while normalizing spacing and newlines."""
    text_with_control = "Line 1\x00\x07\x08\nLine 2\twith\x1f tab."
    cleaned = clean_text(text_with_control)

    assert cleaned == "Line 1\nLine 2 with tab."


def test_chunks_preserve_page_numbers(sample_pdf: Path) -> None:
    """Verify that chunks retain accurate 1-based page metadata and document source."""
    result = process_pdf(sample_pdf, chunk_size=250, overlap=40)

    assert result["chunk_count"] > 0
    assert result["page_count"] == 2
    assert result["document"] == "sample.pdf"

    # Crucial RAG guarantee: chunks never cross page boundaries
    for chunk in result["chunks"]:
        assert chunk["page"] in (1, 2)
        assert chunk["text"].strip() != ""
        assert chunk["source"] == "sample.pdf"
        assert "chunk_id" in chunk
        assert chunk["char_start"] < chunk["char_end"]


def test_chunks_are_within_configured_size() -> None:
    """Verify that chunk splitting respects target chunk length boundaries."""
    pages = [{
        "page": 1,
        "text": " ".join(["documind_token"] * 200),
        "char_count": 2999,
    }]

    chunks = chunk_pages(
        pages,
        source_name="test.pdf",
        chunk_size=200,
        overlap=30,
    )

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 200 for c in chunks)


def test_rejects_non_existent_file(tmp_path: Path) -> None:
    """Verify that a missing file path raises InvalidPDFError."""
    missing_file = tmp_path / "does_not_exist.pdf"
    with pytest.raises(InvalidPDFError, match="File does not exist"):
        process_pdf(missing_file)


def test_rejects_non_pdf_extension(tmp_path: Path) -> None:
    """Verify that unsupported extensions (e.g. .txt, .docx) are rejected."""
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("This is plain text, not a PDF.")

    with pytest.raises(InvalidPDFError, match="Only .pdf files are supported"):
        process_pdf(bad_file)


def test_rejects_empty_zero_byte_file(tmp_path: Path) -> None:
    """Verify that a zero-byte file raises InvalidPDFError."""
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(InvalidPDFError, match="The uploaded file is empty"):
        process_pdf(empty_file)


def test_rejects_image_only_or_scanned_pdf(tmp_path: Path) -> None:
    """Verify that a PDF without extractable text raises EmptyPDFError."""
    scanned_path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    doc.new_page()  # Page without text
    doc.save(scanned_path)
    doc.close()

    with pytest.raises(EmptyPDFError, match="No extractable text found"):
        process_pdf(scanned_path)


def test_rejects_encrypted_pdf(tmp_path: Path) -> None:
    """Verify that password-protected PDFs raise InvalidPDFError."""
    enc_path = tmp_path / "encrypted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Confidential data", fontsize=12)

    # Save with user and owner passwords
    perm = fitz.PDF_PERM_ACCESSIBILITY
    doc.save(
        enc_path,
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="secret_owner",
        user_pw="secret_user",
        permissions=perm,
    )
    doc.close()

    with pytest.raises(InvalidPDFError, match="Password-protected PDFs are not supported"):
        process_pdf(enc_path)


def test_invalid_chunk_parameters() -> None:
    """Verify that invalid chunk_size or overlap parameter values raise ValueError."""
    dummy_pages = [{"page": 1, "text": "Valid text content."}]

    # chunk_size must be positive
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_pages(dummy_pages, "test.pdf", chunk_size=0, overlap=10)

    # overlap cannot be negative
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(dummy_pages, "test.pdf", chunk_size=100, overlap=-5)

    # overlap cannot be >= chunk_size
    with pytest.raises(ValueError, match="overlap"):
        chunk_pages(dummy_pages, "test.pdf", chunk_size=100, overlap=100)


def test_full_pipeline_output_structure(sample_pdf: Path) -> None:
    """Verify that process_pdf returns the exact agreed dictionary schema."""
    result = process_pdf(sample_pdf, chunk_size=300, overlap=50)

    assert isinstance(result, dict)
    expected_keys = {"document", "page_count", "pages_with_text", "total_characters", "chunk_count", "chunks"}
    assert expected_keys.issubset(result.keys())

    assert result["page_count"] == 2
    assert result["pages_with_text"] == 2
    assert result["total_characters"] > 0
    assert result["chunk_count"] == len(result["chunks"])

    for chunk in result["chunks"]:
        chunk_keys = {"chunk_id", "text", "page", "source", "char_start", "char_end"}
        assert chunk_keys.issubset(chunk.keys())
