"""
DocuMind - Module 1: PDF Processing and NLP Preprocessing
=========================================================
Assigned to: Person 1 (Document Ingestion & Chunking Pipeline)
Description:
    Provides the ingestion pipeline for the DocuMind RAG question-answering system.
    Validates uploaded PDF documents, extracts clean text page-by-page, performs
    conservative text normalization, and divides document text into overlapping
    semantic chunks while strictly preserving page numbers and source metadata
    for Person 2 (retrieval indexing) and Person 3 (answer grounding & citations).

Responsibilities:
    1. Validate uploaded PDF files (existence, extension, size, encryption, integrity).
    2. Extract text page-by-page while preserving 1-based page numbers.
    3. Normalize and clean extracted text (Unicode NFC, whitespace, control chars).
    4. Create paragraph- and sentence-aware overlapping chunks.
    5. Return structured document metadata and chunk records for semantic search.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # Fallback for older PyMuPDF versions

# Configure module-level logger
logger = logging.getLogger("DocuMind-PDFProcessor")

# ---------------------------------------------------------------------------
# Configuration Defaults
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE: int = 1200      # Target character length per chunk (~200-250 words)
DEFAULT_CHUNK_OVERLAP: int = 180    # Overlap characters between consecutive chunks (~15%)
MAX_PDF_SIZE_MB: int = 50           # Maximum permitted upload size in megabytes

# ---------------------------------------------------------------------------
# Custom Exception Hierarchy
# ---------------------------------------------------------------------------

class PDFProcessingError(Exception):
    """Base exception for all document ingestion and processing errors."""


class InvalidPDFError(PDFProcessingError):
    """Raised when a file does not exist, has an unsupported format, is corrupted, or is encrypted."""


class EmptyPDFError(PDFProcessingError):
    """Raised when the PDF has no pages or contains no extractable text (e.g., scanned / image-only)."""


# ---------------------------------------------------------------------------
# Step 1: Validation
# ---------------------------------------------------------------------------

def validate_pdf(file_path: Union[str, Path]) -> Path:
    """
    Validate file existence, file extension, non-empty size, maximum size limit,
    encryption status, and PyMuPDF readability.

    Args:
        file_path: Relative or absolute path to the PDF file.

    Returns:
        Path: Resolved, validated Path object.

    Raises:
        InvalidPDFError: If the file is missing, not a PDF, empty, over-size,
                         password-protected, or corrupted.
    """
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise InvalidPDFError(f"File does not exist or is not a regular file: {path}")

    if path.suffix.lower() != ".pdf":
        raise InvalidPDFError(f"Unsupported file type '{path.suffix}'. Only .pdf files are supported.")

    file_size = path.stat().st_size
    if file_size == 0:
        raise InvalidPDFError("The uploaded file is empty (0 bytes).")

    max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        raise InvalidPDFError(
            f"PDF file size ({file_size / (1024 * 1024):.2f} MB) exceeds the {MAX_PDF_SIZE_MB} MB limit."
        )

    # Validate that PyMuPDF can open and parse the PDF structure
    try:
        with fitz.open(str(path)) as doc:
            if doc.is_encrypted:
                raise InvalidPDFError("Password-protected PDFs are not supported.")
            if doc.page_count == 0:
                raise InvalidPDFError("The PDF contains no pages.")
    except InvalidPDFError:
        raise
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise InvalidPDFError(f"The file is not a valid or readable PDF: {exc}") from exc

    return path.resolve()


# ---------------------------------------------------------------------------
# Step 2: Text Cleaning & Normalization
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text.

    Transformations:
    1. Unicode NFC normalization (combining characters composed).
    2. Non-breaking spaces (\\u00a0) replaced with standard spaces.
    3. Control characters stripped while preserving newlines and tabs.
    4. Consecutive horizontal whitespace collapsed to single space.
    5. Excessive blank lines collapsed to double newlines (preserving paragraph structure).
    6. Leading and trailing whitespace stripped.

    Args:
        text: Raw text string extracted from PDF.

    Returns:
        str: Normalized, cleaned text.
    """
    if not text:
        return ""

    # Canonical Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Replace non-breaking spaces and zero-width spaces
    text = text.replace("\u00a0", " ").replace("\u200b", "")

    # Remove non-printable control characters except newline and tab
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    # Collapse horizontal whitespace (spaces and tabs on the same line)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ consecutive newlines into 2 (preserves paragraph boundaries)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Step 3: Page-by-Page Text Extraction
# ---------------------------------------------------------------------------

def extract_pdf_pages(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract and clean text from every page of a PDF document.

    Page numbers are 1-based to align with real-world document page references
    and user-facing citations.

    Args:
        file_path: Path to the target PDF file.

    Returns:
        List of dicts:
            [
                {
                    "page": 1,
                    "text": "Cleaned page text...",
                    "char_count": 1250,
                    "raw_char_count": 1320
                },
                ...
            ]

    Raises:
        InvalidPDFError: If validation fails.
        EmptyPDFError: If no extractable text is found across all pages (scanned image-only).
        PDFProcessingError: If PyMuPDF encounters an unexpected read error.
    """
    path = validate_pdf(file_path)
    pages: List[Dict[str, Any]] = []

    try:
        with fitz.open(str(path)) as doc:
            for page_idx, page in enumerate(doc):
                # get_text("text", sort=True) ensures natural reading order (top-to-bottom, left-to-right)
                raw_text = page.get_text("text", sort=True) or ""
                cleaned = clean_text(raw_text)

                pages.append({
                    "page": page_idx + 1,
                    "text": cleaned,
                    "char_count": len(cleaned),
                    "raw_char_count": len(raw_text),
                })
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise PDFProcessingError(f"Failed during text extraction: {exc}") from exc

    # If the document has pages but zero characters extracted across all of them,
    # it is almost certainly a scanned/image-only PDF requiring an OCR pipeline.
    if not any(p["text"] for p in pages):
        raise EmptyPDFError(
            "No extractable text found in the PDF. The file may be a scanned "
            "or image-only document, which requires optical character recognition (OCR)."
        )

    return pages


# ---------------------------------------------------------------------------
# Step 4: Semantic Boundary-Aware Chunking
# ---------------------------------------------------------------------------

def _find_chunk_end(text: str, start: int, target: int) -> int:
    """
    Find the most natural breakpoint for a chunk near the target boundary.

    Priority order:
    1. Paragraph break (\\n\\n)
    2. Sentence terminal (. ! ? followed by space, newline, or end-of-string)
    3. Line break (\\n)
    4. Word whitespace ( )
    5. Hard target split (fallback for unbreakable tokens like long URLs)

    Search window is constrained between 70% and 100% of the target length
    to avoid chunks becoming too small.
    """
    if target >= len(text):
        return len(text)

    # Minimum threshold to avoid cutting off prematurely
    min_split = start + int((target - start) * 0.70)
    sub = text[min_split:target + 1]

    # 1. Prefer paragraph boundary
    para_idx = text.rfind("\n\n", min_split, target + 1)
    if para_idx >= min_split:
        return para_idx

    # 2. Prefer sentence boundary
    sentence_matches = list(re.finditer(r'[.!?](?:["\')\]]*)?(?=\s|$)', sub))
    if sentence_matches:
        return min_split + sentence_matches[-1].end()

    # 3. Prefer newline break
    line_idx = text.rfind("\n", min_split, target + 1)
    if line_idx >= min_split:
        return line_idx

    # 4. Prefer word boundary (whitespace)
    space_idx = text.rfind(" ", min_split, target + 1)
    if space_idx >= min_split:
        return space_idx

    # 5. Fallback: hard slice at target
    return target


def chunk_pages(
    pages: List[Dict[str, Any]],
    source_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Partition extracted pages into overlapping, character-bounded chunks.

    Design Rules:
    1. Chunks NEVER span across page boundaries. This ensures that every retrieved
       chunk is unambiguously mapped to exactly one original PDF page for citations.
    2. Overlap is preserved within each page to avoid information loss at boundaries.
    3. Every chunk is assigned a deterministic, reproducible `chunk_id`.

    Args:
        pages: List of page dicts from extract_pdf_pages().
        source_name: Name of the source document file (e.g. 'paper.pdf').
        chunk_size: Maximum character count per chunk.
        overlap: Character overlap count between consecutive chunks.

    Returns:
        List of chunk dictionaries formatted for retrieval indexing:
            [
                {
                    "chunk_id": "sha256prefix-p1-c1",
                    "text": "Extracted chunk content...",
                    "page": 1,
                    "source": "paper.pdf",
                    "char_start": 0,
                    "char_end": 1184
                },
                ...
            ]
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive (received {chunk_size}).")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be non-negative and strictly less than chunk_size ({chunk_size})."
        )

    chunks: List[Dict[str, Any]] = []
    source_hash = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:10]
    global_chunk_idx = 0

    for page in pages:
        page_num = page["page"]
        text = page["text"].strip()

        if not text:
            # Skip empty pages (e.g. blank cover sheets)
            continue

        start = 0
        text_len = len(text)

        while start < text_len:
            target = min(start + chunk_size, text_len)
            end = _find_chunk_end(text, start, target)

            chunk_text = text[start:end].strip()

            if chunk_text:
                global_chunk_idx += 1
                chunk_id = f"{source_hash}-p{page_num}-c{global_chunk_idx}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "page": page_num,
                    "source": source_name,
                    "char_start": start,
                    "char_end": end,
                })

            if end >= text_len:
                break

            # Calculate next starting position with overlap
            next_start = max(0, end - overlap)

            # Prevent infinite loop if boundary selection makes no forward progress
            if next_start <= start:
                next_start = end

            start = next_start

    return chunks


# ---------------------------------------------------------------------------
# Step 5: High-Level Public Pipeline API
# ---------------------------------------------------------------------------

def process_pdf(
    file_path: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """
    Execute the end-to-end PDF processing and NLP chunking pipeline.

    This is the primary function invoked by Person 2 (retriever indexing)
    and Person 4 (web UI upload handler).

    Pipeline stages:
    1. Validate PDF file format, size, encryption, and readability.
    2. Extract text page-by-page while tracking page numbers.
    3. Clean and normalize whitespace, characters, and formatting.
    4. Segment into semantic overlapping chunks with citation metadata.

    Args:
        file_path: Path to the uploaded or local PDF.
        chunk_size: Target character length per chunk (default: 1200).
        overlap: Character overlap between consecutive chunks (default: 180).

    Returns:
        Dict[str, Any]: Structured document processing result:
            {
                "document": "research_paper.pdf",
                "page_count": 12,
                "pages_with_text": 12,
                "total_characters": 28450,
                "chunk_count": 30,
                "chunks": [
                    {
                        "chunk_id": "3a8b9c...-p1-c1",
                        "text": "...",
                        "page": 1,
                        "source": "research_paper.pdf",
                        "char_start": 0,
                        "char_end": 1180
                    },
                    ...
                ]
            }

    Raises:
        InvalidPDFError: If the file is unreadable, invalid, or corrupted.
        EmptyPDFError: If no extractable text is found in the document.
        PDFProcessingError: Base error for unexpected processing failures.
    """
    path = validate_pdf(file_path)
    pages = extract_pdf_pages(path)

    chunks = chunk_pages(
        pages=pages,
        source_name=path.name,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise EmptyPDFError("No usable text chunks could be generated from this PDF.")

    total_characters = sum(p["char_count"] for p in pages)
    pages_with_text = sum(1 for p in pages if p["text"])

    return {
        "document": path.name,
        "page_count": len(pages),
        "pages_with_text": pages_with_text,
        "total_characters": total_characters,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


# ---------------------------------------------------------------------------
# Step 6: Command-Line Interface (CLI Demonstration)
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint for testing and validating PDF ingestion from the terminal."""
    parser = argparse.ArgumentParser(
        prog="pdf_processor",
        description="DocuMind Module 1 - Extract and chunk text from PDF files for semantic search.",
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to process.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Target character length per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Character overlap between adjacent chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of chunks to preview in terminal output (default: 3).",
    )

    args = parser.parse_args()

    try:
        print(f"\n[DocuMind] Processing document: {args.pdf_path}")
        result = process_pdf(
            args.pdf_path,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

        print("\n" + "=" * 65)
        print(" DOCUMENT INGESTION SUMMARY")
        print("=" * 65)
        print(f" Document Name     : {result['document']}")
        print(f" Total Pages       : {result['page_count']}")
        print(f" Pages with Text   : {result['pages_with_text']}")
        print(f" Total Characters  : {result['total_characters']:,}")
        print(f" Chunks Generated  : {result['chunk_count']}")
        print(f" Chunk Size Setting: {args.chunk_size} chars")
        print(f" Overlap Setting   : {args.overlap} chars")

        preview_count = min(args.preview, len(result["chunks"]))
        print("\n" + "=" * 65)
        print(f" CHUNK PREVIEWS (Showing first {preview_count} of {result['chunk_count']})")
        print("=" * 65)

        for chunk in result["chunks"][:preview_count]:
            print(f"\n[Chunk ID] : {chunk['chunk_id']}")
            print(f"[Page]     : {chunk['page']}")
            print(f"[Source]   : {chunk['source']}")
            print(f"[Range]    : chars {chunk['char_start']}..{chunk['char_end']} (length: {len(chunk['text'])} chars)")
            print(f"[Snippet]  :\n{chunk['text'][:300]}...")
            print("-" * 65)

        print("\n[SUCCESS] Document ingestion completed successfully.")

    except PDFProcessingError as exc:
        print(f"\n[ERROR] Ingestion failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
