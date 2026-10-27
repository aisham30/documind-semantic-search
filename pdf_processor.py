"""
DocuMind - Module 1: PDF Processing and NLP Preprocessing
=========================================================
Assigned to: Person 1
Root-level wrapper exposing backend.pdf_processor for top-level usage.
"""

from backend.pdf_processor import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MAX_PDF_SIZE_MB,
    EmptyPDFError,
    InvalidPDFError,
    PDFProcessingError,
    chunk_pages,
    clean_text,
    extract_pdf_pages,
    main,
    process_pdf,
    validate_pdf,
)

__all__ = [
    "process_pdf",
    "extract_pdf_pages",
    "chunk_pages",
    "clean_text",
    "validate_pdf",
    "PDFProcessingError",
    "InvalidPDFError",
    "EmptyPDFError",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "MAX_PDF_SIZE_MB",
    "main",
]

if __name__ == "__main__":
    main()
