"""
DocuMind Backend Package
========================
Module 1: PDF Processing and NLP Preprocessing
"""

from typing import Any

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
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import pdf_processor
        return getattr(pdf_processor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
