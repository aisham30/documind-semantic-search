"""
DocuMind - Integration Test: Module 1 (PDF Ingestion) -> Module 2 (Retriever)
=============================================================================
Validates that chunks produced by Person 1's pdf_processor.process_pdf()
seamlessly feed into Person 2's DocumentRetriever.build_index() and allow
semantic search and citation page tracking without errors.
"""

from pathlib import Path
import pytest

from backend.pdf_processor import process_pdf
from retriever import DocumentRetriever, _DEPS_AVAILABLE


@pytest.fixture
def sample_pdf_path() -> Path:
    sample_file = Path(__file__).resolve().parent.parent / "sample_pdfs" / "research_paper.pdf"
    if not sample_file.exists():
        from sample_pdfs.create_sample_pdf import create_sample_paper
        return create_sample_paper()
    return sample_file


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="Sentence-transformers / FAISS not installed")
def test_p1_to_p2_integration(sample_pdf_path: Path):
    """Verify that process_pdf chunks can be ingested and queried by DocumentRetriever."""
    # 1. Person 1 produces chunks
    pdf_result = process_pdf(sample_pdf_path, chunk_size=800, overlap=100)
    assert pdf_result["chunk_count"] > 0
    chunks = pdf_result["chunks"]

    # 2. Person 2 consumes chunks
    retriever = DocumentRetriever()
    indexed = retriever.build_index(chunks)
    assert indexed == len(chunks)

    # 3. Person 2 executes semantic search
    results = retriever.semantic_search("What is cosine similarity in FAISS?", k=3)
    assert len(results) > 0

    for r in results:
        assert "text" in r
        assert "page" in r
        assert isinstance(r["page"], int)
        assert r["page"] in (1, 2, 3)
        assert "score" in r
        assert r["retrieval_method"] == "semantic"


@pytest.mark.skipif(not _DEPS_AVAILABLE, reason="scikit-learn not installed")
def test_p1_to_p2_tfidf_integration(sample_pdf_path: Path):
    """Verify that process_pdf chunks work with TF-IDF search baseline."""
    pdf_result = process_pdf(sample_pdf_path, chunk_size=800, overlap=100)
    chunks = pdf_result["chunks"]

    retriever = DocumentRetriever()
    retriever.build_index(chunks)

    tfidf_results = retriever.tfidf_search("chunking overlap", k=2)
    assert len(tfidf_results) > 0
    assert tfidf_results[0]["retrieval_method"] == "tfidf"
    assert tfidf_results[0]["page"] in (1, 2, 3)
