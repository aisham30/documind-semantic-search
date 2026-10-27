"""Retrieval adapter module providing the retrieve(question) interface.

This decouples the answer generation, claim verification, and UI from the retrieval layer,
allowing Person 2's retrieval pipeline to be plugged in seamlessly.
"""

from typing import Any, Callable, Dict, List, Optional

# Default sample passages provided for offline demonstration and testing
DEFAULT_SAMPLE_CHUNKS: List[Dict[str, Any]] = [
    {
        "document_id": "documind_whitepaper.pdf",
        "chunk_id": "chunk_001",
        "page_number": 1,
        "text": (
            "DocuMind AI is an intelligent PDF question-answering system. "
            "It indexes uploaded PDF documents and provides factual, cited answers."
        ),
    },
    {
        "document_id": "documind_whitepaper.pdf",
        "chunk_id": "chunk_002",
        "page_number": 3,
        "text": (
            "The generation layer uses Google Gemini with temperature 0.0 to produce grounded, "
            "hallucination-free answers with precise page citations."
        ),
    },
]

# Global hook for Person 2's retriever function
_custom_retriever: Optional[Callable[..., List[Dict[str, Any]]]] = None


def set_retriever(retriever_fn: Callable[..., List[Dict[str, Any]]]) -> None:
    """Register Person 2's custom retrieval function."""
    global _custom_retriever
    _custom_retriever = retriever_fn


def reset_retriever() -> None:
    """Reset the retriever hook back to default sample passages."""
    global _custom_retriever
    _custom_retriever = None


def get_available_collections() -> List[str]:
    """Return names of the 6 NLP educational dataset collections."""
    from app.retrieval.document_loader import get_document_loader
    return get_document_loader().get_collection_names()


def get_collection_stats() -> Dict[str, Any]:
    """Return summary statistics of loaded collections and documents."""
    from app.retrieval.document_loader import get_document_loader
    return get_document_loader().get_stats()


def retrieve(
    question: str,
    collection: Optional[str] = None,
    top_k: int = 3,
    min_score: float = 0.08,
    use_collections: bool = False,
) -> List[Dict[str, Any]]:
    """
    Primary retrieval interface for DocuMind AI.

    Parameters:
        question: Natural language question from the user.
        collection: Optional collection name to strictly filter retrieval.
                    Supports 'All Collections' or any of the 6 educational domains.
        top_k: Number of candidate chunks to retrieve.
        min_score: Relevance confidence threshold. If evidence is insufficient,
                   returns empty list [] to trigger the refusal guardrail.
        use_collections: If True, uses the indexed educational collections even
                         if collection is None.

    Returns:
        List of chunk dictionaries formatted as:
        [
            {
                "document_id": str,
                "chunk_id": str,
                "page_number": int | None,
                "text": str
            },
            ...
        ]
    """
    if _custom_retriever is not None:
        try:
            return _custom_retriever(question)
        except Exception:
            # Fall back on retriever error
            pass

    # If a specific collection is requested or use_collections is explicitly enabled
    if collection is not None or use_collections:
        try:
            from app.retrieval.semantic_retriever import get_semantic_retriever
            retriever = get_semantic_retriever()
            return retriever.retrieve(
                query=question,
                collection_name=collection,
                top_k=top_k,
                min_relevance_threshold=min_score,
            )
        except Exception as e:
            # Graceful fallback to default sample passages on index failure
            pass

    return DEFAULT_SAMPLE_CHUNKS

