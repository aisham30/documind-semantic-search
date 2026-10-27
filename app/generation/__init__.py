"""RAG Answer Generation module for DocuMind AI."""

from app.generation.generator import (
    generate_answer,
    generate_without_rag,
    get_gemini_client,
)
from app.generation.citation_validator import validate_citations
from app.generation.prompts import (
    build_rag_prompt,
    format_context_chunks,
    INSUFFICIENT_EVIDENCE_PHRASE,
    SYSTEM_INSTRUCTION,
    build_claim_verification_prompt,
    build_without_rag_prompt,
)
from app.generation.mock_client import create_mock_client, simulate_grounded_response
from app.generation.claim_verifier import (
    split_into_claims,
    verify_claim,
    verify_all_claims,
    compute_trust_score,
)
from app.generation.retrieval_adapter import (
    retrieve,
    set_retriever,
    reset_retriever,
)

__all__ = [
    "generate_answer",
    "generate_without_rag",
    "get_gemini_client",
    "validate_citations",
    "build_rag_prompt",
    "format_context_chunks",
    "INSUFFICIENT_EVIDENCE_PHRASE",
    "SYSTEM_INSTRUCTION",
    "build_claim_verification_prompt",
    "build_without_rag_prompt",
    "create_mock_client",
    "simulate_grounded_response",
    "split_into_claims",
    "verify_claim",
    "verify_all_claims",
    "compute_trust_score",
    "retrieve",
    "set_retriever",
    "reset_retriever",
]

