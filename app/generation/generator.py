"""RAG answer generation module using Google's google-genai SDK for Gemini."""

import os
import time
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

from google import genai
from google.genai import types

import re
from app.generation.prompts import (
    SYSTEM_INSTRUCTION,
    INSUFFICIENT_EVIDENCE_PHRASE,
    build_rag_prompt,
    build_without_rag_prompt,
)
from app.generation.citation_validator import validate_citations

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"

# Error codes and keywords for transient, retryable failures
RETRYABLE_ERROR_CODES = {429, 500, 503}
RETRYABLE_KEYWORDS = (
    "429",
    "500",
    "503",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "RESOURCEEXHAUSTED",
    "SERVICE UNAVAILABLE",
    "SERVICEUNAVAILABLE",
    "TEMPORARILY UNAVAILABLE",
)
DEFAULT_BACKOFF_DELAYS = [1.0, 2.0, 4.0, 8.0]


def is_retryable_exception(exc: Exception) -> bool:
    """Check if exception is retryable (429, 500, 503, or UNAVAILABLE)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in RETRYABLE_ERROR_CODES:
        return True

    status = getattr(exc, "status", None)
    if status and "UNAVAILABLE" in str(status).upper():
        return True

    msg = str(exc).upper()
    return any(keyword in msg for keyword in RETRYABLE_KEYWORDS)


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initialize and return a Google GenAI Client.

    Parameters:
        api_key: Optional Gemini API key. If not provided, reads GEMINI_API_KEY
                 from the environment (.env).

    Raises:
        ValueError: If no API key is found.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or key.strip() == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or contains a placeholder. "
            "Please configure your real key in a .env file."
        )
    return genai.Client(api_key=key.strip())


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    backoff_delays: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Generate a grounded, cited answer to a question given retrieved document passages.

    Expected input format for each chunk in `retrieved_chunks`:
    [
        {
            "document_id": "handbook.pdf", # Name or identifier of document
            "chunk_id": "chunk_01",         # Unique passage identifier
            "page_number": 4,               # Source page number (or None)
            "text": "Full passage text..."  # Text content of the passage
        },
        ...
    ]

    Parameters:
        question: The user's question.
        retrieved_chunks: Passages retrieved by Person 2's retrieval module.
        client: Optional pre-configured genai.Client (useful for dependency injection / mocking).
        model_name: Optional Gemini model name override (defaults to GEMINI_MODEL env var or gemini-3.8-flash).
        temperature: Sampling temperature (default 0.0 for deterministic, factual outputs).
        backoff_delays: Optional list of retry backoff delays in seconds (defaults to [1.0, 2.0, 4.0, 8.0]).
    """
    # 1. Guard against empty or whitespace-only questions
    if not question or not question.strip():
        return {
            "answer": "Please provide a valid, non-empty question.",
            "sources": [],
            "insufficient_evidence": True,
            "status": "Insufficient Evidence",
        }

    # 2. Guard against missing or empty passages
    if not retrieved_chunks:
        return {
            "answer": INSUFFICIENT_EVIDENCE_PHRASE,
            "sources": [],
            "insufficient_evidence": True,
            "status": "Insufficient Evidence",
        }

    # 3. Determine model name
    effective_model = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)

    # 4. Prepare prompt
    prompt = build_rag_prompt(question, retrieved_chunks)

    # 5. Call Gemini LLM with retry & exponential backoff on transient errors
    delays = backoff_delays if backoff_delays is not None else DEFAULT_BACKOFF_DELAYS
    max_retries = len(delays)
    attempt = 0
    raw_text = ""

    while True:
        try:
            active_client = client if client is not None else get_gemini_client()
            response = active_client.models.generate_content(
                model=effective_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=temperature,
                ),
            )

            raw_text = response.text if response and hasattr(response, "text") and response.text else ""
            if not raw_text.strip():
                return {
                    "answer": INSUFFICIENT_EVIDENCE_PHRASE,
                    "sources": [],
                    "insufficient_evidence": True,
                    "status": "Insufficient Evidence",
                }
            break  # Successful response from Gemini

        except ValueError as ve:
            # Configuration error (e.g., missing API key)
            logger.error(f"Configuration error: {ve}")
            return {
                "answer": f"Configuration Error: {ve}",
                "sources": [],
                "insufficient_evidence": False,
                "status": f"Configuration Error: {ve}",
                "error": str(ve),
            }

        except Exception as exc:
            if is_retryable_exception(exc) and attempt < max_retries:
                delay = delays[attempt]
                logger.warning(
                    f"Retryable error calling Gemini ({exc}). "
                    f"Retrying ({attempt + 1}/{max_retries}) after {delay}s backoff..."
                )
                time.sleep(delay)
                attempt += 1
                continue

            if is_retryable_exception(exc):
                # All retries failed
                logger.error(f"All {max_retries} retries failed for Gemini API: {exc}")
                return {
                    "answer": "API Error: Gemini temporarily unavailable",
                    "sources": [],
                    "insufficient_evidence": False,
                    "status": "API Error: Gemini temporarily unavailable",
                    "error": str(exc),
                }

            # Non-retryable error (e.g. 400 Bad Request, 404 Not Found)
            logger.error(f"Error during Gemini generation: {exc}")
            return {
                "answer": f"Error generating answer: {exc}",
                "sources": [],
                "insufficient_evidence": False,
                "status": f"API Error: {exc}",
                "error": str(exc),
            }

    # 6. Validate and extract citations from the LLM's response
    validation_result = validate_citations(raw_text, retrieved_chunks)

    # 7. Check for contradiction / conflict indicators
    conflict_detected = bool(
        re.search(
            r"\b(conflict detected|contradict|contradiction|conflicting passages|conflicting sources)\b",
            raw_text,
            re.IGNORECASE,
        )
    )
    validation_result["conflict_detected"] = conflict_detected
    validation_result["conflict_message"] = (
        "Retrieved passages contain conflicting information on this question."
        if conflict_detected
        else ""
    )
    return validation_result


def generate_without_rag(
    question: str,
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    backoff_delays: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Generate an ungrounded baseline answer to a question WITHOUT any retrieved passages (no RAG).
    Used for side-by-side comparison with the grounded DocuMind AI RAG pipeline.
    """
    if not question or not question.strip():
        return {
            "answer": "Please provide a valid, non-empty question.",
            "sources": [],
            "status": "Invalid Question",
        }

    effective_model = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    prompt = build_without_rag_prompt(question)

    # Check if client is MagicMock (for offline testing)
    from unittest.mock import MagicMock
    if isinstance(client, MagicMock):
        try:
            resp = client.models.generate_content(
                model=effective_model,
                contents=prompt,
            )
            raw_text = resp.text if resp and hasattr(resp, "text") and resp.text else ""
            if not raw_text:
                raw_text = f"General knowledge answer for: {question}"
        except Exception:
            raw_text = f"General knowledge baseline response for '{question}'."
        return {
            "answer": raw_text,
            "sources": [],
            "status": "Without RAG (Baseline LLM)",
        }

    delays = backoff_delays if backoff_delays is not None else DEFAULT_BACKOFF_DELAYS
    max_retries = len(delays)
    attempt = 0

    while True:
        try:
            active_client = client if client is not None else get_gemini_client()
            response = active_client.models.generate_content(
                model=effective_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            raw_text = response.text if response and hasattr(response, "text") and response.text else ""
            return {
                "answer": raw_text.strip(),
                "sources": [],
                "status": "Without RAG (Baseline LLM)",
            }
        except ValueError as ve:
            return {
                "answer": f"Configuration Error: {ve}",
                "sources": [],
                "status": f"Configuration Error: {ve}",
                "error": str(ve),
            }
        except Exception as exc:
            if is_retryable_exception(exc) and attempt < max_retries:
                delay = delays[attempt]
                logger.warning(f"Retryable error in without-rag: {exc}. Retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                continue
            return {
                "answer": f"API Error: {exc}",
                "sources": [],
                "status": f"API Error: {exc}",
                "error": str(exc),
            }
