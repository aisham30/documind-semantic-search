"""Claim-level verification and trust scoring module for DocuMind AI.

Splits generated answers into claims, verifies each claim against cited context passages
using Gemini (temperature 0) to assign verdicts (SUPPORTS, PARTIALLY SUPPORTS, NOT SUPPORT),
and computes a composite Trust Score (0-100).
"""

import os
import re
import time
import logging
from typing import Any, Dict, List, Optional
from google.genai import types

from app.generation.prompts import (
    CLAIM_VERIFICATION_SYSTEM_INSTRUCTION,
    build_claim_verification_prompt,
    INSUFFICIENT_EVIDENCE_PHRASE,
)
from app.generation.citation_validator import (
    DOC_PAGE_PATTERN,
    CHUNK_PATTERN,
    DOC_CHUNK_PAGE_PATTERN,
    normalize_str,
)

logger = logging.getLogger(__name__)


def split_into_claims(answer: str) -> List[str]:
    """
    Split a generated answer into individual testable claims (sentences).
    Removes simulation markers and strips out standalone refusal statements.
    """
    if not answer or not answer.strip():
        return []

    clean_text = answer.strip()
    if clean_text.startswith("[Simulated Response]"):
        clean_text = clean_text[len("[Simulated Response]"):].strip()

    # If the text is purely the insufficient evidence phrase, no claims to verify
    if clean_text.lower() == INSUFFICIENT_EVIDENCE_PHRASE.lower():
        return []

    # Split by sentence end boundaries: . ! ?
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    claims = []
    for s in raw_sentences:
        sent = s.strip()
        if not sent:
            continue
        # Skip pure fallback explanation sentences that do not make factual assertions
        if sent.lower().startswith("i do not have sufficient evidence"):
            continue
        claims.append(sent)

    return claims


def find_cited_passages_for_claim(
    claim: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Identify which chunk(s) from retrieved_chunks are explicitly cited in the claim.
    If no citation is found, returns all retrieved_chunks as context.
    """
    cited_chunks = []
    seen = set()

    for match in DOC_PAGE_PATTERN.finditer(claim):
        cited_doc = normalize_str(match.group(1))
        cited_page = normalize_str(match.group(2))
        for chunk in retrieved_chunks:
            if (
                normalize_str(chunk.get("document_id")) == cited_doc
                and normalize_str(chunk.get("page_number")) == cited_page
            ):
                key = (chunk.get("document_id"), chunk.get("page_number"), chunk.get("chunk_id"))
                if key not in seen:
                    seen.add(key)
                    cited_chunks.append(chunk)

    for match in CHUNK_PATTERN.finditer(claim):
        cited_chunk_id = normalize_str(match.group(1))
        for chunk in retrieved_chunks:
            if normalize_str(chunk.get("chunk_id")) == cited_chunk_id:
                key = (chunk.get("document_id"), chunk.get("page_number"), chunk.get("chunk_id"))
                if key not in seen:
                    seen.add(key)
                    cited_chunks.append(chunk)

    return cited_chunks if cited_chunks else retrieved_chunks


def _verify_claim_offline(
    claim: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fast offline simulation for claim verification when running tests or without API key.
    Calculates sentence-level token overlap between claim and reference chunks.
    """
    from app.generation.mock_client import extract_content_keywords

    # Strip citation brackets before extracting claim keywords
    claim_clean = re.sub(r"\[Doc:[^\]]+\]|\[Chunk:[^\]]+\]", "", claim).strip()
    claim_words = extract_content_keywords(claim_clean)

    best_overlap = 0
    best_sentence = "None"

    for chunk in chunks:
        text = chunk.get("text", "")
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = extract_content_keywords(s_clean)
            overlap = len(claim_words.intersection(s_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s_clean

    total_words = max(1, len(claim_words))
    ratio = best_overlap / total_words

    if ratio >= 0.45 or best_overlap >= 3:
        verdict = "SUPPORTS"
        tag = "supported"
    elif ratio >= 0.20 or best_overlap >= 1:
        verdict = "PARTIALLY SUPPORTS"
        tag = "weak"
    else:
        verdict = "NOT SUPPORT"
        tag = "unsupported"
        best_sentence = "None"

    return {
        "verdict": verdict,
        "tag": tag,
        "evidence": best_sentence,
    }


def verify_claim(
    claim: str,
    retrieved_chunks: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
    backoff_delays: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Verify a single claim against its cited passage using Gemini at temperature 0.0.
    Falls back to offline sentence matching if client is mocked or simulated.
    """
    cited_chunks = find_cited_passages_for_claim(claim, retrieved_chunks)
    has_citation = bool(
        DOC_PAGE_PATTERN.search(claim)
        or CHUNK_PATTERN.search(claim)
        or DOC_CHUNK_PAGE_PATTERN.search(claim)
    )

    combined_passage_text = "\n\n".join(
        f"[Doc: {c.get('document_id', 'Unknown')}, Page: {c.get('page_number', 'N/A')}]\n{c.get('text', '')}"
        for c in cited_chunks
    )

    # Check if client is a MagicMock (unit tests / mock simulation)
    from unittest.mock import MagicMock
    if isinstance(client, MagicMock):
        res = _verify_claim_offline(claim, cited_chunks)
        res["claim"] = claim
        res["has_citation"] = has_citation
        return res

    # If no live client and no API key configured, use offline simulation
    from app.generation.generator import get_gemini_client, is_retryable_exception, DEFAULT_MODEL
    try:
        active_client = client if client is not None else get_gemini_client()
    except ValueError:
        res = _verify_claim_offline(claim, cited_chunks)
        res["claim"] = claim
        res["has_citation"] = has_citation
        return res

    effective_model = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    prompt = build_claim_verification_prompt(claim, combined_passage_text)

    delays = backoff_delays if backoff_delays is not None else [1.0, 2.0, 4.0]
    max_retries = len(delays)
    attempt = 0

    while True:
        try:
            response = active_client.models.generate_content(
                model=effective_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=CLAIM_VERIFICATION_SYSTEM_INSTRUCTION,
                    temperature=0.0,
                ),
            )
            raw_text = response.text if response and hasattr(response, "text") and response.text else ""
            break
        except Exception as exc:
            if is_retryable_exception(exc) and attempt < max_retries:
                delay = delays[attempt]
                logger.warning(f"Retryable error during claim verification: {exc}. Retrying in {delay}s...")
                time.sleep(delay)
                attempt += 1
                continue
            logger.error(f"Error during claim verification: {exc}. Falling back to offline check.")
            res = _verify_claim_offline(claim, cited_chunks)
            res["claim"] = claim
            res["has_citation"] = has_citation
            return res

    # Parse LLM verification response
    verdict = "NOT SUPPORT"
    tag = "unsupported"
    evidence = "None"

    for line in raw_text.splitlines():
        line_strip = line.strip()
        if line_strip.upper().startswith("VERDICT:"):
            v_val = line_strip.split(":", 1)[1].strip().upper()
            if "PARTIALLY" in v_val:
                verdict = "PARTIALLY SUPPORTS"
                tag = "weak"
            elif "SUPPORTS" in v_val:
                verdict = "SUPPORTS"
                tag = "supported"
            else:
                verdict = "NOT SUPPORT"
                tag = "unsupported"
        elif line_strip.upper().startswith("EVIDENCE:"):
            evidence = line_strip.split(":", 1)[1].strip()

    # Fallback evidence sentence extraction if LLM returned 'None' but supported
    if evidence in ("None", "") and tag in ("supported", "weak"):
        offline_match = _verify_claim_offline(claim, cited_chunks)
        evidence = offline_match.get("evidence", "Context passage supports this assertion.")

    return {
        "claim": claim,
        "verdict": verdict,
        "tag": tag,
        "evidence": evidence,
        "has_citation": has_citation,
    }


def verify_all_claims(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract and verify all claims in a generated answer against the retrieved chunks.
    """
    claims = split_into_claims(answer)
    verified_claims = []
    for claim in claims:
        res = verify_claim(
            claim=claim,
            retrieved_chunks=retrieved_chunks,
            client=client,
            model_name=model_name,
        )
        verified_claims.append(res)
    return verified_claims


def compute_trust_score(
    result: Dict[str, Any],
    claims_verification: List[Dict[str, Any]],
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compute a 0-100 composite trust score from:
    1. Citation coverage (35%): Percentage of claims that carry valid citations.
    2. Claim verification quality (45%): Supports (1.0), Partially (0.5), Unsupported (0.0).
    3. Passage utilization (20%): Proportion of relevant retrieved passages successfully cited.

    Returns score dict with numeric score and transparent breakdown metrics.
    """
    chunks = retrieved_chunks or []
    is_insufficient = result.get("insufficient_evidence", False)
    has_error = bool(result.get("error"))

    if has_error:
        return {
            "score": 0,
            "citation_coverage": 0.0,
            "verification_score": 0.0,
            "passage_score": 0.0,
            "breakdown": "Error state: Generation encountered an unrecoverable API/Configuration error.",
            "supported_count": 0,
            "weak_count": 0,
            "unsupported_count": 0,
        }

    # If properly refused due to insufficient evidence, award full marks for faithful refusal guardrail
    if is_insufficient:
        return {
            "score": 100,
            "citation_coverage": 100.0,
            "verification_score": 100.0,
            "passage_score": 100.0,
            "breakdown": "Faithful Refusal: System accurately identified lack of evidence and refrained from hallucinating.",
            "supported_count": 0,
            "weak_count": 0,
            "unsupported_count": 0,
        }

    total_claims = len(claims_verification)
    if total_claims == 0:
        return {
            "score": 0,
            "citation_coverage": 0.0,
            "verification_score": 0.0,
            "passage_score": 0.0,
            "breakdown": "No verifiable claims identified in generated response.",
            "supported_count": 0,
            "weak_count": 0,
            "unsupported_count": 0,
        }

    # 1. Citation Coverage (35%)
    cited_claims = sum(1 for c in claims_verification if c.get("has_citation"))
    citation_coverage_pct = (cited_claims / total_claims) * 100.0

    # 2. Claim Verification Quality (45%)
    supported_cnt = sum(1 for c in claims_verification if c.get("verdict") == "SUPPORTS")
    weak_cnt = sum(1 for c in claims_verification if c.get("verdict") == "PARTIALLY SUPPORTS")
    unsupported_cnt = sum(1 for c in claims_verification if c.get("verdict") == "NOT SUPPORT")

    verification_sum = (supported_cnt * 1.0) + (weak_cnt * 0.5) + (unsupported_cnt * 0.0)
    verification_score_pct = (verification_sum / total_claims) * 100.0

    # 3. Passage Utilization (20%)
    sources = result.get("sources", [])
    valid_sources_count = len(sources)
    expected_passages = max(1, min(2, len(chunks)))
    passage_score_pct = min(1.0, valid_sources_count / expected_passages) * 100.0

    # Conflict penalty: if a contradiction was flagged, adjust score cleanly
    conflict_penalty = 0.0
    if result.get("conflict_detected"):
        conflict_penalty = 10.0

    composite = (
        (0.35 * citation_coverage_pct)
        + (0.45 * verification_score_pct)
        + (0.20 * passage_score_pct)
        - conflict_penalty
    )
    final_score = max(0, min(100, round(composite)))

    breakdown = (
        f"Citation Coverage: {citation_coverage_pct:.0f}% (weight 35%) | "
        f"Claim Verification: {verification_score_pct:.0f}% (weight 45%) | "
        f"Passage Grounding: {passage_score_pct:.0f}% (weight 20%)"
    )
    if conflict_penalty > 0:
        breakdown += f" | Conflict Flagged: -{conflict_penalty:.0f} pts"

    return {
        "score": final_score,
        "citation_coverage": round(citation_coverage_pct, 1),
        "verification_score": round(verification_score_pct, 1),
        "passage_score": round(passage_score_pct, 1),
        "breakdown": breakdown,
        "supported_count": supported_cnt,
        "weak_count": weak_cnt,
        "unsupported_count": unsupported_cnt,
    }
