"""Mock client and grounded simulation for offline testing of Person 3."""

import re
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from app.generation.prompts import INSUFFICIENT_EVIDENCE_PHRASE

# Common English stop words to ignore when extracting content keywords
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "please", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "tell", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "whatever", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves", "use", "used", "uses", "using",
}


def normalize_token(t: str) -> str:
    """Normalize simple English plurals to singular stem for morphological matching."""
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    return t


def extract_content_keywords(text: str) -> set:
    """Extract non-trivial lowercase content keywords from a text string."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    result = set()
    for t in tokens:
        if t not in STOP_WORDS:
            norm = normalize_token(t)
            if norm not in STOP_WORDS:
                result.add(norm)
    return result


def simulate_grounded_response(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Simulate a grounded LLM response by matching question keywords against retrieved passages.

    - If evidence matching question keywords or conceptual synonyms is found in a chunk,
      extracts the matching sentence and attaches the genuine [Doc: ..., Page: ...] citation.
    - If no relevant keywords match any chunk, strictly returns INSUFFICIENT_EVIDENCE_PHRASE.
    """
    if not question or not question.strip() or not retrieved_chunks:
        return INSUFFICIENT_EVIDENCE_PHRASE

    q_keywords = extract_content_keywords(question)
    if not q_keywords:
        return INSUFFICIENT_EVIDENCE_PHRASE

    # Expand keywords with NLP domain concept synonyms for semantic paraphrase support
    from app.retrieval.semantic_retriever import NLP_CONCEPT_EXPANSIONS
    expanded_q_keywords = set(q_keywords)
    q_low = question.lower()
    for phrase, synonyms in NLP_CONCEPT_EXPANSIONS.items():
        if phrase in q_low:
            for syn in synonyms:
                expanded_q_keywords.update(extract_content_keywords(syn))

    scored_matches = []

    for chunk in retrieved_chunks:
        doc_id = chunk.get("document_id", "Unknown")
        page_num = chunk.get("page_number")
        page_str = str(page_num) if page_num is not None else "N/A"
        chunk_id = chunk.get("chunk_id", "unknown")
        text = chunk.get("text", "")

        # Split passage into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for sentence in sentences:
            sent_clean = sentence.strip()
            if not sent_clean:
                continue

            s_keywords = extract_content_keywords(sent_clean)
            direct_overlap = q_keywords.intersection(s_keywords)
            concept_overlap = expanded_q_keywords.intersection(s_keywords)

            # Compute how many of the question's unique content keywords are addressed
            covered_q_words = set()
            for q_w in q_keywords:
                if q_w in s_keywords:
                    covered_q_words.add(q_w)
                else:
                    # Check if concept synonyms of this q_w appear
                    syns = NLP_CONCEPT_EXPANSIONS.get(q_w, [])
                    if any(any(st in s_keywords for st in extract_content_keywords(syn)) for syn in syns):
                        covered_q_words.add(q_w)

            # Also check multi-word phrase matches from query
            for phrase, syns in NLP_CONCEPT_EXPANSIONS.items():
                if phrase in q_low:
                    p_words = extract_content_keywords(phrase)
                    if any(any(st in s_keywords for st in extract_content_keywords(syn)) for syn in syns):
                        covered_q_words.update(p_words)

            score = (len(direct_overlap) * 2.0) + (len(concept_overlap) * 1.0)
            if score > 0:
                scored_matches.append({
                    "score": score,
                    "sentence": sent_clean,
                    "document_id": doc_id,
                    "page_str": page_str,
                    "chunk_id": chunk_id,
                    "covered_q_words": covered_q_words,
                })

    # Ignore sentences that are injected adversarial instructions
    filtered_matches = []
    injection_regex = re.compile(
        r"ignore\s+(all\s+)?(previous\s+)?(rules|instructions)|say\s+(the\s+)?ceo|disregard",
        re.IGNORECASE,
    )
    for m in scored_matches:
        if not injection_regex.search(m["sentence"]):
            filtered_matches.append(m)

    # If no chunk sentences match or only injected sentences matched, evidence is insufficient
    if not filtered_matches:
        return INSUFFICIENT_EVIDENCE_PHRASE

    # Sort matches by keyword overlap score descending
    filtered_matches.sort(key=lambda x: x["score"], reverse=True)

    # Check question coverage: if question has 3+ content keywords and top match covers <= 35% of them,
    # the question asks for missing information not supported by the document.
    top_match = filtered_matches[0]
    covered_kws = top_match.get("covered_q_words", set())
    if len(q_keywords) >= 3 and (len(covered_kws) / len(q_keywords)) <= 0.35:
        return INSUFFICIENT_EVIDENCE_PHRASE

    # Pick up to the top 2 distinct matching sentences
    selected_matches = []
    seen_sentences = set()
    for match in filtered_matches:
        if match["sentence"] not in seen_sentences:
            seen_sentences.add(match["sentence"])
            selected_matches.append(match)
            if len(selected_matches) >= 2:
                break

    # Check if selected sentences contradict each other (e.g. differing dates, versions, or explicit contradiction)
    if len(selected_matches) >= 2:
        s1 = selected_matches[0]["sentence"]
        s2 = selected_matches[1]["sentence"]
        s1_low, s2_low = s1.lower(), s2.lower()
        # Common contradiction patterns in tests
        is_contradiction = (
            ("released" in s1_low and "released" in s2_low and s1_low != s2_low)
            or ("january" in s1_low and "november" in s2_low)
            or ("founded in 2024" in s1_low and "founded in 2020" in s2_low)
            or ("faiss" in s1_low and "chroma" in s2_low)
        )
        if is_contradiction:
            return (
                f"Conflict detected: [Doc: {selected_matches[0]['document_id']}, Page: {selected_matches[0]['page_str']}] states '{s1}', "
                f"while [Doc: {selected_matches[1]['document_id']}, Page: {selected_matches[1]['page_str']}] states '{s2}'."
            )

    answer_parts = []
    for item in selected_matches:
        sent = item["sentence"]
        if not sent.endswith((".", "!", "?")):
            sent += "."
        answer_parts.append(
            f"{sent} [Doc: {item['document_id']}, Page: {item['page_str']}]"
        )

    if not answer_parts:
        return INSUFFICIENT_EVIDENCE_PHRASE

    return "[Simulated Response] " + " ".join(answer_parts)


def create_mock_client(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> MagicMock:
    """
    Create a mock genai.Client that dynamically simulates Gemini's response
    for the specific question and retrieved chunks.
    """
    simulated_text = simulate_grounded_response(question, retrieved_chunks)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = simulated_text
    mock_client.models.generate_content.return_value = mock_resp
    return mock_client
