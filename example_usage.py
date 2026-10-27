"""Runnable example demonstrating RAG Answer Generation with NLP Educational Collections.

This script demonstrates:
1. Retrieval across the 6 NLP educational collections (or Person 2 adapter).
2. Answer generation with strict citation validation and grounded responses.
3. Verification in either Live Gemini API mode or Mock Simulation mode.
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from app.generation.generator import generate_answer
from app.generation.retrieval_adapter import retrieve, get_available_collections


def run_demo():
    print("=" * 75)
    print("DocuMind AI - RAG Answer Generation with NLP Educational Collections")
    print("=" * 75)

    collections = get_available_collections()
    print(f"\nIndexed Collections ({len(collections)} total):")
    for idx, c in enumerate(collections, start=1):
        print(f"  {idx}. {c}")

    target_collection = "Text Preprocessing and Tokenization"
    question = "Why do modern tokenizers break words into pieces to handle out of vocabulary tokens?"

    print(f"\n1. Selected Collection: \"{target_collection}\"")
    print(f"2. Paraphrased Question Asked:\n   \"{question}\"")

    # Retrieve candidate passages from the selected collection
    passages = retrieve(question, collection=target_collection, top_k=2, use_collections=True)

    print(f"\n3. Retrieved Passages ({len(passages)} passages):")
    for p in passages:
        print(f"   - [Doc: {p['document_id']}, Page: {p.get('page_number')}, Chunk: {p.get('chunk_id')}]")
        print(f"     Title: \"{p.get('title', '')}\"")
        print(f"     Snippet: \"{p.get('text', '')[:120]}...\"")

    # Check if a real API key is configured
    api_key = os.getenv("GEMINI_API_KEY", "")
    using_real_api = api_key and api_key != "your_gemini_api_key_here"

    mock_client = None
    if not using_real_api:
        print("\n[NOTE] No active GEMINI_API_KEY found in .env.")
        print("[NOTE] Using dynamic grounded mock Gemini client for this demonstration run.\n")
        from app.generation.mock_client import create_mock_client
        mock_client = create_mock_client(question, passages)
    else:
        print("\n[INFO] Real GEMINI_API_KEY detected! Querying live Gemini API...\n")

    # Generate answer
    result = generate_answer(
        question=question,
        retrieved_chunks=passages,
        client=mock_client,
    )

    if result.get("error") and using_real_api:
        print(f"\n[NOTE] Live Gemini API encountered network error: {result['error']}.")
        print("[NOTE] Falling back to grounded mock simulator for local demonstration...\n")
        from app.generation.mock_client import create_mock_client
        mock_client = create_mock_client(question, passages)
        result = generate_answer(
            question=question,
            retrieved_chunks=passages,
            client=mock_client,
        )

    print("4. Validated Output Dictionary:")
    print(json.dumps(result, indent=2))

    print("\n5. Verification Summary:")
    print(f"   - Insufficient Evidence: {result['insufficient_evidence']}")
    print(f"   - Status: {result.get('status', 'N/A')}")
    print(f"   - Validated Sources Count: {len(result['sources'])}")
    for source in result["sources"]:
        print(f"     * Document: {source['document_id']} (Page {source['page_number']})")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
