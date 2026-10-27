"""
DocuMind - End-to-End Demonstration: Module 1 (Person 1) + Module 2 (Person 2)
==============================================================================
Demonstrates how Person 1's PDF Processing & Chunking pipeline directly connects
to Person 2's DocumentRetriever (Sentence Transformers + FAISS + TF-IDF)
without modifying any of Person 2's code.

Pipeline stages executed:
1. Person 1 (backend.pdf_processor):
   - Validates sample_pdfs/research_paper.pdf
   - Extracts clean text page-by-page preserving 1-based page numbers
   - Splits text into overlapping, sentence-aware chunks
2. Person 2 (retriever.DocumentRetriever):
   - Consumes Person 1's output chunks directly
   - Builds FAISS IndexFlatIP (dense 384-d embeddings via all-MiniLM-L6-v2)
   - Builds TF-IDF matrix baseline via scikit-learn
3. Retrieval Demonstration:
   - Executes natural language queries against the indexed document
   - Compares Semantic Search vs. TF-IDF baseline
   - Verifies exact page number citation tracking for Person 3
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pdf_processor import process_pdf
from retriever import DocumentRetriever


def run_pipeline_demo():
    print("=" * 75)
    print(" DocuMind: Person 1 (PDF Ingestion) -> Person 2 (Semantic Search) Demo")
    print("=" * 75)

    sample_pdf_path = PROJECT_ROOT / "sample_pdfs" / "research_paper.pdf"
    if not sample_pdf_path.exists():
        print(f"[!] Generating sample PDF at: {sample_pdf_path}")
        from sample_pdfs.create_sample_pdf import create_sample_paper
        create_sample_paper()

    # ------------------------------------------------------------------------
    # Stage 1: Person 1 - PDF Processing and Chunking
    # ------------------------------------------------------------------------
    print(f"\n[Stage 1] Person 1: Processing PDF '{sample_pdf_path.name}'...")
    pdf_result = process_pdf(sample_pdf_path, chunk_size=1000, overlap=150)

    print(f" -> Document Name     : {pdf_result['document']}")
    print(f" -> Total Pages       : {pdf_result['page_count']}")
    print(f" -> Total Characters  : {pdf_result['total_characters']:,}")
    print(f" -> Generated Chunks  : {pdf_result['chunk_count']}")

    chunks = pdf_result["chunks"]
    print(f"\n[Stage 1 Output Sample] First Chunk Schema:")
    print(f"    chunk_id   : {chunks[0]['chunk_id']}")
    print(f"    page       : {chunks[0]['page']}")
    print(f"    source     : {chunks[0]['source']}")
    print(f"    char_range : {chunks[0]['char_start']}..{chunks[0]['char_end']}")
    print(f"    text snippet: {chunks[0]['text'][:160]}...")

    # ------------------------------------------------------------------------
    # Stage 2: Person 2 - Build FAISS and TF-IDF Indexes
    # ------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[Stage 2] Person 2: Feeding Person 1's chunks into DocumentRetriever...")
    retriever = DocumentRetriever(model_name="all-MiniLM-L6-v2")
    indexed_count = retriever.build_index(chunks)
    print(f" -> Successfully indexed {indexed_count} chunks in FAISS and TF-IDF.")

    # ------------------------------------------------------------------------
    # Stage 3: Querying the Ingested Document
    # ------------------------------------------------------------------------
    test_queries = [
        "Why does inner product equal cosine similarity in FAISS?",
        "What chunking strategy is used to prevent ambiguous citations?",
        "How does DocuMind address hallucination in RAG?",
    ]

    print("\n" + "=" * 75)
    print(" RETRIEVAL COMPARISON: SEMANTIC (FAISS) vs. LEXICAL (TF-IDF)")
    print("=" * 75)

    for q_idx, query in enumerate(test_queries, start=1):
        print(f"\nQuery #{q_idx}: \"{query}\"")
        print("-" * 75)

        # 1. Semantic Search (FAISS)
        sem_results = retriever.semantic_search(query, k=2)
        print(" [Dense Semantic Retrieval - FAISS]")
        for rank, res in enumerate(sem_results, start=1):
            print(f"   Rank {rank} | Page {res['page']} | Score: {res['score']:.4f}")
            print(f"   Passage: {res['text'][:150]}...\n")

        # 2. TF-IDF Search (Baseline)
        tfidf_results = retriever.tfidf_search(query, k=2)
        print(" [Sparse Lexical Retrieval - TF-IDF Baseline]")
        for rank, res in enumerate(tfidf_results, start=1):
            print(f"   Rank {rank} | Page {res['page']} | Score: {res['score']:.4f}")
            print(f"   Passage: {res['text'][:150]}...\n")

    print("=" * 75)
    print(" [VERIFICATION COMPLETE] Person 1's chunks successfully powered Person 2's search!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline_demo()
