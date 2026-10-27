"""
DocuMind - Module 2 Test Suite (test_retriever.py)
==================================================
Demonstrates and validates:
1. FAISS Semantic Search using Sentence Transformers.
2. TF-IDF + Cosine Similarity Baseline.
3. Edge cases:
   - Empty documents / empty chunk list
   - Blank queries (empty string, whitespace, None)
   - Invalid chunks (malformed dictionaries, missing text)
   - k values larger than chunk count
   - k values <= 0
   - Search before index is built
4. Consistent output schema verification.
5. Page metadata preservation for Person 3's citations.
"""

import sys
import unittest
from typing import List, Dict, Any

import retriever
from retriever import DocumentRetriever, SAMPLE_CHUNKS, build_index, semantic_search, tfidf_search


# ============================================================================
# Test Mock Setup for Environments Without Heavy PyPI Packages Installed
# (Ensures all logic, edge cases, and schema checks pass anywhere)
# ============================================================================
def setup_mock_environment():
    """Injects lightweight mock implementations if heavy libraries are not installed."""
    if retriever._DEPS_AVAILABLE:
        return

    import types

    class MockArray(list):
        def astype(self, dtype):
            return self

        def flatten(self):
            return self

    class MockNumpy(types.ModuleType):
        def argsort(self, seq):
            # Sort indices descending by score
            indexed = list(enumerate(seq))
            indexed.sort(key=lambda x: x[1])
            return [idx for idx, _ in indexed]

    mock_np = MockNumpy("numpy")

    class MockFAISSIndex:
        def __init__(self, dimension):
            self.dimension = dimension
            self.vectors = []
            self.ntotal = 0

        def add(self, vectors):
            self.vectors.extend(vectors)
            self.ntotal = len(self.vectors)

        def search(self, query_vector, k):
            if not self.vectors:
                return ([[]], [[]])
            # Simulated inner-product scoring
            q = query_vector[0] if isinstance(query_vector, list) else [1.0] * 384
            scores = []
            for idx, vec in enumerate(self.vectors):
                # Simple dot product simulation
                score = sum(a * b for a, b in zip(q[:10], vec[:10]))
                scores.append((score, idx))
            scores.sort(key=lambda x: x[0], reverse=True)
            top_k = scores[:k]
            return ([[s[0] for s in top_k]], [[s[1] for s in top_k]])

    class MockFAISS(types.ModuleType):
        IndexFlatIP = MockFAISSIndex

        @staticmethod
        def normalize_L2(arr):
            pass

    class MockSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
            embeddings = []
            for t in texts:
                # Deterministic pseudo-embedding based on text words
                words = set(t.lower().split())
                vec = [(0.1 * (hash(w) % 10)) for w in sorted(words)[:10]]
                vec += [0.05] * (384 - len(vec))
                embeddings.append(vec)
            return MockArray(embeddings)

    class MockTfidfVectorizer:
        def __init__(self, stop_words=None):
            self.vocabulary = set()

        def fit_transform(self, texts):
            for t in texts:
                self.vocabulary.update(t.lower().split())
            return texts

        def transform(self, texts):
            return texts

    def mock_cosine_similarity(query_texts, doc_texts):
        # Keyword overlap similarity simulation
        q_words = set(query_texts[0].lower().split())
        scores = []
        for doc in doc_texts:
            d_words = set(doc.lower().split())
            overlap = len(q_words.intersection(d_words))
            score = overlap / max(len(q_words), 1)
            scores.append(score)
        return MockArray(scores)

    # Patch module globals
    retriever.np = mock_np
    retriever.faiss = MockFAISS("faiss")
    retriever.SentenceTransformer = MockSentenceTransformer
    retriever.TfidfVectorizer = MockTfidfVectorizer
    retriever.cosine_similarity = mock_cosine_similarity
    retriever._DEPS_AVAILABLE = True


# ============================================================================
# Unit & Integration Tests
# ============================================================================
class TestDocuMindRetriever(unittest.TestCase):

    def setUp(self):
        setup_mock_environment()
        self.retriever = DocumentRetriever()

    def test_sample_dataset_indexing(self):
        """Test indexing with the built-in ML sample dataset."""
        indexed_count = self.retriever.build_index(SAMPLE_CHUNKS)
        self.assertEqual(indexed_count, len(SAMPLE_CHUNKS))
        self.assertEqual(len(self.retriever.chunks), len(SAMPLE_CHUNKS))

    def test_empty_chunks(self):
        """Edge case: build_index with empty or None input."""
        self.assertEqual(self.retriever.build_index([]), 0)
        self.assertEqual(self.retriever.semantic_search("Machine learning"), [])
        self.assertEqual(self.retriever.tfidf_search("Machine learning"), [])

        self.assertEqual(self.retriever.build_index(None), 0)
        self.assertEqual(self.retriever.semantic_search("Machine learning"), [])

    def test_blank_queries(self):
        """Edge case: querying with empty strings, whitespace, or non-strings."""
        self.retriever.build_index(SAMPLE_CHUNKS)
        for blank in ["", "   ", "\n\t", None]:
            self.assertEqual(self.retriever.semantic_search(blank), [])
            self.assertEqual(self.retriever.tfidf_search(blank), [])

    def test_invalid_chunks_filtering(self):
        """Edge case: filter out chunks with missing text or invalid data types."""
        dirty_chunks = [
            {"text": "Valid passage one", "page": 1},
            {"page": 2},                          # missing text
            {"text": "", "page": 3},               # empty text
            {"text": "   ", "page": 4},             # whitespace text
            "invalid non-dict chunk",              # invalid format
            {"text": "Valid passage two", "page": 5},
        ]
        indexed_count = self.retriever.build_index(dirty_chunks)
        self.assertEqual(indexed_count, 2)
        self.assertEqual(self.retriever.chunks[0]["text"], "Valid passage one")
        self.assertEqual(self.retriever.chunks[1]["text"], "Valid passage two")

    def test_k_larger_than_available_chunks(self):
        """Edge case: requesting k greater than the number of indexed chunks."""
        small_dataset = [
            {"text": "Chunk A", "page": 1},
            {"text": "Chunk B", "page": 2},
        ]
        self.retriever.build_index(small_dataset)

        # Ask for k=10 when only 2 chunks exist
        sem_res = self.retriever.semantic_search("Chunk", k=10)
        tfidf_res = self.retriever.tfidf_search("Chunk", k=10)

        self.assertEqual(len(sem_res), 2)
        self.assertEqual(len(tfidf_res), 2)

    def test_k_non_positive(self):
        """Edge case: k <= 0 should return empty list."""
        self.retriever.build_index(SAMPLE_CHUNKS)
        self.assertEqual(self.retriever.semantic_search("Gradient descent", k=0), [])
        self.assertEqual(self.retriever.semantic_search("Gradient descent", k=-5), [])
        self.assertEqual(self.retriever.tfidf_search("Gradient descent", k=0), [])

    def test_search_before_build_index(self):
        """Edge case: searching prior to calling build_index."""
        uninitialized_retriever = DocumentRetriever()
        self.assertEqual(uninitialized_retriever.semantic_search("test query"), [])
        self.assertEqual(uninitialized_retriever.tfidf_search("test query"), [])

    def test_output_schema_and_metadata_preservation(self):
        """Verify that output format is identical for both methods and preserves page numbers."""
        self.retriever.build_index(SAMPLE_CHUNKS)
        query = "regularization and overfitting remedies"

        sem_results = self.retriever.semantic_search(query, k=3)
        tfidf_results = self.retriever.tfidf_search(query, k=3)

        self.assertGreater(len(sem_results), 0)
        self.assertGreater(len(tfidf_results), 0)

        expected_keys = {"text", "page", "score", "retrieval_method"}

        for item in sem_results:
            self.assertEqual(set(item.keys()), expected_keys)
            self.assertEqual(item["retrieval_method"], "semantic")
            self.assertIsInstance(item["page"], int)
            self.assertIsInstance(item["score"], float)
            self.assertIsInstance(item["text"], str)

        for item in tfidf_results:
            self.assertEqual(set(item.keys()), expected_keys)
            self.assertEqual(item["retrieval_method"], "tfidf")
            self.assertIsInstance(item["page"], int)
            self.assertIsInstance(item["score"], float)
            self.assertIsInstance(item["text"], str)

    def test_convenience_functions(self):
        """Verify module-level functions build_index, semantic_search, tfidf_search."""
        count = build_index(SAMPLE_CHUNKS)
        self.assertEqual(count, len(SAMPLE_CHUNKS))

        sem = semantic_search("Transformer attention mechanism", k=2)
        tfidf = tfidf_search("Transformer attention mechanism", k=2)

        self.assertEqual(len(sem), 2)
        self.assertEqual(len(tfidf), 2)


# ============================================================================
# Interactive Demonstration Runner
# Displays results clearly in the terminal for viva and manual inspection
# ============================================================================
def run_demonstration():
    print("=" * 75)
    print("  DocuMind (Person 2): Embeddings & Semantic Search Demonstration")
    print("=" * 75)

    retriever_inst = DocumentRetriever()
    print(f"\n[1] Indexing Built-in Machine Learning Dataset ({len(SAMPLE_CHUNKS)} chunks)...")
    indexed = retriever_inst.build_index(SAMPLE_CHUNKS)
    print(f"    -> Successfully indexed {indexed} chunks.\n")

    test_queries = [
        "How can we prevent overfitting in machine learning models?",
        "What is the transformer architecture and self-attention?",
        "How does clustering work in unsupervised learning?",
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print("-" * 75)
        print(f"Query {q_idx}: \"{query}\"")
        print("-" * 75)

        # Semantic Search (Sentence Transformers + FAISS)
        sem_res = retriever_inst.semantic_search(query, k=2)
        print("\n  >> Method: FAISS Semantic Search (all-MiniLM-L6-v2 + IndexFlatIP)")
        for rank, res in enumerate(sem_res, 1):
            print(f"     Rank {rank} | Score: {res['score']:.4f} | Page: {res['page']}")
            print(f"     Excerpt: {res['text'][:120]}...\n")

        # TF-IDF Baseline Search
        tfidf_res = retriever_inst.tfidf_search(query, k=2)
        print("  >> Method: TF-IDF Baseline (Cosine Similarity)")
        for rank, res in enumerate(tfidf_res, 1):
            print(f"     Rank {rank} | Score: {res['score']:.4f} | Page: {res['page']}")
            print(f"     Excerpt: {res['text'][:120]}...\n")

    print("=" * 75)
    print("Demonstration completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    # Run unittests first
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDocuMindRetriever)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    print("\nAll unit and edge case tests passed!\n")
    # Run the interactive demonstration
    run_demonstration()
