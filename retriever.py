"""
DocuMind - Module 2: Embeddings & Semantic Search
=================================================
Assigned to: Person 2
Description:
    Converts document chunks into dense vector embeddings using Sentence Transformers
    (all-MiniLM-L6-v2) and provides fast cosine similarity retrieval via FAISS (IndexFlatIP).
    Also includes a classical TF-IDF + Cosine Similarity baseline using scikit-learn.

Viva Concepts Covered:
    1. Dense vs. Sparse Embeddings (Sentence Transformers vs. TF-IDF)
    2. L2 Normalization + Inner Product (IP) = Cosine Similarity in FAISS
    3. Document chunking & metadata preservation for RAG citations (P1 -> P2 -> P3)
"""

from typing import List, Dict, Any, Optional
import logging

# Configure logger for clean debugging and demonstration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("DocuMind-Retriever")

# Flags for third-party library availability
_DEPS_AVAILABLE = True
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    _DEPS_AVAILABLE = False
    _IMPORT_ERROR_MSG = str(e)


# ============================================================================
# Built-in Sample Machine Learning Dataset
# Allows Person 2 to test the module independently without waiting for Person 1
# ============================================================================
SAMPLE_CHUNKS: List[Dict[str, Any]] = [
    {
        "text": (
            "Supervised learning is a machine learning paradigm where models are trained "
            "on labeled data. Common algorithms include linear regression, logistic "
            "regression, support vector machines (SVM), decision trees, and neural networks. "
            "The model learns a mapping function from input features to target labels."
        ),
        "page": 1,
    },
    {
        "text": (
            "Unsupervised learning deals with unlabeled datasets. The goal is to discover "
            "underlying patterns, groupings, or representations. Popular techniques include "
            "k-means clustering, hierarchical clustering, and Principal Component Analysis (PCA) "
            "for dimensionality reduction."
        ),
        "page": 2,
    },
    {
        "text": (
            "Overfitting occurs when a machine learning model learns the training data and noise "
            "too well, failing to generalize to unseen test data. Common remedies include "
            "regularization (L1 Lasso, L2 Ridge), dropout in neural networks, early stopping, "
            "and cross-validation."
        ),
        "page": 3,
    },
    {
        "text": (
            "Gradient descent is an iterative optimization algorithm used to minimize the loss "
            "function in machine learning and deep learning models. Variants include stochastic "
            "gradient descent (SGD), mini-batch gradient descent, and adaptive optimizers like Adam."
        ),
        "page": 4,
    },
    {
        "text": (
            "The Transformer architecture relies on the self-attention mechanism to capture "
            "long-range dependencies in natural language without recurrent connections. "
            "Popular transformer models include BERT for masked language modeling and GPT for "
            "autoregressive text generation."
        ),
        "page": 5,
    },
    {
        "text": (
            "Evaluation metrics for classification models include accuracy, precision, recall, "
            "and F1-score. When dealing with imbalanced datasets, accuracy can be misleading, "
            "making precision, recall, and the area under the ROC curve (AUC-ROC) preferred metrics."
        ),
        "page": 6,
    },
]


class DocumentRetriever:
    """
    Handles embedding generation, FAISS vector indexing, semantic search,
    and TF-IDF baseline search for document chunks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever.

        Args:
            model_name: HuggingFace model identifier for sentence-transformers.
                        'all-MiniLM-L6-v2' maps sentences to a 384-dimensional dense vector space.
        """
        self.model_name = model_name
        self.chunks: List[Dict[str, Any]] = []

        # FAISS components
        self.model: Optional[Any] = None
        self.dimension: Optional[int] = None
        self.index: Optional[Any] = None

        # TF-IDF components
        self.tfidf_vectorizer: Optional[Any] = None
        self.tfidf_matrix: Optional[Any] = None

        self._initialized = False

    def _ensure_dependencies(self):
        """Verify that necessary dependencies are available."""
        if not _DEPS_AVAILABLE:
            raise ImportError(
                f"Missing required dependency: {_IMPORT_ERROR_MSG}.\n"
                "Please install dependencies with: pip install -r requirements.txt"
            )

    def _load_model(self):
        """Lazy loader for SentenceTransformer to optimize memory and startup."""
        if self.model is None:
            self._ensure_dependencies()
            logger.info(f"Loading SentenceTransformer model: '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")

    def _validate_and_clean_chunks(self, chunks: Any) -> List[Dict[str, Any]]:
        """
        Validates and cleans incoming document chunks.
        Discards invalid formats and chunks with empty text.

        Accepts chunk format: {"text": str, "page": int or Any}
        """
        if not chunks or not isinstance(chunks, list):
            logger.warning("Invalid or empty chunks list provided to retriever.")
            return []

        valid_chunks: List[Dict[str, Any]] = []
        for idx, item in enumerate(chunks):
            if not isinstance(item, dict):
                logger.warning(f"Chunk at index {idx} skipped: expected dict, got {type(item).__name__}.")
                continue

            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                logger.warning(f"Chunk at index {idx} skipped: 'text' field is empty or missing.")
                continue

            # Preserve page information, default to 1 if missing or invalid
            page = item.get("page", 1)

            valid_chunks.append({
                "text": text.strip(),
                "page": page
            })

        return valid_chunks

    def build_index(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Builds both the FAISS semantic vector index and the TF-IDF index.

        Steps:
        1. Clean and validate chunks.
        2. Encode texts using Sentence Transformers into 384-dimensional embeddings.
        3. L2-normalize embeddings so inner-product search equals cosine similarity.
        4. Populate faiss.IndexFlatIP.
        5. Fit scikit-learn TfidfVectorizer on chunk texts.

        Args:
            chunks: List of dictionaries with 'text' and 'page' keys.

        Returns:
            int: Number of successfully indexed chunks.
        """
        self._ensure_dependencies()
        self.chunks = self._validate_and_clean_chunks(chunks)

        if not self.chunks:
            logger.warning("No valid chunks to index. Index remains empty.")
            self.index = None
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self._initialized = False
            return 0

        texts = [chunk["text"] for chunk in self.chunks]

        # 1. Build FAISS Semantic Index
        self._load_model()
        logger.info(f"Generating dense embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype("float32")

        # L2-normalization for cosine similarity via inner-product
        # Mathematical relation: cos(u, v) = (u . v) / (||u|| * ||v||)
        # When ||u|| = 1 and ||v|| = 1, cos(u, v) = u . v (Inner Product)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        logger.info(f"FAISS IndexFlatIP built with {self.index.ntotal} vectors.")

        # 2. Build TF-IDF Baseline Index
        logger.info("Building TF-IDF baseline index...")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info("TF-IDF matrix built successfully.")

        self._initialized = True
        return len(self.chunks)

    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic retrieval using Sentence Transformers and FAISS.

        Args:
            query: The user's question or search phrase.
            k: Number of top results to return.

        Returns:
            List of dicts: [
                {
                    "text": str,
                    "page": int,
                    "score": float (cosine similarity),
                    "retrieval_method": "semantic"
                },
                ...
            ]
        """
        # Edge case handling
        if not self._initialized or self.index is None or not self.chunks:
            logger.warning("Semantic search attempted on empty or unbuilt index.")
            return []

        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty or blank query provided to semantic_search.")
            return []

        if k <= 0:
            return []

        self._ensure_dependencies()

        # Clamp k to available chunks
        actual_k = min(k, len(self.chunks))

        # Generate and normalize query embedding
        query_embedding = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Inner-product search directly calculates cosine similarity
        scores, indices = self.index.search(query_embedding, actual_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS padding indicator if fewer results than k
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "score": round(float(score), 4),
                "retrieval_method": "semantic"
            })

        return results

    def tfidf_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs baseline keyword retrieval using TF-IDF and Cosine Similarity.

        Args:
            query: The user's question or search phrase.
            k: Number of top results to return.

        Returns:
            List of dicts matching the semantic_search output format: [
                {
                    "text": str,
                    "page": int,
                    "score": float (cosine similarity),
                    "retrieval_method": "tfidf"
                },
                ...
            ]
        """
        # Edge case handling
        if not self._initialized or self.tfidf_vectorizer is None or not self.chunks:
            logger.warning("TF-IDF search attempted on empty or unbuilt index.")
            return []

        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty or blank query provided to tfidf_search.")
            return []

        if k <= 0:
            return []

        self._ensure_dependencies()

        # Clamp k to available chunks
        actual_k = min(k, len(self.chunks))

        # Transform query into TF-IDF vector space
        query_vec = self.tfidf_vectorizer.transform([query.strip()])

        # Compute cosine similarities between query vector and chunk matrix
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Sort descending by score
        top_indices = np.argsort(similarities)[::-1][:actual_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "score": round(float(similarities[idx]), 4),
                "retrieval_method": "tfidf"
            })

        return results


# ============================================================================
# Module-level convenience functions for easy integration by P1 and P3
# ============================================================================
_global_retriever = DocumentRetriever()

def build_index(chunks: List[Dict[str, Any]]) -> int:
    """Convenience function: builds index on global retriever instance."""
    return _global_retriever.build_index(chunks)

def semantic_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function: queries global retriever via semantic search."""
    return _global_retriever.semantic_search(query, k=k)

def tfidf_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function: queries global retriever via TF-IDF search."""
    return _global_retriever.tfidf_search(query, k=k)
