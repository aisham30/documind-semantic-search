"""Semantic and hybrid retrieval engine for NLP educational collections.

Features:
1. TF-IDF + Subword N-gram Indexing + NLP Conceptual Semantic Expansion.
2. Cosine similarity scoring supporting paraphrased queries by meaning.
3. Strict collection-level dataset filtering.
4. Relevance confidence thresholding: returns empty chunks when queries are unrelated
   or lack evidence, triggering the 'Insufficient Evidence' / 'Unsupported Question' guardrail.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from app.retrieval.document_loader import DocumentLoader, get_document_loader

# Common English stop words
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
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "whatever", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves",
}

# Domain semantic concept synonym thesaurus for paraphrasing in NLP
NLP_CONCEPT_EXPANSIONS = {
    # Text preprocessing
    "break words": ["tokenization", "subword", "segmentation", "bpe", "wordpiece"],
    "break words into pieces": ["subword", "tokenization", "bpe", "wordpiece", "sentencepiece"],
    "breaking words": ["subword", "tokenization", "bpe", "wordpiece"],
    "out of vocabulary": ["oov", "unknown words", "unseen", "subword", "byte-pair"],
    "oov": ["out-of-vocabulary", "subword", "bpe", "wordpiece", "sentencepiece"],
    "base form": ["lemma", "lemmatization", "stem", "stemming", "canonical"],
    "root form": ["lemma", "lemmatization", "stem", "stemming", "root"],
    "dictionary form": ["lemma", "lemmatization", "wordnet"],
    "porter": ["stemmer", "stemming", "snowball", "suffix"],
    "zero probability": ["smoothing", "laplace", "kneser-ney", "unseen"],
    "markov": ["n-gram", "probability", "unigram", "bigram", "trigram"],
    "full stop": ["sentence boundary", "segmentation", "abbreviation", "period"],

    # Sentiment analysis
    "polarity": ["sentiment", "valence", "positive", "negative", "vader"],
    "opinion mining": ["sentiment analysis", "subjectivity", "aspect-based"],
    "valence": ["sentiment", "polarity", "vader", "shifter"],
    "vader": ["lexicon", "valence", "rule-based", "shifters"],
    "aspect": ["absa", "attribute", "fine-grained", "dependency"],
    "aspect-based": ["absa", "ate", "asc", "aspect", "dependency"],
    "multilabel": ["multi-label", "sigmoid", "binary cross-entropy", "overlapping"],

    # Embeddings & Language Models
    "word vectors": ["embeddings", "word2vec", "glove", "fasttext", "dense"],
    "continuous vector": ["embedding", "dense", "word2vec", "glove"],
    "skip-gram": ["word2vec", "cbow", "negative sampling"],
    "polysemy": ["multiple senses", "ambiguity", "static embeddings", "contextual"],
    "multiple meanings": ["polysemy", "contextual", "bert", "senses"],
    "self-attention": ["transformer", "multi-head", "query key value", "scaled dot-product"],
    "masked language model": ["mlm", "bert", "roberta", "bidirectional"],
    "causal language model": ["clm", "gpt", "autoregressive", "next-token"],
    "scaling laws": ["chinchilla", "kaplan", "compute-optimal", "tokens"],
    "compute optimal": ["chinchilla", "scaling laws", "parameters"],
    "rlhf": ["alignment", "ppo", "reward model", "preference", "dpo"],
    "preference optimization": ["dpo", "rlhf", "alignment", "bradley-terry"],

    # NER & IE
    "entity recognition": ["ner", "named entity", "sequence labeling", "bio"],
    "sequence labeling": ["bio", "crf", "viterbi", "hmm", "ner"],
    "bio scheme": ["begin inside outside", "tagging", "ner", "span"],
    "nested entity": ["overlapping", "hierarchical", "span-based", "biaffine"],
    "triples": ["relation extraction", "knowledge graph", "subject predicate object"],
    "relation extraction": ["triples", "entity linking", "hearst", "tacred"],
    "knowledge graph": ["kg", "triples", "entities", "link prediction"],

    # RAG & Semantic Search
    "hallucination": ["rag", "grounding", "citation", "faithfulness", "verification"],
    "mitigate hallucination": ["rag", "retrieval-augmented", "grounding", "citation"],
    "chunking": ["sliding window", "recursive", "semantic chunking", "parent-child"],
    "vector search": ["dense retrieval", "embedding", "bi-encoder", "ann", "faiss"],
    "bi-encoder": ["dual-encoder", "dense passage", "cosine similarity", "cross-encoder"],
    "approximate nearest neighbor": ["ann", "hnsw", "ivf-pq", "faiss"],
    "hnsw": ["graph index", "ann", "nearest neighbor", "skip connections"],
    "hybrid search": ["bm25", "dense vector", "reciprocal rank fusion", "rrf"],
    "lexical gap": ["bm25", "hybrid search", "sparse", "exact match"],
    "ragas": ["faithfulness", "answer relevance", "context precision", "context recall"],

    # Machine Translation & Summarization
    "machine translation": ["nmt", "seq2seq", "transformer", "bahdanau", "bleu"],
    "seq2seq": ["encoder-decoder", "bottleneck", "attention", "lstm"],
    "information bottleneck": ["seq2seq", "fixed-length vector", "bahdanau", "attention"],
    "beam search": ["decoding", "hypotheses", "length penalty", "greedy"],
    "bleu": ["sacrebleu", "n-gram precision", "brevity penalty", "translation metric"],
    "extractive": ["verbatim sentences", "textrank", "lexrank", "bertsum"],
    "abstractive": ["novel sentences", "paraphrase", "bart", "t5", "pegasus"],
    "rouge": ["rouge-1", "rouge-2", "rouge-l", "summarization metric"],
    "lost in the middle": ["long-document", "sparse attention", "longformer", "bigbird"],
    "map-reduce": ["chunk-and-combine", "summarization", "long-document"],
}


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase words and alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())


def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Extract character or word n-grams."""
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def extract_char_ngrams(token: str, n: int = 3) -> List[str]:
    """Extract subword character n-grams to handle morphological variations."""
    if len(token) < n:
        return [token]
    return [token[i:i+n] for i in range(len(token) - n + 1)]


class SemanticRetriever:
    """
    In-memory semantic and hybrid retrieval engine with conceptual expansion.
    Indexes all educational chunks and allows fast, filtered cosine similarity search.
    """

    def __init__(self, loader: Optional[DocumentLoader] = None):
        self.loader = loader or get_document_loader()
        self.chunks = self.loader.get_chunks()
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[np.ndarray] = []
        self._build_index()

    def _expand_query_concepts(self, query: str) -> List[str]:
        """Expand natural language query with NLP domain synonyms and concepts."""
        q_low = query.lower()
        expanded_terms = []
        for phrase, synonyms in NLP_CONCEPT_EXPANSIONS.items():
            if phrase in q_low:
                expanded_terms.extend(synonyms)
        return expanded_terms

    def _build_index(self) -> None:
        """Construct TF-IDF + Subword + Conceptual Vector Index across all chunks."""
        doc_term_freqs: List[Dict[str, float]] = []
        doc_freqs: Dict[str, int] = {}
        n_docs = max(1, len(self.chunks))

        for chunk in self.chunks:
            # Combine title, section, source, and text for rich indexing
            full_text = f"{chunk.get('title', '')} {chunk.get('section_title', '')} {chunk.get('text', '')}"
            tokens = tokenize_text(full_text)
            term_counts: Dict[str, float] = {}

            # 1. Word unigrams (excluding stop words)
            for token in tokens:
                if token not in STOP_WORDS:
                    term_counts[token] = term_counts.get(token, 0.0) + 1.0
                    # Subword 3-character n-grams for morphological flexibility
                    for c_gram in extract_char_ngrams(token, 3):
                        c_key = f"char_{c_gram}"
                        term_counts[c_key] = term_counts.get(c_key, 0.0) + 0.3

            # 2. Word bigrams for syntactic pairs
            for bigram in extract_ngrams(tokens, 2):
                b_tokens = bigram.split()
                if not all(t in STOP_WORDS for t in b_tokens):
                    b_key = f"bi_{bigram}"
                    term_counts[b_key] = term_counts.get(b_key, 0.0) + 1.5

            doc_term_freqs.append(term_counts)
            for term in term_counts:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1

        # Build vocabulary & compute Smooth IDF
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(doc_freqs.keys()))}
        self.idf = {
            term: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            for term, df in doc_freqs.items()
        }

        # Build normalized unit L2 document vectors
        vocab_size = len(self.vocabulary)
        self.doc_vectors = []

        for tf_dict in doc_term_freqs:
            vec = np.zeros(vocab_size, dtype=np.float32)
            for term, count in tf_dict.items():
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    # Log-scaled TF * IDF
                    vec[idx] = (1.0 + math.log(count)) * self.idf[term]

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            self.doc_vectors.append(vec)

    def _vectorize_query(self, query: str) -> np.ndarray:
        """Vectorize a user query including semantic concept expansion."""
        vocab_size = len(self.vocabulary)
        vec = np.zeros(vocab_size, dtype=np.float32)
        tokens = tokenize_text(query)

        term_counts: Dict[str, float] = {}
        for token in tokens:
            if token not in STOP_WORDS:
                term_counts[token] = term_counts.get(token, 0.0) + 1.5
                for c_gram in extract_char_ngrams(token, 3):
                    c_key = f"char_{c_gram}"
                    term_counts[c_key] = term_counts.get(c_key, 0.0) + 0.4

        for bigram in extract_ngrams(tokens, 2):
            b_tokens = bigram.split()
            if not all(t in STOP_WORDS for t in b_tokens):
                b_key = f"bi_{bigram}"
                term_counts[b_key] = term_counts.get(b_key, 0.0) + 2.0

        # Inject conceptual synonyms
        expanded_concepts = self._expand_query_concepts(query)
        for syn in expanded_concepts:
            for s_token in tokenize_text(syn):
                if s_token not in STOP_WORDS:
                    term_counts[s_token] = term_counts.get(s_token, 0.0) + 1.2
                    for c_gram in extract_char_ngrams(s_token, 3):
                        c_key = f"char_{c_gram}"
                        term_counts[c_key] = term_counts.get(c_key, 0.0) + 0.3

        for term, count in term_counts.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                vec[idx] = (1.0 + math.log(count)) * self.idf.get(term, 1.0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: int = 3,
        min_relevance_threshold: float = 0.08,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant passages for a query.

        Parameters:
            query: The user's natural language question.
            collection_name: If specified, strictly restricts retrieval to passages
                             belonging to this collection.
            top_k: Maximum number of passages to return.
            min_relevance_threshold: Confidence threshold. If the highest score
                                     is below this value (indicating unrelated query
                                     or missing evidence), returns empty list [].

        Returns:
            List of chunk dicts formatted with document_id, chunk_id, page_number, text, score.
        """
        if not query or not query.strip():
            # If no query provided, return default first passages of collection
            if collection_name and collection_name.strip() not in ("", "All Collections", "All Datasets"):
                col_chunks = [c for c in self.chunks if c.get("category", "").lower() == collection_name.strip().lower()]
                return col_chunks[:top_k]
            return self.chunks[:top_k]

        q_vec = self._vectorize_query(query)
        if np.linalg.norm(q_vec) == 0:
            return []

        # Determine target collection filter
        is_filtered = bool(
            collection_name and collection_name.strip() not in ("", "All Collections", "All Datasets")
        )
        target_col = collection_name.strip().lower() if is_filtered else ""

        scored_chunks: List[Tuple[float, Dict[str, Any]]] = []

        for idx, doc_vec in enumerate(self.doc_vectors):
            chunk = self.chunks[idx]

            # Strict collection filtering
            if is_filtered:
                chunk_cat = chunk.get("category", "").strip().lower()
                if chunk_cat != target_col:
                    continue

            similarity = float(np.dot(q_vec, doc_vec))
            if similarity > 0.0:
                scored_chunks.append((similarity, chunk))

        if not scored_chunks:
            return []

        # Sort descending by similarity
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_score = scored_chunks[0][0]

        # Guardrail check: if even the top score is below threshold, return empty
        # to trigger "Insufficient Evidence" / "Unsupported Question"
        if top_score < min_relevance_threshold:
            return []

        results = []
        for sim, chunk in scored_chunks[:top_k]:
            chunk_copy = dict(chunk)
            chunk_copy["retrieval_score"] = round(sim, 4)
            results.append(chunk_copy)

        return results


# Global singleton instance for retriever
_global_retriever: Optional[SemanticRetriever] = None


def get_semantic_retriever() -> SemanticRetriever:
    """Get or initialize singleton SemanticRetriever instance."""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = SemanticRetriever()
    return _global_retriever
