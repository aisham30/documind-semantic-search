"""Automated unit and integration tests for the six NLP dataset collections.

Tests all six collections:
1. Text Preprocessing and Tokenization
2. Sentiment Analysis and Text Classification
3. Word Embeddings and Language Models
4. Named Entity Recognition and Information Extraction
5. RAG, Semantic Search and Question Answering
6. Machine Translation and Text Summarization

For each collection, validates:
- Relevant questions retrieve correct documents and produce grounded, cited answers.
- Paraphrased questions retrieve by semantic meaning and produce cited answers.
- Unrelated questions return "Insufficient Evidence" / "Unsupported Question".
- Missing-information questions return "Insufficient Evidence".
- Dataset selector restricts retrieval strictly to the chosen collection.
- Document loader and indexing verification (18+ documents across all 6 collections).
"""

import unittest
from unittest.mock import MagicMock

from app.retrieval.document_loader import DocumentLoader, get_document_loader
from app.retrieval.semantic_retriever import SemanticRetriever, get_semantic_retriever
from app.generation.retrieval_adapter import retrieve
from app.generation.generator import generate_answer
from app.generation.mock_client import create_mock_client
from app.generation.prompts import INSUFFICIENT_EVIDENCE_PHRASE


class TestNLPCollectionsAndRetrieval(unittest.TestCase):
    """Test suite for the 6 NLP educational collections and semantic retrieval."""

    @classmethod
    def setUpClass(cls):
        cls.loader = get_document_loader()
        cls.retriever = get_semantic_retriever()
        cls.collection_names = [
            "Text Preprocessing and Tokenization",
            "Sentiment Analysis and Text Classification",
            "Word Embeddings and Language Models",
            "Named Entity Recognition and Information Extraction",
            "RAG, Semantic Search and Question Answering",
            "Machine Translation and Text Summarization",
        ]

    def test_document_loader_loads_all_six_collections(self):
        """Verify that all six collections and at least 3 documents per collection are loaded."""
        loaded_names = self.loader.get_collection_names()
        for name in self.collection_names:
            self.assertIn(name, loaded_names, f"Collection '{name}' was not loaded.")
            col = self.loader.get_collection(name)
            self.assertIsNotNone(col)
            self.assertGreaterEqual(
                len(col.documents), 3,
                f"Collection '{name}' has fewer than 3 documents ({len(col.documents)} found)."
            )
            # Verify every document has title, id, category, source, and pages
            for doc in col.documents:
                self.assertTrue(doc.id, "Document missing ID")
                self.assertTrue(doc.title, "Document missing title")
                self.assertEqual(doc.category, name, "Document category mismatch")
                self.assertTrue(doc.source, "Document missing source metadata")
                self.assertGreater(len(doc.chunks), 0, f"Document '{doc.id}' has no pages/chunks")

    def test_indexing_contains_all_documents(self):
        """Verify that the semantic retriever indexed all 18+ documents across the 6 collections."""
        stats = self.loader.get_stats()
        self.assertEqual(stats["total_collections"], 6)
        self.assertGreaterEqual(stats["total_documents"], 18)
        self.assertGreaterEqual(stats["total_passages"], 54)
        self.assertEqual(len(self.retriever.chunks), stats["total_passages"])

    # -------------------------------------------------------------------------
    # Collection 1: Text Preprocessing and Tokenization
    # -------------------------------------------------------------------------
    def test_collection_1_text_preprocessing(self):
        col_name = "Text Preprocessing and Tokenization"

        # 1. Relevant question: Porter stemmer vs lemmatization
        rel_q = "What is the difference between Porter stemming and WordNet lemmatization?"
        passages = retrieve(rel_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages), 0)
        self.assertEqual(passages[0]["category"], col_name)
        client = create_mock_client(rel_q, passages)
        ans = generate_answer(rel_q, passages, client=client)
        self.assertFalse(ans["insufficient_evidence"])
        self.assertGreater(len(ans["sources"]), 0)
        self.assertEqual(ans["sources"][0]["document_id"], "doc_prep_001")

        # 2. Paraphrased question: breaking words into subwords to avoid out-of-vocabulary
        para_q = "Why do modern tokenizers break words into pieces to handle out of vocabulary tokens?"
        passages_para = retrieve(para_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages_para), 0)
        self.assertEqual(passages_para[0]["category"], col_name)
        client_para = create_mock_client(para_q, passages_para)
        ans_para = generate_answer(para_q, passages_para, client=client_para)
        self.assertFalse(ans_para["insufficient_evidence"])
        self.assertGreater(len(ans_para["sources"]), 0)

        # 3. Unrelated question: Astronomy / Geography
        unrelated_q = "What is the diameter of planet Jupiter?"
        passages_unrelated = retrieve(unrelated_q, collection=col_name, top_k=2)
        client_unrelated = create_mock_client(unrelated_q, passages_unrelated)
        ans_unrelated = generate_answer(unrelated_q, passages_unrelated, client=client_unrelated)
        self.assertTrue(ans_unrelated["insufficient_evidence"])
        self.assertEqual(ans_unrelated["sources"], [])

        # 4. Missing-information question within the topic
        missing_q = "What was the exact financial cost of training the WordPiece tokenizer in dollars?"
        client_missing = create_mock_client(missing_q, passages)
        ans_missing = generate_answer(missing_q, passages, client=client_missing)
        self.assertTrue(ans_missing["insufficient_evidence"])

    # -------------------------------------------------------------------------
    # Collection 2: Sentiment Analysis and Text Classification
    # -------------------------------------------------------------------------
    def test_collection_2_sentiment_analysis(self):
        col_name = "Sentiment Analysis and Text Classification"

        # 1. Relevant question: VADER rules and shifters
        rel_q = "How does VADER handle valence shifters and punctuation for sentiment?"
        passages = retrieve(rel_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages), 0)
        self.assertEqual(passages[0]["category"], col_name)
        client = create_mock_client(rel_q, passages)
        ans = generate_answer(rel_q, passages, client=client)
        self.assertFalse(ans["insufficient_evidence"])
        self.assertGreater(len(ans["sources"]), 0)
        self.assertEqual(ans["sources"][0]["document_id"], "doc_sa_001")

        # 2. Paraphrased question: fine-grained opinion mining for specific product attributes
        para_q = "How does aspect based opinion mining evaluate specific attributes of a product?"
        passages_para = retrieve(para_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages_para), 0)
        self.assertEqual(passages_para[0]["category"], col_name)
        client_para = create_mock_client(para_q, passages_para)
        ans_para = generate_answer(para_q, passages_para, client=client_para)
        self.assertFalse(ans_para["insufficient_evidence"])
        self.assertGreater(len(ans_para["sources"]), 0)

        # 3. Unrelated question: Cooking recipe
        unrelated_q = "How many grams of butter are required for chocolate chip cookies?"
        passages_unrelated = retrieve(unrelated_q, collection=col_name, top_k=2)
        client_unrelated = create_mock_client(unrelated_q, passages_unrelated)
        ans_unrelated = generate_answer(unrelated_q, passages_unrelated, client=client_unrelated)
        self.assertTrue(ans_unrelated["insufficient_evidence"])

        # 4. Missing-information question
        missing_q = "What is the phone number of the developer of VADER?"
        client_missing = create_mock_client(missing_q, passages)
        ans_missing = generate_answer(missing_q, passages, client=client_missing)
        self.assertTrue(ans_missing["insufficient_evidence"])

    # -------------------------------------------------------------------------
    # Collection 3: Word Embeddings and Language Models
    # -------------------------------------------------------------------------
    def test_collection_3_word_embeddings_and_lms(self):
        col_name = "Word Embeddings and Language Models"

        # 1. Relevant question: Word2Vec Skip-gram vs CBOW
        rel_q = "What is the difference between CBOW and Skip-gram in Word2Vec?"
        passages = retrieve(rel_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages), 0)
        self.assertEqual(passages[0]["category"], col_name)
        client = create_mock_client(rel_q, passages)
        ans = generate_answer(rel_q, passages, client=client)
        self.assertFalse(ans["insufficient_evidence"])
        self.assertGreater(len(ans["sources"]), 0)
        self.assertEqual(ans["sources"][0]["document_id"], "doc_emb_001")

        # 2. Paraphrased question: resolving multiple word senses / polysemy with attention
        para_q = "How does contextual self attention resolve multiple meanings and polysemy in sentences?"
        passages_para = retrieve(para_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages_para), 0)
        self.assertEqual(passages_para[0]["category"], col_name)
        client_para = create_mock_client(para_q, passages_para)
        ans_para = generate_answer(para_q, passages_para, client=client_para)
        self.assertFalse(ans_para["insufficient_evidence"])
        self.assertGreater(len(ans_para["sources"]), 0)

        # 3. Unrelated question
        unrelated_q = "Who won the football World Cup in 1994?"
        passages_unrelated = retrieve(unrelated_q, collection=col_name, top_k=2)
        client_unrelated = create_mock_client(unrelated_q, passages_unrelated)
        ans_unrelated = generate_answer(unrelated_q, passages_unrelated, client=client_unrelated)
        self.assertTrue(ans_unrelated["insufficient_evidence"])

        # 4. Missing-information question
        missing_q = "What brand of coffee did the authors of GloVe drink during development?"
        client_missing = create_mock_client(missing_q, passages)
        ans_missing = generate_answer(missing_q, passages, client=client_missing)
        self.assertTrue(ans_missing["insufficient_evidence"])

    # -------------------------------------------------------------------------
    # Collection 4: Named Entity Recognition and Information Extraction
    # -------------------------------------------------------------------------
    def test_collection_4_ner_and_ie(self):
        col_name = "Named Entity Recognition and Information Extraction"

        # 1. Relevant question: BIO tagging scheme
        rel_q = "Explain the BIO tagging scheme and entity sequence labeling in NER."
        passages = retrieve(rel_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages), 0)
        self.assertEqual(passages[0]["category"], col_name)
        client = create_mock_client(rel_q, passages)
        ans = generate_answer(rel_q, passages, client=client)
        self.assertFalse(ans["insufficient_evidence"])
        self.assertGreater(len(ans["sources"]), 0)
        self.assertEqual(ans["sources"][0]["document_id"], "doc_ner_001")

        # 2. Paraphrased question: relation extraction and knowledge graph triples
        para_q = "How are subject predicate object triples extracted to build a knowledge graph?"
        passages_para = retrieve(para_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages_para), 0)
        self.assertEqual(passages_para[0]["category"], col_name)
        client_para = create_mock_client(para_q, passages_para)
        ans_para = generate_answer(para_q, passages_para, client=client_para)
        self.assertFalse(ans_para["insufficient_evidence"])
        self.assertGreater(len(ans_para["sources"]), 0)

        # 3. Unrelated question
        unrelated_q = "What is the boiling point of ethanol at sea level?"
        passages_unrelated = retrieve(unrelated_q, collection=col_name, top_k=2)
        client_unrelated = create_mock_client(unrelated_q, passages_unrelated)
        ans_unrelated = generate_answer(unrelated_q, passages_unrelated, client=client_unrelated)
        self.assertTrue(ans_unrelated["insufficient_evidence"])

        # 4. Missing-information question
        missing_q = "What was the date of birth of the CoNLL evaluation script author's grandmother?"
        client_missing = create_mock_client(missing_q, passages)
        ans_missing = generate_answer(missing_q, passages, client=client_missing)
        self.assertTrue(ans_missing["insufficient_evidence"])

    # -------------------------------------------------------------------------
    # Collection 5: RAG, Semantic Search and Question Answering
    # -------------------------------------------------------------------------
    def test_collection_5_rag_and_semantic_search(self):
        col_name = "RAG, Semantic Search and Question Answering"

        # 1. Relevant question: Bi-Encoder vs Cross-Encoder
        rel_q = "Compare Bi-Encoder and Cross-Encoder architectures in dense retrieval."
        passages = retrieve(rel_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages), 0)
        self.assertEqual(passages[0]["category"], col_name)
        client = create_mock_client(rel_q, passages)
        ans = generate_answer(rel_q, passages, client=client)
        self.assertFalse(ans["insufficient_evidence"])
        self.assertGreater(len(ans["sources"]), 0)
        self.assertEqual(ans["sources"][0]["document_id"], "doc_rag_002")

        # 2. Paraphrased question: mitigating hallucinations with citations and external knowledge
        para_q = "How does injecting external knowledge passages mitigate hallucinations and provide citations?"
        passages_para = retrieve(para_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages_para), 0)
        self.assertEqual(passages_para[0]["category"], col_name)
        client_para = create_mock_client(para_q, passages_para)
        ans_para = generate_answer(para_q, passages_para, client=client_para)
        self.assertFalse(ans_para["insufficient_evidence"])
        self.assertGreater(len(ans_para["sources"]), 0)

        # 3. Unrelated question
        unrelated_q = "What is the airspeed velocity of an unladen swallow?"
        passages_unrelated = retrieve(unrelated_q, collection=col_name, top_k=2)
        client_unrelated = create_mock_client(unrelated_q, passages_unrelated)
        ans_unrelated = generate_answer(unrelated_q, passages_unrelated, client=client_unrelated)
        self.assertTrue(ans_unrelated["insufficient_evidence"])

        # 4. Missing-information question
        missing_q = "What is the serial number of the hard drive holding the FAISS vector index?"
        client_missing = create_mock_client(missing_q, passages)
        ans_missing = generate_answer(missing_q, passages, client=client_missing)
        self.assertTrue(ans_missing["insufficient_evidence"])

    # -------------------------------------------------------------------------
    # Collection 6: Machine Translation and Text Summarization
    # -------------------------------------------------------------------------
    def test_collection_6_translation_and_summarization(self):
        col_name = "Machine Translation and Text Summarization"

        # 1. Relevant question: Extractive vs Abstractive Summarization
        rel_q = "What is the difference between extractive and abstractive summarization?"
        passages = retrieve(rel_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages), 0)
        self.assertEqual(passages[0]["category"], col_name)
        client = create_mock_client(rel_q, passages)
        ans = generate_answer(rel_q, passages, client=client)
        self.assertFalse(ans["insufficient_evidence"])
        self.assertGreater(len(ans["sources"]), 0)
        self.assertEqual(ans["sources"][0]["document_id"], "doc_trans_002")

        # 2. Paraphrased question: beam search and length penalty in translation decoding
        para_q = "How does beam search explore multiple translation hypotheses with length penalty?"
        passages_para = retrieve(para_q, collection=col_name, top_k=2)
        self.assertGreater(len(passages_para), 0)
        self.assertEqual(passages_para[0]["category"], col_name)
        client_para = create_mock_client(para_q, passages_para)
        ans_para = generate_answer(para_q, passages_para, client=client_para)
        self.assertFalse(ans_para["insufficient_evidence"])
        self.assertGreater(len(ans_para["sources"]), 0)

        # 3. Unrelated question
        unrelated_q = "What is the capital city of Madagascar?"
        passages_unrelated = retrieve(unrelated_q, collection=col_name, top_k=2)
        client_unrelated = create_mock_client(unrelated_q, passages_unrelated)
        ans_unrelated = generate_answer(unrelated_q, passages_unrelated, client=client_unrelated)
        self.assertTrue(ans_unrelated["insufficient_evidence"])

        # 4. Missing-information question
        missing_q = "What was the exact battery percentage of the laptop when BLEU was formulated?"
        client_missing = create_mock_client(missing_q, passages)
        ans_missing = generate_answer(missing_q, passages, client=client_missing)
        self.assertTrue(ans_missing["insufficient_evidence"])

    # -------------------------------------------------------------------------
    # Dataset Selector Restriction Test
    # -------------------------------------------------------------------------
    def test_dataset_selector_restricts_retrieval(self):
        """Verify that selecting a specific dataset strictly restricts retrieval to its own documents."""
        # Query about BLEU metric (from Machine Translation)
        query = "How is BLEU score calculated for machine translation evaluation?"

        # 1. Search in its correct collection:
        correct_passages = retrieve(query, collection="Machine Translation and Text Summarization", top_k=3)
        self.assertGreater(len(correct_passages), 0)
        for p in correct_passages:
            self.assertEqual(p["category"], "Machine Translation and Text Summarization")

        # 2. Restrict to a DIFFERENT collection ("Text Preprocessing and Tokenization"):
        # The retriever MUST NOT return any chunks from Machine Translation!
        restricted_passages = retrieve(query, collection="Text Preprocessing and Tokenization", top_k=3)
        for p in restricted_passages:
            self.assertEqual(p["category"], "Text Preprocessing and Tokenization")
            # Must not be trans_001
            self.assertNotEqual(p["document_id"], "doc_trans_001")


if __name__ == "__main__":
    unittest.main()
