"""
Script to generate sample research and technical PDFs for DocuMind testing.
"""

from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz

SAMPLE_DIR = Path(__file__).resolve().parent

def create_sample_paper():
    pdf_path = SAMPLE_DIR / "research_paper.pdf"
    doc = fitz.open()

    rect = fitz.Rect(50, 50, 550, 750)

    # Page 1: Abstract & Introduction
    page1 = doc.new_page()
    text_p1 = (
        "DocuMind: Accurate Document Question Answering using Semantic Retrieval and RAG\n\n"
        "Abstract:\n"
        "Retrieval-Augmented Generation (RAG) addresses large language model hallucination by grounding "
        "model responses in verified external document collections. In this paper, we propose DocuMind, "
        "a modular architecture combining robust PDF text preprocessing, dense semantic vector retrieval "
        "via FAISS, and citation-grounded generation.\n\n"
        "1. Introduction\n"
        "Natural Language Processing has witnessed significant breakthroughs with Transformer architectures. "
        "However, applying standard generative language models directly to domain-specific documents presents "
        "two critical challenges: factual hallucination and the lack of exact citation references. "
        "Standard search engines rely on lexical keyword matching such as BM25 or TF-IDF, which fail when users "
        "formulate queries using synonyms or conceptual questions not explicitly containing query keywords. "
        "Dense retrieval maps queries and passages into a shared vector embedding space where semantic similarity "
        "is measured by cosine similarity or inner product.\n\n"
        "DocuMind resolves these limitations by executing a four-stage sequential pipeline: "
        "First, documents are parsed page-by-page, normalizing Unicode representations and tracking 1-based page "
        "metadata. Second, text is partitioned into overlapping windows preserving sentence boundaries. "
        "Third, dense vector embeddings are indexed using FAISS IndexFlatIP. Fourth, relevant passages are passed "
        "with strict attribution constraints to the generation model."
    )
    page1.insert_textbox(rect, text_p1, fontsize=11)

    # Page 2: PDF Preprocessing & Chunking Strategy
    page2 = doc.new_page()
    text_p2 = (
        "2. Document Preprocessing and Ingestion Pipeline\n\n"
        "The first stage in RAG systems is document ingestion. Unstructured PDF files present unique challenges "
        "including non-standard font encodings, hyphenated line wraps, non-breaking spaces, and varying page layouts.\n\n"
        "2.1 Text Extraction and Normalization\n"
        "We employ PyMuPDF (fitz) to extract text content in reading order. Raw extracted text undergoes canonical "
        "Unicode NFC normalization, non-breaking space replacement, and removal of unprintable control characters. "
        "Unlike lossy cleaners that strip mathematical symbols and punctuation, our cleaner conservatively preserves "
        "punctuation marks, parentheses, and technical symbols essential for semantic understanding.\n\n"
        "2.2 Page-Constrained Chunking\n"
        "Chunking strategies greatly influence downstream retrieval accuracy. Overly large chunks dilute specific semantic "
        "signals, while overly small chunks fragment conceptual coherence. DocuMind implements a paragraph- and "
        "sentence-aware sliding window with a target size of 1,200 characters and an overlap of 180 characters. "
        "Critically, chunks never cross page boundaries. This design decision ensures that every citation presented "
        "to end users points unambiguously to a single verified page number in the source document."
    )
    page2.insert_textbox(rect, text_p2, fontsize=11)

    # Page 3: Semantic Retrieval & Embeddings
    page3 = doc.new_page()
    text_p3 = (
        "3. Semantic Vector Search and FAISS Indexing\n\n"
        "3.1 Embedding Model\n"
        "Extracted chunks are transformed into 384-dimensional dense vectors using the SentenceTransformer "
        "'all-MiniLM-L6-v2' model. MiniLM offers an optimal trade-off between inference speed and representation quality, "
        "making it suitable for real-time document search on consumer CPUs.\n\n"
        "3.2 Cosine Similarity through L2 Normalization and Inner Product\n"
        "Cosine similarity measures the cosine of the angle between two multi-dimensional vectors. "
        "By performing L2 normalization on both chunk embeddings and query embeddings prior to indexing, the denominator "
        "in the cosine similarity formula simplifies to 1. Consequently, computing cosine similarity reduces to an inner "
        "product (dot product). In FAISS, this is implemented using IndexFlatIP, leveraging AVX-256 SIMD parallel instructions "
        "for ultra-fast nearest-neighbor retrieval without needing clustering approximation.\n\n"
        "3.3 Baseline Comparison\n"
        "For empirical evaluation, DocuMind maintains a classical TF-IDF baseline with scikit-learn. "
        "Experimental runs demonstrate that while TF-IDF excels on exact technical term lookup, dense semantic search "
        "achieves significantly higher recall on conceptual queries and natural language paraphrases."
    )
    page3.insert_textbox(rect, text_p3, fontsize=11)

    doc.save(pdf_path)
    doc.close()
    print(f"[CREATED] Sample PDF created at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_sample_paper()
