"""Web interface for DocuMind AI's RAG Answer Generation & Verification module using Streamlit.

Features:
1. Six NLP Educational Dataset Collections with interactive dataset selector.
2. Semantic & hybrid passage retrieval supporting paraphrasing and meaning-based search.
3. Dataset-restricted retrieval ensuring query isolation to the selected collection.
4. Grounded answer generation with strict citation validation and passage inspection.
5. Strict refusal guardrail returning "Unsupported Question" / "Insufficient Evidence" on missing facts.
6. Claim-level verification with sentence-level evidence extraction and tags (supported / weak / unsupported).
7. 0-100 Trust Score with transparent breakdown metrics.
8. Contradiction warning banner for conflicting sources.
9. "With RAG vs Without RAG" side-by-side comparison panel.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load any .env variables
load_dotenv()

from app.generation.generator import generate_answer, generate_without_rag
from app.generation.mock_client import create_mock_client
from app.generation.claim_verifier import verify_all_claims, compute_trust_score
from app.generation.citation_validator import normalize_str
from app.generation.retrieval_adapter import (
    retrieve,
    get_available_collections,
    get_collection_stats,
)
from app.retrieval.document_loader import get_document_loader

# Page layout & title
st.set_page_config(
    page_title="DocuMind AI - Answer Generation & Verification",
    page_icon="🧠",
    layout="wide",
)

# Preset questions tailored to each educational collection
COLLECTION_PRESETS = {
    "All Collections": [
        ("📄 Stem vs Lemma (Prep)", "What is the difference between Porter stemming and WordNet lemmatization?"),
        ("💬 VADER Valence (SA)", "How does VADER handle valence shifters and punctuation for sentiment?"),
        ("🧠 Self-Attention (LM)", "How does contextual self attention resolve multiple meanings and polysemy in sentences?"),
        ("❓ Unsupported Question", "What is the capital of France?"),
    ],
    "Text Preprocessing and Tokenization": [
        ("📄 Stem vs Lemma", "What is the difference between Porter stemming and WordNet lemmatization?"),
        ("🔤 Subwords (Paraphrased)", "Why do modern tokenizers break words into pieces to handle out of vocabulary tokens?"),
        ("❓ Missing Info", "What was the exact financial cost of training the WordPiece tokenizer in dollars?"),
        ("🚫 Unrelated", "What is the diameter of planet Jupiter?"),
    ],
    "Sentiment Analysis and Text Classification": [
        ("💬 VADER Valence", "How does VADER handle valence shifters and punctuation for sentiment?"),
        ("🎯 Aspect (Paraphrased)", "How does aspect based opinion mining evaluate specific attributes of a product?"),
        ("❓ Missing Info", "What is the phone number of the developer of VADER?"),
        ("🚫 Unrelated", "How many grams of butter are required for chocolate chip cookies?"),
    ],
    "Word Embeddings and Language Models": [
        ("🧠 Skip-Gram vs CBOW", "What is the difference between CBOW and Skip-gram in Word2Vec?"),
        ("🔄 Polysemy (Paraphrased)", "How does contextual self attention resolve multiple meanings and polysemy in sentences?"),
        ("❓ Missing Info", "What brand of coffee did the authors of GloVe drink during development?"),
        ("🚫 Unrelated", "Who won the football World Cup in 1994?"),
    ],
    "Named Entity Recognition and Information Extraction": [
        ("🏷️ BIO Scheme", "Explain the BIO tagging scheme and entity sequence labeling in NER."),
        ("🕸️ Triples (Paraphrased)", "How are subject predicate object triples extracted to build a knowledge graph?"),
        ("❓ Missing Info", "What was the date of birth of the CoNLL evaluation script author's grandmother?"),
        ("🚫 Unrelated", "What is the boiling point of ethanol at sea level?"),
    ],
    "RAG, Semantic Search and Question Answering": [
        ("🔍 Bi vs Cross Encoder", "Compare Bi-Encoder and Cross-Encoder architectures in dense retrieval."),
        ("🛡️ Grounding (Paraphrased)", "How does injecting external knowledge passages mitigate hallucinations and provide citations?"),
        ("❓ Missing Info", "What is the serial number of the hard drive holding the FAISS vector index?"),
        ("🚫 Unrelated", "What is the airspeed velocity of an unladen swallow?"),
    ],
    "Machine Translation and Text Summarization": [
        ("📝 Extractive vs Abstractive", "What is the difference between extractive and abstractive summarization?"),
        ("🌐 Beam Search (Paraphrased)", "How does beam search explore multiple translation hypotheses with length penalty?"),
        ("❓ Missing Info", "What was the exact battery percentage of the laptop when BLEU was formulated?"),
        ("🚫 Unrelated", "What is the capital city of Madagascar?"),
    ],
}


def check_api_status():
    """Detect whether a live Gemini API key is configured in .env."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    is_live = bool(key and key != "your_gemini_api_key_here")
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    return is_live, model


def render_claim_badge(tag: str):
    """Render a colored badge for claim verification status."""
    if tag == "supported":
        return "🟢 **Supported**"
    elif tag == "weak":
        return "🟡 **Weak / Partially Supported**"
    else:
        return "🔴 **Unsupported**"


def main():
    st.title("🧠 DocuMind AI — RAG Answer Generation & Verification")
    st.caption("College NLP Project • Multi-Collection Answer Generation, Semantic Retrieval & Guardrails")

    is_live, model_name = check_api_status()
    available_collections = ["All Collections"] + get_available_collections()
    loader = get_document_loader()
    stats = get_collection_stats()

    # Sidebar: Dataset Selector, System Status, & Retrieved Passages
    with st.sidebar:
        st.subheader("📚 NLP Dataset Selector")
        selected_dataset = st.selectbox(
            "Choose Active Dataset Collection:",
            options=available_collections,
            index=0,
            help="Selecting a collection strictly restricts retrieval to its educational documents.",
        )

        # Show collection details
        if selected_dataset != "All Collections":
            col_obj = loader.get_collection(selected_dataset)
            if col_obj:
                st.caption(f"**Description:** {col_obj.description}")
                st.info(f"📁 **Documents:** {len(col_obj.documents)} | 📄 **Passages:** {sum(len(d.chunks) for d in col_obj.documents)}")
        else:
            st.caption(
                f"Indexing **{stats.get('total_collections', 6)} Collections**, "
                f"**{stats.get('total_documents', 18)} Documents**, and **{stats.get('total_passages', 54)} Passages**."
            )

        st.divider()
        st.subheader("⚙️ System Status")
        if is_live:
            st.success(f"🟢 **Live Gemini API Mode**\n\nModel: `{model_name}`\n\nReal API calls & claim verification enabled.")
        else:
            st.warning(
                "🟡 **Mock / Simulation Mode**\n\n"
                "No live `GEMINI_API_KEY` configured in `.env`.\n\n"
                "Responses & verifications are produced by Person 3's offline grounded simulator."
            )

        st.divider()
        st.subheader("📖 Indexed Reference Passages")
        target_filter = None if selected_dataset == "All Collections" else selected_dataset
        active_sample_passages = retrieve("", collection=target_filter, top_k=3, use_collections=True)
        st.caption(f"Top indexed passages in `{selected_dataset}`:")
        for idx, chunk in enumerate(active_sample_passages, start=1):
            with st.expander(
                f"Passage {idx}: {chunk.get('document_id', 'doc')} (Page {chunk.get('page_number', 'N/A')})",
                expanded=(idx == 1),
            ):
                st.markdown(f"**Title:** *{chunk.get('title', 'N/A')}*")
                st.markdown(f"**Chunk ID:** `{chunk.get('chunk_id', idx)}`")
                st.markdown(f"**Page:** `{chunk.get('page_number', 'N/A')}`")
                if chunk.get("source"):
                    st.caption(f"**Source:** {chunk.get('source')}")
                st.write(chunk.get("text", ""))

    # Main Area: Dataset Indicator Banner
    if selected_dataset != "All Collections":
        st.info(f"📂 **Active Dataset Restricted To:** `{selected_dataset}` — Retrieval is restricted exclusively to documents in this collection.")

    # Question Input Section
    st.subheader("Ask a Question")

    # Session state for the input text box
    if "question_input_field" not in st.session_state:
        st.session_state.question_input_field = (
            "What is the difference between Porter stemming and WordNet lemmatization?"
        )

    # Quick test question presets adapted to selected collection
    st.markdown("**Quick Test Presets for Selected Collection:**")
    current_presets = COLLECTION_PRESETS.get(
        selected_dataset, COLLECTION_PRESETS["All Collections"]
    )
    p_cols = st.columns(len(current_presets))
    for idx, (label, preset_q) in enumerate(current_presets):
        with p_cols[idx]:
            if st.button(label, use_container_width=True, key=f"btn_preset_{idx}"):
                st.session_state.question_input_field = preset_q

    # The text input is bound directly to question_input_field in session state
    question = st.text_input(
        "Enter your question about the documents:",
        placeholder="Type any custom question or paraphrase here...",
        key="question_input_field",
    )

    action_col1, action_col2 = st.columns([2, 1])
    with action_col1:
        ask_button = st.button("🚀 Ask DocuMind AI (With RAG)", type="primary", use_container_width=True)
    with action_col2:
        compare_button = st.button("⚖️ With RAG vs Without RAG", type="secondary", use_container_width=True)

    if ask_button or compare_button:

        if not question or not question.strip():
            st.warning("Please enter a question.")
            return

        # Fetch passages via the retrieve interface restricted to selected collection
        retrieval_col = None if selected_dataset == "All Collections" else selected_dataset
        passages = retrieve(
            question=question,
            collection=retrieval_col,
            top_k=3,
            min_score=0.08,
            use_collections=True,
        )

        with st.spinner("Processing question with Person 3 pipeline..."):
            client_to_use = None
            if not is_live:
                client_to_use = create_mock_client(question, passages)

            # 1. Generate grounded answer
            result = generate_answer(
                question=question,
                retrieved_chunks=passages,
                client=client_to_use,
            )

            # 2. If compare mode was selected, generate ungrounded baseline
            without_rag_result = None
            if compare_button:
                without_rag_result = generate_without_rag(
                    question=question,
                    client=client_to_use,
                )

        st.divider()

        # Contradiction warning banner if conflicting sources were detected
        if result.get("conflict_detected"):
            st.warning(
                "⚠️ **Contradiction Detected Across Passages!** "
                "Retrieved documents contain conflicting information on this subject. "
                "DocuMind AI has preserved and cited both perspectives rather than silently choosing one."
            )

        raw_answer = result.get("answer", "")
        display_answer = raw_answer
        if display_answer.startswith("[Simulated Response]"):
            display_answer = display_answer[len("[Simulated Response]"):].strip()

        status_text = result.get("status", "")
        has_error = bool(result.get("error") or status_text.startswith("API Error") or status_text.startswith("Configuration Error"))
        is_refusal = bool(result.get("insufficient_evidence"))

        # Mode Indicator
        if not is_live:
            st.info("🧪 **Simulation Mode:** Generated using Person 3 offline grounded mock simulator.")
        else:
            st.success(f"🟢 **Live Gemini Result:** Generated via model `{model_name}`.")

        # If "With RAG vs Without RAG" comparison is requested, render side-by-side
        if compare_button and without_rag_result is not None:
            st.subheader("⚖️ With RAG vs Without RAG Comparison")
            c_col1, c_col2 = st.columns(2)

            with c_col1:
                st.markdown("### 🧠 With RAG (DocuMind AI)")
                st.caption(f"Grounded exclusively on retrieved passages from `{selected_dataset}`.")
                if has_error:
                    st.error(f"❌ {status_text}\n\n*{display_answer}*")
                elif is_refusal:
                    st.warning(f"⚠️ **Insufficient Evidence Refusal**\n\n> *{display_answer}*")
                else:
                    st.success("✅ **Grounded & Cited**")
                    st.markdown(display_answer)

            with c_col2:
                st.markdown("### 🌐 Without RAG (Raw LLM Baseline)")
                st.caption("Un-augmented model response without document context (prone to hallucination).")
                st.info(without_rag_result.get("answer", "No answer."))

            st.divider()

        # Primary Answer Display
        st.subheader("Grounded Answer & Citations")

        if has_error:
            st.error(f"❌ **Status: {status_text or 'API Error'}**")
            st.markdown(f"> *{display_answer}*")
            st.caption("A transient API or configuration failure occurred. This is an ERROR, not an Insufficient Evidence refusal.")
            return

        elif is_refusal:
            st.warning("⚠️ **Status: Insufficient Evidence in Selected Dataset / Unsupported Question**")
            st.markdown(f"> *{display_answer}*")
            st.caption(
                f"Because the retrieved passages in '{selected_dataset}' do not contain sufficient evidence to answer this question, "
                "DocuMind AI correctly refrained from hallucinating an answer."
            )
        else:
            st.success("✅ **Status: Answer Grounded with Verified Citations**")
            st.markdown(f"### Answer\n{display_answer}")

            # Validated Sources with Document & Passage Snippet Inspection
            st.markdown("#### 📖 Validated Citations & Passage Context")
            sources = result.get("sources", [])
            if sources:
                src_cols = st.columns(min(len(sources), 3))
                for idx, src in enumerate(sources):
                    col = src_cols[idx % len(src_cols)]

                    # Find matching chunk text in retrieved passages
                    matching_text = ""
                    matching_title = ""
                    matching_source = ""
                    for p in passages:
                        if (
                            normalize_str(p.get("document_id")) == normalize_str(src.get("document_id"))
                            and str(p.get("page_number")) == str(src.get("page_number"))
                        ) or normalize_str(p.get("chunk_id")) == normalize_str(src.get("chunk_id")):
                            matching_text = p.get("text", "")
                            matching_title = p.get("title", "")
                            matching_source = p.get("source", "")
                            break

                    with col:
                        st.info(
                            f"**Citation #{idx+1}**\n\n"
                            f"📄 Document ID: `{src.get('document_id', 'Unknown')}`\n\n"
                            f"{f'📖 Title: *{matching_title}*' + chr(10) + chr(10) if matching_title else ''}"
                            f"📑 Page: `{src.get('page_number', 'N/A')}`\n\n"
                            f"🔖 Chunk: `{src.get('chunk_id', 'Unknown')}`"
                        )
                        if matching_text:
                            with st.expander("🔍 View Cited Passage", expanded=False):
                                if matching_source:
                                    st.caption(f"**Source:** {matching_source}")
                                st.write(matching_text)
            else:
                st.write("No valid citations were identified.")

        # Claim-Level Verification & Trust Score Section (if not a raw error)
        st.divider()
        st.subheader("🔍 Claim-Level Verification & Evidence Analysis")

        with st.spinner("Verifying factual claims against reference passages..."):
            claims_verification = verify_all_claims(
                answer=display_answer,
                retrieved_chunks=passages,
                client=client_to_use,
            )
            trust_metrics = compute_trust_score(
                result=result,
                claims_verification=claims_verification,
                retrieved_chunks=passages,
            )

        # Trust Score Display
        t_col1, t_col2 = st.columns([1, 2])
        with t_col1:
            st.metric(
                label="🎯 DocuMind Trust Score",
                value=f"{trust_metrics['score']} / 100",
                delta="Faithful Grounding" if trust_metrics['score'] >= 75 else "Low Grounding / Weak Evidence",
            )
        with t_col2:
            st.markdown("**Trust Score Formula & Breakdown:**")
            st.progress(trust_metrics["score"] / 100.0)
            st.caption(trust_metrics["breakdown"])

        # Detailed Claim Breakdown
        if claims_verification:
            st.markdown("#### Individual Claim Verdicts & Extracted Evidence")
            for idx, c in enumerate(claims_verification, start=1):
                badge = render_claim_badge(c["tag"])
                with st.expander(f"Claim {idx}: {badge} — {c['claim'][:70]}...", expanded=True):
                    st.markdown(f"**Exact Claim:** \"{c['claim']}\"")
                    st.markdown(f"**Verification Verdict:** {badge}")
                    st.markdown(f"**Cited Evidence from Passage:**")
                    if c["evidence"] and c["evidence"] != "None":
                        st.success(f"“{c['evidence']}”")
                    else:
                        st.warning("No supporting sentence found in the cited context passage.")
        else:
            if is_refusal:
                st.info("System refrained from making any factual claims due to lack of evidence.")

        # Raw output JSON viewer
        with st.expander("🔍 View Raw Output Dictionary (JSON)"):
            debug_payload = {
                "selected_collection": selected_dataset,
                "retrieved_passages_count": len(passages),
                "generation_result": result,
                "claims_verification": claims_verification,
                "trust_metrics": trust_metrics,
            }
            st.json(debug_payload)


if __name__ == "__main__":
    main()
