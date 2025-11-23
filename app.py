"""
ELI5 Paper Summarizer - Streamlit UI.

Run with: streamlit run app.py
"""
from src.summarizer import summarize_paper
from src.embeddings import create_retriever
from src.chunker import chunk_by_section, prepare_chunks_for_embedding, get_total_tokens
from src.pdf_processor import process_paper, PaperContent, extract_text_from_pdf, detect_sections
import streamlit as st
import tempfile
from pathlib import Path

# Page config must be first Streamlit command
st.set_page_config(
    page_title="ELI5 Paper Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Sample papers for quick testing
SAMPLE_PAPERS = {
    "Select a sample paper...": "",
    "Attention Is All You Need (Transformers)": "1706.03762",
    "BERT": "1810.04805",
    "GPT-2": "1901.02860",
    "ResNet": "1512.03385",
    "Adam Optimizer": "1412.6980",
}


@st.cache_data(ttl=3600, show_spinner=False)
def process_complete_pipeline(arxiv_url: str, chunking_strategy: str):
    """Cache the entire pipeline including summarization."""
    # Step 1-2: Process paper
    paper = process_paper(arxiv_url)

    # Step 3: Chunk
    chunks = chunk_by_section(paper.sections, strategy=chunking_strategy)
    texts, metadatas = prepare_chunks_for_embedding(chunks)

    # Step 4: Embed
    retriever = create_retriever(texts, metadatas, paper.title)
    all_chunks = retriever.get_all_chunks()

    # Step 5: Summarize
    result = summarize_paper(all_chunks)

    return {
        'paper_title': paper.title,
        'authors': paper.authors,
        'sections': list(paper.sections.keys()),
        'total_tokens': get_total_tokens(chunks),
        'num_chunks': len(chunks),
        'result': result
    }


def process_uploaded_pdf(uploaded_file) -> PaperContent:
    """Process an uploaded PDF file."""
    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / uploaded_file.name

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    full_text = extract_text_from_pdf(str(temp_path))
    sections = detect_sections(full_text)

    # Try to extract title from first lines
    lines = full_text.split("\n")[:10]
    title = next((l.strip()
                 for l in lines if len(l.strip()) > 10), "Uploaded Paper")

    return PaperContent(
        title=title,
        authors=[],
        abstract=sections.get("Abstract", ""),
        full_text=full_text,
        sections=sections,
        pdf_path=str(temp_path),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def process_uploaded_pipeline(pdf_hash: str, pdf_bytes: bytes, chunking_strategy: str):
    """Cache uploaded PDF processing based on file hash."""
    import tempfile
    from pathlib import Path

    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / f"upload_{pdf_hash}.pdf"

    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    # Extract text
    full_text = extract_text_from_pdf(str(temp_path))
    sections = detect_sections(full_text)

    # Try to extract title
    lines = full_text.split("\n")[:10]
    title = next((l.strip()
                 for l in lines if len(l.strip()) > 10), "Uploaded Paper")

    # Chunk
    chunks = chunk_by_section(sections, strategy=chunking_strategy)
    texts, metadatas = prepare_chunks_for_embedding(chunks)

    # Embed
    retriever = create_retriever(texts, metadatas, title)
    all_chunks = retriever.get_all_chunks()

    # Summarize
    result = summarize_paper(all_chunks)

    return {
        'paper_title': title,
        'authors': [],
        'sections': list(sections.keys()),
        'total_tokens': get_total_tokens(chunks),
        'num_chunks': len(chunks),
        'result': result
    }


def main():
    # Header
    st.title("📚 ELI5 Paper Summarizer")
    st.markdown(
        "Transform complex academic papers into **layered summaries** — "
        "from expert-level technical detail to explanations a 10-year-old could understand."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Chunking strategy selector
        st.subheader("🔧 Chunking Strategy")
        chunking_strategy = st.selectbox(
            "How to split the paper:",
            options=["recursive", "semantic", "simple"],
            index=0,
            help=(
                "• Recursive: Splits on paragraphs, then sentences (recommended)\n"
                "• Semantic: Groups similar sentences together\n"
                "• Simple: Basic paragraph splitting"
            )
        )

        st.divider()

        # Sample papers
        st.subheader("🔬 Sample Papers")
        sample_selection = st.selectbox(
            "Try a famous paper:",
            options=list(SAMPLE_PAPERS.keys()),
            help="Select a well-known paper to test the summarizer",
        )

        if sample_selection != "Select a sample paper...":
            sample_id = SAMPLE_PAPERS[sample_selection]
            if st.button(f"Load: {sample_selection[:30]}..."):
                st.session_state.arxiv_url = sample_id
                st.rerun()

        st.divider()

        # About section
        st.subheader("ℹ️ About")
        st.markdown("""
        **Three Summary Levels:**
        - 📚 **Technical**: For researchers (500-800 words)
        - 📖 **Simplified**: For educated non-experts (300-500 words)
        - 🧒 **ELI5**: For complete beginners (150-250 words)
        
        **Powered by:**
        - Groq (Llama 3.1 70B) - Free!
        - ChromaDB for retrieval
        - PyMuPDF for PDF processing
        """)

        st.divider()
        st.markdown("Made with ❤️ by [Sidd](https://github.com/Siddhesh513)")

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔗 From arXiv URL")
        arxiv_url = st.text_input(
            "Enter arXiv URL or paper ID:",
            value=st.session_state.get("arxiv_url", ""),
            placeholder="e.g., 2301.00001 or https://arxiv.org/abs/2301.00001",
            help="Supports arXiv URLs and paper IDs",
        )

    with col2:
        st.subheader("📁 Upload PDF")
        uploaded_file = st.file_uploader(
            "Or upload a PDF file:",
            type=["pdf"],
            help="Upload any research paper PDF",
        )

    # Process button
    st.divider()

    process_clicked = st.button(
        "🚀 Generate Summaries",
        type="primary",
        use_container_width=True,
        disabled=not (arxiv_url or uploaded_file),
    )

    if process_clicked:
        if not arxiv_url and not uploaded_file:
            st.error("Please provide an arXiv URL or upload a PDF file.")
            return

        try:
            # Progress tracking
            progress = st.progress(0, text="Starting...")

            # Step 1: Process paper
            progress.progress(10, text="📄 Fetching and extracting paper...")

            if arxiv_url:
                # Use cached complete pipeline
                cached_result = process_complete_pipeline(
                    arxiv_url, chunking_strategy)

                # Display paper info
                st.success(f"**{cached_result['paper_title']}** ✨")
                if cached_result['authors']:
                    st.caption(
                        f"Authors: {', '.join(cached_result['authors'][:5])}{'...' if len(cached_result['authors']) > 5 else ''}")

                # Show stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks", cached_result['num_chunks'])
                col2.metric("Tokens", f"{cached_result['total_tokens']:,}")
                col3.metric("Sections", len(cached_result['sections']))

                progress.progress(100, text="✅ Complete!")

                # Store results
                st.session_state.result = cached_result['result']
                st.session_state.paper_title = cached_result['paper_title']
            else:
                # PDF upload with caching
                import hashlib
                pdf_bytes = uploaded_file.getbuffer().tobytes()
                pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

                cached_result = process_uploaded_pipeline(
                    pdf_hash, pdf_bytes, chunking_strategy)

                st.success(f"**{cached_result['paper_title']}** ✨")

                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks", cached_result['num_chunks'])
                col2.metric("Tokens", f"{cached_result['total_tokens']:,}")
                col3.metric("Sections", len(cached_result['sections']))

                progress.progress(100, text="✅ Complete!")

                st.session_state.result = cached_result['result']
                st.session_state.paper_title = cached_result['paper_title']
        except Exception as e:
            st.error(f"❌ Error processing paper: {str(e)}")
            st.exception(e)
            return

    # Display results if available
    if "result" in st.session_state:
        result = st.session_state.result

        st.divider()
        st.subheader(
            f"📝 Summaries: {st.session_state.get('paper_title', 'Paper')[:50]}...")

        # Tabs for different summary levels
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ Key Findings",
            "📚 Technical",
            "📖 Simplified",
            "🧒 ELI5"
        ])

        with tab1:
            st.markdown("### Key Findings")
            st.markdown(
                "*Quick reference - main contributions, results, and limitations*")
            st.markdown(result.key_findings)
            st.divider()
            st.download_button(
                "📥 Download Key Findings",
                result.key_findings,
                file_name="key_findings.md",
                mime="text/markdown",
            )

        with tab2:
            st.markdown("### Technical Summary")
            st.markdown("*For researchers and domain experts*")
            st.markdown(result.technical)
            st.divider()
            st.download_button(
                "📥 Download Technical Summary",
                result.technical,
                file_name="technical_summary.md",
                mime="text/markdown",
            )

        with tab3:
            st.markdown("### Simplified Summary")
            st.markdown("*For educated non-experts*")
            st.markdown(result.simplified)
            st.divider()
            st.download_button(
                "📥 Download Simplified Summary",
                result.simplified,
                file_name="simplified_summary.md",
                mime="text/markdown",
            )

        with tab4:
            st.markdown("### ELI5 Summary")
            st.markdown("*Explain Like I'm 5 — for complete beginners*")
            st.markdown(result.eli5)
            st.divider()
            st.download_button(
                "📥 Download ELI5 Summary",
                result.eli5,
                file_name="eli5_summary.md",
                mime="text/markdown",
            )

        # Summary stats
        st.divider()
        st.caption(
            f"📊 Stats: {result.chunks_used} chunks ({chunking_strategy}) | "
            f"{result.token_count:,} input tokens | "
            f"Technical: {len(result.technical.split())} words | "
            f"Simplified: {len(result.simplified.split())} words | "
            f"ELI5: {len(result.eli5.split())} words"
        )


if __name__ == "__main__":
    main()
