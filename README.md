# 📚 ELI5 Paper Summarizer

Transform complex academic papers into **layered summaries** — from expert-level technical detail to explanations a 10-year-old could understand.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Problem

Academic papers are dense and inaccessible:
- Researchers spend hours deciphering papers outside their expertise
- Students struggle with jargon-heavy content
- Industry practitioners need quick understanding of new developments
- Existing summarizers produce one-size-fits-all summaries

## 💡 Solution

ELI5 Paper Summarizer generates **three distinct summary levels**:

| Level | Audience | Length | Style |
|-------|----------|--------|-------|
| 📚 **Technical** | Researchers, experts | 500-800 words | Precise, metrics-focused |
| 📖 **Simplified** | Educated non-experts | 300-500 words | Accessible, contextual |
| 🧒 **ELI5** | Complete beginners | 150-250 words | Analogies, simple language |

## ✨ Features

- 🔗 **arXiv Integration**: Paste any arXiv URL or paper ID
- 📁 **PDF Upload**: Process any research paper PDF
- 🎨 **Web UI**: Beautiful Streamlit interface with tabs
- 💻 **CLI Support**: Command-line interface for automation
- 🆓 **100% Free**: Uses Groq's free Llama 3.1 70B API
- 📊 **Progress Tracking**: Real-time status updates

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Siddhesh513/eli5-paper-summarizer.git
cd eli5-paper-summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Free API Key

1. Go to https://console.groq.com/keys
2. Sign up (free)
3. Create API Key
4. Copy the key

### 3. Configure

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Run

**Web UI (Recommended):**
```bash
streamlit run app.py
```

**Command Line:**
```bash
python main.py --url 1706.03762
python main.py --file paper.pdf
python main.py --url 2301.00001 --level eli5
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   arXiv URL     │     │   PDF Upload    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
        ┌─────────────────────────┐
        │    PDF Processor        │
        │  (PyMuPDF + arxiv API)  │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │   Section-Aware Chunker │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │   Vector Store          │
        │   (ChromaDB)            │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │   Summary Generator     │
        │   (Groq Llama 3.1 70B)  │
        └────────────┬────────────┘
                     ▼
    ┌────────────────┼────────────────┐
    ▼                ▼                ▼
┌────────┐     ┌──────────┐     ┌────────┐
│Technical│    │Simplified│     │  ELI5  │
└────────┘     └──────────┘     └────────┘
```

## 📁 Project Structure

```
eli5-paper-summarizer/
├── app.py                 # Streamlit web UI
├── main.py                # CLI entry point
├── src/
│   ├── pdf_processor.py   # PDF fetching & extraction
│   ├── chunker.py         # Section-aware chunking
│   ├── embeddings.py      # ChromaDB vector store
│   ├── summarizer.py      # LLM summary generation
│   └── prompts.py         # Prompt templates
├── config/
│   └── settings.py        # Configuration
├── tests/
│   └── test_processor.py  # Unit tests
├── requirements.txt
└── README.md
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq (Llama 3.1 70B) - Free! |
| **Vector Store** | ChromaDB |
| **PDF Processing** | PyMuPDF |
| **Orchestration** | LangChain |
| **Frontend** | Streamlit |
| **Tokenization** | tiktoken |

## 📊 Example Output

**Input**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

**ELI5 Summary**:
> Imagine you're reading a really long story and trying to remember what happened at the beginning while you're at the end. That's hard, right? Computers have the same problem!
>
> Scientists created something called a "Transformer" — it's like giving the computer a special pair of glasses that lets it look at ALL parts of a story at the same time, instead of reading word by word.
>
> Before this, computers were like reading through a tiny keyhole. Now they can see the whole page at once! This makes them much better at translating languages and understanding what we write.
>
> This invention is now used in things like Google Translate and ChatGPT!

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 🚀 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Add `GROQ_API_KEY` to Secrets

## 🔮 Future Improvements

- [ ] Multi-paper comparison
- [ ] Citation extraction
- [ ] Key findings bullets
- [ ] Audio narration of ELI5
- [ ] Browser extension
- [ ] Fine-tuned summarization model

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss changes.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

Made with ❤️ by [Sidd](https://github.com/Siddhesh513)

**Weekend Project** 🚀 Built in 2 days to demonstrate GenAI/RAG skills.
