# 📚 ELI5 Paper Summarizer

Transform complex academic papers into **layered summaries** — from expert-level technical detail to explanations a 10-year-old could understand.

## 🚧 Work in Progress

This is the Day 1 version with core backend functionality:
- ✅ arXiv paper fetching
- ✅ PDF text extraction
- ✅ Section-aware chunking
- ✅ Vector embeddings with ChromaDB
- ✅ Three-level summary generation
- ✅ CLI interface

Coming in Day 2:
- 🔲 Streamlit web UI
- 🔲 PDF file upload
- 🔲 Error handling improvements
- 🔲 Deployment to Streamlit Cloud

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/eli5-paper-summarizer.git
cd eli5-paper-summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key (get free key from https://console.groq.com/keys)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run
python main.py --url 1706.03762
```

## 📊 Summary Levels

| Level | Audience | Length |
|-------|----------|--------|
| 📚 Technical | Researchers | 500-800 words |
| 📖 Simplified | General audience | 300-500 words |
| 🧒 ELI5 | Complete beginners | 150-250 words |

## 🛠️ Tech Stack

- **LLM**: Groq (Llama 3.1 70B) - Free!
- **Embeddings**: HuggingFace sentence-transformers - Free!
- **Vector Store**: ChromaDB
- **PDF Processing**: PyMuPDF
- **Orchestration**: LangChain

---
Day 1 of weekend project 🚀
