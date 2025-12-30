"""
Configuration settings for ELI5 Paper Summarizer.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Key (FREE!)
# Get yours at: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM Configuration
LLM_PROVIDER = "groq"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Chunking Configuration
CHUNK_SIZE = 1500  # tokens
CHUNK_OVERLAP = 200  # tokens
MAX_CHUNKS_FOR_DIRECT = 15  # Use map-reduce above this

# Summary Length Targets (words)
TECHNICAL_LENGTH = (500, 800)
SIMPLIFIED_LENGTH = (300, 500)
ELI5_LENGTH = (150, 250)

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "paper_chunks"

# Rate Limiting
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Supported arXiv URL patterns
ARXIV_PATTERNS = [
    r"arxiv\.org/abs/(\d+\.\d+)",
    r"arxiv\.org/pdf/(\d+\.\d+)",
    r"^(\d+\.\d+)$",  # Just the ID
]

# Processing Mode Configuration
TOKEN_BUDGETS = {
    'quick': 500,      # Abstract only
    'smart': 2500,     # Abstract + key sections
    'deep': 12000      # Full paper
}

# Smart Mode Configuration
SMART_MODE_CONFIG = {
    'min_sections': 2,
    'max_sections': 5,
    'reserved_for_abstract': 300,
    'section_similarity_threshold': 0.3,
    'section_preview_words': 200,  # For embedding
}

# Embedding Model for Section Ranking
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Non-content sections to skip in Smart mode
SKIP_SECTIONS = {"Abstract", "Preamble", "References", "Acknowledgments",
                 "Acknowledgements", "Appendix", "Appendices", "Supplementary"}
