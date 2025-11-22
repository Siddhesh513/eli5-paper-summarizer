"""
Configuration settings for ELI5 Paper Summarizer.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Provider: "openai", "groq", "google", "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # Default to free Groq

# Model Configuration per provider
LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",  # Free, fast, high quality
    "google": "gemini-1.5-flash",        # Free tier available
    "ollama": "llama3.1",                # Local, fully free
}
LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODELS.get(
    LLM_PROVIDER, "llama-3.1-70b-versatile"))

# Embedding Configuration
# For free embeddings, use HuggingFace sentence-transformers (local)
EMBEDDING_PROVIDER = os.getenv(
    "EMBEDDING_PROVIDER", "huggingface")  # "openai" or "huggingface"
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Free local model

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
