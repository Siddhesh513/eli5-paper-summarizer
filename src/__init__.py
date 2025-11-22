"""
ELI5 Paper Summarizer - Core modules.
"""
from .pdf_processor import process_paper, PaperContent
from .chunker import chunk_by_section, prepare_chunks_for_embedding, Chunk
from .embeddings import PaperVectorStore, create_retriever
from .summarizer import PaperSummarizer, summarize_paper, SummaryResult

__all__ = [
    "process_paper",
    "PaperContent",
    "chunk_by_section",
    "prepare_chunks_for_embedding",
    "Chunk",
    "PaperVectorStore",
    "create_retriever",
    "PaperSummarizer",
    "summarize_paper",
    "SummaryResult",
]
