"""
Intelligent Chunking Module.
Section-aware text chunking with metadata preservation.
"""
from dataclasses import dataclass
from typing import Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """Container for a text chunk with metadata."""
    content: str
    section: str
    chunk_index: int
    total_chunks_in_section: int
    token_count: int
    metadata: dict


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer selection

    Returns:
        Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into chunks respecting token limits.

    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks

    Returns:
        List of text chunks
    """
    # Approximate chars per token (conservative estimate)
    chars_per_token = 3.5
    chunk_size_chars = int(max_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    return splitter.split_text(text)


def chunk_by_section(
    sections: dict[str, str],
    max_tokens: int = CHUNK_SIZE,
    min_section_tokens: int = 100
) -> list[Chunk]:
    """
    Chunk paper content while preserving section boundaries.

    Args:
        sections: Dictionary mapping section names to content
        max_tokens: Maximum tokens per chunk
        min_section_tokens: Minimum tokens to keep section as separate chunk

    Returns:
        List of Chunk objects with metadata
    """
    all_chunks = []

    # Priority order for sections (most important first)
    section_priority = [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Experiments",
        "Discussion",
        "Conclusion",
        "Related Work",
        "Preamble",
    ]

    # Process sections in priority order, then any remaining
    processed = set()
    ordered_sections = []

    for section_name in section_priority:
        if section_name in sections:
            ordered_sections.append((section_name, sections[section_name]))
            processed.add(section_name)

    for section_name, content in sections.items():
        if section_name not in processed:
            ordered_sections.append((section_name, content))

    # Chunk each section
    for section_name, content in ordered_sections:
        if not content or not content.strip():
            continue

        section_tokens = count_tokens(content)

        if section_tokens <= max_tokens:
            # Section fits in one chunk
            chunk = Chunk(
                content=content.strip(),
                section=section_name,
                chunk_index=0,
                total_chunks_in_section=1,
                token_count=section_tokens,
                metadata={
                    "section": section_name,
                    "is_complete_section": True,
                }
            )
            all_chunks.append(chunk)
        else:
            # Split section into multiple chunks
            sub_chunks = chunk_text(content, max_tokens)
            total_sub = len(sub_chunks)

            for idx, sub_content in enumerate(sub_chunks):
                chunk = Chunk(
                    content=sub_content.strip(),
                    section=section_name,
                    chunk_index=idx,
                    total_chunks_in_section=total_sub,
                    token_count=count_tokens(sub_content),
                    metadata={
                        "section": section_name,
                        "is_complete_section": False,
                        "part": f"{idx + 1}/{total_sub}",
                    }
                )
                all_chunks.append(chunk)

    return all_chunks


def prepare_chunks_for_embedding(chunks: list[Chunk]) -> tuple[list[str], list[dict]]:
    """
    Prepare chunks for embedding into vector store.

    Args:
        chunks: List of Chunk objects

    Returns:
        Tuple of (texts, metadatas) for ChromaDB
    """
    texts = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # Prepend section context to content
        context_prefix = f"[{chunk.section}] "
        enriched_content = context_prefix + chunk.content

        texts.append(enriched_content)
        metadatas.append({
            "chunk_id": i,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "total_in_section": chunk.total_chunks_in_section,
            "token_count": chunk.token_count,
            **chunk.metadata,
        })

    return texts, metadatas


def get_total_tokens(chunks: list[Chunk]) -> int:
    """Calculate total tokens across all chunks."""
    return sum(chunk.token_count for chunk in chunks)


def estimate_cost(total_tokens: int, model: str = "gpt-4o-mini") -> float:
    """
    Estimate API cost for processing.

    Args:
        total_tokens: Total input tokens
        model: Model name

    Returns:
        Estimated cost in USD
    """
    # Approximate pricing (as of 2024)
    pricing = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # per 1K tokens
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }

    rates = pricing.get(model, pricing["gpt-4o-mini"])

    # Estimate output as 30% of input for summaries
    estimated_output = total_tokens * 0.3

    input_cost = (total_tokens / 1000) * rates["input"]
    output_cost = (estimated_output / 1000) * rates["output"]

    # Multiply by 3 for three summary levels
    return (input_cost + output_cost) * 3


if __name__ == "__main__":
    # Quick test
    test_sections = {
        "Abstract": "This is a test abstract. " * 50,
        "Introduction": "This is the introduction. " * 200,
        "Methods": "These are the methods. " * 300,
        "Results": "These are the results. " * 150,
        "Conclusion": "This is the conclusion. " * 50,
    }

    chunks = chunk_by_section(test_sections)
    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks:
        print(f"  [{chunk.section}] Part {chunk.chunk_index + 1}/{chunk.total_chunks_in_section}: {chunk.token_count} tokens")

    total = get_total_tokens(chunks)
    cost = estimate_cost(total)
    print(f"\nTotal tokens: {total}")
    print(f"Estimated cost: ${cost:.4f}")
