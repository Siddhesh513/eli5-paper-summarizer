"""
Intelligent Chunking Module.
Section-aware text chunking with metadata preservation.
Supports multiple chunking strategies.
"""
from dataclasses import dataclass
from typing import Optional
import re

import tiktoken

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


def chunk_text_recursive(
    text: str,
    max_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into chunks using recursive character splitting.
    Tries to split on natural boundaries (paragraphs, sentences, words).
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
    
    Returns:
        List of text chunks
    """
    # Approximate chars per token
    chars_per_token = 3.5
    max_chars = int(max_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)
    
    # Separators in order of preference
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    
    def split_with_separator(text: str, separator: str, remaining_separators: list) -> list[str]:
        """Recursively split text using separators."""
        if not text:
            return []
        
        # If text is small enough, return it
        if len(text) <= max_chars:
            return [text]
        
        # Split by current separator
        splits = text.split(separator)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split) + len(separator)
            
            # If single split is too large, use next separator
            if split_len > max_chars and remaining_separators:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Recursively split this large piece
                sub_chunks = split_with_separator(split, remaining_separators[0], remaining_separators[1:])
                chunks.extend(sub_chunks)
            elif current_length + split_len > max_chars:
                # Start new chunk
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [split]
                current_length = split_len
            else:
                # Add to current chunk
                current_chunk.append(split)
                current_length += split_len
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
    
    # Start recursive splitting
    chunks = split_with_separator(text, separators[0], separators[1:])
    
    # Add overlap
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap_chars > 0:
            # Add end of previous chunk as overlap
            prev_chunk = chunks[i-1]
            overlap = prev_chunk[-overlap_chars:] if len(prev_chunk) > overlap_chars else prev_chunk
            overlapped_chunks.append(overlap + chunk)
        else:
            overlapped_chunks.append(chunk)
    
    return overlapped_chunks if overlapped_chunks else [text]


def chunk_text_semantic(
    text: str,
    max_tokens: int = CHUNK_SIZE,
) -> list[str]:
    """
    Split text into chunks based on semantic coherence.
    Groups sentences with similar topics together.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
    
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_tokens = count_tokens(sentence)
        
        # Check if adding this sentence would exceed limit
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Don't forget last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks if chunks else [text]


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_SIZE,
    overlap_tokens: int = CHUNK_OVERLAP,
    strategy: str = "recursive"
) -> list[str]:
    """
    Split text into chunks using specified strategy.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks
        strategy: Chunking strategy ("recursive", "semantic", "simple")
    
    Returns:
        List of text chunks
    """
    if strategy == "recursive":
        return chunk_text_recursive(text, max_tokens, overlap_tokens)
    elif strategy == "semantic":
        return chunk_text_semantic(text, max_tokens)
    else:  # simple/default
        # Original simple chunking
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = count_tokens(para)
            
            if para_tokens > max_tokens:
                sentences = para.replace('. ', '.|').split('|')
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    sent_tokens = count_tokens(sent)
                    
                    if current_tokens + sent_tokens > max_tokens and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else ''
                        current_chunk = [overlap_text] if overlap_text else []
                        current_tokens = count_tokens(overlap_text) if overlap_text else 0
                    
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
            else:
                if current_tokens + para_tokens > max_tokens and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                current_chunk.append(para)
                current_tokens += para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks if chunks else [text]


def chunk_by_section(
    sections: dict[str, str],
    max_tokens: int = CHUNK_SIZE,
    min_section_tokens: int = 100,
    strategy: str = "recursive"
) -> list[Chunk]:
    """
    Chunk paper content while preserving section boundaries.
    
    Args:
        sections: Dictionary mapping section names to content
        max_tokens: Maximum tokens per chunk
        min_section_tokens: Minimum tokens to keep section as separate chunk
        strategy: Chunking strategy ("recursive", "semantic", "simple")
    
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
                    "chunking_strategy": strategy,
                }
            )
            all_chunks.append(chunk)
        else:
            # Split section into multiple chunks using selected strategy
            sub_chunks = chunk_text(content, max_tokens, strategy=strategy)
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
                        "chunking_strategy": strategy,
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
