"""
Smart Paper Processor Module.
Intelligent tiered processing with three modes: Quick, Smart, and Deep.
"""
from dataclasses import dataclass
from typing import Literal, Optional
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunker import Chunk, count_tokens, chunk_by_section
from src.pdf_processor import PaperContent
from config.settings import (
    TOKEN_BUDGETS,
    SMART_MODE_CONFIG,
    EMBEDDING_MODEL,
    SKIP_SECTIONS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ProcessingMode = Literal["quick", "smart", "deep"]


@dataclass
class ProcessingConfig:
    """Configuration for smart processing modes."""
    mode: ProcessingMode = "smart"
    max_tokens: int = TOKEN_BUDGETS["deep"]
    top_n_sections: int = SMART_MODE_CONFIG["max_sections"]
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    chunking_strategy: str = "recursive"


class SmartPaperProcessor:
    """
    Intelligent paper processor with three processing modes.

    Modes:
    - Quick: Abstract only (fastest, minimal context)
    - Smart: Abstract + top N most relevant sections (balanced)
    - Deep: Full paper (existing flow, most comprehensive)
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        embedding_model: str = EMBEDDING_MODEL
    ):
        """
        Initialize processor.

        Args:
            config: Processing configuration
            embedding_model: Sentence transformer model for section ranking
        """
        self.config = config or ProcessingConfig()
        self._embedder: Optional[SentenceTransformer] = None
        self._embedding_model_name = embedding_model

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load embedding model (only when needed for Smart mode)."""
        if self._embedder is None:
            logger.info(f"Loading embedding model: {self._embedding_model_name}")
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def process(
        self,
        paper: PaperContent,
        mode: Optional[ProcessingMode] = None
    ) -> list[Chunk]:
        """
        Main processing entry point.

        Args:
            paper: PaperContent object from process_paper()
            mode: Override the configured mode

        Returns:
            List of Chunk objects compatible with existing pipeline
        """
        processing_mode = mode or self.config.mode

        logger.info(f"Processing paper in {processing_mode.upper()} mode")

        if processing_mode == "quick":
            return self._process_quick(paper)
        elif processing_mode == "smart":
            return self._process_smart(paper)
        else:  # deep
            return self._process_deep(paper)

    def _process_quick(self, paper: PaperContent) -> list[Chunk]:
        """
        Quick mode: Abstract only.

        Strategy: Extract abstract, create single chunk.
        Fallback: If no abstract, use first N paragraphs.
        """
        abstract = paper.abstract or paper.sections.get("Abstract", "")

        # Fallback: extract first ~500 words if no abstract
        if not abstract or len(abstract.strip()) < 50:
            logger.warning("No abstract found, extracting opening content")
            abstract = self._extract_opening_content(paper.full_text, max_words=500)

        token_count = count_tokens(abstract)

        logger.info(f"Quick mode: Using abstract ({token_count} tokens)")

        return [
            Chunk(
                content=abstract.strip(),
                section="Abstract",
                chunk_index=0,
                total_chunks_in_section=1,
                token_count=token_count,
                metadata={
                    "section": "Abstract",
                    "is_complete_section": True,
                    "processing_mode": "quick",
                    "chunking_strategy": "none",
                }
            )
        ]

    def _process_smart(self, paper: PaperContent) -> list[Chunk]:
        """
        Smart mode: Abstract + top N most relevant sections.

        Algorithm:
        1. Extract abstract
        2. Embed abstract
        3. Embed all section titles + first 200 words
        4. Compute cosine similarity
        5. Rank sections by relevance
        6. Select top N sections within token budget
        7. Create chunks from selected sections
        """
        try:
            chunks = []

            # Step 1: Get abstract
            abstract = paper.abstract or paper.sections.get("Abstract", "")
            if not abstract or len(abstract.strip()) < 50:
                logger.warning("No abstract found, extracting opening content")
                abstract = self._extract_opening_content(paper.full_text, max_words=300)

            # Add abstract as first chunk
            abstract_tokens = count_tokens(abstract)
            chunks.append(
                Chunk(
                    content=abstract.strip(),
                    section="Abstract",
                    chunk_index=0,
                    total_chunks_in_section=1,
                    token_count=abstract_tokens,
                    metadata={
                        "section": "Abstract",
                        "is_complete_section": True,
                        "processing_mode": "smart",
                        "relevance_score": 1.0,  # Abstract always most relevant
                        "chunking_strategy": "none",
                    }
                )
            )

            logger.info(f"Smart mode: Abstract ({abstract_tokens} tokens)")

            # Step 2-5: Rank sections by relevance to abstract
            section_rankings = self._rank_sections_by_relevance(
                query_text=abstract,
                sections=paper.sections
            )

            # Step 6: Select sections within token budget
            budget = TOKEN_BUDGETS.get(self.config.mode, TOKEN_BUDGETS["smart"])
            remaining_budget = budget - abstract_tokens
            selected_sections = self._select_sections_within_budget(
                section_rankings,
                paper.sections,
                budget=remaining_budget,
                max_sections=self.config.top_n_sections
            )

            logger.info(f"Smart mode: Selected {len(selected_sections)} sections: {selected_sections}")

            # Step 7: Create chunks from selected sections
            if selected_sections:
                selected_section_dict = {
                    name: paper.sections[name]
                    for name in selected_sections
                }

                section_chunks = chunk_by_section(
                    selected_section_dict,
                    max_tokens=self.config.chunk_size,
                    strategy=self.config.chunking_strategy
                )

                # Enrich metadata with relevance scores
                for chunk in section_chunks:
                    chunk.metadata["processing_mode"] = "smart"
                    chunk.metadata["relevance_score"] = section_rankings.get(
                        chunk.section, 0.0
                    )

                chunks.extend(section_chunks)

            total_tokens = sum(c.token_count for c in chunks)
            logger.info(f"Smart mode: Total {len(chunks)} chunks ({total_tokens} tokens)")

            return chunks

        except Exception as e:
            logger.warning(f"Smart mode failed: {e}. Falling back to Deep mode.")
            return self._process_deep(paper)

    def _process_deep(self, paper: PaperContent) -> list[Chunk]:
        """
        Deep mode: Full paper processing (existing flow).

        Simply delegates to existing chunk_by_section.
        """
        logger.info("Deep mode: Processing full paper")

        chunks = chunk_by_section(
            paper.sections,
            max_tokens=self.config.chunk_size,
            strategy=self.config.chunking_strategy
        )

        # Add processing mode to metadata
        for chunk in chunks:
            chunk.metadata["processing_mode"] = "deep"

        total_tokens = sum(c.token_count for c in chunks)
        logger.info(f"Deep mode: Total {len(chunks)} chunks ({total_tokens} tokens)")

        return chunks

    def _rank_sections_by_relevance(
        self,
        query_text: str,
        sections: dict[str, str]
    ) -> dict[str, float]:
        """
        Rank sections by semantic similarity to query.

        Args:
            query_text: Text to compare against (typically abstract)
            sections: Dict of section_name -> content

        Returns:
            Dict of section_name -> similarity_score (0-1)
        """
        # Embed query (abstract)
        query_embedding = self.embedder.encode(query_text, convert_to_numpy=True)

        rankings = {}
        section_embeddings = []
        section_names = []

        # Process sections
        for section_name, content in sections.items():
            # Skip abstract itself and non-content sections
            if section_name in SKIP_SECTIONS:
                continue

            if not content or len(content.strip()) < 100:
                continue

            # Embed section preview (title + first 200 words)
            section_preview = self._create_section_preview(section_name, content)
            section_embedding = self.embedder.encode(
                section_preview,
                convert_to_numpy=True
            )

            section_embeddings.append(section_embedding)
            section_names.append(section_name)

        # Compute cosine similarities
        if section_embeddings:
            section_embeddings = np.array(section_embeddings)
            similarities = self._cosine_similarity(
                query_embedding,
                section_embeddings
            )

            for name, score in zip(section_names, similarities):
                rankings[name] = float(score)

        return rankings

    def _select_sections_within_budget(
        self,
        rankings: dict[str, float],
        sections: dict[str, str],
        budget: int,
        max_sections: int
    ) -> list[str]:
        """
        Select top sections within token budget.

        Strategy: Greedy selection by relevance score.

        Args:
            rankings: Section name -> relevance score
            sections: Section name -> content
            budget: Maximum total tokens
            max_sections: Maximum number of sections

        Returns:
            List of selected section names
        """
        # Sort by relevance score (descending)
        sorted_sections = sorted(
            rankings.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected = []
        total_tokens = 0

        for section_name, score in sorted_sections:
            if len(selected) >= max_sections:
                break

            section_content = sections.get(section_name, "")
            section_tokens = count_tokens(section_content)

            # Check if adding this section exceeds budget
            if total_tokens + section_tokens > budget:
                # Try to fit partially (truncate to remaining budget)
                remaining = budget - total_tokens
                if remaining > 500:  # Minimum viable chunk
                    # Will be handled by chunker
                    selected.append(section_name)
                break

            selected.append(section_name)
            total_tokens += section_tokens

        return selected

    @staticmethod
    def _create_section_preview(section_name: str, content: str, max_words: int = None) -> str:
        """Create preview of section for embedding (title + beginning)."""
        if max_words is None:
            max_words = SMART_MODE_CONFIG.get("section_preview_words", 200)

        words = content.split()[:max_words]
        preview = " ".join(words)
        return f"{section_name}: {preview}"

    @staticmethod
    def _extract_opening_content(text: str, max_words: int = 500) -> str:
        """Extract opening paragraphs when no abstract available."""
        paragraphs = text.split("\n\n")
        content_words = []

        for para in paragraphs:
            content_words.extend(para.split())
            if len(content_words) >= max_words:
                break

        return " ".join(content_words[:max_words])

    @staticmethod
    def _cosine_similarity(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents."""
        # Normalize
        query_norm = query / np.linalg.norm(query)
        docs_norm = documents / np.linalg.norm(documents, axis=1, keepdims=True)

        # Dot product
        similarities = np.dot(docs_norm, query_norm)
        return similarities


# Convenience function for backward compatibility
def process_paper_smart(
    paper: PaperContent,
    mode: ProcessingMode = "deep",
    **kwargs
) -> list[Chunk]:
    """
    Process paper with specified mode.

    Drop-in replacement for chunk_by_section() with mode support.
    """
    config = ProcessingConfig(mode=mode, **kwargs)
    processor = SmartPaperProcessor(config)
    return processor.process(paper)
