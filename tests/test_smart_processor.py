"""
Tests for Smart Paper Processor.
"""
import pytest
from src.smart_processor import SmartPaperProcessor, ProcessingConfig
from src.pdf_processor import PaperContent
from src.chunker import Chunk


@pytest.fixture
def sample_paper():
    """Create a sample paper for testing."""
    return PaperContent(
        title="Test Paper on Machine Learning",
        authors=["Test Author"],
        abstract="This paper presents a novel approach to machine learning using neural networks. We demonstrate improved performance on benchmark datasets.",
        full_text="Abstract\nThis paper presents a novel approach...\n\nIntroduction\nMachine learning has become...\n\nMethods\nWe used a neural network...\n\nResults\nOur experiments show...\n\nConclusion\nIn conclusion...",
        sections={
            "Abstract": "This paper presents a novel approach to machine learning using neural networks. We demonstrate improved performance on benchmark datasets.",
            "Introduction": "Machine learning has become increasingly important in recent years. " * 50,
            "Methods": "We used a neural network architecture with three hidden layers. The model was trained using backpropagation. " * 50,
            "Results": "Our experiments show that the proposed method achieves 95% accuracy on the test set. This represents a 5% improvement over baseline. " * 50,
            "Conclusion": "In conclusion, we have presented a novel approach to machine learning that demonstrates superior performance. Future work will explore additional architectures. " * 30,
            "References": "1. Smith et al. (2020)\n2. Jones et al. (2021)\n" * 20,
        }
    )


@pytest.fixture
def minimal_paper():
    """Create a minimal paper with just abstract."""
    return PaperContent(
        title="Minimal Paper",
        authors=["Minimal Author"],
        abstract="This is a very short abstract.",
        full_text="This is a very short abstract.",
        sections={"Abstract": "This is a very short abstract."}
    )


class TestSmartProcessor:
    """Test suite for SmartPaperProcessor."""

    def test_quick_mode_with_abstract(self, sample_paper):
        """Test Quick mode extracts abstract correctly."""
        config = ProcessingConfig(mode="quick")
        processor = SmartPaperProcessor(config)

        chunks = processor.process(sample_paper)

        # Should return exactly one chunk
        assert len(chunks) == 1
        assert chunks[0].section == "Abstract"
        assert chunks[0].content == sample_paper.abstract.strip()
        assert chunks[0].metadata["processing_mode"] == "quick"
        assert chunks[0].token_count < 600  # Within quick mode budget

    def test_quick_mode_fallback(self):
        """Test fallback when no abstract available."""
        paper_no_abstract = PaperContent(
            title="No Abstract Paper",
            authors=["Test"],
            abstract="",
            full_text="This is the start of a paper without an abstract section. " * 100,
            sections={}
        )

        config = ProcessingConfig(mode="quick")
        processor = SmartPaperProcessor(config)

        chunks = processor.process(paper_no_abstract)

        # Should still return a chunk with opening content
        assert len(chunks) == 1
        assert chunks[0].section == "Abstract"
        assert len(chunks[0].content) > 0

    def test_smart_mode_section_ranking(self, sample_paper):
        """Test that sections are ranked by relevance."""
        config = ProcessingConfig(mode="smart")
        processor = SmartPaperProcessor(config)

        chunks = processor.process(sample_paper)

        # Should have abstract + some sections
        assert len(chunks) > 1

        # First chunk should be abstract
        assert chunks[0].section == "Abstract"
        assert chunks[0].metadata["processing_mode"] == "smart"

        # Check that relevance scores are present
        for chunk in chunks:
            assert "relevance_score" in chunk.metadata
            assert 0 <= chunk.metadata["relevance_score"] <= 1

        # References should not be included (filtered out)
        section_names = {chunk.section for chunk in chunks}
        assert "References" not in section_names

    def test_smart_mode_budget_enforcement(self, sample_paper):
        """Test token budget is respected."""
        config = ProcessingConfig(mode="smart")
        processor = SmartPaperProcessor(config)

        chunks = processor.process(sample_paper)

        # Calculate total tokens
        total_tokens = sum(chunk.token_count for chunk in chunks)

        # Should be within budget (with some tolerance for chunking overhead)
        from config.settings import TOKEN_BUDGETS
        budget = TOKEN_BUDGETS["smart"]
        assert total_tokens <= budget * 1.2  # 20% tolerance for chunking

    def test_deep_mode_compatibility(self, sample_paper):
        """Verify Deep mode produces same output as original."""
        config = ProcessingConfig(mode="deep")
        processor = SmartPaperProcessor(config)

        chunks = processor.process(sample_paper)

        # Should process all sections
        assert len(chunks) > 0

        # All chunks should have deep mode metadata
        for chunk in chunks:
            assert chunk.metadata["processing_mode"] == "deep"

        # Should include most sections (except maybe preamble)
        section_names = {chunk.section for chunk in chunks}
        assert "Introduction" in section_names
        assert "Methods" in section_names
        assert "Results" in section_names

    def test_chunk_structure_compatibility(self, sample_paper):
        """Ensure all modes return compatible Chunk objects."""
        modes = ["quick", "smart", "deep"]

        for mode in modes:
            config = ProcessingConfig(mode=mode)
            processor = SmartPaperProcessor(config)
            chunks = processor.process(sample_paper)

            # Verify all chunks are Chunk objects
            assert all(isinstance(c, Chunk) for c in chunks)

            # Verify required fields
            for chunk in chunks:
                assert isinstance(chunk.content, str)
                assert isinstance(chunk.section, str)
                assert isinstance(chunk.chunk_index, int)
                assert isinstance(chunk.total_chunks_in_section, int)
                assert isinstance(chunk.token_count, int)
                assert isinstance(chunk.metadata, dict)
                assert "section" in chunk.metadata
                assert "processing_mode" in chunk.metadata

    def test_graceful_degradation(self):
        """Test Smart mode falls back to Deep on errors."""
        # Create paper that might cause issues with smart processing
        problematic_paper = PaperContent(
            title="Problematic Paper",
            authors=["Test"],
            abstract="",  # No abstract
            full_text="Some text",
            sections={}  # No sections
        )

        config = ProcessingConfig(mode="smart")
        processor = SmartPaperProcessor(config)

        # Should not raise exception, should fall back gracefully
        chunks = processor.process(problematic_paper)

        # Should return some chunks (from fallback)
        assert len(chunks) > 0

    def test_mode_token_usage_differences(self, sample_paper):
        """Test that different modes use different amounts of tokens."""
        quick_config = ProcessingConfig(mode="quick")
        smart_config = ProcessingConfig(mode="smart")
        deep_config = ProcessingConfig(mode="deep")

        quick_processor = SmartPaperProcessor(quick_config)
        smart_processor = SmartPaperProcessor(smart_config)
        deep_processor = SmartPaperProcessor(deep_config)

        quick_chunks = quick_processor.process(sample_paper)
        smart_chunks = smart_processor.process(sample_paper)
        deep_chunks = deep_processor.process(sample_paper)

        quick_tokens = sum(c.token_count for c in quick_chunks)
        smart_tokens = sum(c.token_count for c in smart_chunks)
        deep_tokens = sum(c.token_count for c in deep_chunks)

        # Quick should use fewer tokens than Smart
        assert quick_tokens < smart_tokens

        # Smart should use fewer tokens than Deep (or equal if paper is small)
        assert smart_tokens <= deep_tokens

    def test_section_preview_creation(self):
        """Test section preview generation for embedding."""
        processor = SmartPaperProcessor()

        section_name = "Introduction"
        content = "This is a test section. " * 100  # Long content

        preview = processor._create_section_preview(section_name, content, max_words=10)

        # Should include section name
        assert section_name in preview

        # Should be limited in length
        words = preview.split()
        assert len(words) <= 15  # section name + max_words

    def test_cosine_similarity_computation(self):
        """Test cosine similarity calculation."""
        import numpy as np
        processor = SmartPaperProcessor()

        # Test vectors
        query = np.array([1.0, 0.0, 0.0])
        documents = np.array([
            [1.0, 0.0, 0.0],  # Identical
            [0.0, 1.0, 0.0],  # Orthogonal
            [0.7, 0.7, 0.0],  # Similar
        ])

        similarities = processor._cosine_similarity(query, documents)

        # Check similarity values
        assert similarities[0] > 0.99  # Nearly identical
        assert abs(similarities[1]) < 0.01  # Nearly orthogonal
        assert 0.5 < similarities[2] < 0.9  # Moderately similar

    def test_opening_content_extraction(self):
        """Test extraction of opening content when no abstract."""
        processor = SmartPaperProcessor()

        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        opening = processor._extract_opening_content(text, max_words=5)

        # Should extract limited words
        words = opening.split()
        assert len(words) == 5

    def test_minimal_paper_handling(self, minimal_paper):
        """Test handling of very minimal papers."""
        modes = ["quick", "smart", "deep"]

        for mode in modes:
            config = ProcessingConfig(mode=mode)
            processor = SmartPaperProcessor(config)

            # Should not raise exception
            chunks = processor.process(minimal_paper)

            # Should return at least one chunk
            assert len(chunks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
