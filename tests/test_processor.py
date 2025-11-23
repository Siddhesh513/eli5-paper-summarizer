"""
Unit tests for ELI5 Paper Summarizer.
Run with: pytest tests/ -v
"""
import pytest
from src.pdf_processor import extract_arxiv_id, detect_sections
from src.chunker import count_tokens, chunk_text, chunk_by_section


class TestArxivIdExtraction:
    """Tests for arXiv ID extraction."""
    
    def test_extract_from_abs_url(self):
        url = "https://arxiv.org/abs/2301.00001"
        assert extract_arxiv_id(url) == "2301.00001"
    
    def test_extract_from_pdf_url(self):
        url = "https://arxiv.org/pdf/2301.00001.pdf"
        assert extract_arxiv_id(url) == "2301.00001"
    
    def test_extract_from_id_only(self):
        assert extract_arxiv_id("2301.00001") == "2301.00001"
    
    def test_extract_from_id_with_version(self):
        assert extract_arxiv_id("2301.00001v2") == "2301.00001"
    
    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            extract_arxiv_id("not-a-valid-url")


class TestSectionDetection:
    """Tests for section detection."""
    
    def test_detect_introduction(self):
        text = "Some preamble\n\n1. Introduction\n\nThis is the intro."
        sections = detect_sections(text)
        assert "Introduction" in sections
        assert "This is the intro." in sections["Introduction"]
    
    def test_detect_multiple_sections(self):
        text = """Abstract

This is abstract.

1. Introduction

This is intro.

2. Methods

These are methods.

3. Conclusion

This is conclusion."""
        
        sections = detect_sections(text)
        assert "Abstract" in sections
        assert "Introduction" in sections
        assert "Methods" in sections
        assert "Conclusion" in sections
    
    def test_case_insensitive(self):
        text = "INTRODUCTION\n\nContent here"
        sections = detect_sections(text)
        assert "Introduction" in sections


class TestChunker:
    """Tests for chunking functionality."""
    
    def test_count_tokens(self):
        text = "Hello, world!"
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < 10  # Should be around 3-4 tokens
    
    def test_chunk_short_text(self):
        text = "This is a short text."
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) >= 1
    
    def test_chunk_long_text(self):
        text = "This is a sentence. " * 500
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) > 1
    
    def test_chunk_by_section_preserves_sections(self):
        sections = {
            "Abstract": "This is the abstract.",
            "Introduction": "This is the introduction. " * 10,
            "Conclusion": "This is the conclusion.",
        }
        
        chunks = chunk_by_section(sections)
        
        # Check all sections are represented
        chunk_sections = set(c.section for c in chunks)
        assert "Abstract" in chunk_sections
        assert "Introduction" in chunk_sections
        assert "Conclusion" in chunk_sections
    
    def test_chunk_metadata(self):
        sections = {"Abstract": "Test content"}
        chunks = chunk_by_section(sections)
        
        assert len(chunks) == 1
        assert chunks[0].section == "Abstract"
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks_in_section == 1


class TestEmbeddings:
    """Tests for vector store."""
    
    def test_create_retriever(self):
        from src.embeddings import create_retriever
        
        texts = ["Test document about AI.", "Another document about ML."]
        metadatas = [{"section": "Intro"}, {"section": "Methods"}]
        
        store = create_retriever(texts, metadatas, "Test Paper")
        
        # Should be able to retrieve
        results = store.retrieve("artificial intelligence", k=1)
        assert len(results) >= 1
    
    def test_get_all_chunks(self):
        from src.embeddings import create_retriever
        
        texts = ["Doc 1", "Doc 2", "Doc 3"]
        metadatas = [{"id": 1}, {"id": 2}, {"id": 3}]
        
        store = create_retriever(texts, metadatas, "Test")
        all_chunks = store.get_all_chunks()
        
        assert len(all_chunks) == 3


class TestIntegration:
    """Integration tests (require API key)."""
    
    @pytest.mark.skip(reason="Requires API key and network")
    def test_full_pipeline_arxiv(self):
        from src.pdf_processor import process_paper
        from src.chunker import chunk_by_section, prepare_chunks_for_embedding
        from src.summarizer import summarize_paper
        
        paper = process_paper("2301.00001")
        assert paper.title
        assert paper.sections
        
        chunks = chunk_by_section(paper.sections)
        assert len(chunks) > 0
        
        texts, metadatas = prepare_chunks_for_embedding(chunks)
        assert len(texts) == len(metadatas)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
