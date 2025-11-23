"""
Summarization Engine.
LangChain-based summary generation with three detail levels.
Uses Groq (free) by default.
"""
from typing import Optional
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.settings import LLM_MODEL, LLM_PROVIDER, GROQ_API_KEY, MAX_CHUNKS_FOR_DIRECT
from src.prompts import (
    TECHNICAL_SUMMARY_PROMPT,
    SIMPLIFIED_SUMMARY_PROMPT,
    ELI5_SUMMARY_PROMPT,
    KEY_FINDINGS_PROMPT,
    MAP_PROMPT,
    REDUCE_PROMPT,
    STYLE_INSTRUCTIONS,
)
from src.chunker import count_tokens


def get_llm(temperature: float = 0.3):
    """
    Get LLM - uses Groq (free) by default.
    """
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=LLM_MODEL,
        temperature=temperature,
        groq_api_key=GROQ_API_KEY,
    )


@dataclass
class SummaryResult:
    """Container for generated summaries."""
    technical: str
    simplified: str
    eli5: str
    key_findings: str
    token_count: int
    chunks_used: int


class PaperSummarizer:
    """
    Multi-level paper summarizer using LangChain.
    """
    
    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = 0.3
    ):
        """
        Initialize summarizer.
        
        Args:
            model: LLM model name
            temperature: Generation temperature (lower = more focused)
        """
        self.llm = get_llm(temperature)
        self.output_parser = StrOutputParser()
        
        # Create chains for each summary level
        self.technical_chain = self._create_chain(TECHNICAL_SUMMARY_PROMPT)
        self.simplified_chain = self._create_chain(SIMPLIFIED_SUMMARY_PROMPT)
        self.eli5_chain = self._create_chain(ELI5_SUMMARY_PROMPT)
        self.key_findings_chain = self._create_chain(KEY_FINDINGS_PROMPT)
    
    def _create_chain(self, prompt_template: str):
        """Create a LangChain LCEL chain."""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.llm | self.output_parser
    
    def _prepare_context(
        self,
        chunks: list[dict],
        max_tokens: int = 12000
    ) -> str:
        """
        Prepare context from chunks, respecting token limits.
        
        Args:
            chunks: List of chunk dicts with 'content' and 'metadata'
            max_tokens: Maximum context tokens
        
        Returns:
            Concatenated context string
        """
        context_parts = []
        current_tokens = 0
        
        for chunk in chunks:
            content = chunk["content"]
            chunk_tokens = count_tokens(content)
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            section = chunk.get("metadata", {}).get("section", "Content")
            context_parts.append(f"### {section}\n{content}")
            current_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)
    
    def summarize(
        self,
        chunks: list[dict],
        use_map_reduce: bool = False
    ) -> SummaryResult:
        """
        Generate all three summary levels.
        
        Args:
            chunks: List of chunk dicts from vector store
            use_map_reduce: Force map-reduce for long documents
        
        Returns:
            SummaryResult with all three summaries
        """
        # Decide on strategy
        if len(chunks) > MAX_CHUNKS_FOR_DIRECT or use_map_reduce:
            return self._summarize_map_reduce(chunks)
        else:
            return self._summarize_direct(chunks)
    
    def _summarize_direct(self, chunks: list[dict]) -> SummaryResult:
        """Direct summarization for shorter papers."""
        context = self._prepare_context(chunks)
        
        # Generate all summaries including key findings
        technical = self.technical_chain.invoke({"context": context})
        simplified = self.simplified_chain.invoke({"context": context})
        eli5 = self.eli5_chain.invoke({"context": context})
        key_findings = self.key_findings_chain.invoke({"context": context})
        
        return SummaryResult(
            technical=technical.strip(),
            simplified=simplified.strip(),
            eli5=eli5.strip(),
            key_findings=key_findings.strip(),
            token_count=count_tokens(context),
            chunks_used=len(chunks),
        )
    
    def _summarize_map_reduce(self, chunks: list[dict]) -> SummaryResult:
        """Map-reduce summarization for longer papers."""
        # Group chunks by section
        sections = {}
        for chunk in chunks:
            section = chunk.get("metadata", {}).get("section", "Content")
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk["content"])
        
        # Map phase: summarize each section
        section_summaries = []
        map_prompt = ChatPromptTemplate.from_template(MAP_PROMPT)
        map_chain = map_prompt | self.llm | self.output_parser
        
        for section_name, contents in sections.items():
            combined_content = "\n\n".join(contents)
            
            # Skip very short sections
            if count_tokens(combined_content) < 50:
                continue
            
            summary = map_chain.invoke({
                "section_name": section_name,
                "content": combined_content[:8000],  # Truncate if needed
            })
            section_summaries.append(f"**{section_name}**: {summary}")
        
        combined_summaries = "\n\n".join(section_summaries)
        
        # Reduce phase: create final summaries
        reduce_prompt = ChatPromptTemplate.from_template(REDUCE_PROMPT)
        reduce_chain = reduce_prompt | self.llm | self.output_parser
        
        technical = reduce_chain.invoke({
            "summaries": combined_summaries,
            "summary_type": "TECHNICAL",
            "style_instructions": STYLE_INSTRUCTIONS["technical"],
        })
        
        simplified = reduce_chain.invoke({
            "summaries": combined_summaries,
            "summary_type": "SIMPLIFIED",
            "style_instructions": STYLE_INSTRUCTIONS["simplified"],
        })
        
        eli5 = reduce_chain.invoke({
            "summaries": combined_summaries,
            "summary_type": "ELI5",
            "style_instructions": STYLE_INSTRUCTIONS["eli5"],
        })
        
        key_findings = reduce_chain.invoke({
            "summaries": combined_summaries,
            "summary_type": "KEY FINDINGS",
            "style_instructions": "Extract key contributions, results, and limitations in bullet format.",
        })
        
        total_tokens = sum(count_tokens(c["content"]) for c in chunks)
        
        return SummaryResult(
            technical=technical.strip(),
            simplified=simplified.strip(),
            eli5=eli5.strip(),
            key_findings=key_findings.strip(),
            token_count=total_tokens,
            chunks_used=len(chunks),
        )
    
    def summarize_single(
        self,
        chunks: list[dict],
        level: str = "technical"
    ) -> str:
        """
        Generate a single summary level.
        
        Args:
            chunks: List of chunk dicts
            level: One of 'technical', 'simplified', 'eli5'
        
        Returns:
            Summary string
        """
        context = self._prepare_context(chunks)
        
        chains = {
            "technical": self.technical_chain,
            "simplified": self.simplified_chain,
            "eli5": self.eli5_chain,
        }
        
        chain = chains.get(level, self.technical_chain)
        return chain.invoke({"context": context}).strip()


def summarize_paper(
    chunks: list[dict],
    model: str = LLM_MODEL
) -> SummaryResult:
    """
    Convenience function to summarize a paper.
    
    Args:
        chunks: List of chunk dicts from vector store or chunker
        model: LLM model to use
    
    Returns:
        SummaryResult with all summaries
    """
    summarizer = PaperSummarizer(model=model)
    return summarizer.summarize(chunks)


if __name__ == "__main__":
    # Quick test with mock data
    test_chunks = [
        {
            "content": "[Abstract] We present a novel transformer architecture for document understanding that achieves state-of-the-art results on multiple benchmarks.",
            "metadata": {"section": "Abstract"}
        },
        {
            "content": "[Introduction] Document understanding is a critical task in NLP. Previous approaches have limitations in handling long documents.",
            "metadata": {"section": "Introduction"}
        },
        {
            "content": "[Methods] Our DocTransformer uses hierarchical attention with O(n log n) complexity. We introduce sparse attention patterns.",
            "metadata": {"section": "Methods"}
        },
        {
            "content": "[Results] DocTransformer achieves 94.2% accuracy on DocVQA, 91.5% on FUNSD, outperforming LayoutLMv3 by 3.2%.",
            "metadata": {"section": "Results"}
        },
        {
            "content": "[Conclusion] We demonstrated that hierarchical attention improves document understanding. Future work will explore multilingual documents.",
            "metadata": {"section": "Conclusion"}
        },
    ]
    
    print("Testing summarizer (requires OPENAI_API_KEY)...")
    try:
        result = summarize_paper(test_chunks)
        print("\n=== TECHNICAL ===")
        print(result.technical[:500] + "...")
        print("\n=== SIMPLIFIED ===")
        print(result.simplified[:500] + "...")
        print("\n=== ELI5 ===")
        print(result.eli5)
        print(f"\nTokens used: {result.token_count}, Chunks: {result.chunks_used}")
    except Exception as e:
        print(f"Test failed (expected if no API key): {e}")
