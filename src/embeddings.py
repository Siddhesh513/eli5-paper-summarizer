"""
Embeddings and Vector Store Module.
ChromaDB integration for semantic retrieval.
Supports both OpenAI (paid) and HuggingFace (free) embeddings.
"""
import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma

from config.settings import (
    CHROMA_PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, 
    OPENAI_API_KEY, EMBEDDING_PROVIDER
)


def get_embeddings():
    """
    Get embeddings model based on configuration.
    Returns HuggingFace (free) or OpenAI embeddings.
    """
    if EMBEDDING_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
    else:
        # Free local embeddings using HuggingFace
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,  # Default: "all-MiniLM-L6-v2"
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )


class PaperVectorStore:
    """
    Vector store for paper chunks using ChromaDB.
    """
    
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME
    ):
        """
        Initialize vector store.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the collection
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self._vectorstore: Optional[Chroma] = None
        self._current_paper_id: Optional[str] = None
    
    def _get_paper_id(self, title: str) -> str:
        """Generate unique ID for a paper based on title."""
        return hashlib.md5(title.encode()).hexdigest()[:12]
    
    def embed_paper(
        self,
        texts: list[str],
        metadatas: list[dict],
        paper_title: str
    ) -> str:
        """
        Embed paper chunks into vector store.
        
        Args:
            texts: List of chunk texts
            metadatas: List of chunk metadata dicts
            paper_title: Title of the paper (for ID generation)
        
        Returns:
            Paper ID
        """
        paper_id = self._get_paper_id(paper_title)
        
        # Add paper_id to all metadata
        for meta in metadatas:
            meta["paper_id"] = paper_id
        
        # Create new collection for this paper
        collection_name = f"{self.collection_name}_{paper_id}"
        
        self._vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name=collection_name,
            persist_directory=self.persist_dir,
        )
        
        self._current_paper_id = paper_id
        return paper_id
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_section: Optional[str] = None
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_section: Optional section name to filter by
        
        Returns:
            List of dicts with 'content' and 'metadata' keys
        """
        if self._vectorstore is None:
            raise ValueError("No paper embedded. Call embed_paper first.")
        
        # Build filter if section specified
        where_filter = None
        if filter_section:
            where_filter = {"section": filter_section}
        
        results = self._vectorstore.similarity_search(
            query,
            k=k,
            filter=where_filter,
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]
    
    def retrieve_by_sections(
        self,
        queries: dict[str, str],
        k_per_section: int = 2
    ) -> list[dict]:
        """
        Retrieve chunks using section-specific queries.
        
        Args:
            queries: Dict mapping section names to queries
            k_per_section: Results per section
        
        Returns:
            Combined list of relevant chunks
        """
        all_results = []
        seen_contents = set()
        
        for section, query in queries.items():
            results = self.retrieve(
                query=query,
                k=k_per_section,
                filter_section=section,
            )
            
            for result in results:
                # Deduplicate
                content_hash = hash(result["content"][:100])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_results.append(result)
        
        return all_results
    
    def get_all_chunks(self) -> list[dict]:
        """
        Retrieve all chunks from the current paper.
        
        Returns:
            List of all chunks with metadata
        """
        if self._vectorstore is None:
            raise ValueError("No paper embedded. Call embed_paper first.")
        
        # Use a broad query to get all documents
        results = self._vectorstore.similarity_search(
            "paper content research study",
            k=100,  # Get all chunks
        )
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]
    
    def clear(self):
        """Clear the current vector store."""
        self._vectorstore = None
        self._current_paper_id = None


def create_retriever(
    texts: list[str],
    metadatas: list[dict],
    paper_title: str
) -> PaperVectorStore:
    """
    Convenience function to create and populate a vector store.
    
    Args:
        texts: List of chunk texts
        metadatas: List of chunk metadata
        paper_title: Title of the paper
    
    Returns:
        Populated PaperVectorStore instance
    """
    store = PaperVectorStore()
    store.embed_paper(texts, metadatas, paper_title)
    return store


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "[Abstract] This paper presents a novel approach to machine learning.",
        "[Introduction] Machine learning has revolutionized many fields.",
        "[Methods] We use a transformer architecture with attention mechanisms.",
        "[Results] Our model achieves 95% accuracy on the benchmark.",
        "[Conclusion] We have demonstrated the effectiveness of our approach.",
    ]
    
    test_metadatas = [
        {"section": "Abstract", "chunk_id": 0},
        {"section": "Introduction", "chunk_id": 1},
        {"section": "Methods", "chunk_id": 2},
        {"section": "Results", "chunk_id": 3},
        {"section": "Conclusion", "chunk_id": 4},
    ]
    
    print("Testing vector store...")
    store = create_retriever(test_texts, test_metadatas, "Test Paper")
    
    results = store.retrieve("What is the accuracy?", k=2)
    print(f"\nQuery: 'What is the accuracy?'")
    for r in results:
        print(f"  [{r['metadata']['section']}] {r['content'][:50]}...")
    
    print("\nVector store test complete!")
