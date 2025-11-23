"""
Embeddings and Vector Store Module.
ChromaDB integration for semantic retrieval.
Uses ChromaDB's default embeddings (no external dependencies).
"""
import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings

from config.settings import CHROMA_PERSIST_DIR, COLLECTION_NAME


class PaperVectorStore:
    """
    Vector store for paper chunks using ChromaDB.
    Uses ChromaDB's built-in default embedding function.
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
        self.client = chromadb.Client()
        self._collection = None
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
        
        # Create collection (ChromaDB uses default embedding)
        collection_name = f"{self.collection_name}_{paper_id}"
        
        # Delete if exists
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self._collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents
        ids = [f"chunk_{i}" for i in range(len(texts))]
        self._collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
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
        if self._collection is None:
            raise ValueError("No paper embedded. Call embed_paper first.")
        
        # Build filter if section specified
        where_filter = None
        if filter_section:
            where_filter = {"section": filter_section}
        
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
        )
        
        chunks = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                chunks.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                })
        
        return chunks
    
    def get_all_chunks(self) -> list[dict]:
        """
        Retrieve all chunks from the current paper.
        
        Returns:
            List of all chunks with metadata
        """
        if self._collection is None:
            raise ValueError("No paper embedded. Call embed_paper first.")
        
        # Get all documents
        results = self._collection.get()
        
        chunks = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents']):
                chunks.append({
                    "content": doc,
                    "metadata": results['metadatas'][i] if results['metadatas'] else {},
                })
        
        return chunks
    
    def clear(self):
        """Clear the current vector store."""
        self._collection = None
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
    

