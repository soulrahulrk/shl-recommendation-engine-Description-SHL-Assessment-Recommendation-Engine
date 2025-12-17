"""
Phase 2: Embeddings + Vector Index
Build semantic search capability using sentence-transformers and FAISS

Embedding strategy:
- Combine: test_name + test_types + description + job_levels
- Model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
- Index: FAISS IndexFlatIP (inner product for cosine similarity with normalized vectors)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class SHLVectorStore:
    """Semantic vector store for SHL assessments"""
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []
        self.url_to_idx: dict[str, int] = {}
        
    def _load_model(self):
        """Load the embedding model"""
        if self.model is None:
            print(f"Loading embedding model: {self.MODEL_NAME}")
            self.model = SentenceTransformer(self.MODEL_NAME)
        return self.model
    
    def _create_embedding_text(self, test: dict) -> str:
        """Create the text to embed for a test"""
        parts = []
        
        # Test name (most important)
        if test.get("test_name"):
            parts.append(f"Assessment: {test['test_name']}")
        
        # Test types (important for matching)
        if test.get("test_types"):
            types_str = ", ".join(test["test_types"])
            parts.append(f"Test Types: {types_str}")
        
        # Category
        if test.get("category"):
            parts.append(f"Category: {test['category']}")
        
        # Description (rich semantic content)
        if test.get("description"):
            parts.append(f"Description: {test['description']}")
        
        # Job levels (context)
        if test.get("job_levels"):
            parts.append(f"Job Levels: {test['job_levels']}")
        
        return " | ".join(parts)
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for consistent matching"""
        url = url.strip()
        url = url.replace("/solutions/products/product-catalog/", "/products/product-catalog/")
        if not url.endswith("/"):
            url += "/"
        return url
    
    def build_index(self, catalog_files: list[str]):
        """Build FAISS index from catalog files"""
        print("=" * 60)
        print("Building Vector Index")
        print("=" * 60)
        
        # Load model
        model = self._load_model()
        
        # Load all tests from catalog files
        all_tests = []
        for file_path in catalog_files:
            path = self.data_dir / file_path
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    tests = data.get("tests", data) if isinstance(data, dict) else data
                    all_tests.extend(tests)
                    print(f"  Loaded {len(tests)} tests from {file_path}")
        
        print(f"\nTotal tests to index: {len(all_tests)}")
        
        # Deduplicate by URL
        seen_urls = set()
        unique_tests = []
        for test in all_tests:
            url = self._normalize_url(test.get("url", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_tests.append(test)
        
        print(f"Unique tests after deduplication: {len(unique_tests)}")
        
        # Create embedding texts
        print("\nCreating embedding texts...")
        texts = []
        for test in unique_tests:
            text = self._create_embedding_text(test)
            texts.append(text)
        
        # Generate embeddings
        print(f"\nGenerating embeddings with {self.MODEL_NAME}...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity via inner product
        )
        
        print(f"Embedding shape: {embeddings.shape}")
        
        # Build FAISS index
        print("\nBuilding FAISS index...")
        self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)  # Inner product (cosine with normalized vectors)
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.metadata = unique_tests
        self.url_to_idx = {self._normalize_url(t["url"]): i for i, t in enumerate(unique_tests)}
        
        print(f"Index built with {self.index.ntotal} vectors")
        
        return self
    
    def save(self, index_file: str = "vector_store.faiss", metadata_file: str = "metadata.pkl"):
        """Save index and metadata to disk"""
        index_path = self.data_dir / index_file
        metadata_path = self.data_dir / metadata_file
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        print(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "url_to_idx": self.url_to_idx,
                "model_name": self.MODEL_NAME,
                "embedding_dim": self.EMBEDDING_DIM
            }, f)
        print(f"Saved metadata to {metadata_path}")
        
        return self
    
    def load(self, index_file: str = "vector_store.faiss", metadata_file: str = "metadata.pkl"):
        """Load index and metadata from disk"""
        index_path = self.data_dir / index_file
        metadata_path = self.data_dir / metadata_file
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")
        
        # Load metadata
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.url_to_idx = data["url_to_idx"]
        print(f"Loaded metadata: {len(self.metadata)} tests")
        
        # Load model
        self._load_model()
        
        return self
    
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Semantic search for assessments matching a query
        
        Args:
            query: Natural language query or job description
            top_k: Number of results to return
            
        Returns:
            List of test dictionaries with similarity scores
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load() or build_index() first.")
        
        # Embed query
        model = self._load_model()
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                test = self.metadata[idx].copy()
                test["similarity_score"] = float(score)
                results.append(test)
        
        return results


def semantic_search(query: str, top_k: int = 10, data_dir: str = "data") -> list[dict]:
    """
    Convenience function for semantic search
    
    Args:
        query: Natural language query
        top_k: Number of results
        data_dir: Directory containing index files
        
    Returns:
        List of matching assessments with scores
    """
    store = SHLVectorStore(data_dir=data_dir)
    store.load()
    return store.search(query, top_k)


def main():
    """Build and test the vector index"""
    store = SHLVectorStore(data_dir="data")
    
    # Build index from enriched catalogs
    store.build_index([
        "catalog_enriched.json",
        "prepackaged_enriched.json"
    ])
    
    # Save to disk
    store.save()
    
    # Test queries
    print("\n" + "=" * 60)
    print("VALIDATION: Testing Semantic Search")
    print("=" * 60)
    
    test_queries = [
        "Java developer who can collaborate with business teams",
        "Entry level sales position requiring communication skills",
        "Python and SQL programming skills assessment",
        "Leadership and management assessment for executives",
        "Customer service skills for retail"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query[:60]}... ---")
        results = store.search(query, top_k=5)
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['similarity_score']:.3f}] {r['test_name']}")
            print(f"      Types: {r.get('test_types', [])}")


if __name__ == "__main__":
    main()
