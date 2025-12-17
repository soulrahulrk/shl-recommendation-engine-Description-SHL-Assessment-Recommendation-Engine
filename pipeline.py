"""
Recommendation Pipeline
Combines semantic search with LLM re-ranking for optimal results

Flow:
1. Query → Vector Store → Top-K candidates (semantic)
2. Candidates → LLM Reranker → Final Top-10 (balanced)
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "llm"))

from vector_store import SHLVectorStore
from rerank import LLMReranker


class RecommendationPipeline:
    """End-to-end recommendation pipeline"""
    
    def __init__(
        self, 
        data_dir: Optional[str] = None,
        use_llm: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialize the pipeline
        
        Args:
            data_dir: Directory containing vector store and metadata
            use_llm: Whether to use LLM re-ranking (False = semantic only)
            api_key: Optional Gemini API key for LLM re-ranking
        """
        self.data_dir = data_dir or str(Path(__file__).parent / "data")
        self.use_llm = use_llm
        
        # Initialize vector store
        self.vector_store = SHLVectorStore(data_dir=self.data_dir)
        self.vector_store.load()
        print(f"✓ Loaded vector store with {len(self.vector_store.metadata)} assessments")
        
        # Initialize LLM reranker if enabled
        self.reranker = None
        if use_llm:
            try:
                self.reranker = LLMReranker(api_key=api_key)
                print("✓ LLM reranker initialized (Groq Llama-3.3-70B)")
            except ValueError as e:
                print(f"⚠ LLM reranker not available: {e}")
                print("  Falling back to semantic-only mode")
                self.use_llm = False
    
    def recommend(
        self, 
        query: str,
        top_k: int = 10,
        candidate_pool: int = 20
    ) -> dict:
        """
        Get assessment recommendations for a query
        
        Args:
            query: Job description or hiring requirements
            top_k: Number of final recommendations
            candidate_pool: Number of candidates to consider from semantic search
            
        Returns:
            Dictionary with recommendations and metadata
        """
        # Step 1: Semantic search for candidates
        candidates = self.vector_store.search(query, top_k=candidate_pool)
        
        result = {
            "query": query,
            "method": "semantic_only",
            "candidates_considered": len(candidates),
            "recommendations": []
        }
        
        # Step 2: LLM re-ranking (if enabled)
        if self.use_llm and self.reranker:
            llm_result = self.reranker.rerank(query, candidates, top_k=top_k)
            
            if llm_result.get("recommendations"):
                result["method"] = "semantic+llm_rerank"
                result["recommendations"] = llm_result["recommendations"]
                result["balance_summary"] = llm_result.get("balance_summary", "")
            else:
                # Fallback to semantic
                result["llm_error"] = llm_result.get("error", "Unknown error")
                result["recommendations"] = self._format_semantic_results(candidates[:top_k])
        else:
            result["recommendations"] = self._format_semantic_results(candidates[:top_k])
        
        return result
    
    def _format_semantic_results(self, candidates: list[dict]) -> list[dict]:
        """Format semantic search results as recommendations"""
        return [
            {
                "rank": i + 1,
                "assessment_name": c["test_name"],
                "url": c["url"],
                "test_types": c.get("test_types", []),
                "duration": c.get("duration"),
                "reason": f"High semantic similarity ({c.get('similarity_score', 0):.2f})"
            }
            for i, c in enumerate(candidates)
        ]
    
    def recommend_batch(
        self,
        queries: list[str],
        top_k: int = 10
    ) -> list[dict]:
        """
        Get recommendations for multiple queries
        
        Args:
            queries: List of job descriptions
            top_k: Number of recommendations per query
            
        Returns:
            List of recommendation results
        """
        results = []
        for i, query in enumerate(queries):
            print(f"Processing query {i+1}/{len(queries)}...")
            result = self.recommend(query, top_k=top_k)
            results.append(result)
        return results


def get_recommendations(
    query: str,
    top_k: int = 10,
    use_llm: bool = True,
    api_key: Optional[str] = None
) -> list[dict]:
    """
    Convenience function to get recommendations
    
    Args:
        query: Job description or hiring requirements
        top_k: Number of recommendations
        use_llm: Whether to use LLM re-ranking
        api_key: Optional Gemini API key
        
    Returns:
        List of recommended assessments
    """
    pipeline = RecommendationPipeline(use_llm=use_llm, api_key=api_key)
    result = pipeline.recommend(query, top_k=top_k)
    return result["recommendations"]


def demo():
    """Run a demo of the recommendation pipeline"""
    print("=" * 70)
    print("SHL ASSESSMENT RECOMMENDATION ENGINE - DEMO")
    print("=" * 70)
    
    # Test queries
    queries = [
        "Looking for a Java developer who can work in teams and solve problems efficiently. Need someone for a senior role.",
        "Hiring for a customer service representative who needs good communication skills and can handle difficult situations.",
        "Entry-level data analyst position requiring SQL and Excel skills with attention to detail."
    ]
    
    # Initialize pipeline
    pipeline = RecommendationPipeline(use_llm=True)
    
    for query in queries:
        print("\n" + "-" * 70)
        print(f"QUERY: {query[:100]}...")
        print("-" * 70)
        
        result = pipeline.recommend(query, top_k=5)
        
        print(f"\nMethod: {result['method']}")
        print(f"Candidates considered: {result['candidates_considered']}")
        
        print("\nRecommendations:")
        for rec in result["recommendations"]:
            print(f"  {rec['rank']}. {rec['assessment_name']}")
            print(f"     Types: {rec['test_types']}")
            print(f"     Reason: {rec.get('reason', 'N/A')}")
        
        if result.get("balance_summary"):
            print(f"\nBalance: {result['balance_summary']}")


if __name__ == "__main__":
    demo()
