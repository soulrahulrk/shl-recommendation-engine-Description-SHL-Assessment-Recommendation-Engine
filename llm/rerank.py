"""
Phase 3: LLM Re-ranking Module
Uses Groq (free tier) to re-rank and balance assessment recommendations

Key responsibilities:
1. Take Top-K candidates from semantic search
2. Re-rank based on query relevance and skill balance
3. Ensure mix of hard skills (K) and soft skills (P, B, A)
4. Generate explanations for recommendations
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from groq import Groq
from dotenv import load_dotenv


class LLMReranker:
    """LLM-based re-ranking for assessment recommendations"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Initialize the reranker with Groq API (FREE)
        
        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var
            model: Model to use. Defaults to llama-3.3-70b-versatile
        """
        # Load .env file if exists
        load_dotenv()
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter. Get FREE key at https://console.groq.com/keys"
            )
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        # Use Llama 3.3 70B (fast and free on Groq)
        self.model = model or "llama-3.3-70b-versatile"
        
        # Load system prompt
        prompt_path = Path(__file__).parent / "prompt.txt"
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            self.system_prompt = self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Default system prompt if file not found"""
        return """You are an HR assessment specialist. Given a job query and candidate assessments,
re-rank and select the TOP 10 most relevant assessments.

RULES:
1. Only select from provided candidates - DO NOT invent assessments
2. Balance technical (K) and behavioral (P, B, A) assessments
3. Return valid JSON with recommendations array

OUTPUT FORMAT:
{
  "recommendations": [
    {"rank": 1, "assessment_name": "...", "url": "...", "test_types": [...], "reason": "..."},
    ...
  ],
  "balance_summary": "..."
}"""
    
    def _format_candidates(self, candidates: list[dict]) -> str:
        """Format candidate assessments for the prompt"""
        formatted = []
        for i, c in enumerate(candidates, 1):
            entry = f"""
{i}. Assessment: {c.get('test_name', 'Unknown')}
   URL: {c.get('url', '')}
   Test Types: {', '.join(c.get('test_types', []))}
   Category: {c.get('category', '')}
   Description: {c.get('description', '')[:300]}...
   Similarity Score: {c.get('similarity_score', 0):.3f}
"""
            formatted.append(entry)
        return "\n".join(formatted)
    
    def _parse_response(self, response_text: str) -> dict:
        """Parse LLM response to extract JSON"""
        # Try to find JSON in the response
        # First, try direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object pattern
        json_match = re.search(r'\{[\s\S]*"recommendations"[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Return error structure if parsing fails
        return {
            "recommendations": [],
            "error": "Failed to parse LLM response",
            "raw_response": response_text[:500]
        }
    
    def _validate_recommendations(
        self, 
        recommendations: list[dict], 
        candidates: list[dict]
    ) -> list[dict]:
        """Validate and fix recommendations against original candidates"""
        # Create lookup by URL and name
        url_lookup = {c.get("url", "").strip(): c for c in candidates}
        name_lookup = {c.get("test_name", "").lower(): c for c in candidates}
        
        validated = []
        seen_urls = set()
        
        for rec in recommendations:
            url = rec.get("url", "").strip()
            name = rec.get("assessment_name", "")
            
            # Find matching candidate
            matched = None
            if url in url_lookup:
                matched = url_lookup[url]
            elif name.lower() in name_lookup:
                matched = name_lookup[name.lower()]
            
            if matched and matched["url"] not in seen_urls:
                # Use original candidate data to ensure correctness
                validated.append({
                    "rank": len(validated) + 1,
                    "assessment_name": matched["test_name"],
                    "url": matched["url"],
                    "test_types": matched.get("test_types", []),
                    "reason": rec.get("reason", "Relevant to query requirements")
                })
                seen_urls.add(matched["url"])
        
        return validated
    
    def rerank(
        self, 
        query: str, 
        candidates: list[dict], 
        top_k: int = 10
    ) -> dict:
        """
        Re-rank candidate assessments using LLM
        
        Args:
            query: The job description or hiring query
            candidates: List of candidate assessments from semantic search
            top_k: Number of recommendations to return
            
        Returns:
            Dictionary with recommendations and metadata
        """
        if not candidates:
            return {"recommendations": [], "error": "No candidates provided"}
        
        # Format the prompt
        user_prompt = f"""
QUERY:
{query}

CANDIDATE ASSESSMENTS (from semantic search):
{self._format_candidates(candidates)}

Please analyze these candidates and provide your re-ranked TOP {top_k} recommendations.
Remember to balance technical and behavioral assessments.
Return your response as valid JSON.
"""
        
        try:
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=2048,
            )
            
            # Extract response
            generated_text = chat_completion.choices[0].message.content
            
            # Parse response
            result = self._parse_response(generated_text)
            
            # Validate recommendations
            if "recommendations" in result:
                result["recommendations"] = self._validate_recommendations(
                    result["recommendations"],
                    candidates
                )[:top_k]
            
            return result
            
        except Exception as e:
            return {
                "recommendations": [],
                "error": f"LLM call failed: {str(e)}"
            }
    
    def rerank_with_fallback(
        self, 
        query: str, 
        candidates: list[dict], 
        top_k: int = 10
    ) -> list[dict]:
        """
        Re-rank with fallback to semantic search results if LLM fails
        
        Args:
            query: The job description or hiring query
            candidates: List of candidate assessments from semantic search
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended assessments
        """
        result = self.rerank(query, candidates, top_k)
        
        if result.get("recommendations"):
            return result["recommendations"]
        
        # Fallback: return top candidates with basic balancing
        print(f"Warning: LLM reranking failed, using fallback. Error: {result.get('error')}")
        
        # Simple balancing: alternate between K-type and non-K-type
        k_type = [c for c in candidates if "K" in c.get("test_type_codes", [])]
        other = [c for c in candidates if "K" not in c.get("test_type_codes", [])]
        
        balanced = []
        for i in range(top_k):
            if i % 2 == 0 and k_type:
                balanced.append(k_type.pop(0))
            elif other:
                balanced.append(other.pop(0))
            elif k_type:
                balanced.append(k_type.pop(0))
        
        return [
            {
                "rank": i + 1,
                "assessment_name": c["test_name"],
                "url": c["url"],
                "test_types": c.get("test_types", []),
                "reason": "Semantically relevant to query"
            }
            for i, c in enumerate(balanced[:top_k])
        ]


def rerank_assessments(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    api_key: Optional[str] = None
) -> list[dict]:
    """
    Convenience function for re-ranking assessments
    
    Args:
        query: Job description or hiring query
        candidates: Candidate assessments from semantic search
        top_k: Number of recommendations
        api_key: Optional Gemini API key
        
    Returns:
        List of re-ranked recommendations
    """
    reranker = LLMReranker(api_key=api_key)
    return reranker.rerank_with_fallback(query, candidates, top_k)


# Test function
def test_reranker():
    """Test the reranker with sample data"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from vector_store import SHLVectorStore
    
    # Load vector store
    store = SHLVectorStore(data_dir=str(Path(__file__).parent.parent / "data"))
    store.load()
    
    # Test query
    query = "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    
    # Get candidates
    candidates = store.search(query, top_k=20)
    print(f"Got {len(candidates)} candidates from semantic search")
    
    # Re-rank
    reranker = LLMReranker()
    result = reranker.rerank(query, candidates, top_k=10)
    
    print("\n" + "=" * 60)
    print("RE-RANKED RECOMMENDATIONS")
    print("=" * 60)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    
    for rec in result.get("recommendations", []):
        print(f"\n{rec['rank']}. {rec['assessment_name']}")
        print(f"   Types: {rec['test_types']}")
        print(f"   Reason: {rec['reason']}")
    
    if result.get("balance_summary"):
        print(f"\nBalance: {result['balance_summary']}")


if __name__ == "__main__":
    test_reranker()
