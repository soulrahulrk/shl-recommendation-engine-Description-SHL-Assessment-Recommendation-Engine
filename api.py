"""
FastAPI Backend for SHL Assessment Recommendations

Main endpoints:
- POST /recommend - Get recommendations
- GET /health - Check system status
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "llm"))

from vector_store import SHLVectorStore

# Try to import LLM reranker
try:
    from rerank import LLMReranker
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM reranker not available")


# Initialize FastAPI
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Semantic search + LLM re-ranking for SHL assessment recommendations",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store: Optional[SHLVectorStore] = None
reranker: Optional["LLMReranker"] = None


# Request/Response Models - Following SHL's API specification
class RecommendRequest(BaseModel):
    query: str = Field(..., description="Job description or hiring requirements")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Looking for Java developers with team collaboration skills"
            }
        }


class AssessmentResponse(BaseModel):
    """Individual assessment in recommendations"""
    url: str = Field(..., description="URL of the assessment on SHL's catalog")
    adaptive_support: str = Field(..., description="Yes/No - Adaptive/IRT support")
    description: str = Field(..., description="Brief description of the assessment")
    duration: Optional[int] = Field(None, description="Duration in minutes")
    remote_support: str = Field(..., description="Yes/No - Remote testing support")
    test_type: list[str] = Field(..., description="Test type categories")
    name: str = Field(..., description="Name of the assessment")


class RecommendResponse(BaseModel):
    """Response format as per SHL specification"""
    recommended_assessments: list[AssessmentResponse]


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    llm_available: bool
    total_assessments: int


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load vector store and initialize reranker on startup"""
    global vector_store, reranker
    
    data_dir = str(Path(__file__).parent / "data")
    
    # Load vector store
    vector_store = SHLVectorStore(data_dir=data_dir)
    try:
        # Check if vector store exists, build if not
        vector_store_path = Path(data_dir) / "vector_store.faiss"
        if not vector_store_path.exists():
            print("Building vector store for first time...")
            vector_store.build_index()
            vector_store.save()
            print(f"✓ Vector store built: {len(vector_store.metadata)} assessments")
        else:
            vector_store.load()
            print(f"✓ Vector store loaded: {len(vector_store.metadata)} assessments")
    except Exception as e:
        print(f"✗ Failed to load vector store: {e}")
        vector_store = None
    
    # Initialize LLM reranker if API key available
    if LLM_AVAILABLE and os.getenv("GROQ_API_KEY"):
        try:
            reranker = LLMReranker()
            print("✓ LLM reranker initialized (Groq Llama-3.3-70B)")
        except Exception as e:
            print(f"⚠ LLM reranker not available: {e}")
            reranker = None
    else:
        print("⚠ LLM reranker disabled (no GROQ_API_KEY)")


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if vector_store else "degraded",
        vector_store_loaded=vector_store is not None,
        llm_available=reranker is not None,
        total_assessments=len(vector_store.metadata) if vector_store else 0
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get assessment recommendations for a job query - Returns 1-10 relevant assessments"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not loaded")
    
    # Semantic search for candidates
    candidate_pool = 20  # Get more candidates for LLM to re-rank
    candidates = vector_store.search(request.query, top_k=candidate_pool)
    
    top_k = 10  # Max 10 recommendations as per spec
    recommendations = []
    
    # LLM re-ranking if available
    if reranker:
        try:
            llm_result = reranker.rerank(request.query, candidates, top_k=top_k)
            if llm_result.get("recommendations"):
                for rec in llm_result["recommendations"]:
                    # Find original candidate data for complete info
                    orig = next((c for c in candidates if c["url"] == rec["url"]), {})
                    recommendations.append(AssessmentResponse(
                        url=rec["url"],
                        name=rec["assessment_name"],
                        description=orig.get("description", "")[:200] if orig.get("description") else "",
                        test_type=rec.get("test_types", []),
                        duration=int(orig.get("duration", "0").split()[0]) if orig.get("duration") and orig.get("duration").split()[0].isdigit() else None,
                        remote_support=orig.get("remote_testing", "No"),
                        adaptive_support=orig.get("adaptive_irt", "No")
                    ))
        except Exception as e:
            print(f"LLM reranking failed: {e}")
    
    # Fallback to semantic results if LLM failed or not available
    if not recommendations:
        for c in candidates[:top_k]:
            duration_str = c.get("duration", "")
            duration_int = None
            if duration_str:
                parts = duration_str.split()
                if parts and parts[0].isdigit():
                    duration_int = int(parts[0])
            
            recommendations.append(AssessmentResponse(
                url=c["url"],
                name=c["test_name"],
                description=c.get("description", "")[:200] if c.get("description") else "",
                test_type=c.get("test_types", []),
                duration=duration_int,
                remote_support=c.get("remote_testing", "No"),
                adaptive_support=c.get("adaptive_irt", "No")
            ))
    
    return RecommendResponse(recommended_assessments=recommendations)


@app.get("/assessments")
async def list_assessments(limit: int = 100, offset: int = 0):
    """List all available assessments"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not loaded")
    
    assessments = vector_store.metadata[offset:offset + limit]
    return {
        "total": len(vector_store.metadata),
        "limit": limit,
        "offset": offset,
        "assessments": assessments
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
