"""
Memory-Optimized Streamlit App for Free Tier Deployments
Designed to work within 512MB RAM limit

Key optimizations:
1. Lazy model loading - only when needed
2. Smaller embedding model option
3. LLM features optional (can disable to save memory)
4. Efficient caching strategy
"""

import sys
import os
from pathlib import Path
from typing import Optional
import gc

import streamlit as st

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


# Page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .recommendation-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .test-type-badge {
        background-color: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 15px;
        font-size: 12px;
        margin-right: 5px;
    }
    .soft-skill {
        background-color: #2196F3;
    }
    .hard-skill {
        background-color: #FF9800;
    }
</style>
""", unsafe_allow_html=True)


# Memory-optimized loading with lazy initialization
@st.cache_resource
def load_vector_store():
    """Load vector store (cached) - only loads pre-built index"""
    data_dir = str(Path(__file__).parent / "data")
    store = SHLVectorStore(data_dir=data_dir)
    
    # Only load pre-built index (don't build if missing)
    vector_store_path = Path(data_dir) / "vector_store.faiss"
    if not vector_store_path.exists():
        st.error("Vector store not found! Please ensure data/vector_store.faiss exists.")
        st.stop()
    
    store.load()
    return store


@st.cache_resource
def load_reranker(_enable_llm: bool = True):
    """Load LLM reranker only if enabled and available (cached)"""
    if not _enable_llm:
        return None
    if not LLM_AVAILABLE:
        return None
    try:
        # Only load if API key exists
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        if not api_key:
            return None
        return LLMReranker()
    except Exception as e:
        st.warning(f"LLM reranker not available: {e}")
        return None


def get_test_type_badge(test_type: str) -> str:
    """Generate badge HTML for test type"""
    soft_skills = ["P", "B", "A", "S"]
    is_soft = any(test_type.startswith(s) or s in test_type for s in soft_skills)
    badge_class = "soft-skill" if is_soft else "hard-skill"
    return f'<span class="test-type-badge {badge_class}">{test_type}</span>'


def recommend(query: str, top_k: int = 10, use_llm: bool = True) -> tuple:
    """Get recommendations for a query"""
    vector_store = load_vector_store()
    reranker = load_reranker(use_llm) if use_llm else None
    
    # Semantic search
    candidate_pool = 20 if use_llm and reranker else top_k
    candidates = vector_store.search(query, top_k=candidate_pool)
    
    # LLM re-ranking if available
    if use_llm and reranker:
        try:
            result = reranker.rerank(query, candidates, top_k=top_k)
            if result.get("recommendations"):
                return result["recommendations"], "semantic+llm_rerank"
        except Exception as e:
            st.warning(f"LLM re-ranking failed: {e}. Using semantic search only.")
    
    # Fallback to semantic results
    recommendations = [
        {
            "rank": i + 1,
            "assessment_name": c["test_name"],
            "url": c["url"],
            "test_types": c.get("test_types", []),
            "duration": c.get("duration"),
            "remote_testing": c.get("remote_testing"),
            "reason": f"Semantic similarity: {c.get('similarity_score', 0):.2f}"
        }
        for i, c in enumerate(candidates[:top_k])
    ]
    return recommendations, "semantic_only"


# Main UI
def main():
    # Header
    st.title("🎯 SHL Assessment Recommendation Engine")
    st.markdown("""
    Find the perfect SHL assessments for your hiring needs. Enter a job description 
    or requirements below, and get intelligent recommendations.
    """)
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Number of recommendations
        top_k = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)
        
        # LLM re-ranking toggle
        use_llm = st.checkbox(
            "Enable LLM Re-ranking", 
            value=True,
            help="Uses AI to improve and balance recommendations. Disable to save memory."
        )
        
        if use_llm and not LLM_AVAILABLE:
            st.warning("⚠️ LLM re-ranking not available (missing dependencies)")
        
        st.divider()
        
        # Info
        st.markdown("### 📊 System Info")
        vector_store = load_vector_store()
        st.metric("Total Assessments", len(vector_store.metadata))
        
        st.divider()
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific about required skills
        - Mention job level (entry, mid, senior)
        - Include both technical and soft skills
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Job Requirements")
        query = st.text_area(
            "Describe the position or paste a job description:",
            height=150,
            placeholder="Example: Looking for a senior Java developer with strong communication skills and team collaboration abilities for our enterprise software team..."
        )
    
    with col2:
        st.subheader("🚀 Quick Examples")
        examples = [
            "Java developer with business collaboration",
            "Entry-level sales with communication",
            "Python & SQL programming skills",
            "Executive leadership assessment",
            "Customer service for retail"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example[:20]}"):
                query = example
                st.rerun()
    
    # Search button
    if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
        if not query or len(query.strip()) < 10:
            st.warning("⚠️ Please enter a more detailed job description (at least 10 characters)")
        else:
            with st.spinner("Finding best assessments..."):
                recommendations, method = recommend(query, top_k=top_k, use_llm=use_llm)
                
                # Display results
                st.success(f"Found {len(recommendations)} recommendations using {method}")
                
                # Results
                st.subheader("📋 Recommended Assessments")
                
                for rec in recommendations:
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{rec['rank']}. {rec['assessment_name']}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        
                        with col_a:
                            # Test types
                            if rec.get('test_types'):
                                badges = " ".join([get_test_type_badge(t) for t in rec['test_types'][:5]])
                                st.markdown(f"**Types:** {badges}", unsafe_allow_html=True)
                        
                        with col_b:
                            if rec.get('duration'):
                                st.markdown(f"**Duration:** {rec['duration']}")
                        
                        with col_c:
                            if rec.get('remote_testing'):
                                st.markdown(f"**Remote:** {rec['remote_testing']}")
                        
                        # Reason
                        if rec.get('reason'):
                            with st.expander("Why this assessment?"):
                                st.markdown(rec['reason'])
                        
                        # Link
                        if rec.get('url'):
                            st.markdown(f"[View Assessment Details →]({rec['url']})")
                        
                        st.divider()
                
                # Download results
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=str(recommendations),
                    file_name="shl_recommendations.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    # Force garbage collection to free memory
    gc.collect()
    main()
