"""
Phase 5: Streamlit Frontend for SHL Assessment Recommendations

Features:
- Text input for job descriptions
- URL input for job postings
- Results display with test types and links
"""

import sys
from pathlib import Path
from typing import Optional

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


# Initialize session state
@st.cache_resource
def load_vector_store():
    """Load vector store (cached) - builds if not exists"""
    data_dir = str(Path(__file__).parent / "data")
    store = SHLVectorStore(data_dir=data_dir)
    
    # Check if vector store exists
    vector_store_path = Path(data_dir) / "vector_store.faiss"
    if not vector_store_path.exists():
        st.info("Building vector store for first time... This takes ~2 minutes.")
        # Build index from JSON data
        store.build_index()
        store.save()
        st.success("Vector store built successfully!")
    else:
        store.load()
    
    return store


@st.cache_resource
def load_reranker():
    """Load LLM reranker if available (cached)"""
    if not LLM_AVAILABLE:
        return None
    try:
        return LLMReranker()
    except Exception as e:
        st.warning(f"LLM reranker not available: {e}")
        return None


def get_test_type_badge(test_type: str) -> str:
    """Generate badge HTML for test type"""
    soft_skills = ["P", "B", "A", "S"]  # Personality, Behavioral, Ability (soft), Simulation
    
    # Check if it's a soft skill type
    is_soft = any(test_type.startswith(s) or s in test_type for s in soft_skills)
    badge_class = "soft-skill" if is_soft else "hard-skill"
    
    return f'<span class="test-type-badge {badge_class}">{test_type}</span>'


def recommend(query: str, top_k: int = 10, use_llm: bool = True) -> list:
    """Get recommendations for a query"""
    vector_store = load_vector_store()
    reranker = load_reranker() if use_llm else None
    
    # Semantic search
    candidate_pool = 20 if use_llm and reranker else top_k
    candidates = vector_store.search(query, top_k=candidate_pool)
    
    # LLM re-ranking if available
    if use_llm and reranker:
        result = reranker.rerank(query, candidates, top_k=top_k)
        if result.get("recommendations"):
            return result["recommendations"], "semantic+llm_rerank"
    
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
    or requirements below, and get intelligent recommendations powered by semantic search 
    and LLM re-ranking.
    """)
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider("Number of recommendations", 1, 20, 10)
        use_llm = st.checkbox("Use LLM re-ranking", value=True, 
                              help="Use AI to re-rank and balance recommendations")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool uses:
        - **Semantic search** with sentence-transformers
        - **LLM re-ranking** with Groq (Llama-3.3-70B)
        - **518 SHL assessments** in the database
        """)
        
        # Status
        st.markdown("---")
        st.markdown("### Status")
        try:
            store = load_vector_store()
            st.success(f"✓ {len(store.metadata)} assessments loaded")
        except:
            st.error("✗ Vector store not loaded")
        
        reranker = load_reranker()
        if reranker:
            st.success("✓ LLM reranker ready")
        else:
            st.warning("⚠ LLM reranker not available")
    
    # Input section
    st.header("📝 Enter Your Requirements")
    
    tab1, tab2 = st.tabs(["Text Input", "Example Queries"])
    
    with tab1:
        query = st.text_area(
            "Job Description or Requirements",
            height=150,
            placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes."
        )
    
    with tab2:
        example_queries = [
            "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment that can be completed in 40 minutes.",
            "Need to assess customer service representatives on communication skills and problem-solving ability.",
            "Entry-level data analyst position requiring SQL and Excel skills with strong attention to detail.",
            "Senior project manager role requiring leadership, stakeholder management, and strategic thinking.",
            "Looking for assessments for a software engineering team lead who needs both technical and people skills."
        ]
        
        selected_example = st.selectbox("Select an example query", [""] + example_queries)
        if selected_example:
            query = selected_example
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        submit = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
    
    # Results section
    if submit and query:
        with st.spinner("Finding best assessments..."):
            recommendations, method = recommend(query, top_k=top_k, use_llm=use_llm)
        
        st.markdown("---")
        st.header("🎯 Recommended Assessments")
        st.caption(f"Method: {method.replace('_', ' ').title()}")
        
        if not recommendations:
            st.warning("No recommendations found. Try a different query.")
        else:
            # Display as cards
            for rec in recommendations:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"### {rec['rank']}. [{rec['assessment_name']}]({rec['url']})")
                        
                        # Test types as badges
                        test_types = rec.get("test_types", [])
                        if test_types:
                            badges_html = " ".join(get_test_type_badge(t) for t in test_types)
                            st.markdown(badges_html, unsafe_allow_html=True)
                        
                        # Reason
                        if rec.get("reason"):
                            st.caption(f"💡 {rec['reason']}")
                    
                    with col2:
                        # Duration and other info
                        if rec.get("duration"):
                            st.metric("Duration", rec["duration"])
                        if rec.get("remote_testing") == "Yes":
                            st.success("🌐 Remote")
                    
                    st.markdown("---")
            
            # Download results
            import json
            results_json = json.dumps(recommendations, indent=2)
            st.download_button(
                "📥 Download Results (JSON)",
                results_json,
                "recommendations.json",
                "application/json"
            )
    
    elif submit:
        st.warning("Please enter a job description or requirements.")


if __name__ == "__main__":
    main()
