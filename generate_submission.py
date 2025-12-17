"""
Generate predictions CSV for the Test-Set
Format: Query, Assessment_url (one row per recommendation)
"""

import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "llm"))

from vector_store import SHLVectorStore

# Try to import LLM reranker
try:
    from rerank import LLMReranker
    reranker = LLMReranker()
    print("✓ LLM reranker loaded")
except Exception as e:
    print(f"⚠ LLM reranker not available: {e}")
    reranker = None


def generate_predictions(dataset_path: str, output_path: str = "submission.csv"):
    """Generate predictions for test set in submission format"""
    
    # Load test set
    df = pd.read_excel(dataset_path, sheet_name="Test-Set")
    
    # Get unique queries
    queries = df["Query"].unique().tolist()
    print(f"✓ Loaded {len(queries)} test queries")
    
    # Load vector store
    data_dir = str(Path(__file__).parent / "data")
    vector_store = SHLVectorStore(data_dir=data_dir)
    vector_store.load()
    print(f"✓ Vector store loaded: {len(vector_store.metadata)} assessments")
    
    # Generate predictions
    results = []
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}...")
        
        # Get candidates from semantic search
        candidates = vector_store.search(query, top_k=20)
        
        # LLM re-ranking if available
        predicted_urls = []
        if reranker:
            try:
                llm_result = reranker.rerank(query, candidates, top_k=10)
                if llm_result.get("recommendations"):
                    predicted_urls = [rec["url"] for rec in llm_result["recommendations"]]
            except Exception as e:
                print(f"  LLM failed: {e}")
        
        # Fallback to semantic results
        if not predicted_urls:
            predicted_urls = [c["url"] for c in candidates[:10]]
        
        # Add to results (one row per recommendation)
        for url in predicted_urls:
            results.append({
                "Query": query,
                "Assessment_url": url
            })
    
    # Save as CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"  Total rows: {len(output_df)}")
    
    return output_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate predictions for SHL test set")
    parser.add_argument("--dataset", default="Gen_AI Dataset.xlsx", 
                       help="Path to dataset Excel file")
    parser.add_argument("--output", default="submission.csv",
                       help="Output CSV file path")
    
    args = parser.parse_args()
    generate_predictions(args.dataset, args.output)
