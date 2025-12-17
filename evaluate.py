"""
Phase 4: Evaluation Module
Computes Mean Recall@10 against the ground truth dataset

Recall@10 = |predicted ∩ actual| / |actual| for each query
Mean Recall@10 = average of all Recall@10 scores
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

import pandas as pd

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "llm"))


def normalize_url(url: str) -> str:
    """Normalize URL for comparison"""
    if not url:
        return ""
    url = url.strip()
    url = unquote(url)
    # Remove trailing slashes and query strings
    url = re.sub(r'\?.*$', '', url)
    url = url.rstrip('/')
    # Ensure https
    url = url.replace('http://', 'https://')
    # Normalize path: remove /solutions/ if present to standardize
    url = url.replace('/solutions/products/', '/products/')
    return url.lower()


def load_ground_truth(file_path: str) -> list[dict]:
    """
    Load ground truth from evaluation dataset
    
    The dataset has multiple rows per query (one per expected URL).
    Format: Query, Assessment_url
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        List of dicts with query and expected URLs
    """
    df = pd.read_excel(file_path, sheet_name="Train-Set")
    
    # Group by query to collect all expected URLs
    query_to_urls = {}
    for _, row in df.iterrows():
        query = str(row.get("Query", "")).strip()
        url = str(row.get("Assessment_url", "")).strip()
        
        if not query or not url:
            continue
        
        if query not in query_to_urls:
            query_to_urls[query] = []
        
        normalized_url = normalize_url(url)
        if normalized_url and normalized_url not in query_to_urls[query]:
            query_to_urls[query].append(normalized_url)
    
    # Convert to list format
    ground_truth = [
        {"query": query, "expected_urls": urls}
        for query, urls in query_to_urls.items()
    ]
    
    return ground_truth


def calculate_recall_at_k(
    predicted_urls: list[str],
    actual_urls: list[str],
    k: int = 10
) -> float:
    """
    Calculate Recall@K for a single query
    
    Args:
        predicted_urls: URLs from the model
        actual_urls: Ground truth URLs
        k: Number of predictions to consider
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if not actual_urls:
        return 0.0
    
    # Normalize all URLs
    predicted_set = set(normalize_url(u) for u in predicted_urls[:k])
    actual_set = set(normalize_url(u) for u in actual_urls)
    
    # Calculate intersection
    hits = len(predicted_set & actual_set)
    
    return hits / len(actual_set)


def evaluate_pipeline(
    pipeline,
    ground_truth: list[dict],
    k: int = 10,
    verbose: bool = True
) -> dict:
    """
    Evaluate the recommendation pipeline against ground truth
    
    Args:
        pipeline: RecommendationPipeline instance
        ground_truth: List of query/expected_urls dicts
        k: Number of recommendations
        verbose: Print progress
        
    Returns:
        Evaluation results with Mean Recall@K
    """
    recalls = []
    results = []
    
    for i, item in enumerate(ground_truth):
        query = item["query"]
        expected_urls = item["expected_urls"]
        
        if verbose:
            print(f"\rEvaluating query {i+1}/{len(ground_truth)}...", end="", flush=True)
        
        # Get recommendations
        result = pipeline.recommend(query, top_k=k)
        predicted_urls = [rec["url"] for rec in result["recommendations"]]
        
        # Calculate recall
        recall = calculate_recall_at_k(predicted_urls, expected_urls, k)
        recalls.append(recall)
        
        results.append({
            "query_index": i,
            "query": query[:100],
            "expected_count": len(expected_urls),
            "recall_at_k": recall,
            "predicted_urls": predicted_urls,
            "expected_urls": expected_urls
        })
    
    if verbose:
        print()  # New line after progress
    
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    return {
        "mean_recall_at_k": mean_recall,
        "k": k,
        "total_queries": len(ground_truth),
        "method": pipeline.recommend("test", top_k=1).get("method", "unknown"),
        "query_results": results
    }


def run_evaluation(
    dataset_path: str,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    save_results: bool = True
) -> dict:
    """
    Run full evaluation and save results
    
    Args:
        dataset_path: Path to evaluation dataset
        use_llm: Whether to use LLM re-ranking
        api_key: Optional Gemini API key
        save_results: Save results to files
        
    Returns:
        Evaluation results
    """
    from pipeline import RecommendationPipeline
    
    print("=" * 60)
    print("SHL RECOMMENDATION ENGINE - EVALUATION")
    print("=" * 60)
    
    # Load ground truth
    print(f"\nLoading ground truth from: {dataset_path}")
    ground_truth = load_ground_truth(dataset_path)
    print(f"✓ Loaded {len(ground_truth)} queries with ground truth")
    
    # Initialize pipeline
    print(f"\nInitializing pipeline (LLM={use_llm})...")
    pipeline = RecommendationPipeline(use_llm=use_llm, api_key=api_key)
    
    # Run evaluation
    print(f"\nEvaluating {len(ground_truth)} queries...")
    results = evaluate_pipeline(pipeline, ground_truth, k=10, verbose=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Method: {results['method']}")
    print(f"Total queries: {results['total_queries']}")
    print(f"Mean Recall@10: {results['mean_recall_at_k']:.4f} ({results['mean_recall_at_k']*100:.2f}%)")
    
    # Breakdown
    recalls = [r["recall_at_k"] for r in results["query_results"]]
    print(f"\nBreakdown:")
    print(f"  Perfect (100%): {sum(1 for r in recalls if r == 1.0)} queries")
    print(f"  Partial (1-99%): {sum(1 for r in recalls if 0 < r < 1.0)} queries")
    print(f"  Zero (0%): {sum(1 for r in recalls if r == 0.0)} queries")
    
    # Save results
    if save_results:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Detailed results saved to: {results_file}")
        
        # Save predictions CSV
        predictions_file = output_dir / "predictions.csv"
        pred_data = []
        for qr in results["query_results"]:
            row = {"query": qr["query"], "recall": qr["recall_at_k"]}
            for j, url in enumerate(qr["predicted_urls"][:10]):
                row[f"pred_{j+1}"] = url
            pred_data.append(row)
        pd.DataFrame(pred_data).to_csv(predictions_file, index=False)
        print(f"✓ Predictions saved to: {predictions_file}")
    
    return results


def compare_methods(dataset_path: str, api_key: Optional[str] = None):
    """Compare semantic-only vs semantic+LLM"""
    from pipeline import RecommendationPipeline
    
    print("=" * 60)
    print("COMPARISON: Semantic-Only vs Semantic+LLM")
    print("=" * 60)
    
    ground_truth = load_ground_truth(dataset_path)
    
    # Semantic only
    print("\n[1/2] Evaluating Semantic-Only...")
    pipeline_semantic = RecommendationPipeline(use_llm=False)
    results_semantic = evaluate_pipeline(pipeline_semantic, ground_truth, verbose=True)
    
    # Semantic + LLM
    print("\n[2/2] Evaluating Semantic+LLM...")
    pipeline_llm = RecommendationPipeline(use_llm=True, api_key=api_key)
    results_llm = evaluate_pipeline(pipeline_llm, ground_truth, verbose=True)
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Semantic-Only Mean Recall@10: {results_semantic['mean_recall_at_k']:.4f} ({results_semantic['mean_recall_at_k']*100:.2f}%)")
    print(f"Semantic+LLM Mean Recall@10:  {results_llm['mean_recall_at_k']:.4f} ({results_llm['mean_recall_at_k']*100:.2f}%)")
    
    improvement = results_llm['mean_recall_at_k'] - results_semantic['mean_recall_at_k']
    print(f"\nImprovement: {improvement*100:+.2f}%")
    
    return {
        "semantic_only": results_semantic,
        "semantic_llm": results_llm,
        "improvement": improvement
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SHL Recommendation Engine")
    parser.add_argument("--dataset", default="data/Gen_AI Dataset.xlsx", 
                       help="Path to evaluation dataset")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM re-ranking")
    parser.add_argument("--compare", action="store_true",
                       help="Compare semantic-only vs semantic+LLM")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_methods(args.dataset)
    else:
        run_evaluation(args.dataset, use_llm=not args.no_llm)
