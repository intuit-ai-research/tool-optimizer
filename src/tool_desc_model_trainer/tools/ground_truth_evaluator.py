#!/usr/bin/env python3
"""
Ground Truth Performance Evaluator

This module evaluates multiple ground truth descriptions for each tool
using training data and selects the best-performing one for SFT training.

Key Features:
- Evaluates each ground truth variant against training queries
- Uses existing tool-level reward system for consistent scoring
- Supports multiple aggregation strategies (mean, median, percentile)
- Caches evaluation results to avoid recomputation
- Provides detailed performance breakdown per variant
"""

import json
import os
import sys
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from pathlib import Path

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
policy_learn_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(os.path.dirname(policy_learn_dir))

# Add to Python path
for path in [current_dir, policy_learn_dir, root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import reward evaluation
from reward_modules.tool_level_reward import compute_score_detailed as compute_tool_reward

class GroundTruthEvaluator:
    """Evaluates multiple ground truth descriptions and selects the best one per tool."""

    def __init__(self,
                 aggregation_method: str = "mean",
                 variance_penalty: float = 0.0,
                 w1: float = 0.5,
                 w2: float = 0.5,
                 max_workers: int = 4,
                 cache_dir: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            aggregation_method: How to aggregate scores across queries ("mean", "median", "p75")
            variance_penalty: Penalty for high variance across queries
            w1: Weight for API selection accuracy
            w2: Weight for parameter generation success
            max_workers: Maximum number of parallel evaluation threads
            cache_dir: Optional directory for caching evaluation results
        """
        self.aggregation_method = aggregation_method
        self.variance_penalty = variance_penalty
        self.w1 = w1
        self.w2 = w2
        self.max_workers = max_workers
        self.cache_dir = cache_dir

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def evaluate_ground_truth_variant(self,
                                    tool_name: str,
                                    variant: Dict[str, str],
                                    queries: List[Dict[str, Any]],
                                    baseline_description: str) -> Dict[str, Any]:
        """
        Evaluate a single ground truth variant against training queries.

        Args:
            tool_name: Name of the tool
            variant: Dict with 'source' and 'description' keys
            queries: List of training queries for this tool
            baseline_description: D0 baseline description

        Returns:
            Dict with evaluation results including aggregated score and details
        """

        self.logger.info(f"   🧪 Evaluating {tool_name} - {variant['source']} on {len(queries)} queries...")

        # Prepare evaluation data structure
        # The tool reward function expects extra_info with query metadata
        extra_info = {
            "tool_name": tool_name,
            "train_queries": queries,
            "heldout_queries": [],  # Use all for evaluation
            "all_queries": queries,
            "aggregation_method": self.aggregation_method,
            "variance_penalty": self.variance_penalty,
            "eval_on_heldout": False,  # Use training queries for ground truth selection
        }

        # Simulate the expected data structure for tool_level_reward
        evaluation_data = {
            "prompt": f"Evaluate ground truth for {tool_name}",
            "ground_truth": variant['description'],
            "extra_info": extra_info
        }

        try:
            # Use tool-level reward function with correct signature
            reward_result = compute_tool_reward(
                data_source="tmdb",
                solution_str=variant['description'],
                ground_truth=variant['description'],  # Not used in tool-level evaluation
                extra_info=extra_info
            )

            # Extract results
            aggregated_score = reward_result.get("reward", 0.0)
            per_query_results = reward_result.get("per_query_results", [])

            # Compute additional statistics
            if per_query_results:
                scores = [r.get("reward", 0.0) for r in per_query_results]
                variance = np.var(scores) if len(scores) > 1 else 0.0
                std_dev = np.std(scores) if len(scores) > 1 else 0.0

                # Success rates
                api_success_rate = np.mean([r.get("api_selection_correct", False) for r in per_query_results])
                param_success_rate = np.mean([r.get("parameter_generation_success", False) for r in per_query_results])

            else:
                scores = [aggregated_score]
                variance = 0.0
                std_dev = 0.0
                api_success_rate = 0.0
                param_success_rate = 0.0

            result = {
                "tool_name": tool_name,
                "variant_source": variant['source'],
                "description": variant['description'],
                "aggregated_score": aggregated_score,
                "variance": variance,
                "std_dev": std_dev,
                "api_success_rate": api_success_rate,
                "param_success_rate": param_success_rate,
                "num_queries_evaluated": len(queries),
                "per_query_scores": scores,
                "per_query_results": per_query_results,
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_params": {
                    "aggregation_method": self.aggregation_method,
                    "variance_penalty": self.variance_penalty,
                    "w1": self.w1,
                    "w2": self.w2
                }
            }

            self.logger.info(f"   ✅ {tool_name} ({variant['source']}): score={aggregated_score:.4f}, "
                           f"api_success={api_success_rate:.2%}, param_success={param_success_rate:.2%}")

            return result

        except Exception as e:
            self.logger.error(f"   ❌ Error evaluating {tool_name} ({variant['source']}): {e}")
            # Return zero score on error
            return {
                "tool_name": tool_name,
                "variant_source": variant['source'],
                "description": variant['description'],
                "aggregated_score": 0.0,
                "variance": 0.0,
                "std_dev": 0.0,
                "api_success_rate": 0.0,
                "param_success_rate": 0.0,
                "num_queries_evaluated": len(queries),
                "per_query_scores": [0.0],
                "per_query_results": [],
                "evaluation_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def select_best_ground_truth(self,
                               tool_name: str,
                               variants: List[Dict[str, str]],
                               queries: List[Dict[str, Any]],
                               baseline_description: str) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Evaluate all ground truth variants and select the best one.

        Args:
            tool_name: Name of the tool
            variants: List of variant dicts with 'source' and 'description' keys
            queries: List of training queries for this tool
            baseline_description: D0 baseline description

        Returns:
            Tuple of (best_variant, evaluation_results)
        """
        if not variants:
            raise ValueError(f"No ground truth variants provided for {tool_name}")

        if len(variants) == 1:
            self.logger.info(f"   🎯 Only one variant for {tool_name}, using {variants[0]['source']}")
            # Still evaluate it for logging
            evaluation_result = self.evaluate_ground_truth_variant(
                tool_name, variants[0], queries, baseline_description
            )
            return variants[0], {"variants": [evaluation_result], "best_variant": evaluation_result}

        self.logger.info(f"   🏆 Evaluating {len(variants)} ground truth variants for {tool_name}...")

        # Evaluate all variants
        evaluation_results = []

        if self.max_workers == 1:
            # Sequential evaluation
            for variant in variants:
                result = self.evaluate_ground_truth_variant(
                    tool_name, variant, queries, baseline_description
                )
                evaluation_results.append(result)
        else:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_variant = {
                    executor.submit(
                        self.evaluate_ground_truth_variant,
                        tool_name, variant, queries, baseline_description
                    ): variant for variant in variants
                }

                for future in as_completed(future_to_variant):
                    result = future.result()
                    evaluation_results.append(result)

        # Sort by aggregated score (descending)
        evaluation_results.sort(key=lambda x: x["aggregated_score"], reverse=True)
        best_result = evaluation_results[0]

        # Find the corresponding variant
        best_variant = next(
            v for v in variants if v['source'] == best_result['variant_source']
        )

        self.logger.info(f"   🥇 Best variant for {tool_name}: {best_variant['source']} "
                        f"(score: {best_result['aggregated_score']:.4f})")

        # Log comparison if multiple variants
        if len(evaluation_results) > 1:
            self.logger.info(f"   📊 Variant comparison for {tool_name}:")
            for i, result in enumerate(evaluation_results):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                self.logger.info(f"      {rank} {result['variant_source']}: {result['aggregated_score']:.4f}")

        return best_variant, {
            "variants": evaluation_results,
            "best_variant": best_result,
            "selection_reason": f"Highest aggregated score: {best_result['aggregated_score']:.4f}"
        }

    def evaluate_all_tools(self,
                          tools_data: Dict[str, Dict[str, Any]],
                          output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate ground truth variants for all tools and select the best ones.

        Args:
            tools_data: Dict mapping tool_name -> {
                'variants': List[Dict[str, str]],
                'queries': List[Dict[str, Any]],
                'baseline_description': str
            }
            output_file: Optional file to save detailed results

        Returns:
            Dict with best variants and evaluation details for all tools
        """
        self.logger.info(f"🚀 Starting ground truth evaluation for {len(tools_data)} tools...")

        results = {
            "evaluation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tools": len(tools_data),
                "evaluation_params": {
                    "aggregation_method": self.aggregation_method,
                    "variance_penalty": self.variance_penalty,
                    "w1": self.w1,
                    "w2": self.w2,
                    "max_workers": self.max_workers
                }
            },
            "tool_results": {},
            "best_variants": {}
        }

        tools_with_multiple_variants = 0
        total_variants_evaluated = 0

        for tool_name, tool_data in tools_data.items():
            variants = tool_data['variants']
            queries = tool_data['queries']
            baseline_description = tool_data.get('baseline_description', '')

            if len(variants) > 1:
                tools_with_multiple_variants += 1
            total_variants_evaluated += len(variants)

            try:
                best_variant, evaluation_details = self.select_best_ground_truth(
                    tool_name, variants, queries, baseline_description
                )

                results["tool_results"][tool_name] = evaluation_details
                results["best_variants"][tool_name] = best_variant

            except Exception as e:
                self.logger.error(f"❌ Failed to evaluate {tool_name}: {e}")
                # Use first variant as fallback
                if variants:
                    results["best_variants"][tool_name] = variants[0]
                    results["tool_results"][tool_name] = {
                        "error": str(e),
                        "fallback_variant": variants[0]['source']
                    }

        # Update summary
        results["evaluation_summary"].update({
            "tools_with_multiple_variants": tools_with_multiple_variants,
            "total_variants_evaluated": total_variants_evaluated,
            "successful_evaluations": len(results["best_variants"]),
            "average_variants_per_tool": total_variants_evaluated / len(tools_data)
        })

        self.logger.info(f"✅ Ground truth evaluation complete!")
        self.logger.info(f"   📊 {len(results['best_variants'])}/{len(tools_data)} tools evaluated successfully")
        self.logger.info(f"   🎯 {tools_with_multiple_variants} tools had multiple variants to choose from")
        self.logger.info(f"   📈 {total_variants_evaluated} total variants evaluated")

        # Save detailed results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"   💾 Detailed results saved to: {output_file}")

        return results


def main():
    """Command-line interface for ground truth evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ground truth descriptions for tool-level training")
    parser.add_argument("--tools-data", required=True, help="JSON file with tools data")
    parser.add_argument("--output", help="Output file for detailed results")
    parser.add_argument("--aggregation-method", default="mean", choices=["mean", "median", "p75"],
                       help="Score aggregation method")
    parser.add_argument("--variance-penalty", type=float, default=0.0,
                       help="Penalty for high variance across queries")
    parser.add_argument("--w1", type=float, default=0.5, help="Weight for API selection")
    parser.add_argument("--w2", type=float, default=0.5, help="Weight for parameter generation")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers")

    args = parser.parse_args()

    # Load tools data
    with open(args.tools_data, 'r') as f:
        tools_data = json.load(f)

    # Create evaluator
    evaluator = GroundTruthEvaluator(
        aggregation_method=args.aggregation_method,
        variance_penalty=args.variance_penalty,
        w1=args.w1,
        w2=args.w2,
        max_workers=args.max_workers
    )

    # Run evaluation
    results = evaluator.evaluate_all_tools(tools_data, args.output)

    # Print summary
    print("\n🏆 Ground Truth Selection Results:")
    for tool_name, variant in results["best_variants"].items():
        print(f"   {tool_name}: {variant['source']}")


if __name__ == "__main__":
    main()