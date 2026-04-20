"""
Unified evaluation statistics combining MCP and LLM metrics
"""
import json
import os
from typing import Dict, Any, Optional


class EvaluationStatistics:
    """
    Combine statistics from MCP call logger and LLM parameter generation logger
    into a unified statistics file
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def generate_statistics(
        self, 
        mcp_stats: Dict[str, Any],
        llm_stats: Dict[str, Any],
        step_wise_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate unified statistics from both loggers
        
        Args:
            mcp_stats: Statistics from MCP call logger
            llm_stats: Statistics from LLM parameter generation logger
            step_wise_stats: Statistics from step-wise evaluation (optional)
            
        Returns:
            Combined statistics dictionary
        """
        result = {
            "summary": {
                "total_api_calls": mcp_stats.get("total_calls", 0),
                "total_parameter_generations": llm_stats.get("total_calls", 0)
            },
            "api_execution": {
                "successful_calls": mcp_stats.get("successful_calls", 0),
                "failed_calls": mcp_stats.get("failed_calls", 0),
                "success_rate": mcp_stats.get("success_rate", 0.0),
                "api_call_counts": mcp_stats.get("api_call_counts", {}),
                "platform_counts": mcp_stats.get("platform_counts", {})
            },
            "parameter_validation": {
                "valid_parameters": llm_stats.get("valid_parameters", 0),
                "invalid_parameters": llm_stats.get("invalid_parameters", 0),
                "parameter_validation_rate": llm_stats.get("parameter_validation_rate", 0.0),
                "validation_error_breakdown": llm_stats.get("parameter_validation_details", {})
            },
            "error_analysis": {
                "error_categories": llm_stats.get("error_categories", {}),
                "mcp_error_analysis": mcp_stats.get("error_analysis", {})
            }
        }
        
        # Add step-wise evaluation metrics if provided
        if step_wise_stats:
            result["step_wise_evaluation"] = {
                "exact_match_accuracy": {
                    "mean": step_wise_stats.get("overall_exact_match_accuracy_mean", 0.0),
                    "std": step_wise_stats.get("overall_exact_match_accuracy_std", 0.0),
                },
                "sel_acc_and_param_valid": {
                    "mean": step_wise_stats.get("sel_acc_and_param_valid_mean", 0.0),
                    "std": step_wise_stats.get("sel_acc_and_param_valid_std", 0.0),
                },
                "sel_acc_and_api_success": {
                    "mean": step_wise_stats.get("sel_acc_and_api_success_mean", 0.0),
                    "std": step_wise_stats.get("sel_acc_and_api_success_std", 0.0),
                },
                "total_scenarios": step_wise_stats.get("total_scenarios", 0),
                "total_runs_all_scenarios": step_wise_stats.get("total_runs_all_scenarios", 0),
                "total_api_requiring_runs": step_wise_stats.get("total_api_requiring_runs", 0),
                "total_correct_exact_matches": step_wise_stats.get("total_correct_exact_matches", 0),
            }
            result["query_level_perfect_execution"] = {
                "perfect_api_selection_rate": step_wise_stats.get("query_perfect_api_selection_rate", 0.0),
                "perfect_parameter_rate": step_wise_stats.get("query_perfect_parameter_rate", 0.0),
                "perfect_api_success_rate": step_wise_stats.get("query_perfect_api_success_rate", 0.0),
                "perfect_sel_acc_and_param_valid_rate": step_wise_stats.get("query_perfect_sel_acc_and_param_valid_rate", 0.0),
                "perfect_sel_acc_and_api_success_rate": step_wise_stats.get("query_perfect_sel_acc_and_api_success_rate", 0.0)
            }

            result["raw_data"] = {
                "overall_exact_match_accuracy": step_wise_stats.get("overall_exact_match_accuracy", []),
                "sel_acc_and_param_valid": step_wise_stats.get("sel_acc_and_param_valid", []),
                "sel_acc_and_api_success": step_wise_stats.get("sel_acc_and_api_success", []),
                
                "perfect_api_selection": step_wise_stats.get("query_perfect_api_selection", []),
                "perfect_parameter": step_wise_stats.get("query_perfect_parameter", []),
                "perfect_api_success": step_wise_stats.get("query_perfect_api_success", []),
                "perfect_sel_acc_and_param_valid": step_wise_stats.get("query_perfect_sel_acc_and_param_valid", []),
                "perfect_sel_acc_and_api_success": step_wise_stats.get("query_perfect_sel_acc_and_api_success", [])
            }

        
        return result
    
    def save_statistics(
        self,
        mcp_stats: Dict[str, Any],
        llm_stats: Dict[str, Any],
        step_wise_stats: Optional[Dict[str, Any]] = None,
        filename: str = "evaluation_statistics.json"
    ) -> None:
        """
        Save unified statistics to file
        
        Args:
            mcp_stats: Statistics from MCP call logger
            llm_stats: Statistics from LLM parameter generation logger
            step_wise_stats: Statistics from step-wise evaluation (optional)
            filename: Output filename (default: evaluation_statistics.json)
        """
        stats = self.generate_statistics(mcp_stats, llm_stats, step_wise_stats)
        
        output_path = os.path.join(self.output_dir, filename)
        try:
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"📊 Unified evaluation statistics saved to: {output_path}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save evaluation statistics: {e}")


# Global instance
_global_eval_stats: Optional[EvaluationStatistics] = None


def initialize_evaluation_statistics(output_dir: str) -> EvaluationStatistics:
    """Initialize the global evaluation statistics"""
    global _global_eval_stats
    _global_eval_stats = EvaluationStatistics(output_dir=output_dir)
    return _global_eval_stats


def get_evaluation_statistics() -> Optional[EvaluationStatistics]:
    """Get the global evaluation statistics instance"""
    return _global_eval_stats


