"""
LLM Call Logger - Logs all LLM parameter generation calls for evaluation
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class LLMCallLogger:
    """Logger for LLM parameter generation calls including prompts, responses, and context"""
    
    def __init__(self, output_dir: Optional[str] = None, enabled: bool = True):
        """
        Initialize LLM call logger
        
        Args:
            output_dir: Directory to save logs. If None, uses current working directory
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.output_dir = output_dir
        self.call_history = []
        
        if self.enabled and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.log_file = os.path.join(self.output_dir, "llm_parameter_generation_log.jsonl")
            self.api_selection_log_file = os.path.join(self.output_dir, "llm_api_selection_log.jsonl")
            print(f"📝 LLM Call Logger initialized - logging to:")
            print(f"   Parameter Generation: {self.log_file}")
            print(f"   API Selection: {self.api_selection_log_file}")
        else:
            self.log_file = None
            self.api_selection_log_file = None
    
    def log_parameter_generation_call(self, 
                                     query_id: Optional[str] = None,
                                     subtask_id: Optional[int] = None,
                                     subtask_input: str = "",
                                     original_query: str = "",
                                     llm_prompt: str = "",
                                     llm_response: Any = None,
                                     golden_api: Optional[Dict[str, Any]] = None,
                                     api_success: Optional[bool] = None,
                                     api_response: Any = None,
                                     api_error_message: str = "",
                                     parameter_quality_evaluation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log a single LLM parameter generation call with API execution results
        
        Args:
            query_id: ID of the query being processed
            subtask_id: ID of the subtask
            subtask_input: The subtask question/input
            original_query: The original full query
            llm_prompt: The prompt sent to LLM
            llm_response: The response from LLM (can be dict, str, or parsed JSON)
            golden_api: Expected golden API with name, description, parameters
            api_success: Whether the API call was successful (None if not executed yet)
            api_response: The response from the API call (None if not executed yet)
            api_error_message: Error message from API (e.g., "HTTP 400: {...}")
            parameter_quality_evaluation: Parameter quality evaluation results including:
                - params_valid: whether params match schema (NEW METRIC)
                - api_error_category: category of API error (PARAMETER_ERROR, EXTERNAL_CONSTRAINT, etc.)
                - api_success: whether API call succeeded (ORIGINAL METRIC)
            
        Returns:
            The log entry that was created
        """
        if not self.enabled:
            return {}
        
        # Parse LLM response if it's a string
        if isinstance(llm_response, str):
            try:
                llm_response_parsed = json.loads(llm_response)
            except:
                llm_response_parsed = {"raw_response": llm_response}
        else:
            llm_response_parsed = llm_response
        
        # Extract api_error_category and api_error_reason to top level
        api_error_category = ""
        api_error_reason = ""
        param_quality_eval_clean = {}
        
        if parameter_quality_evaluation:
            api_error_category = parameter_quality_evaluation.get("api_error_category", "")
            api_error_reason = parameter_quality_evaluation.get("api_error_reason", "")
            # Create clean version without api_error_category and api_error_reason
            param_quality_eval_clean = {
                k: v for k, v in parameter_quality_evaluation.items()
                if k not in ["api_error_category", "api_error_reason"]
            }
        
        log_entry = {
            "query_id": query_id,
            "subtask_id": subtask_id,
            "subtask_input": subtask_input,
            "original_query": original_query,
            "llm_prompt": llm_prompt,
            "llm_response": llm_response_parsed,
            "golden_api": golden_api or {},
            "api_success": api_success,
            "api_response": api_response,
            "api_error_message": api_error_message,
            "api_error_category": api_error_category,
            "api_error_reason": api_error_reason,
            "parameter_quality_evaluation": param_quality_eval_clean,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.call_history.append(log_entry)
        
        # Write to file immediately (append mode)
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"⚠️  Warning: Failed to write LLM call log: {e}")
        
        # Print summary
        status = "✅" if api_success else "❌" if api_success is not None else "⏳"
        param_status = ""
        if parameter_quality_evaluation:
            params_valid = parameter_quality_evaluation.get("params_valid", None)
            if params_valid is not None:
                param_status = f" | Params: {'✅' if params_valid else '❌'}"
        print(f"   {status} Parameter Generation Log: Q{query_id} ST{subtask_id} | API: {api_success}{param_status}")
        
        return log_entry

    def log_api_selection_call(self,
                              query_id: Optional[str] = None,
                              subtask_id: Optional[int] = None,
                              subtask_input: str = "",
                              original_query: str = "",
                              llm_prompt: str = "",
                              llm_response: Any = None,
                              available_tools: List[Dict[str, Any]] = None,
                              selected_provider: Optional[str] = None,
                              selected_api: Optional[str] = None,
                              expected_golden_api: str = "",
                              exact_match_accuracy: float = 0.0,
                              reasoning: str = "",
                              model_name: str = "gpt-4.1-2025-04-14",
                              temperature: float = 0.2,
                              max_tokens: int = 32768,
                              llm_seed: Optional[int] = None,
                              previous_context: List[Dict[str, Any]] = None,
                              selection_confidence: Optional[str] = None,
                              error_type: Optional[str] = None,
                              error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Log a single LLM API selection call with results

        Args:
            query_id: ID of the query being processed
            subtask_id: ID of the subtask
            subtask_input: The subtask question/input
            original_query: The original full query
            llm_prompt: The prompt sent to LLM for API selection
            llm_response: The response from LLM (can be dict, str, or parsed JSON)
            available_tools: List of tools available for selection
            selected_provider: Provider selected by LLM (None if not provided)
            selected_api: API name selected by LLM (None if selection failed)
            expected_golden_api: Ground truth API name
            exact_match_accuracy: 1.0 if selected == expected, 0.0 otherwise
            reasoning: Selection reasoning/explanation
            model_name: LLM model used
            temperature: Temperature parameter used
            max_tokens: Max tokens parameter used
            llm_seed: Random seed used for reproducibility
            previous_context: Context from previous subtask executions
            selection_confidence: Confidence level if provided by LLM
            error_type: Error category if selection failed
            error_message: Detailed error message if applicable

        Returns:
            The log entry that was created
        """
        if not self.enabled:
            return {}

        # Parse LLM response if it's a string
        if isinstance(llm_response, str):
            try:
                llm_response_parsed = json.loads(llm_response)
            except:
                llm_response_parsed = {"raw_response": llm_response}
        else:
            llm_response_parsed = llm_response or {}

        log_entry = {
            "query_id": query_id,
            "subtask_id": subtask_id,
            "subtask_input": subtask_input,
            "original_query": original_query,
            "llm_prompt": llm_prompt,
            "llm_response": llm_response_parsed,
            "available_tools": available_tools or [],
            "selected_provider": selected_provider,
            "selected_api": selected_api,
            "expected_golden_api": expected_golden_api,
            "exact_match_accuracy": exact_match_accuracy,
            "reasoning": reasoning,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "previous_context": previous_context or []
        }

        # Only include optional fields if they have values
        if llm_seed is not None:
            log_entry["llm_seed"] = llm_seed
        if selection_confidence is not None:
            log_entry["selection_confidence"] = selection_confidence
        if error_type is not None:
            log_entry["error_type"] = error_type
        if error_message is not None:
            log_entry["error_message"] = error_message

        # Add to history
        self.call_history.append(log_entry)

        # Write to API selection log file (separate from parameter generation)
        if self.api_selection_log_file:
            try:
                with open(self.api_selection_log_file, 'a') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"⚠️  Warning: Failed to write API selection log: {e}")

        # Print summary
        status = "✅" if exact_match_accuracy == 1.0 else "❌"
        confidence_str = f" | Confidence: {selection_confidence}" if selection_confidence else ""
        error_str = f" | Error: {error_type}" if error_type else ""
        print(
            f"   {status} API Selection Log: Q{query_id} ST{subtask_id} | "
            f"Selected: ({selected_provider}, {selected_api}) | Expected: {expected_golden_api}"
            f"{confidence_str}{error_str}"
        )

        return log_entry

    def save_summary(self, summary_file: Optional[str] = None) -> None:
        """
        Save detailed call history (without aggregate statistics)
        Statistics are saved separately in evaluation_statistics.json
        
        Args:
            summary_file: Path to summary file. If None, saves to output_dir
        """
        if not self.enabled or not self.call_history:
            return
        
        # Build detailed call history only
        call_history = []
        for entry in self.call_history:
            history_entry = {
                "query_id": entry.get("query_id"),
                "subtask_id": entry.get("subtask_id"),
                "subtask_input": entry.get("subtask_input", ""),
                "original_query": entry.get("original_query", ""),
                "llm_prompt": entry.get("llm_prompt", ""),
                "llm_response": entry.get("llm_response", {}),
                "golden_api": entry.get("golden_api", {}),
                "api_success": entry.get("api_success"),
                "api_response": entry.get("api_response"),
                "api_error_message": entry.get("api_error_message", ""),
                "api_error_category": entry.get("api_error_category", ""),
                "api_error_reason": entry.get("api_error_reason", ""),
                "parameter_quality_evaluation": entry.get("parameter_quality_evaluation", {}),
                "timestamp": entry.get("timestamp", "")
            }
            call_history.append(history_entry)
        
        # Determine output file
        if summary_file is None and self.output_dir:
            summary_file = os.path.join(self.output_dir, "llm_parameter_generation_summary.json")
        
        if summary_file:
            try:
                with open(summary_file, 'w') as f:
                    json.dump(call_history, f, indent=2, ensure_ascii=False)
                print(f"💾 LLM call history saved to: {summary_file}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to save LLM call summary: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged API calls and parameter validation"""
        if not self.call_history:
            return {
                "total_calls": 0,
                "successful_api_calls": 0,
                "failed_api_calls": 0,
                "api_success_rate": 0.0,
                "valid_parameters": 0,
                "invalid_parameters": 0,
                "parameter_validation_rate": 0.0,
                "error_categories": {},
                "parameter_validation_details": {}
            }
        
        total_calls = len(self.call_history)
        successful_api_calls = sum(1 for entry in self.call_history if entry.get("api_success", False))
        failed_api_calls = total_calls - successful_api_calls
        
        # Parameter validation statistics
        valid_parameters = 0
        invalid_parameters = 0
        error_categories = {}
        validation_error_counts = {
            "missing_required": 0,
            "type_mismatch": 0,
            "unexpected_params": 0,
            "empty_or_hallucination": 0  # LLM generated empty/None values or hallucinated non-existent IDs
        }
        
        for entry in self.call_history:
            param_eval = entry.get("parameter_quality_evaluation", {})
            params_valid = param_eval.get("params_valid", False)
            
            if params_valid:
                valid_parameters += 1
            else:
                invalid_parameters += 1
                
                # Count validation error types ONLY for invalid parameters
                validation_details = param_eval.get("parameter_validation_details", {})
                has_schema_error = False
                
                if validation_details.get("required_params_missing"):
                    validation_error_counts["missing_required"] += 1
                    has_schema_error = True
                if validation_details.get("type_mismatches"):
                    validation_error_counts["type_mismatch"] += 1
                    has_schema_error = True
                if validation_details.get("unexpected_params"):
                    validation_error_counts["unexpected_params"] += 1
                    has_schema_error = True
                
                # If no schema error but params are invalid, it must be empty/hallucination
                if not has_schema_error:
                    validation_error_counts["empty_or_hallucination"] += 1
            
            # Count error categories for ALL entries (not just invalid)
            error_cat = entry.get("api_error_category", "NONE")
            error_categories[error_cat] = error_categories.get(error_cat, 0) + 1
        
        return {
            "total_calls": total_calls,
            "successful_api_calls": successful_api_calls,
            "failed_api_calls": failed_api_calls,
            "api_success_rate": successful_api_calls / total_calls if total_calls > 0 else 0.0,
            "valid_parameters": valid_parameters,
            "invalid_parameters": invalid_parameters,
            "parameter_validation_rate": valid_parameters / total_calls if total_calls > 0 else 0.0,
            "error_categories": error_categories,
            "parameter_validation_details": validation_error_counts
        }


# Global logger instance
_global_llm_logger: Optional[LLMCallLogger] = None


def initialize_global_llm_logger(output_dir: str, enabled: bool = True) -> LLMCallLogger:
    """Initialize the global LLM logger"""
    global _global_llm_logger
    _global_llm_logger = LLMCallLogger(output_dir=output_dir, enabled=enabled)
    return _global_llm_logger


def get_global_llm_logger() -> Optional[LLMCallLogger]:
    """Get the global LLM logger instance"""
    return _global_llm_logger
