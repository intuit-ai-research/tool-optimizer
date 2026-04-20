"""
MCP Pattern Analyzer for extracting contrastive patterns from MCP call history.

This analyzer extracts positive and negative usage patterns from evaluation results
that include MCP call logs and question-level outcomes, enabling contrastive learning
for description improvement.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, Counter

from ..interfaces.pattern_interface import (
    PatternAnalyzer, PatternAnalysisInput, PatternAnalysisResult, 
    ToolPattern, PatternSourceType
)
from ..interfaces.dataset_interface import StandardizedTool
from ..core.registries import PatternAnalyzerRegistry
from ..core.prompt_manager import ContrastivePromptManager

logger = logging.getLogger(__name__)


class MCPPatternAnalyzer(PatternAnalyzer):
    """
    Pattern analyzer that extracts contrastive learning patterns from MCP call history.
    
    Analyzes both successful and failed tool usage to identify:
    1. Tool selection patterns (which tools are correctly/incorrectly chosen)
    2. Tool usage patterns (which parameter combinations succeed/fail)
    3. Answer quality patterns (which tool usage leads to correct/incorrect answers)
    """
    
    def __init__(self, min_samples_per_tool: int = 2, max_error_samples: int = 10,
                 selection_threshold: float = 0.9, params_valid_threshold: float = 0.7,
                 extract_guidelines: bool = False, llm_config: Dict[str, Any] = None,
                 prompt_config: Dict[str, Any] = None, guideline_constraints: Dict[str, Any] = None):
        """
        Initialize MCP pattern analyzer.
        
        Args:
            min_samples_per_tool: Minimum samples needed to generate patterns for a tool
            max_error_samples: Maximum error samples to include per tool (to avoid noise)
            selection_threshold: Minimum exact_match_accuracy for positive selection samples (default 0.9)
            params_valid_threshold: Minimum params_valid rate for positive usage samples (default 0.7)
            extract_guidelines: Whether to extract guidelines from patterns using LLM
            llm_config: Configuration for LLM calls (model, temperature, etc.)
            prompt_config: Configuration for prompt templates and paths
            guideline_constraints: Configuration for guideline count limits (max_selection_guidelines, max_usage_guidelines)
        """
        self.min_samples_per_tool = min_samples_per_tool
        self.max_error_samples = max_error_samples
        self.selection_threshold = selection_threshold
        self.params_valid_threshold = params_valid_threshold
        self.extract_guidelines = extract_guidelines
        self.llm_config = llm_config or {
            "model_name": "gpt-4",
            "temperature": 0.3,
            "max_tokens": 600
        }
        self.prompt_config = prompt_config or {}
        self.guideline_constraints = guideline_constraints or {
            "max_selection_guidelines": 10,
            "max_usage_guidelines": 10,
            "use_constrained_prompts": True
        }
        
        # Initialize prompt manager
        prompt_base_dir = self.prompt_config.get("base_dir", "prompts")
        self.prompt_manager = ContrastivePromptManager(prompt_base_dir)
    
    @property
    def supported_source_types(self) -> List[PatternSourceType]:
        """Return supported source types."""
        return [PatternSourceType.MCP_LOGS]
    
    def analyze_usage_patterns(self, analysis_input: PatternAnalysisInput) -> PatternAnalysisResult:
        """
        Analyze MCP usage patterns from standardized input.
        
        Args:
            analysis_input: Standardized pattern analysis input
            
        Returns:
            PatternAnalysisResult with standardized tool patterns
        """
        return self._analyze_patterns_new_interface(analysis_input)
    
    def analyze_usage_patterns_legacy(self, evaluation_results: Dict[str, Any], 
                                     tools: List[StandardizedTool]) -> Dict[str, Any]:
        """
        Legacy interface for backward compatibility.
        
        Args:
            evaluation_results: Results from evaluation containing output_directory
            tools: List of tools being analyzed
            
        Returns:
            Dictionary containing extracted contrastive patterns
        """
        # Create standardized input
        analysis_input = PatternAnalysisInput(
            tools=tools,
            evaluation_results=evaluation_results,
            source_type=PatternSourceType.MCP_LOGS
        )
        
        # Analyze using new interface
        result = self._analyze_patterns_new_interface(analysis_input)
        
        # Convert back to legacy format
        return self._convert_to_legacy_format(result)
    
    def _analyze_patterns_new_interface(self, analysis_input: PatternAnalysisInput) -> PatternAnalysisResult:
        """Core pattern analysis implementation using new interface."""
        try:
            # Get output directory from evaluation results
            output_dir = analysis_input.get_output_directory()
            if not output_dir:
                logger.warning("No output_directory found in evaluation results")
                return self._empty_result(analysis_input)
            
            output_path = Path(output_dir)
            if not output_path.exists():
                logger.warning(f"Output directory does not exist: {output_path}")
                return self._empty_result(analysis_input)
            
            # Parse MCP calls and steps
            mcp_calls = self._parse_mcp_calls(output_path)
            steps_data = self._parse_steps_data(output_path)
            
            if not mcp_calls:
                logger.warning("No MCP calls found in evaluation results")
                return self._empty_result(analysis_input)
            
            # Extract tool patterns using legacy method
            legacy_tool_patterns = self._extract_tool_patterns(mcp_calls, steps_data, analysis_input.tools)
            
            # Convert to new ToolPattern format and create tool lookup
            tools_by_name = {tool.name: tool for tool in analysis_input.tools}
            tool_patterns = {}
            for tool_name, legacy_data in legacy_tool_patterns.items():
                tool_pattern = self._convert_legacy_tool_data(tool_name, legacy_data)
                
                # Extract guidelines if enabled
                if self.extract_guidelines:
                    # Get the tool object for this tool name
                    tool_obj = tools_by_name.get(tool_name)
                    tool_pattern = self._extract_guidelines_for_tool(tool_name, tool_pattern, tool_obj)
                
                tool_patterns[tool_name] = tool_pattern
            
            result = PatternAnalysisResult(
                tool_patterns=tool_patterns,
                total_tools_analyzed=len(analysis_input.tools),
                total_data_points=len(mcp_calls),
                data_source=PatternSourceType.MCP_LOGS
            )
            
            logger.info(f"Extracted patterns for {len(tool_patterns)} tools from {len(mcp_calls)} MCP calls")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing MCP usage patterns: {e}")
            return self._empty_result(analysis_input)
    
    def extract_tool_insights(self, tool_name: str, 
                             patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract insights for specific tool from patterns.
        
        Args:
            tool_name: Name of the tool
            patterns: Usage patterns dictionary from analyze_usage_patterns
            
        Returns:
            Tool-specific insights with positive and negative samples
        """
        tool_patterns = patterns.get("tool_patterns", {}).get(tool_name, {})
        
        if not tool_patterns:
            return {
                "tool_name": tool_name,
                "has_patterns": False,
                "positive_samples": [],
                "negative_samples": [],
                "insights": "No usage data available for this tool"
            }
        
        insights = {
            "tool_name": tool_name,
            "has_patterns": True,
            # Provide both separate and combined pattern collections
            "selection_positive_samples": tool_patterns.get("selection_positive_samples", []),
            "selection_negative_samples": tool_patterns.get("selection_negative_samples", []),
            "usage_positive_samples": tool_patterns.get("usage_positive_samples", []),
            "usage_negative_samples": tool_patterns.get("usage_negative_samples", []),
            # Legacy fields for backward compatibility (populated from usage patterns)
            "positive_samples": tool_patterns.get("positive_samples", []),
            "negative_samples": tool_patterns.get("negative_samples", []),
            "usage_stats": tool_patterns.get("usage_stats", {}),
            "common_errors": tool_patterns.get("common_errors", []),
            "success_indicators": tool_patterns.get("success_indicators", []),
            "parameter_patterns": tool_patterns.get("parameter_patterns", {}),
            "insights": self._generate_tool_insights(tool_patterns)
        }
        
        return insights
    
    def _parse_mcp_calls(self, output_path: Path) -> List[Dict[str, Any]]:
        """Parse MCP calls from mcp_call_log.jsonl file."""
        # Try both possible filenames
        mcp_calls_file = output_path / "mcp_call_log.jsonl"
        if not mcp_calls_file.exists():
            mcp_calls_file = output_path / "mcp_calls.jsonl"
        
        mcp_calls = []
        
        if not mcp_calls_file.exists():
            logger.warning(f"MCP calls file not found: {mcp_calls_file}")
            return mcp_calls
        
        try:
            with open(mcp_calls_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        call_data = json.loads(line)
                        mcp_calls.append(call_data)
        except Exception as e:
            logger.error(f"Error parsing MCP calls file: {e}")
        
        return mcp_calls
    
    def _parse_steps_data(self, output_path: Path) -> List[Dict[str, Any]]:
        """Parse question steps from step results file."""
        # Try multiple possible filenames
        steps_file = output_path / "step_wise_eval_results.json"
        is_jsonl = False
        
        if not steps_file.exists():
            steps_file = output_path / "steps.jsonl"
            is_jsonl = True
            
        if not steps_file.exists():
            steps_file = output_path / "steps.json"
            is_jsonl = False
        
        steps_data = []
        
        if not steps_file.exists():
            logger.warning(f"Steps file not found: {steps_file}")
            return steps_data
        
        try:
            if is_jsonl:
                # Parse JSONL format
                with open(steps_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            step_data = json.loads(line)
                            steps_data.append(step_data)
            else:
                # Parse JSON format
                with open(steps_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        steps_data = data
                    elif isinstance(data, dict):
                        # If it's a dict, try to extract the list
                        steps_data = data.get('results', data.get('steps', [data]))
        except Exception as e:
            logger.error(f"Error parsing steps file: {e}")
        
        return steps_data
    
    def _normalize_steps_data(self, steps_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Normalize different step data formats into a consistent structure.
        
        Handles both:
        1. Simple format: [{"question_id": ..., "correct": ...}, ...]
        2. Complex format: [{"main_results": {"step_wise_results": [...]}, ...}]
        """
        question_outcomes = {}
        
        if not steps_data:
            return question_outcomes
        
        # Check if this is the complex nested format
        if isinstance(steps_data, list) and len(steps_data) > 0:
            first_item = steps_data[0]
            
            # Handle complex step_wise_eval format
            if "main_results" in first_item:
                for data_item in steps_data:
                    # Extract query data to get golden_api from subtasks
                    query_data = data_item.get("query_data", {})
                    subtasks = query_data.get("subtasks", [])

                    # Create mapping of subtask_id to golden_api
                    golden_api_map = {}
                    for i, subtask in enumerate(subtasks, 1):  # subtasks are 1-indexed
                        golden_api_map[i] = subtask.get("golden_api", "")

                    main_results = data_item.get("main_results", {})
                    step_wise_results = main_results.get("step_wise_results", [])

                    for step_result in step_wise_results:
                        subtask_id = step_result.get("subtask_id")
                        query_id = step_result.get("query_id", data_item.get("query_id", "unknown"))

                        if not subtask_id:
                            continue

                        # Extract evaluation data from statistics and runs
                        statistics = step_result.get("statistics", {})
                        runs = step_result.get("runs", [])

                        # Get expected golden API from the subtasks mapping
                        expected_golden_api = golden_api_map.get(subtask_id, "")

                        # Use subtask_id as key (MCP calls use subtask_id to match)
                        # Store both forms for compatibility
                        question_outcomes[subtask_id] = {
                            "question_id": query_id,
                            "subtask_id": subtask_id,
                            "statistics": statistics,
                            "runs": runs,
                            "scenario": step_result.get("scenario"),
                            "expected_golden_api": expected_golden_api,
                            "total_runs": len(runs)
                        }

                        # Also create composite key for more specific matching
                        composite_key = f"{query_id}_{subtask_id}"
                        question_outcomes[composite_key] = question_outcomes[subtask_id]
            
            # Handle simple format with question_id
            elif "question_id" in first_item:
                for step in steps_data:
                    qid = step.get("question_id")
                    if qid:
                        question_outcomes[qid] = step
            
            # Handle format with subtask_id directly
            elif "subtask_id" in first_item:
                for step in steps_data:
                    sid = step.get("subtask_id")
                    if sid:
                        question_outcomes[sid] = step
        
        return question_outcomes
    
    def _extract_tool_patterns(self, mcp_calls: List[Dict[str, Any]], 
                              steps_data: List[Dict[str, Any]], 
                              tools: List[StandardizedTool]) -> Dict[str, Dict[str, Any]]:
        """Extract contrastive patterns for each tool."""
        
        # Normalize steps_data to handle different formats
        question_outcomes = self._normalize_steps_data(steps_data)
        
        # Group MCP calls by tool name
        tool_calls = defaultdict(list)
        for call in mcp_calls:
            # Extract tool name from call_signature if not directly available
            tool_name = call.get("tool_name")
            if not tool_name and "call_signature" in call:
                call_sig = call["call_signature"]
                # Handle both dict and string formats
                if isinstance(call_sig, dict):
                    tool_name = call_sig.get("api_name") or call_sig.get("tool_name")
                elif isinstance(call_sig, str) and "(" in call_sig:
                    # Parse from string format: "tool_name(params)"
                    tool_name = call_sig.split("(")[0].strip()
            
            if tool_name:
                tool_calls[tool_name].append(call)
        
        # Create tool name mapping for available tools
        available_tools = {tool.name: tool for tool in tools}
        
        tool_patterns = {}
        
        for tool_name, calls in tool_calls.items():
            if len(calls) < self.min_samples_per_tool:
                continue
            
            patterns = self._analyze_tool_calls(tool_name, calls, question_outcomes, available_tools)
            # Check if we have any meaningful patterns (selection or usage)
            has_selection_patterns = patterns["selection_positive_samples"] or patterns["selection_negative_samples"]
            has_usage_patterns = patterns["usage_positive_samples"] or patterns["usage_negative_samples"]
            if has_selection_patterns or has_usage_patterns:
                tool_patterns[tool_name] = patterns
        
        # Also analyze tools that were never called but are available
        for tool_name, tool in available_tools.items():
            if tool_name not in tool_patterns:
                # This tool was never used - could indicate selection issues
                tool_patterns[tool_name] = {
                    "positive_samples": [],
                    "negative_samples": [],
                    "usage_stats": {"never_used": True, "total_calls": 0},
                    "common_errors": ["Tool never selected despite being available"],
                    "success_indicators": [],
                    "parameter_patterns": {},
                    "insights": "This tool was never used in the evaluation"
                }
        
        return tool_patterns
    
    def _analyze_tool_calls(self, tool_name: str, calls: List[Dict[str, Any]], 
                           question_outcomes: Dict[str, Dict[str, Any]], 
                           available_tools: Dict[str, StandardizedTool]) -> Dict[str, Any]:
        """Analyze calls for a specific tool to extract positive and negative patterns."""

        # Separate collections for different pattern types
        selection_positive_samples = []
        selection_negative_samples = []
        usage_positive_samples = []
        usage_negative_samples = []

        # Legacy collections for backward compatibility
        positive_samples = []
        negative_samples = []
        common_errors = []
        success_indicators = []
        parameter_patterns = defaultdict(int)

        # Track skipped samples
        skipped_samples_count = 0
        
        # Analyze each call
        for call in calls:
            # Handle different ID field names - use subtask_id as primary key to match question_outcomes
            subtask_id = call.get("subtask_id")
            query_id = call.get("query_id")

            # Try different matching strategies
            qid = subtask_id or call.get("qid") or query_id
            
            # Extract result/response and status
            result_text = ""
            status = call.get("status", "unknown")
            error = call.get("error")
            
            if "response" in call:
                response = call["response"]
                if isinstance(response, dict):
                    # Handle structured response format: {"success": bool, "data": ..., "error": ...}
                    if "success" in response:
                        status = "success" if response.get("success") else "failed"
                    if "error" in response and response.get("error"):
                        error = response.get("error")
                    # Extract data from response
                    if "data" in response:
                        result_text = str(response.get("data", ""))
                    else:
                        result_text = str(response.get("result", response.get("text", "")))
                else:
                    result_text = str(response)
            elif "result" in call:
                result = call["result"]
                if isinstance(result, dict):
                    result_text = result.get("text", str(result))
                else:
                    result_text = str(result)
            
            # Extract parameters from multiple possible locations
            kwargs = call.get("kwargs", {})
            if not kwargs and "call_signature" in call:
                call_sig = call["call_signature"]
                if isinstance(call_sig, dict):
                    kwargs = call_sig.get("parameters", {})
            if not kwargs and "metadata" in call:
                metadata = call["metadata"]
                if isinstance(metadata, dict):
                    kwargs = metadata.get("parameters", metadata.get("kwargs", {}))
            
            duration_ms = call.get("duration_ms", 0)
            
            # Classify for both selection and usage patterns separately
            is_positive_selection = self._is_positive_selection_sample(call, qid, question_outcomes, self.selection_threshold)
            is_positive_usage = self._is_positive_usage_sample(call, qid, question_outcomes, self.params_valid_threshold)

            # Skip this sample entirely if we can't classify it due to missing evaluation data or golden_api=""
            if is_positive_selection is None and is_positive_usage is None:
                # Both classifications returned None (no evaluation data or golden_api="") - skip this sample
                skipped_samples_count += 1
                if qid and qid in question_outcomes:
                    expected_api = question_outcomes[qid].get("expected_golden_api", "")
                    if expected_api == "":
                        logger.debug(f"Skipping MCP call for {tool_name} - golden_api is empty for qid: {qid} (no API usage expected)")
                    else:
                        logger.debug(f"Skipping MCP call for {tool_name} - missing evaluation data for qid: {qid}")
                else:
                    logger.debug(f"Skipping MCP call for {tool_name} - no evaluation data for qid: {qid}")
                continue

            # Create enhanced sample data with evaluation metrics
            sample = self._create_enhanced_sample_data(call, qid, question_outcomes, kwargs, result_text, status, error, duration_ms)

            # Add to selection collections (only if we have selection evaluation data)
            if is_positive_selection is not None:
                if is_positive_selection:
                    selection_positive_samples.append(sample)
                else:
                    if len(selection_negative_samples) < self.max_error_samples:
                        selection_negative_samples.append(sample)

            # Add to usage collections (only if we have usage evaluation data)
            if is_positive_usage is not None:
                if is_positive_usage:
                    usage_positive_samples.append(sample)
                    success_indicators.append(self._extract_success_indicator(call))
                else:
                    if len(usage_negative_samples) < self.max_error_samples:
                        usage_negative_samples.append(sample)
                    common_errors.append(self._extract_error_pattern(call))

                # Populate legacy collections from usage patterns (only if we have usage evaluation data)
                if is_positive_usage:
                    positive_samples.append(sample)
                else:
                    if len(negative_samples) < self.max_error_samples:
                        negative_samples.append(sample)
            
            # Track parameter patterns
            for param_name, param_value in kwargs.items():
                param_key = f"{param_name}:{type(param_value).__name__}"
                parameter_patterns[param_key] += 1
        
        # Clean up error patterns and success indicators
        common_errors = list(set(filter(None, common_errors)))[:5]  # Top 5 unique errors
        success_indicators = list(set(filter(None, success_indicators)))[:5]  # Top 5 unique indicators

        # Warn if too many samples were skipped due to missing evaluation data or golden_api=""
        if skipped_samples_count > 0:
            skip_percentage = (skipped_samples_count / len(calls)) * 100 if calls else 0
            if skip_percentage > 50:
                logger.warning(f"High percentage of samples skipped for {tool_name}: {skip_percentage:.1f}% ({skipped_samples_count}/{len(calls)}) - missing evaluation data or golden_api=''")
            elif skip_percentage > 20:
                logger.info(f"Some samples skipped for {tool_name}: {skip_percentage:.1f}% ({skipped_samples_count}/{len(calls)}) - missing evaluation data or golden_api=''")
        
        return {
            # New separate pattern collections
            "selection_positive_samples": selection_positive_samples,
            "selection_negative_samples": selection_negative_samples,
            "usage_positive_samples": usage_positive_samples,
            "usage_negative_samples": usage_negative_samples,

            # Legacy collections for backward compatibility
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,

            # Usage statistics with separate metrics
            "usage_stats": {
                "total_calls": len(calls),
                "skipped_calls": skipped_samples_count,
                "analyzed_calls": len(calls) - skipped_samples_count,
                # Legacy stats
                "positive_calls": len(positive_samples),
                "negative_calls": len(negative_samples),
                "success_rate": len(positive_samples) / len(calls) if calls else 0,
                # New separate stats (calculated from analyzed calls, not total calls)
                "selection_positive_calls": len(selection_positive_samples),
                "selection_negative_calls": len(selection_negative_samples),
                "selection_success_rate": len(selection_positive_samples) / (len(selection_positive_samples) + len(selection_negative_samples)) if (len(selection_positive_samples) + len(selection_negative_samples)) > 0 else 0,
                "usage_positive_calls": len(usage_positive_samples),
                "usage_negative_calls": len(usage_negative_samples),
                "usage_success_rate": len(usage_positive_samples) / (len(usage_positive_samples) + len(usage_negative_samples)) if (len(usage_positive_samples) + len(usage_negative_samples)) > 0 else 0,
            },
            "common_errors": common_errors,
            "success_indicators": success_indicators,
            "parameter_patterns": dict(parameter_patterns),
            "insights": f"Tool used {len(calls)} times ({skipped_samples_count} skipped, {len(calls) - skipped_samples_count} analyzed) - Selection: {len(selection_positive_samples)} positive, {len(selection_negative_samples)} negative; Usage: {len(usage_positive_samples)} positive, {len(usage_negative_samples)} negative"
        }
    

    def _is_positive_selection_sample(self, call: Dict[str, Any], qid: Optional[str],
                                    question_outcomes: Dict[str, Dict[str, Any]],
                                    selection_threshold: float = 0.9) -> Optional[bool]:
        """
        Determine if call is positive for API selection patterns.

        Based on exact_match_accuracy from step-wise evaluation results.

        Args:
            call: MCP call data
            qid: Query/subtask ID
            question_outcomes: Evaluation outcomes with statistics
            selection_threshold: Minimum accuracy threshold for positive selection (default 0.9)

        Returns:
            True if positive selection sample, False if negative, None if no evaluation data (skip)
        """
        # Skip if no evaluation data - return None to indicate we should skip this sample
        if not qid or qid not in question_outcomes:
            return None

        outcome = question_outcomes[qid]

        # Skip if golden_api is empty (no API usage expected)
        expected_api = outcome.get("expected_golden_api", "")
        if expected_api == "":
            # No API should be used - skip selection pattern analysis for this subtask
            return None

        # Extract exact_match_accuracy from runs or statistics
        runs = outcome.get("runs", [])
        if runs:
            # Get accuracy from individual runs
            accuracies = [run.get("exact_match_accuracy", 0.0) for run in runs]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        else:
            # Fallback to statistics aggregate
            stats = outcome.get("statistics", {})
            exact_matches = stats.get("correct_exact_matches", 0)
            total_runs = stats.get("total_runs", 1)
            avg_accuracy = exact_matches / total_runs if total_runs > 0 else 0.0

        return avg_accuracy >= selection_threshold

    def _is_positive_usage_sample(self, call: Dict[str, Any], qid: Optional[str],
                                question_outcomes: Dict[str, Dict[str, Any]],
                                params_valid_threshold: float = 0.7) -> Optional[bool]:
        """
        Determine if call is positive for API usage/parameter patterns.

        Based on params_valid from parameter_quality_evaluation.

        Args:
            call: MCP call data
            qid: Query/subtask ID
            question_outcomes: Evaluation outcomes with parameter validation
            params_valid_threshold: Minimum params valid rate for positive usage (default 0.7)

        Returns:
            True if positive usage sample, False if negative, None if no evaluation data (skip)
        """
        # Skip if no evaluation data - return None to indicate we should skip this sample
        if not qid or qid not in question_outcomes:
            return None

        outcome = question_outcomes[qid]

        # Skip if golden_api is empty (no API usage expected)
        expected_api = outcome.get("expected_golden_api", "")
        if expected_api == "":
            # No API should be used - skip usage pattern analysis for this subtask
            return None

        # Extract params_valid from runs
        runs = outcome.get("runs", [])
        if runs:
            # Get params_valid from parameter quality evaluation
            valid_params = []
            for run in runs:
                param_eval = run.get("parameter_quality_evaluation", {})
                valid_params.append(param_eval.get("params_valid", False))

            valid_rate = sum(valid_params) / len(valid_params) if valid_params else 0.0
        else:
            # Fallback: check for technical errors in the call
            raise ValueError(f"No evaluation data found for qid: {qid}")

        return valid_rate >= params_valid_threshold

    def _has_technical_errors(self, call: Dict[str, Any]) -> bool:
        """Check if call has technical errors (for fallback usage classification)."""
        error = call.get("error")
        if error:
            return True

        # Check response for errors
        result_text = ""
        if "response" in call:
            response = call["response"]
            if isinstance(response, dict):
                result_text = str(response.get("result", response.get("text", "")))
            else:
                result_text = str(response)
        elif "result" in call:
            result = call["result"]
            result_text = str(result)

        if "Unknown tool" in result_text:
            return True
        if "Error:" in result_text or "error" in result_text.lower():
            return True

        status = call.get("status", "")
        if status and status != "success" and "success" not in status.lower():
            return True

        return False

    def _create_enhanced_sample_data(self, call: Dict[str, Any], qid: Optional[str],
                                   question_outcomes: Dict[str, Dict[str, Any]],
                                   kwargs: Dict[str, Any], result_text: str,
                                   status: str, error: Optional[str], duration_ms: int) -> Dict[str, Any]:
        """Create enhanced sample data with evaluation metrics."""
        sample = {
            "tool_name": call.get("call_signature", {}).get("api_name", "unknown"),
            "kwargs": kwargs,
            "result_preview": result_text[:200] + "..." if len(result_text) > 200 else result_text,
            "status": status,
            "error": error,
            "duration_ms": duration_ms,
            "question_id": qid,
            "subtask_id": call.get("subtask_id"),
        }

        # Add evaluation metrics if available
        if qid and qid in question_outcomes:
            outcome = question_outcomes[qid]

            # Add selection metrics
            runs = outcome.get("runs", [])
            if runs:
                # Get metrics from runs
                accuracies = [run.get("exact_match_accuracy", 0.0) for run in runs]
                sample["exact_match_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0
                sample["expected_api"] = runs[0].get("expected_golden_api", "unknown") if runs else "unknown"
                sample["selected_api"] = runs[0].get("selected_api", sample["tool_name"]) if runs else sample["tool_name"]

                # Get parameter validation metrics
                param_evaluations = [run.get("parameter_quality_evaluation", {}) for run in runs]
                if param_evaluations:
                    valid_params = [pe.get("params_valid", False) for pe in param_evaluations]
                    sample["params_valid"] = sum(valid_params) / len(valid_params) if valid_params else 0.0
                    sample["api_success"] = any(run.get("api_success", False) for run in runs)

                    # Add parameter details
                    if param_evaluations[0]:
                        pe = param_evaluations[0]
                        sample["parameter_validation_details"] = {
                            "required_params_present": pe.get("required_params_present", []),
                            "required_params_missing": pe.get("required_params_missing", []),
                            "optional_params_present": pe.get("optional_params_present", []),
                            "parameter_validation_errors": pe.get("parameter_validation_errors", [])
                        }
            else:
                # Fallback to statistics
                stats = outcome.get("statistics", {})
                exact_matches = stats.get("correct_exact_matches", 0)
                total_runs = stats.get("total_runs", 1)
                sample["exact_match_accuracy"] = exact_matches / total_runs if total_runs > 0 else 0.0
                sample["expected_api"] = outcome.get("expected_golden_api", "unknown")
                sample["selected_api"] = sample["tool_name"]

        return sample

    def _extract_error_pattern(self, call: Dict[str, Any]) -> Optional[str]:
        """Extract error pattern from a failed call."""
        # Extract result/response text
        result_text = ""
        if "response" in call:
            response = call["response"]
            if isinstance(response, dict):
                result_text = str(response.get("result", response.get("text", "")))
            else:
                result_text = str(response)
        elif "result" in call:
            result = call["result"]
            if isinstance(result, dict):
                result_text = result.get("text", str(result))
            else:
                result_text = str(result)
        
        error = call.get("error")
        
        if error:
            return f"Error: {error}"
        elif "Unknown tool" in result_text:
            return "Unknown tool error"
        elif "Error:" in result_text or "error" in result_text.lower():
            # Extract specific error from result
            if "Error:" in result_text:
                error_part = result_text.split("Error:")[-1].strip()[:100]
            else:
                error_part = result_text[:100]
            return f"Execution error: {error_part}"
        
        return None
    
    def _extract_success_indicator(self, call: Dict[str, Any]) -> Optional[str]:
        """Extract success indicator from a successful call."""
        # Extract result/response text
        result_text = ""
        if "response" in call:
            response = call["response"]
            if isinstance(response, dict):
                result_text = str(response.get("result", response.get("text", "")))
            else:
                result_text = str(response)
        elif "result" in call:
            result = call["result"]
            if isinstance(result, dict):
                result_text = result.get("text", str(result))
            else:
                result_text = str(result)
        
        kwargs = call.get("kwargs", {})
        
        # Success indicators
        if result_text and not any(err in result_text for err in ["Error", "Unknown", "Failed"]):
            if len(result_text) > 50:  # Substantial response
                return "Provided substantial response"
            elif result_text.strip():
                return "Provided valid response"
        
        # Parameter usage indicators
        if kwargs:
            return f"Successfully used {len(kwargs)} parameters"
        
        return "Completed without error"
    
    def _generate_tool_insights(self, tool_patterns: Dict[str, Any]) -> str:
        """Generate human-readable insights for a tool using separate pattern analysis."""
        usage_stats = tool_patterns.get("usage_stats", {})
        total_count = usage_stats.get("total_calls", 0)

        if usage_stats.get("never_used"):
            return "This tool was never used during evaluation. Consider improving discoverability or description clarity."

        if total_count == 0:
            return "No usage data available."

        # Get separate success rates
        selection_success_rate = usage_stats.get("selection_success_rate", 0.0)
        usage_success_rate = usage_stats.get("usage_success_rate", 0.0)

        insights = []

        # Selection insights
        if selection_success_rate >= 0.8:
            insights.append("Good API selection accuracy")
        elif selection_success_rate >= 0.5:
            insights.append("Moderate API selection accuracy - may need clearer selection guidelines")
        else:
            insights.append("Poor API selection accuracy - tool likely being chosen incorrectly")

        # Usage insights
        if usage_success_rate >= 0.8:
            insights.append("Good parameter usage")
        elif usage_success_rate >= 0.5:
            insights.append("Moderate parameter usage - may need clearer usage guidelines")
        else:
            insights.append("Poor parameter usage - likely parameter validation issues")

        # Add common error information
        common_errors = tool_patterns.get("common_errors", [])
        if common_errors:
            insights.append(f"Common issues: {common_errors[0]}")

        return ". ".join(insights)
    
    def _extract_guidelines_for_tool(self, tool_name: str, tool_pattern: ToolPattern, tool=None) -> ToolPattern:
        """
        Extract guidelines from tool patterns using LLM analysis.
        
        Args:
            tool_name: Name of the tool
            tool_pattern: Tool pattern with raw positive/negative samples
            tool: Optional StandardizedTool object for accessing description and parameters
            
        Returns:
            Updated tool pattern with extracted guidelines
        """
        if not self.extract_guidelines:
            return tool_pattern
            
        try:
            # Extract selection guidelines
            selection_guidelines = self._extract_selection_guidelines(tool_name, tool_pattern, tool)
            
            # Extract usage guidelines  
            usage_guidelines = self._extract_usage_guidelines(tool_name, tool_pattern, tool)
            
            # Update tool pattern with guidelines
            tool_pattern.selection_guidelines = selection_guidelines
            tool_pattern.usage_guidelines = usage_guidelines
            
            logger.info(f"Extracted {len(selection_guidelines)} selection + {len(usage_guidelines)} usage guidelines for {tool_name}")
            
        except Exception as e:
            logger.error(f"Error extracting guidelines for {tool_name}: {e}")
            tool_pattern.selection_guidelines = []
            tool_pattern.usage_guidelines = []
            
        return tool_pattern
    
    def _extract_selection_guidelines(self, tool_name: str, tool_pattern: ToolPattern, tool=None) -> List[str]:
        """Extract API selection guidelines using LLM analysis."""
        try:
            import json
            from ..core.llm_client import llm_call
            import time
            
            # Use selection-specific samples for API selection guideline extraction
            positive_samples = tool_pattern.selection_positive_samples or []
            negative_samples = tool_pattern.selection_negative_samples or []

            if not positive_samples and not negative_samples:
                return []

            # Format samples for LLM analysis with selection-focused metrics
            positive_formatted = []
            for sample in positive_samples:
                expected_api = sample.get("expected_api", "unknown")
                selected_api = sample.get("selected_api", sample.get("tool_name", "unknown"))
                accuracy = sample.get("exact_match_accuracy", 0.0)
                formatted = f"Query: {sample.get('kwargs', {})}, Expected API: {expected_api}, Selected API: {selected_api}, Accuracy: {accuracy:.2f}"
                positive_formatted.append(formatted)

            negative_formatted = []
            for sample in negative_samples:
                expected_api = sample.get("expected_api", "unknown")
                selected_api = sample.get("selected_api", sample.get("tool_name", "unknown"))
                accuracy = sample.get("exact_match_accuracy", 0.0)
                error_info = f", Error: {sample.get('error')}" if sample.get('error') else ""
                formatted = f"Query: {sample.get('kwargs', {})}, Expected API: {expected_api}, Selected API: {selected_api}, Accuracy: {accuracy:.2f}{error_info}"
                negative_formatted.append(formatted)
            
            # Load prompt template from file (check custom config first)
            pattern_config = self.prompt_config.get("pattern_analysis", {})
            selection_template_path = pattern_config.get("selection_guidelines")

            if selection_template_path:
                prompt_template = self.prompt_manager.load_template_from_path(selection_template_path)
            else:
                # Choose prompt template based on configuration
                use_constrained = self.guideline_constraints.get("use_constrained_prompts", True)
                template_name = "selection_guidelines_constrained" if use_constrained else "selection_guidelines"
                prompt_template = self.prompt_manager.load_pattern_analysis_prompt(template_name)
            
            positive_patterns_text = chr(10).join(positive_formatted) if positive_formatted else "No positive patterns available"
            negative_patterns_text = chr(10).join(negative_formatted) if negative_formatted else "No negative patterns available"
            
            # Get existing selection guidelines if available (for iterative refinement)
            existing_guidelines = tool_pattern.selection_guidelines if hasattr(tool_pattern, 'selection_guidelines') else []

            # Get max guidelines constraint (only used for constrained prompts)
            use_constrained = self.guideline_constraints.get("use_constrained_prompts", True)
            max_guidelines = self.guideline_constraints.get("max_selection_guidelines", 5) if use_constrained else None

            # Format prompt using template with tool object, existing guidelines, and optional constraint
            prompt = self.prompt_manager.format_pattern_analysis_prompt(
                prompt_template, tool_name, positive_patterns_text, negative_patterns_text,
                tool=tool, existing_guidelines=existing_guidelines, max_guidelines=max_guidelines
            )
            
            # Load system prompt from file (check custom config first)
            system_template_path = pattern_config.get("system_selection")
            
            if system_template_path:
                system_prompt = self.prompt_manager.load_template_from_path(system_template_path)
            else:
                system_prompt = self.prompt_manager.load_system_prompt("system_selection")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            time.sleep(0.1)  # Rate limiting
            content = llm_call(
                client=self.llm_config["client"],
                messages=messages,
                model=self.llm_config["model_name"],
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
            )
            
            # Parse JSON response
            try:
                # Handle code blocks
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                result = json.loads(content)
                guidelines = result.get("revised_guidelines", [])
                
                logger.info(f"Extracted {len(guidelines)} selection guidelines for {tool_name}")
                return guidelines
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse selection guidelines JSON for {tool_name}: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting selection guidelines for {tool_name}: {e}")
            return []
    
    def _extract_usage_guidelines(self, tool_name: str, tool_pattern: ToolPattern, tool=None) -> List[str]:
        """Extract API usage guidelines using LLM analysis."""
        try:
            import json
            from ..core.llm_client import llm_call
            import time
            
            # Use usage-specific samples for API parameter usage guideline extraction
            positive_samples = tool_pattern.usage_positive_samples or []
            negative_samples = tool_pattern.usage_negative_samples or []

            if not positive_samples and not negative_samples:
                return []

            # Format samples focusing on parameter usage with validation metrics
            positive_formatted = []
            for sample in positive_samples:
                kwargs = sample.get('kwargs', {})
                result_preview = sample.get('result_preview', '')[:100]
                params_valid = sample.get('params_valid', 'unknown')
                api_success = sample.get('api_success', 'unknown')
                param_details = sample.get('parameter_validation_details', {})
                required_present = param_details.get('required_params_present', [])
                formatted = f"Parameters: {kwargs}, Params Valid: {params_valid}, API Success: {api_success}, Required Present: {required_present}, Result: {result_preview}..."
                positive_formatted.append(formatted)

            negative_formatted = []
            for sample in negative_samples:
                kwargs = sample.get('kwargs', {})
                result_preview = sample.get('result_preview', '')[:100]
                params_valid = sample.get('params_valid', 'unknown')
                param_details = sample.get('parameter_validation_details', {})
                missing_params = param_details.get('required_params_missing', [])
                validation_errors = param_details.get('parameter_validation_errors', [])
                error_info = f", Error: {sample.get('error')}" if sample.get('error') else ""
                formatted = f"Parameters: {kwargs}, Params Valid: {params_valid}, Missing: {missing_params}, Validation Errors: {validation_errors}, Result: {result_preview}...{error_info}"
                negative_formatted.append(formatted)
            
            # Load prompt template from file (check custom config first)
            pattern_config = self.prompt_config.get("pattern_analysis", {})
            usage_template_path = pattern_config.get("usage_guidelines")
            
            if usage_template_path:
                prompt_template = self.prompt_manager.load_template_from_path(usage_template_path)
            else:
                # Choose prompt template based on configuration
                use_constrained = self.guideline_constraints.get("use_constrained_prompts", True)
                template_name = "usage_guidelines_constrained" if use_constrained else "usage_guidelines"
                prompt_template = self.prompt_manager.load_pattern_analysis_prompt(template_name)
            
            positive_patterns_text = chr(10).join(positive_formatted) if positive_formatted else "No positive patterns available"
            negative_patterns_text = chr(10).join(negative_formatted) if negative_formatted else "No negative patterns available"
            
            # Get existing usage guidelines if available (for iterative refinement)
            existing_guidelines = tool_pattern.usage_guidelines if hasattr(tool_pattern, 'usage_guidelines') else []

            # Get max guidelines constraint (only used for constrained prompts)
            use_constrained = self.guideline_constraints.get("use_constrained_prompts", True)
            max_guidelines = self.guideline_constraints.get("max_usage_guidelines", 5) if use_constrained else None

            # Format prompt using template with tool object, existing guidelines, and optional constraint
            prompt = self.prompt_manager.format_pattern_analysis_prompt(
                prompt_template, tool_name, positive_patterns_text, negative_patterns_text,
                tool=tool, existing_guidelines=existing_guidelines, max_guidelines=max_guidelines
            )
            
            # Load system prompt from file (check custom config first)
            system_template_path = pattern_config.get("system_usage")
            
            if system_template_path:
                system_prompt = self.prompt_manager.load_template_from_path(system_template_path)
            else:
                system_prompt = self.prompt_manager.load_system_prompt("system_usage")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            time.sleep(0.1)  # Rate limiting
            content = llm_call(
                client=self.llm_config["client"],
                messages=messages,
                model=self.llm_config["model_name"],
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
            )
            
            # Parse JSON response
            try:
                # Handle code blocks
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                result = json.loads(content)
                guidelines = result.get("revised_guidelines", [])
                
                logger.info(f"Extracted {len(guidelines)} usage guidelines for {tool_name}")
                return guidelines
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse usage guidelines JSON for {tool_name}: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting usage guidelines for {tool_name}: {e}")
            return []
    
    def _convert_legacy_tool_data(self, tool_name: str, legacy_data: Dict[str, Any]) -> ToolPattern:
        """Convert legacy tool data format to new ToolPattern."""
        usage_stats = legacy_data.get('usage_stats', {})
        
        # Calculate confidence based on data availability
        confidence = 0.0
        total_calls = usage_stats.get('total_calls', 0)
        positive_samples = legacy_data.get('positive_samples', [])
        negative_samples = legacy_data.get('negative_samples', [])
        
        
        return ToolPattern(
            tool_name=tool_name,
            total_calls=total_calls,

            # New separate pattern collections
            selection_positive_samples=legacy_data.get('selection_positive_samples', []),
            selection_negative_samples=legacy_data.get('selection_negative_samples', []),
            selection_success_rate=usage_stats.get('selection_success_rate', 0.0),

            usage_positive_samples=legacy_data.get('usage_positive_samples', []),
            usage_negative_samples=legacy_data.get('usage_negative_samples', []),
            usage_success_rate=usage_stats.get('usage_success_rate', 0.0),

            # Legacy fields for backward compatibility
            success_rate=usage_stats.get('success_rate', 0.0),
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            success_indicators=legacy_data.get('success_indicators', []),
            common_errors=legacy_data.get('common_errors', []),
            parameter_patterns=legacy_data.get('parameter_patterns', {}),

            # Metadata
            data_source=PatternSourceType.MCP_LOGS,
            insights=legacy_data.get('insights', '')
        )
    
    def _convert_to_legacy_format(self, result: PatternAnalysisResult) -> Dict[str, Any]:
        """Convert new PatternAnalysisResult back to legacy format."""
        legacy_tool_patterns = {}
        
        for tool_name, tool_pattern in result.tool_patterns.items():
            legacy_tool_patterns[tool_name] = {
                'positive_samples': tool_pattern.positive_samples,
                'negative_samples': tool_pattern.negative_samples,
                'usage_stats': {
                    'total_calls': tool_pattern.total_calls,
                    'success_rate': tool_pattern.success_rate,
                    'positive_calls': len(tool_pattern.positive_samples),
                    'negative_calls': len(tool_pattern.negative_samples)
                },
                'common_errors': tool_pattern.common_errors,
                'success_indicators': tool_pattern.success_indicators,
                'parameter_patterns': tool_pattern.parameter_patterns,
                'insights': tool_pattern.insights
            }
        
        return {
            'mcp_data_available': True,
            'total_mcp_calls': result.total_data_points,
            'total_questions': 0,  # Not available in new format
            'tool_patterns': legacy_tool_patterns,
            'evaluation_summary': {
                'accuracy': 0.0,  # Not available in new format
                'avg_execution_time': 0.0,  # Not available in new format
                'tools_with_patterns': len([p for p in result.tool_patterns.values() 
                                          if p.has_positive_patterns() or p.has_negative_patterns()])
            }
        }
    
    def _empty_result(self, analysis_input: PatternAnalysisInput) -> PatternAnalysisResult:
        """Return empty result in new format."""
        return PatternAnalysisResult(
            tool_patterns={},
            total_tools_analyzed=len(analysis_input.tools),
            total_data_points=0,
            data_source=analysis_input.source_type
        )
    
    def _calculate_accuracy(self, steps_data: List[Dict[str, Any]]) -> float:
        """Calculate overall accuracy from steps data."""
        if not steps_data:
            return 0.0
        
        correct_count = sum(1 for step in steps_data if step.get("correct", False))
        return correct_count / len(steps_data)
    
    def _calculate_avg_time(self, steps_data: List[Dict[str, Any]]) -> float:
        """Calculate average execution time from steps data."""
        if not steps_data:
            return 0.0
        
        times = [step.get("execution_time", 0) for step in steps_data if step.get("execution_time")]
        return sum(times) / len(times) if times else 0.0
    
    def _empty_patterns(self) -> Dict[str, Any]:
        """Return empty patterns structure when no data is available."""
        return {
            "mcp_data_available": False,
            "total_mcp_calls": 0,
            "total_questions": 0,
            "tool_patterns": {},
            "evaluation_summary": {
                "accuracy": 0.0,
                "avg_execution_time": 0.0,
                "tools_with_patterns": 0
            }
        }


# Register the MCP analyzer
PatternAnalyzerRegistry.register("mcp_analyzer", MCPPatternAnalyzer)


