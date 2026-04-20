"""
One-Step CICL Strategy - Combines API selection and usage analysis in a single LLM call.

This strategy extracts both selection patterns (from step_wise_eval) and usage patterns 
(from MCP calls) and combines them in one prompt for description improvement.
"""

import copy
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(iterable, total=None, desc=None, unit=None):
        return iterable

from ..interfaces.improvement_interface import (
    ImprovementStrategy, ImprovementMethod, ImprovementStage, 
    ImprovementContext, ImprovementResult
)
from ..interfaces.dataset_interface import StandardizedTool
from ..interfaces.pattern_interface import PatternAnalysisInput, PatternSourceType
from ..core.registries import ImprovementStrategyRegistry, PatternAnalyzerRegistry
from ..core.prompt_manager import ContrastivePromptManager
from ..core.llm_client import create_openai_client, llm_call

logger = logging.getLogger(__name__)


class OneStepCICLStrategy(ImprovementStrategy):
    """
    One-step CICL strategy that combines API selection and usage analysis.
    
    Unlike the two-step approach (extract guidelines first, then improve), this
    strategy provides raw selection + usage samples directly to the LLM for 
    description improvement in a single step.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3,
                 max_tokens: int = 2048, pattern_analyzer: str = "mcp_analyzer",
                 dataset_name: str = None, min_pattern_threshold: int = 1,
                 prompt_template_path: str = None, eval_results_path: str = None,
                 openai_api_key: str = None, **kwargs):
        """
        Initialize one-step CICL strategy.
        
        Args:
            model_name: LLM model to use
            temperature: Sampling temperature  
            max_tokens: Maximum tokens in response
            pattern_analyzer: Pattern analyzer to use
            dataset_name: Dataset name for prompts
            min_pattern_threshold: Minimum patterns needed to improve a tool
            prompt_template_path: Path to one-step CICL prompt template
            eval_results_path: Path to step_wise_eval results for API selection data
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pattern_analyzer_name = pattern_analyzer
        self.dataset_name = dataset_name
        self.min_pattern_threshold = min_pattern_threshold
        self.prompt_template_path = prompt_template_path
        self.eval_results_path = eval_results_path
        self.openai_api_key = openai_api_key
        self.kwargs = kwargs

        # Initialize components
        self.client = create_openai_client(api_key=openai_api_key)
        self.pattern_analyzer = None
        self.prompt_manager = ContrastivePromptManager()
        self.api_selection_data = None  # Will load from eval_results_path
    
    @property
    def name(self) -> str:
        return "onestep_cicl_strategy"
    
    def get_method_type(self) -> ImprovementMethod:
        """Return the improvement method type."""
        return ImprovementMethod.LLM_BASED
    
    def get_supported_stages(self) -> List[ImprovementStage]:
        """Return supported improvement stages."""
        return [ImprovementStage.DATA_DEPENDENT]
    
    def can_handle_tools(self, tools: List[StandardizedTool]) -> bool:
        """Check if this strategy can handle the given tools."""
        return len(tools) > 0
    
    def improve_descriptions(self, context: ImprovementContext) -> ImprovementResult:
        """
        Improve descriptions using one-step CICL approach.
        
        This combines:
        1. API selection patterns (from step_wise_eval)
        2. API usage patterns (from MCP calls)
        into a single prompt for each tool.
        """
        logger.info(f"Starting one-step CICL improvement with {len(context.tools)} tools")
        
        # Initialize pattern analyzer
        if not self.pattern_analyzer:
            logger.info(f"Initializing pattern analyzer: {self.pattern_analyzer_name}")
            self.pattern_analyzer = PatternAnalyzerRegistry.create(
                self.pattern_analyzer_name,
                extract_guidelines=False,  # One-step doesn't need guideline extraction
                llm_config={
                    "model_name": self.model_name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "client": self.client,
                }
            )
        
        # Load API selection data from step_wise_eval results
        self._load_api_selection_data()
        
        # Analyze usage patterns from MCP calls
        logger.info("Analyzing usage patterns from MCP call history...")
        analysis_input = PatternAnalysisInput(
            tools=context.tools,
            evaluation_results=context.evaluation_results,
            source_type=PatternSourceType.MCP_LOGS
        )
        pattern_result = self.pattern_analyzer.analyze_usage_patterns(analysis_input)
        
        logger.info(f"Found patterns for {len(pattern_result.tool_patterns)} tools")
        logger.info(f"Tools with sufficient data: {pattern_result.summary['tools_with_sufficient_data']}")
        
        # Improve each tool
        improved_tools = []
        improvement_stats = {
            "tools_with_patterns": 0,
            "tools_improved": 0,
            "tools_failed": 0,
            "total_positive_selection": 0,
            "total_negative_selection": 0,
            "total_positive_usage": 0,
            "total_negative_usage": 0
        }
        
        tool_iterator = _tqdm(
            context.tools,
            total=len(context.tools),
            desc="Improving tools (one-step CICL)",
            unit="tool"
        )
        
        for tool in tool_iterator:
            try:
                # Get tool pattern from MCP analysis
                tool_pattern = pattern_result.get_tool_pattern(tool.name)
                
                # Get API selection data for this tool
                selection_data = self._get_selection_data_for_tool(tool.name)
                
                # Skip tools without sufficient data
                if not tool_pattern or not tool_pattern.has_sufficient_data():
                    logger.debug(f"Skipping {tool.name}: insufficient usage data")
                    improved_tools.append(tool)
                    continue
                
                improvement_stats["tools_with_patterns"] += 1
                improvement_stats["total_positive_usage"] += len(tool_pattern.positive_samples)
                improvement_stats["total_negative_usage"] += len(tool_pattern.negative_samples)
                improvement_stats["total_positive_selection"] += len(selection_data["positive"])
                improvement_stats["total_negative_selection"] += len(selection_data["negative"])
                
                # Improve tool using combined selection + usage data
                improved_tool = self._improve_tool_with_combined_patterns(
                    tool, tool_pattern, selection_data, context
                )
                
                if improved_tool:
                    improved_tools.append(improved_tool)
                    improvement_stats["tools_improved"] += 1
                    logger.info(f"✓ Improved {tool.name}")
                else:
                    improved_tools.append(tool)
                    improvement_stats["tools_failed"] += 1
                    logger.warning(f"✗ Failed to improve {tool.name}")
                    
            except Exception as e:
                logger.error(f"Error improving {tool.name}: {e}", exc_info=True)
                improved_tools.append(tool)
                improvement_stats["tools_failed"] += 1
        
        logger.info(f"One-step CICL improvement complete: {improvement_stats['tools_improved']} improved, "
                   f"{improvement_stats['tools_failed']} failed")
        
        return ImprovementResult(
            improved_tools=improved_tools,
            metadata={
                "strategy": self.name,
                "method": self.method.value,
                "stage": self.stage.value,
                "model": self.model_name,
                "temperature": self.temperature,
                "improvement_stats": improvement_stats,
                "pattern_summary": pattern_result.summary
            },
            stage=self.stage
        )
    
    def _load_api_selection_data(self):
        """Load API selection data from step_wise_eval results."""
        if not self.eval_results_path:
            logger.warning("No eval_results_path provided, API selection data unavailable")
            self.api_selection_data = {}
            return
        
        try:
            results_file = Path(self.eval_results_path) / "step_wise_eval_results.json"
            logger.info(f"Loading API selection data from: {results_file}")
            
            with open(results_file, 'r') as f:
                eval_results = json.load(f)
            
            # Extract selection data per API
            api_selection = {}
            
            for result_item in eval_results:
                main_results = result_item.get("main_results", {})
                step_wise_results = main_results.get("step_wise_results", [])
                
                for step in step_wise_results:
                    expected_api = step.get("expected_golden_api")
                    if not expected_api:
                        continue
                    
                    if expected_api not in api_selection:
                        api_selection[expected_api] = {"positive": [], "negative": []}
                    
                    # Get runs data
                    runs = step.get("runs", [])
                    scenario = step.get("scenario", {})
                    subtask_input = scenario.get("subtask_input", "")
                    
                    for run in runs:
                        selected_api = run.get("selected_api")
                        exact_match = run.get("exact_match_correct", False)
                        
                        sample = {
                            "query": subtask_input,
                            "expected_api": expected_api,
                            "selected_api": selected_api
                        }
                        
                        if exact_match and selected_api == expected_api:
                            # Positive: API correctly selected
                            api_selection[expected_api]["positive"].append(sample)
                        elif not exact_match or selected_api != expected_api:
                            # Negative: API should have been selected but wasn't
                            api_selection[expected_api]["negative"].append(sample)
            
            self.api_selection_data = api_selection
            logger.info(f"Loaded API selection data for {len(api_selection)} APIs")
            
        except Exception as e:
            logger.error(f"Failed to load API selection data: {e}")
            self.api_selection_data = {}
    
    def _get_selection_data_for_tool(self, tool_name: str) -> Dict[str, List]:
        """Get API selection data for a specific tool."""
        if not self.api_selection_data or tool_name not in self.api_selection_data:
            return {"positive": [], "negative": []}
        return self.api_selection_data[tool_name]
    
    def _improve_tool_with_combined_patterns(self, tool: StandardizedTool, 
                                            tool_pattern, selection_data: Dict,
                                            context: ImprovementContext) -> Optional[StandardizedTool]:
        """
        Improve tool using combined selection + usage patterns in one step.
        """
        try:
            # Load prompt template
            if self.prompt_template_path:
                prompt_template = self.prompt_manager.load_template_from_path(self.prompt_template_path)
            else:
                # Fallback to dataset-based template
                prompt_template = self.prompt_manager.load_template(
                    self.dataset_name or "generic", "data_dep_v1"
                )
            
            # Format API selection samples
            positive_selection_formatted = []
            for sample in selection_data["positive"][:10]:  # Limit to 10
                formatted = f"Query: {sample['query']}, Expected API: {sample['expected_api']}, Correctly Selected: {sample['selected_api']}"
                positive_selection_formatted.append(formatted)
            
            negative_selection_formatted = []
            for sample in selection_data["negative"][:10]:  # Limit to 10
                formatted = f"Query: {sample['query']}, Expected API: {sample['expected_api']}, Incorrectly Selected: {sample.get('selected_api', 'None')}"
                negative_selection_formatted.append(formatted)
            
            # Format API usage samples
            positive_usage_formatted = []
            for sample in tool_pattern.positive_samples[:10]:  # Limit to 10
                kwargs = sample.get('kwargs', {})
                result = sample.get('result_preview', '')[:100]
                formatted = f"Parameters: {kwargs}, Result: {result}"
                positive_usage_formatted.append(formatted)
            
            negative_usage_formatted = []
            for sample in tool_pattern.negative_samples[:10]:  # Limit to 10
                kwargs = sample.get('kwargs', {})
                error = sample.get('error', 'Unknown error')
                formatted = f"Parameters: {kwargs}, Error: {error}"
                negative_usage_formatted.append(formatted)
            
            # Format parameters
            required_params = tool.parameters.get("required", [])
            optional_params = tool.parameters.get("optional", [])
            required_text = "\n".join([f"- {p}" for p in required_params]) if required_params else "None"
            optional_text = "\n".join([f"- {p}" for p in optional_params]) if optional_params else "None"
            
            # Create format dict for the prompt
            format_dict = {
                "api_name": tool.name,
                "current_description": tool.description,
                "required_parameters_text": required_text,
                "optional_parameters_text": optional_text,
                "list of all positive_sample_api_selection": "\n".join(positive_selection_formatted) if positive_selection_formatted else "No positive selection samples available.",
                "list of all negative_sample_api_selection": "\n".join(negative_selection_formatted) if negative_selection_formatted else "No negative selection samples available.",
                "list of all positive_sample_api_usage": "\n".join(positive_usage_formatted) if positive_usage_formatted else "No positive usage samples available.",
                "list of all negative_sample_api_usage": "\n".join(negative_usage_formatted) if negative_usage_formatted else "No negative usage samples available.",
                "usage_guidelines_text": ""  # Not used in one-step
            }
            
            # Format the prompt
            formatted_prompt = prompt_template.format(**format_dict)
            
            # Call LLM (with internal extraction retry logic)
            improved_description = self._call_llm(formatted_prompt, tool.name)

            # Track if fallback was used
            improvement_method = "onestep_cicl"
            if not improved_description:
                # Fallback: use original D1 tool unchanged (preserves original metadata for consistency)
                logger.warning(f"Description extraction failed for {tool.name}, using original D1 tool unchanged")
                return tool  # Return original tool with original improvement_method metadata

            # Create improved tool
            improved_tool = StandardizedTool(
                name=tool.name,
                description=improved_description,
                parameters=tool.parameters,
                format_specific=tool.format_specific,
                metadata={
                    **(tool.metadata or {}),
                    "improvement_method": improvement_method,
                    "improvement_stage": "data_dependent",
                    "improvement_timestamp": time.time(),
                    "original_description": tool.description,
                    "pattern_stats": {
                        "positive_selection": len(selection_data["positive"]),
                        "negative_selection": len(selection_data["negative"]),
                        "positive_usage": len(tool_pattern.positive_samples),
                        "negative_usage": len(tool_pattern.negative_samples),
                        "success_rate": tool_pattern.success_rate,
                        "total_calls": tool_pattern.total_calls
                    },
                    "dataset": self.dataset_name,
                    "model": self.model_name,
                    "temperature": self.temperature
                }
            )
            
            return improved_tool
            
        except Exception as e:
            logger.error(f"Error improving {tool.name}: {e}", exc_info=True)
            return None
    
    def _call_llm(self, prompt: str, tool_name: str = "unknown") -> str:
        """Call LLM and parse response with retry logic."""
        messages = [
            {"role": "system", "content": "You are an expert API documentation specialist. Improve API descriptions based on usage patterns."},
            {"role": "user", "content": prompt}
        ]

        # Try up to 2 times: original call + 1 retry
        for attempt in range(1, 3):  # attempt 1, then attempt 2
            try:
                if attempt > 1:
                    logger.info(f"Retrying LLM call for tool: {tool_name} (attempt {attempt})")

                time.sleep(0.1)  # Rate limiting
                content = llm_call(
                    client=self.client,
                    messages=messages,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                # Try ALL extraction methods on this response
                extracted = self._try_all_extraction_methods(content, tool_name, attempt)

                if extracted:
                    if attempt > 1:
                        logger.info(f"LLM retry succeeded for tool: {tool_name}")
                    return extracted
                else:
                    logger.warning(f"All extraction methods failed for tool: {tool_name} (attempt {attempt})")
                    # Continue to next attempt (retry LLM call)

            except Exception as e:
                logger.error(f"LLM call attempt {attempt} failed for tool {tool_name}: {e}")
                # Continue to next attempt

        # Both attempts failed
        logger.error(f"All LLM attempts failed for tool: {tool_name}")
        return ""

    def _get_extraction_methods(self):
        """Override to include JSON hunting method specific to OneStep CICL."""
        return [
            ("JSON parsing", self._extract_with_json_parsing),
            ("Regex extraction", self._extract_with_regex),
            ("JSON hunting", self._extract_with_json_hunting)
        ]

    # JSON parsing and regex methods inherited from base class

    def _extract_with_json_hunting(self, content: str) -> str:
        """Extract by hunting for JSON objects in the response."""
        # Look for JSON object in the response
        start_idx = content.find('{')
        if start_idx != -1:
            end_idx = content.rfind('}')
            if end_idx != -1 and end_idx > start_idx:
                json_content = content[start_idx:end_idx + 1]
                try:
                    result = json.loads(json_content)
                    return result.get("improved_description", "")
                except json.JSONDecodeError:
                    pass
        return ""


# Register the strategy
ImprovementStrategyRegistry.register("onestep_cicl_strategy", OneStepCICLStrategy)

