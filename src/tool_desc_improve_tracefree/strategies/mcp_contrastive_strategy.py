"""
MCP Contrastive Learning Strategy for D1->D2 Description Improvement.

This strategy uses MCP call history to improve tool descriptions through contrastive learning,
supporting both direct pattern application and two-stage guideline extraction approaches.
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


class MCPContrastiveStrategy(ImprovementStrategy):
    """
    Contrastive learning strategy that improves descriptions using MCP call history.
    
    This strategy:
    1. Uses MCP pattern analyzer to extract positive/negative usage patterns
    2. Applies contrastive learning similar to CICL in the original pipeline
    3. Generates improved descriptions using LLM with pattern-informed prompts
    """
    
    def __init__(self, prompt_template_path: str, model_name: str = "gpt-4",
                 temperature: float = 0.3, max_tokens: int = 800,
                 pattern_analyzer: str = "mcp_analyzer", dataset_name: str = None,
                 min_pattern_threshold: int = 1, extract_guidelines: bool = False,
                 max_selection_guidelines: int = 5, max_usage_guidelines: int = 5,
                 use_constrained_prompts: bool = True, openai_api_key: str = None, **kwargs):
        """
        Initialize MCP contrastive strategy.

        Args:
            prompt_template_path: Path to prompt template file (required)
            model_name: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            pattern_analyzer: Pattern analyzer to use for extracting MCP patterns
            dataset_name: Override dataset name (otherwise auto-detect)
            min_pattern_threshold: Minimum patterns needed to improve a tool
            extract_guidelines: Whether to extract guidelines from patterns (two-stage approach)
            max_selection_guidelines: Maximum number of selection guidelines to generate (default: 10)
            max_usage_guidelines: Maximum number of usage guidelines to generate (default: 10)
            use_constrained_prompts: Whether to use constrained prompt templates with count limits (default: True)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pattern_analyzer_name = pattern_analyzer
        self.dataset_name = dataset_name
        self.min_pattern_threshold = min_pattern_threshold
        self.extract_guidelines = extract_guidelines
        self.max_selection_guidelines = max_selection_guidelines
        self.max_usage_guidelines = max_usage_guidelines
        self.use_constrained_prompts = use_constrained_prompts
        self.prompt_template_path = prompt_template_path
        self.openai_api_key = openai_api_key
        self.kwargs = kwargs

        # Initialize components
        self.client = create_openai_client(api_key=openai_api_key)
        self.pattern_analyzer = None
        self.prompt_manager = ContrastivePromptManager()
    
    @property
    def name(self) -> str:
        return "mcp_contrastive_strategy"
    
    def get_method_type(self) -> ImprovementMethod:
        return ImprovementMethod.LLM_BASED
    
    def get_supported_stages(self) -> List[ImprovementStage]:
        return [ImprovementStage.DATA_DEPENDENT]
    
    def can_handle_tools(self, tools: List[StandardizedTool]) -> bool:
        """Check if this strategy can handle the given tools."""
        return len(tools) > 0
    
    def estimate_improvement_time(self, context: ImprovementContext) -> float:
        """Estimate time needed for improvement in seconds."""
        # Estimate ~5-10 seconds per tool (LLM calls + pattern analysis)
        base_time_per_tool = 7.5
        
        # Add extra time for guideline extraction (two-stage approach)
        if self.extract_guidelines:
            base_time_per_tool += 5.0
        
        return len(context.tools) * base_time_per_tool
    
    def improve_descriptions(self, context: ImprovementContext) -> ImprovementResult:
        """
        Main method to improve tool descriptions using MCP contrastive learning.
        
        Args:
            context: Improvement context with tools, evaluation results, and guidelines
            
        Returns:
            ImprovementResult with improved tools
        """
        try:
            logger.info(f"Starting MCP contrastive improvement for {len(context.tools)} tools")
            
            # Validate context for MCP data
            if not context.evaluation_results:
                error_msg = "Context validation failed - no evaluation results available"
                logger.error(error_msg)
                return ImprovementResult(
                    improved_tools=context.tools,
                    improvement_metadata={},
                    success=False,
                    error_message=error_msg
                )
            
            # Check for MCP output directory (support both keys)
            mcp_output_dir = context.evaluation_results.get("mcp_output_dir") or \
                             context.evaluation_results.get("output_directory")
            if not mcp_output_dir:
                error_msg = "Context validation failed - no MCP data available for contrastive learning"
                logger.error(error_msg)
                return ImprovementResult(
                    improved_tools=context.tools,
                    improvement_metadata={},
                    success=False,
                    error_message=error_msg
                )
            
            # Initialize pattern analyzer with guideline extraction configuration
            if not self.pattern_analyzer:
                analyzer_config = {
                    'min_samples_per_tool': self.min_pattern_threshold,
                    'extract_guidelines': self.extract_guidelines,
                    'guideline_constraints': {
                        'max_selection_guidelines': self.max_selection_guidelines,
                        'max_usage_guidelines': self.max_usage_guidelines,
                        'use_constrained_prompts': self.use_constrained_prompts,
                    },
                    'llm_config': {
                        'model_name': self.model_name,
                        'temperature': self.temperature,
                        'max_tokens': 2048,  # Increased from 600 to avoid JSON truncation
                        'client': self.client,
                    }
                }
                self.pattern_analyzer = PatternAnalyzerRegistry.create(
                    self.pattern_analyzer_name, 
                    **analyzer_config
                )
            
            # Create standardized analysis input
            analysis_input = PatternAnalysisInput(
                tools=context.tools,
                evaluation_results=context.evaluation_results,
                source_type=PatternSourceType.MCP_LOGS,
                dataset_name=self.dataset_name or self._detect_dataset(context)
            )
            
            # Extract usage patterns (with or without guidelines)
            logger.info("Extracting MCP usage patterns for contrastive learning...")
            pattern_result = self.pattern_analyzer.analyze_usage_patterns(analysis_input)
            
            if not pattern_result.tool_patterns:
                error_msg = "Failed to extract MCP usage patterns"
                logger.error(error_msg)
                return ImprovementResult(
                    improved_tools=context.tools,
                    improvement_metadata={},
                    success=False,
                    error_message=error_msg
                )
            
            # Auto-detect dataset if not specified
            dataset_name = self.dataset_name or self._detect_dataset(context)
            
            # Improve each tool using contrastive patterns
            improved_tools = []
            tool_patterns_data = {}  # Collect patterns/guidelines for separate file
            improvement_stats = {
                "tools_with_patterns": 0,
                "tools_improved": 0,
                "tools_failed": 0,
                "total_positive_samples": 0,
                "total_negative_samples": 0
            }
            
            tool_iterator = _tqdm(
                context.tools,
                total=len(context.tools),
                desc="Improving tools",
                unit="tool"
            )

            for tool in tool_iterator:
                try:
                    # Get tool pattern from results
                    tool_pattern = pattern_result.get_tool_pattern(tool.name)
                    
                    # Skip tools without sufficient patterns
                    if not tool_pattern or not tool_pattern.has_sufficient_data():
                        improved_tools.append(tool)
                        continue
                    
                    improvement_stats["tools_with_patterns"] += 1
                    improvement_stats["total_positive_samples"] += len(tool_pattern.positive_samples)
                    improvement_stats["total_negative_samples"] += len(tool_pattern.negative_samples)
                    
                    # Collect pattern/guideline data for this tool (for separate file)
                    tool_patterns_data[tool.name] = {
                        "positive_samples_count": len(tool_pattern.positive_samples),
                        "negative_samples_count": len(tool_pattern.negative_samples),
                        "success_rate": tool_pattern.success_rate,
                        "total_calls": tool_pattern.total_calls,
                        "selection_guidelines": tool_pattern.selection_guidelines if tool_pattern.selection_guidelines else [],
                        "usage_guidelines": tool_pattern.usage_guidelines if tool_pattern.usage_guidelines else [],
                        "common_errors": tool_pattern.common_errors[:5] if tool_pattern.common_errors else [],
                        "success_indicators": tool_pattern.success_indicators[:5] if tool_pattern.success_indicators else []
                    }
                    
                    # Improve tool using patterns or guidelines
                    improved_tool = self._improve_tool_with_pattern(
                        tool, tool_pattern, context, dataset_name
                    )
                    
                    if improved_tool:
                        improved_tools.append(improved_tool)
                        improvement_stats["tools_improved"] += 1
                        approach = "two-stage" if self.extract_guidelines else "direct"
                        logger.info(f"Successfully improved {tool.name} using {approach} contrastive patterns")
                    else:
                        improved_tools.append(tool)  # Keep original on failure
                        improvement_stats["tools_failed"] += 1
                        logger.warning(f"Failed to improve {tool.name}, keeping original")
                        
                except Exception as e:
                    logger.error(f"Error improving tool {tool.name}: {e}")
                    improved_tools.append(tool)  # Keep original on error
                    improvement_stats["tools_failed"] += 1
            
            # Create result metadata
            improvement_metadata = {
                "strategy": "mcp_contrastive_strategy",
                "model": self.model_name,
                "dataset": dataset_name,
                "pattern_analyzer": self.pattern_analyzer_name,
                "stage": context.stage.value,
                "stats": improvement_stats,
                "approach": "two-stage" if self.extract_guidelines else "direct",
                "guideline_extraction_enabled": self.extract_guidelines,
                "mcp_summary": {
                    "total_mcp_calls": pattern_result.total_data_points,
                    "total_questions": 0,  # Not available in new interface
                    "tools_with_usage_data": len(pattern_result.tool_patterns)
                },
                # Store patterns/guidelines for saving to separate file
                "tool_patterns": tool_patterns_data
            }
            
            success = improvement_stats["tools_improved"] > 0
            
            logger.info(f"MCP contrastive improvement completed: {improvement_stats['tools_improved']}/{len(context.tools)} tools improved")
            
            return ImprovementResult(
                improved_tools=improved_tools,
                improvement_metadata=improvement_metadata,
                success=success
            )
            
        except Exception as e:
            logger.error(f"Error in MCP contrastive improvement: {e}")
            return ImprovementResult(
                improved_tools=context.tools,
                improvement_metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _improve_tool_with_pattern(self, tool: StandardizedTool, tool_pattern, 
                                  context: ImprovementContext, dataset_name: str) -> Optional[StandardizedTool]:
        """
        Improve tool using pattern data (either raw patterns or extracted guidelines).
        
        Args:
            tool: Tool to improve
            tool_pattern: ToolPattern object with patterns and optional guidelines
            context: Improvement context
            dataset_name: Dataset name for prompts
            
        Returns:
            Improved tool or None if improvement failed
        """
        try:
            # Load prompt template from provided path
            prompt_template = self.prompt_manager.load_template_from_path(self.prompt_template_path)
            logger.debug(f"Loaded prompt from path: {self.prompt_template_path}")
            
            # Format prompt based on whether we have guidelines or raw patterns
            if self.extract_guidelines and (tool_pattern.selection_guidelines or tool_pattern.usage_guidelines):
                # Two-stage approach: use extracted guidelines
                formatted_prompt = self.prompt_manager.format_prompt_with_guidelines(
                    prompt_template, tool, context.guidelines or [],
                    tool_pattern.selection_guidelines, tool_pattern.usage_guidelines
                )
                improvement_method = "mcp_contrastive_two_stage"
            else:
                # Direct approach: use raw patterns
                positive_patterns = self.prompt_manager.format_samples(tool_pattern.positive_samples, "positive")
                negative_patterns = self.prompt_manager.format_samples(tool_pattern.negative_samples, "negative")
                
                formatted_prompt = self.prompt_manager.format_prompt_with_patterns(
                    prompt_template, tool, context.guidelines or [],
                    positive_patterns, negative_patterns, 
                    tool_pattern.common_errors, tool_pattern.success_indicators
                )
                improvement_method = "mcp_contrastive_direct"
            
            # Call LLM for description improvement (with internal extraction retry logic)
            improved_description = self._call_llm(formatted_prompt, tool.name)

            if not improved_description:
                # Fallback: use original D1 tool unchanged (preserves original metadata for consistency)
                logger.warning(f"Description extraction failed for {tool.name}, using original D1 tool unchanged")
                return tool  # Return original tool with original improvement_method metadata

            # Create improved tool with lightweight metadata (detailed patterns saved separately)
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
                    # Keep only summary stats in tool metadata
                    "pattern_stats": {
                        "positive_samples": len(tool_pattern.positive_samples),
                        "negative_samples": len(tool_pattern.negative_samples),
                        "selection_guidelines": len(tool_pattern.selection_guidelines) if tool_pattern.selection_guidelines else 0,
                        "usage_guidelines": len(tool_pattern.usage_guidelines) if tool_pattern.usage_guidelines else 0,
                        "success_rate": tool_pattern.success_rate,
                        "total_calls": tool_pattern.total_calls
                    },
                    "dataset": dataset_name,
                    "model": self.model_name,
                    # Note: Full guidelines saved in separate patterns file
                    "_patterns_file_note": "See <output>_patterns.json for detailed guidelines and patterns"
                }
            )
            
            return improved_tool
            
        except Exception as e:
            logger.error(f"Error in contrastive improvement for {tool.name}: {e}")
            return None
    
    def _call_llm(self, prompt: str, tool_name: str = "unknown") -> str:
        """
        Make LLM API call for description improvement with retry logic.

        Args:
            prompt: Formatted prompt for LLM
            tool_name: Name of tool being improved (for logging)

        Returns:
            Improved description or empty string if failed after retry
        """
        messages = [
            {"role": "system", "content": "You are an expert API documentation specialist focused on creating comprehensive, practical API descriptions using real usage insights."},
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
        """Override to include line parsing method specific to MCP Contrastive."""
        return [
            ("JSON parsing", self._extract_with_json_parsing),
            ("Regex extraction", self._extract_with_regex),
            ("Line-by-line extraction", self._extract_with_line_parsing)
        ]

    # JSON parsing and regex methods inherited from base class

    def _extract_with_line_parsing(self, content: str) -> str:
        """Extract using line-by-line parsing."""
        if '"improved_description"' not in content:
            return ""

        lines = content.split('\n')
        capture_next = False
        description_lines = []

        for line in lines:
            if '"improved_description"' in line:
                # Try to extract from same line first
                if ':' in line:
                    after_colon = line.split(':', 1)[1].strip()
                    if after_colon.startswith('"') and len(after_colon) > 1:
                        # Single line description
                        desc = after_colon.strip('"').strip()
                        if desc and not desc.startswith('{'):
                            return desc
                capture_next = True
                continue

            if capture_next and line.strip():
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    description_lines.append(line.strip('"'))
                elif line.startswith('"'):
                    description_lines.append(line[1:])
                elif line.endswith('"') and description_lines:
                    description_lines.append(line[:-1])
                    break
                elif description_lines:
                    description_lines.append(line)
                elif not line.startswith('{') and not line.startswith('}'):
                    description_lines.append(line)

        if description_lines:
            extracted_desc = '\n'.join(description_lines).strip()
            if extracted_desc:
                return extracted_desc

        return ""

    def _detect_dataset(self, context: ImprovementContext) -> str:
        """Auto-detect dataset from context."""
        # Simple heuristic based on evaluation results structure
        eval_keys = context.evaluation_results.keys() if context.evaluation_results else []
        
        if any("dbqa" in str(key).lower() for key in eval_keys):
            return "dbqa"
        elif any("tmdb" in str(key).lower() for key in eval_keys):
            return "tmdb"
        elif any("spotify" in str(key).lower() for key in eval_keys):
            return "spotify"
        else:
            return "generic"


# Register the strategy
ImprovementStrategyRegistry.register("mcp_contrastive_strategy", MCPContrastiveStrategy)
