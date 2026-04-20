"""
Advanced Strategy Template with LLM, Pattern Analysis, and Evaluation Integration.

This template shows advanced patterns for implementing sophisticated improvement strategies.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..interfaces.improvement_interface import (
    ImprovementStrategy, ImprovementMethod, ImprovementStage, 
    ImprovementContext, ImprovementResult
)
from ..interfaces.dataset_interface import StandardizedTool
from ..interfaces.pattern_interface import PatternAnalysisInput, PatternSourceType
from ..core.registries import ImprovementStrategyRegistry, PatternAnalyzerRegistry
from ..core.llm_client import create_openai_client, llm_call

logger = logging.getLogger(__name__)


class AdvancedStrategyTemplate(ImprovementStrategy):
    """
    Advanced template with LLM integration, pattern analysis, and evaluation support.
    
    Features demonstrated:
    - LLM-based improvement
    - Pattern analysis integration  
    - Prompt template management
    - Evaluation data utilization
    - Multi-stage support
    - Custom configuration
    
    TODO: Customize for your specific strategy
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3,
                 max_tokens: int = 800, pattern_analyzer: str = "basic",
                 dataset_name: str = None, prompt_template_path: str = None,
                 custom_param: str = "default", openai_api_key: str = None, **kwargs):
        """
        Initialize advanced strategy with comprehensive configuration.
        
        Args:
            model_name: LLM model to use
            temperature: LLM sampling temperature
            max_tokens: Maximum LLM response tokens
            pattern_analyzer: Pattern analyzer to use
            dataset_name: Dataset name for prompt selection
            prompt_template_path: Explicit prompt template path
            custom_param: Your custom parameter
            **kwargs: Additional parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pattern_analyzer_name = pattern_analyzer
        self.dataset_name = dataset_name
        self.prompt_template_path = prompt_template_path
        self.custom_param = custom_param
        self.openai_api_key = openai_api_key
        self.kwargs = kwargs

        # Initialize components
        self.client = create_openai_client(api_key=openai_api_key)
        self.pattern_analyzer = None
        
        logger.info(f"Initialized {self.name} with model={model_name}, dataset={dataset_name}")
    
    @property
    def name(self) -> str:
        """Strategy name identifier."""
        # TODO: Change to your strategy name
        return "advanced_strategy_template"
    
    def get_method_type(self) -> ImprovementMethod:
        """Return the improvement method type."""
        return ImprovementMethod.LLM_BASED  # Since this template uses LLM
    
    def get_supported_stages(self) -> List[ImprovementStage]:
        """Return supported improvement stages."""
        # This template supports both stages
        return [ImprovementStage.DATA_INDEPENDENT, ImprovementStage.DATA_DEPENDENT]
    
    def can_handle_tools(self, tools: List[StandardizedTool]) -> bool:
        """Check if this strategy can handle the given tools."""
        return len(tools) > 0
    
    def estimate_improvement_time(self, context: ImprovementContext) -> float:
        """Estimate time needed for improvement in seconds."""
        # Estimate based on LLM calls
        base_time_per_tool = 5.0  # seconds per tool for LLM call
        return len(context.tools) * base_time_per_tool
    
    def improve_descriptions(self, context: ImprovementContext) -> ImprovementResult:
        """
        🎯 MAIN METHOD: Advanced improvement with LLM and pattern analysis.
        
        This template demonstrates:
        1. Context validation
        2. Pattern analysis (for data-dependent stages)
        3. Prompt template loading
        4. LLM-based improvement
        5. Comprehensive error handling
        6. Rich metadata collection
        """
        try:
            logger.info(f"Starting {self.name} improvement for {len(context.tools)} tools")
            
            # Step 1: Validate context
            if not self._validate_context(context):
                return ImprovementResult(
                    improved_tools=context.tools,
                    improvement_metadata={},
                    success=False,
                    error_message="Context validation failed"
                )
            
            # Step 2: Initialize pattern analyzer (for data-dependent stages)
            pattern_analysis_result = None
            if context.stage == ImprovementStage.DATA_DEPENDENT:
                pattern_analysis_result = self._analyze_patterns(context)
                if not pattern_analysis_result:
                    logger.warning("Pattern analysis failed, continuing without patterns")
            
            # Step 3: Load prompt template
            prompt_template = self._load_prompt_template(context)
            
            # Step 4: Improve each tool
            improved_tools = []
            improvement_stats = {"succeeded": 0, "failed": 0, "total_llm_calls": 0}
            
            for tool in context.tools:
                try:
                    improved_tool = self._improve_single_tool_advanced(
                        tool, context, pattern_analysis_result, prompt_template
                    )
                    improved_tools.append(improved_tool)
                    improvement_stats["succeeded"] += 1
                    improvement_stats["total_llm_calls"] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to improve {tool.name}: {e}")
                    improved_tools.append(tool)  # Keep original
                    improvement_stats["failed"] += 1
            
            # Step 5: Create comprehensive metadata
            improvement_metadata = {
                "strategy": self.name,
                "method_type": self.get_method_type().value,
                "stage": context.stage.value,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "dataset_name": self.dataset_name or "auto-detected",
                "improvement_stats": improvement_stats,
                "pattern_analysis_available": pattern_analysis_result is not None,
                "total_tools": len(context.tools),
                "timestamp": time.time(),
                # TODO: Add your custom metadata
                "custom_metric": self._calculate_improvement_quality(context.tools, improved_tools),
            }
            
            logger.info(f"Completed {self.name}: {improvement_stats['succeeded']}/{len(context.tools)} tools improved")
            
            return ImprovementResult(
                improved_tools=improved_tools,
                improvement_metadata=improvement_metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}")
            return ImprovementResult(
                improved_tools=context.tools,
                improvement_metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _validate_context(self, context: ImprovementContext) -> bool:
        """Validate context for advanced strategy requirements."""
        if not context.tools:
            logger.error("No tools provided in context")
            return False
        
        # Data-dependent validation
        if context.stage == ImprovementStage.DATA_DEPENDENT:
            if not context.evaluation_results:
                logger.warning("Data-dependent stage without evaluation results")
                # Continue anyway for this template
        
        return True
    
    def _analyze_patterns(self, context: ImprovementContext) -> Optional[Any]:
        """
        Analyze usage patterns for data-dependent improvement.
        
        TODO: Customize pattern analysis for your needs
        """
        try:
            if not self.pattern_analyzer:
                self.pattern_analyzer = PatternAnalyzerRegistry.create(
                    self.pattern_analyzer_name,
                    min_samples_per_tool=1
                )
            
            analysis_input = PatternAnalysisInput(
                tools=context.tools,
                evaluation_results=context.evaluation_results or {},
                source_type=PatternSourceType.EVALUATION_METRICS,  # TODO: Choose appropriate type
                dataset_name=self.dataset_name or "generic"
            )
            
            result = self.pattern_analyzer.analyze_usage_patterns(analysis_input)
            logger.info(f"Pattern analysis completed: {result.total_tools_analyzed} tools analyzed")
            return result
            
        except Exception as e:
            logger.warning(f"Pattern analysis failed: {e}")
            return None
    
    def _load_prompt_template(self, context: ImprovementContext) -> str:
        """
        Load prompt template for LLM calls.
        
        TODO: Customize prompt template loading
        """
        try:
            # Option 1: Explicit template path
            if self.prompt_template_path:
                with open(self.prompt_template_path, 'r') as f:
                    return f.read()
            
            # Option 2: Dataset-based template
            dataset_name = self.dataset_name or self._detect_dataset(context)
            template_path = Path(f"prompts/datasets/{dataset_name}/your_template.txt")
            
            if template_path.exists():
                with open(template_path, 'r') as f:
                    return f.read()
            
            # Option 3: Default template
            return self._get_default_prompt_template()
            
        except Exception as e:
            logger.warning(f"Failed to load prompt template: {e}")
            return self._get_default_prompt_template()
    
    def _get_default_prompt_template(self) -> str:
        """
        Get default prompt template.
        
        TODO: Customize your default prompt
        """
        return """You are an expert at improving tool descriptions.

Current Tool:
Name: {tool_name}
Description: {current_description}
Parameters: {parameters}

Guidelines:
{guidelines}

Task: Improve the tool description to be more clear, accurate, and helpful.
Provide ONLY the improved description text."""
    
    def _improve_single_tool_advanced(self, tool: StandardizedTool, 
                                     context: ImprovementContext,
                                     pattern_analysis_result: Optional[Any],
                                     prompt_template: str) -> StandardizedTool:
        """
        Improve a single tool using advanced LLM-based approach.
        
        TODO: Customize the improvement logic
        """
        # Prepare prompt variables
        prompt_vars = {
            "tool_name": tool.name,
            "current_description": tool.description,
            "parameters": self._format_parameters(tool.parameters),
            "guidelines": "\\n".join(context.guidelines or ["Use best practices"]),
        }
        
        # Add pattern-based variables if available
        if pattern_analysis_result and pattern_analysis_result.tool_patterns:
            tool_pattern = pattern_analysis_result.tool_patterns.get(tool.name)
            if tool_pattern:
                prompt_vars["usage_patterns"] = self._format_patterns(tool_pattern)
            else:
                prompt_vars["usage_patterns"] = "No usage patterns available"
        else:
            prompt_vars["usage_patterns"] = "No usage patterns available"
        
        # Format prompt
        formatted_prompt = prompt_template.format(**prompt_vars)
        
        # Call LLM
        improved_description = self._call_llm(formatted_prompt)
        
        # Create improved tool
        improved_tool = StandardizedTool(
            name=tool.name,
            description=improved_description,
            parameters=tool.parameters,
            format_specific=tool.format_specific,
            metadata={
                **(tool.metadata or {}),
                "improvement_method": self.name,
                "improvement_timestamp": time.time(),
                "original_description": tool.description,
                "model_used": self.model_name,
                "temperature": self.temperature,
                "pattern_analysis_used": pattern_analysis_result is not None,
                # TODO: Add your custom metadata
            }
        )
        
        return improved_tool
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM for description improvement.

        TODO: Customize LLM call parameters
        """
        try:
            messages = [
                {"role": "system", "content": "You are an expert tool documentation specialist."},
                {"role": "user", "content": prompt}
            ]

            return llm_call(
                client=self.client,
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM improvement failed: {str(e)}")
    
    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format tool parameters for prompt."""
        if not parameters:
            return "No parameters"
        
        formatted = []
        for name, param_def in parameters.items():
            if isinstance(param_def, dict):
                param_type = param_def.get('type', 'unknown')
                param_desc = param_def.get('description', 'No description')
                formatted.append(f"- {name} ({param_type}): {param_desc}")
            else:
                formatted.append(f"- {name}: {param_def}")
        
        return "\\n".join(formatted)
    
    def _format_patterns(self, tool_pattern: Any) -> str:
        """
        Format usage patterns for prompt.
        
        TODO: Customize pattern formatting
        """
        # This is a placeholder - customize based on your pattern structure
        if hasattr(tool_pattern, 'positive_samples') and tool_pattern.positive_samples:
            return f"Found {len(tool_pattern.positive_samples)} usage examples"
        return "No specific usage patterns found"
    
    def _detect_dataset(self, context: ImprovementContext) -> str:
        """Auto-detect dataset name from context."""
        # TODO: Implement your dataset detection logic
        if context.tools:
            tool_names = " ".join([tool.name for tool in context.tools]).lower()
            if "clinvar" in tool_names or "ensembl" in tool_names:
                return "dbqa"
            elif "movie" in tool_names or "actor" in tool_names:
                return "tmdb"
        
        return "generic"
    
    def _calculate_improvement_quality(self, original_tools: List[StandardizedTool], 
                                      improved_tools: List[StandardizedTool]) -> float:
        """
        Calculate a custom improvement quality metric.
        
        TODO: Implement your quality metric
        """
        if not original_tools or not improved_tools:
            return 0.0
        
        # Example: improvement in average description length
        orig_avg_len = sum(len(tool.description) for tool in original_tools) / len(original_tools)
        improved_avg_len = sum(len(tool.description) for tool in improved_tools) / len(improved_tools)
        
        return improved_avg_len / orig_avg_len if orig_avg_len > 0 else 1.0


# TODO: Register your strategy
# ImprovementStrategyRegistry.register("advanced_strategy_template", AdvancedStrategyTemplate)
