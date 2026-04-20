"""
Simple Strategy Template for easy implementation of improve_descriptions().

Copy this file and customize the TODO sections to create your own strategy.
"""

import logging
from typing import List, Dict, Any, Optional

from ..interfaces.improvement_interface import (
    ImprovementStrategy, ImprovementMethod, ImprovementStage, 
    ImprovementContext, ImprovementResult
)
from ..interfaces.dataset_interface import StandardizedTool
from ..core.registries import ImprovementStrategyRegistry

logger = logging.getLogger(__name__)


class SimpleStrategyTemplate(ImprovementStrategy):
    """
    Template for creating simple improvement strategies.
    
    TODO: Rename this class to YourStrategyName
    TODO: Update the docstring to describe your strategy
    """
    
    def __init__(self, **kwargs):
        """
        Initialize your strategy.
        
        TODO: Add your custom parameters here
        """
        # TODO: Store your configuration parameters
        self.custom_param = kwargs.get('custom_param', 'default_value')
        self.another_param = kwargs.get('another_param', 42)
        
        logger.info(f"Initialized {self.name} with custom_param={self.custom_param}")
    
    @property
    def name(self) -> str:
        """Strategy name identifier."""
        # TODO: Change this to your strategy name
        return "simple_strategy_template"
    
    def get_method_type(self) -> ImprovementMethod:
        """Return the improvement method type."""
        # TODO: Choose appropriate method type
        return ImprovementMethod.CUSTOM  # or LLM_BASED, RULE_BASED, etc.
    
    def get_supported_stages(self) -> List[ImprovementStage]:
        """Return supported improvement stages."""
        # TODO: Specify which stages your strategy supports
        return [ImprovementStage.DATA_INDEPENDENT]  # or DATA_DEPENDENT, etc.
    
    def can_handle_tools(self, tools: List[StandardizedTool]) -> bool:
        """Check if this strategy can handle the given tools."""
        # TODO: Add your validation logic
        return len(tools) > 0  # Basic validation
    
    def improve_descriptions(self, context: ImprovementContext) -> ImprovementResult:
        """
        🎯 MAIN METHOD: Implement your improvement logic here.
        
        This is the core method that contains your unique improvement algorithm.
        
        Args:
            context: ImprovementContext with:
                - tools: List[StandardizedTool] to improve
                - evaluation_results: Dict with evaluation data  
                - guidelines: List[str] with context guidelines
                - stage: Current improvement stage
        
        Returns:
            ImprovementResult with improved tools and metadata
        """
        try:
            logger.info(f"Starting {self.name} improvement for {len(context.tools)} tools")
            
            # TODO: Add your validation logic
            if not self._validate_context(context):
                return ImprovementResult(
                    improved_tools=context.tools,
                    improvement_metadata={},
                    success=False,
                    error_message="Context validation failed"
                )
            
            improved_tools = []
            
            # TODO: Implement your improvement logic for each tool
            for tool in context.tools:
                try:
                    improved_tool = self._improve_single_tool(tool, context)
                    improved_tools.append(improved_tool)
                except Exception as e:
                    logger.warning(f"Failed to improve {tool.name}: {e}")
                    improved_tools.append(tool)  # Keep original on failure
            
            # TODO: Create your improvement metadata
            improvement_metadata = {
                "strategy": self.name,
                "method_type": self.get_method_type().value,
                "tools_processed": len(improved_tools),
                "custom_metric": self._calculate_custom_metric(improved_tools),
                # Add your custom metadata here
            }
            
            logger.info(f"Completed {self.name}: processed {len(improved_tools)} tools")
            
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
        """
        Validate that context has required information for this strategy.
        
        TODO: Add your specific validation requirements
        """
        if not context.tools:
            logger.error("No tools provided in context")
            return False
        
        # TODO: Add your validation logic
        # Example: Check for required evaluation results
        # if context.stage == ImprovementStage.DATA_DEPENDENT:
        #     if not context.evaluation_results:
        #         logger.error("Data-dependent stage requires evaluation results")
        #         return False
        
        return True
    
    def _improve_single_tool(self, tool: StandardizedTool, 
                            context: ImprovementContext) -> StandardizedTool:
        """
        Improve a single tool's description.
        
        TODO: Implement your core improvement algorithm here
        
        Args:
            tool: Tool to improve
            context: Improvement context
            
        Returns:
            Improved StandardizedTool
        """
        # TODO: Implement your improvement logic
        
        # Example: Simple rule-based improvement
        original_description = tool.description
        improved_description = self._apply_your_algorithm(original_description, context)
        
        # Create improved tool with metadata
        improved_tool = StandardizedTool(
            name=tool.name,
            description=improved_description,
            parameters=tool.parameters,
            format_specific=tool.format_specific,
            metadata={
                **(tool.metadata or {}),
                "improvement_method": self.name,
                "improvement_timestamp": self._get_timestamp(),
                "original_description": original_description,
                # TODO: Add your custom metadata
            }
        )
        
        return improved_tool
    
    def _apply_your_algorithm(self, description: str, context: ImprovementContext) -> str:
        """
        Apply your specific improvement algorithm.
        
        TODO: Replace this with your actual improvement logic
        """
        # Example: Simple string-based improvement
        improved = description
        
        # Example improvements (replace with your logic):
        if not improved.endswith('.'):
            improved += '.'
        
        if len(improved) < 50:
            improved += " Provides enhanced functionality for users."
        
        # TODO: Implement your algorithm here
        # Examples:
        # - LLM-based improvement: call LLM with custom prompts
        # - Rule-based improvement: apply domain-specific rules
        # - Pattern-based improvement: use learned patterns
        # - Retrieval-based: enhance with external knowledge
        
        return improved
    
    def _calculate_custom_metric(self, tools: List[StandardizedTool]) -> float:
        """
        Calculate your custom improvement metric.
        
        TODO: Replace with your metric calculation
        """
        # Example: average description length
        if not tools:
            return 0.0
        return sum(len(tool.description) for tool in tools) / len(tools)
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()


# TODO: Register your strategy with a unique name
# ImprovementStrategyRegistry.register("your_strategy_name", YourStrategyClass)

# Example registration (uncomment and modify):
# ImprovementStrategyRegistry.register("simple_strategy_template", SimpleStrategyTemplate)
