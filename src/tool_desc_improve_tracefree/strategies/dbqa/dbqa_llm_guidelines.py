"""
DBQA-specific LLM-based improvement strategy using guidelines.
"""

from typing import List, Dict, Any
from pathlib import Path

from ...core.registries import ImprovementStrategyRegistry
from ..generic_llm_strategy import GenericLLMStrategy


class DBQALLMGuidelinesStrategy(GenericLLMStrategy):
    """DBQA-specific LLM strategy that inherits from GenericLLMStrategy."""
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3, 
                 max_tokens: int = 500, **kwargs):
        """
        Initialize DBQA LLM guidelines strategy.
        
        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        # Initialize parent with DBQA as fixed dataset
        super().__init__(
            model_name=model_name,
            temperature=temperature, 
            max_tokens=max_tokens,
            dataset_name="dbqa",  # Fixed to DBQA
            **kwargs
        )
    
    @property
    def name(self) -> str:
        """Override strategy name."""
        return "dbqa_llm_guidelines"
    
    def _detect_dataset(self, context) -> str:
        """Override dataset detection - always return DBQA."""
        return "dbqa"


# Register the DBQA-specific strategy
ImprovementStrategyRegistry.register("dbqa_llm_guidelines", DBQALLMGuidelinesStrategy, 
                                   "DBQA-specific LLM-based improvement using guidelines")
