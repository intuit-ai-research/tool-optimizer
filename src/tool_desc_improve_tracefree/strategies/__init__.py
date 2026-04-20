"""
Improvement strategies for description improvement pipeline.
"""

# Generic strategies
from .file_based_guidelines import FileBasedGuidelinesProvider
from .basic_pattern_analyzer import BasicPatternAnalyzer
from .generic_llm_strategy import GenericLLMStrategy

# MCP contrastive learning components
from .mcp_pattern_analyzer import MCPPatternAnalyzer
from .mcp_contrastive_strategy import MCPContrastiveStrategy
from .onestep_cicl_strategy import OneStepCICLStrategy

# Dataset-specific strategies
from .dbqa import DBQALLMGuidelinesStrategy

# Legacy strategy removed - use GenericLLMStrategy instead

__all__ = [
    'FileBasedGuidelinesProvider', 
    'BasicPatternAnalyzer', 
    'GenericLLMStrategy',
    'MCPPatternAnalyzer',
    'MCPContrastiveStrategy',
    'OneStepCICLStrategy',
    'DBQALLMGuidelinesStrategy'
]
