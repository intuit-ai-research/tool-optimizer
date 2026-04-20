"""
Description Improvement Pipeline

A modular system for improving tool descriptions using various strategies
including data-independent guidelines and data-dependent contrastive learning.
"""

# Import key components for easy access
from .strategies import (
    GenericLLMStrategy,
    MCPPatternAnalyzer,
    MCPContrastiveStrategy,
    FileBasedGuidelinesProvider,
    BasicPatternAnalyzer,
    DBQALLMGuidelinesStrategy
)

from .datasets import (
    MCPYamlDataset
)

from .pipelines import (
    GenericImprovementPipeline
)

from .config.config import (
    PipelineConfig
)

__version__ = "1.0.0"

__all__ = [
    # Strategies
    'GenericLLMStrategy',
    'MCPPatternAnalyzer', 
    'MCPContrastiveStrategy',
    'FileBasedGuidelinesProvider',
    'BasicPatternAnalyzer',
    'DBQALLMGuidelinesStrategy',
    
    # Datasets
    'MCPYamlDataset',
    
    # Pipeline
    'GenericImprovementPipeline',
    
    # Config
    'PipelineConfig'
]
