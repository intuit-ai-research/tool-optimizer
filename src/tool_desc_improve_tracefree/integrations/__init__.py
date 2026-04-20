"""
Integration modules for connecting description improvement with external systems.
"""

from .evaluation_integration import (
    AbstractionEvaluationAdapter,
    GenericEvaluationAdapter, 
    MockEvaluationAdapter,
    create_evaluation_adapter
)

# Register evaluation adapters
from ..core.registries import EvaluationRegistry

EvaluationRegistry.register(
    "abstraction_subprocess",
    AbstractionEvaluationAdapter,
    "Integration with evaluation abstraction system via subprocess",
    supports_mcp=True,
    supports_parallel=False
)

EvaluationRegistry.register(
    "generic_command",
    GenericEvaluationAdapter,
    "Generic evaluation via configurable command template",
    supports_mcp=True,
    supports_parallel=True
)

EvaluationRegistry.register(
    "mock",
    MockEvaluationAdapter,
    "Mock evaluation for testing and development",
    supports_mcp=True,
    supports_parallel=True
)

__all__ = [
    'AbstractionEvaluationAdapter',
    'GenericEvaluationAdapter', 
    'MockEvaluationAdapter',
    'create_evaluation_adapter'
]
