"""
Registry systems for datasets, improvement strategies, and evaluation systems.
"""

from typing import Dict, Type, Any, List, Optional
from ..interfaces.dataset_interface import DatasetInterface, EvaluationInterface
from ..interfaces.improvement_interface import ImprovementStrategy, GuidelinesProvider, PatternAnalyzer


class RegistryError(Exception):
    """Exception raised for registry-related errors."""
    pass


class DatasetRegistry:
    """Registry for dataset implementations."""
    
    _datasets: Dict[str, Type[DatasetInterface]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: Type[DatasetInterface], 
                 description: str = "", **metadata) -> None:
        """Register a dataset implementation."""
        cls._datasets[name] = dataset_class
        cls._metadata[name] = {
            'description': description,
            'class_name': dataset_class.__name__,
            **metadata
        }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> DatasetInterface:
        """Create dataset instance."""
        if name not in cls._datasets:
            raise RegistryError(f"Dataset '{name}' not registered. Available: {list(cls._datasets.keys())}")
        
        dataset_class = cls._datasets[name]
        return dataset_class(**kwargs)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets."""
        return list(cls._datasets.keys())
    
    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        if name not in cls._metadata:
            raise RegistryError(f"Dataset '{name}' not registered")
        return cls._metadata[name]


class ImprovementStrategyRegistry:
    """Registry for improvement strategy implementations."""
    
    _strategies: Dict[str, Type[ImprovementStrategy]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[ImprovementStrategy],
                 description: str = "", **metadata) -> None:
        """Register an improvement strategy."""
        cls._strategies[name] = strategy_class
        cls._metadata[name] = {
            'description': description,
            'class_name': strategy_class.__name__,
            **metadata
        }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> ImprovementStrategy:
        """Create strategy instance."""
        if name not in cls._strategies:
            raise RegistryError(f"Strategy '{name}' not registered. Available: {list(cls._strategies.keys())}")
        
        strategy_class = cls._strategies[name]
        return strategy_class(**kwargs)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def get_compatible_strategies(cls, tools: List[Any], stage: str = None) -> List[str]:
        """Get strategies compatible with given tools and stage."""
        compatible = []
        for name, strategy_class in cls._strategies.items():
            try:
                # Create temporary instance to check compatibility
                strategy = strategy_class()
                if strategy.can_handle_tools(tools):
                    if stage is None or stage in [s.value for s in strategy.get_supported_stages()]:
                        compatible.append(name)
            except Exception:
                continue  # Skip if instantiation fails
        return compatible


class EvaluationRegistry:
    """Registry for evaluation system implementations."""
    
    _evaluators: Dict[str, Type[EvaluationInterface]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, evaluator_class: Type[EvaluationInterface],
                 description: str = "", **metadata) -> None:
        """Register an evaluation system."""
        cls._evaluators[name] = evaluator_class
        cls._metadata[name] = {
            'description': description,
            'class_name': evaluator_class.__name__,
            **metadata
        }
    
    @classmethod
    def create(cls, name: str, **kwargs) -> EvaluationInterface:
        """Create evaluator instance."""
        if name not in cls._evaluators:
            raise RegistryError(f"Evaluator '{name}' not registered. Available: {list(cls._evaluators.keys())}")
        
        evaluator_class = cls._evaluators[name]
        return evaluator_class(**kwargs)
    
    @classmethod
    def list_evaluators(cls) -> List[str]:
        """List all registered evaluators."""
        return list(cls._evaluators.keys())


class GuidelinesRegistry:
    """Registry for guidelines providers."""
    
    _providers: Dict[str, Type[GuidelinesProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[GuidelinesProvider]) -> None:
        """Register a guidelines provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> GuidelinesProvider:
        """Create guidelines provider instance."""
        if name not in cls._providers:
            raise RegistryError(f"Guidelines provider '{name}' not registered")
        
        provider_class = cls._providers[name]
        return provider_class(**kwargs)


class PatternAnalyzerRegistry:
    """Registry for pattern analyzers."""
    
    _analyzers: Dict[str, Type[PatternAnalyzer]] = {}
    
    @classmethod
    def register(cls, name: str, analyzer_class: Type[PatternAnalyzer]) -> None:
        """Register a pattern analyzer."""
        cls._analyzers[name] = analyzer_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> PatternAnalyzer:
        """Create pattern analyzer instance."""
        if name not in cls._analyzers:
            raise RegistryError(f"Pattern analyzer '{name}' not registered")
        
        analyzer_class = cls._analyzers[name]
        return analyzer_class(**kwargs)


# Decorator functions for easy registration
def register_dataset(name: str, description: str = "", **metadata):
    """Decorator to register a dataset class."""
    def decorator(cls):
        DatasetRegistry.register(name, cls, description, **metadata)
        return cls
    return decorator


def register_improvement_strategy(name: str, description: str = "", **metadata):
    """Decorator to register an improvement strategy class."""
    def decorator(cls):
        ImprovementStrategyRegistry.register(name, cls, description, **metadata)
        return cls
    return decorator


def register_evaluator(name: str, description: str = "", **metadata):
    """Decorator to register an evaluation system class."""
    def decorator(cls):
        EvaluationRegistry.register(name, cls, description, **metadata)
        return cls
    return decorator
