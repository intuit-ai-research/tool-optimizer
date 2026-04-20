"""
Flexible configuration system for description improvement.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml
import os
import argparse


@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""
    name: str                               # Dataset type (e.g., "mcp_yaml", "openapi")
    tools_path: str                         # Path to tools configuration
    queries_path: str = ""                  # Path to queries/questions
    params: Dict[str, Any] = field(default_factory=dict)  # Dataset-specific parameters


@dataclass
class ImprovementTurnConfig:
    """Configuration for a single improvement turn."""
    strategy: str                               # Strategy name
    stage: str                                  # "data_independent" or "data_dependent"
    params: Dict[str, Any] = field(default_factory=dict)  # Strategy parameters


@dataclass
class ImprovementConfig:
    """Configuration for improvement process."""
    # Turn-based configuration (D0->D1->D2)
    turn_1: Optional[ImprovementTurnConfig] = None    # D0 -> D1 (data independent)
    turn_2: Optional[ImprovementTurnConfig] = None    # D1 -> D2 (data dependent)
    
    # Supporting configuration
    guidelines_provider: str = "file_based"     # How to load guidelines
    guidelines_path: str = ""                   # Path to guidelines directory
    guidelines_file: str = ""                   # Specific guidelines file to load (optional)
    pattern_analyzer: str = "basic"             # Pattern analysis method
    max_turns: int = 2                          # Maximum turns (D0->D1->D2)
    
    def get_turn_config(self, turn_number: int) -> Optional[ImprovementTurnConfig]:
        """Get configuration for specific turn."""
        if turn_number == 1:
            return self.turn_1
        elif turn_number == 2:
            return self.turn_2
        else:
            return None
    
    def has_turn(self, turn_number: int) -> bool:
        """Check if turn is configured."""
        return self.get_turn_config(turn_number) is not None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation integration."""
    enabled: bool = False                       # Whether to evaluate each iteration
    evaluator: str = "subprocess"               # Evaluation method
    eval_config_template: str = ""              # Template for evaluation config
    dataset_for_eval: str = ""                  # Dataset to use for evaluation
    params: Dict[str, Any] = field(default_factory=dict)  # Evaluator-specific params


@dataclass
class OutputConfig:
    """Configuration for output handling."""
    base_dir: str = "improved_configs"          # Base output directory
    version_format: str = "D{iteration}"        # Format for version names (D0, D1, D2)
    save_intermediate: bool = True              # Save intermediate results
    save_metadata: bool = True                  # Save improvement metadata
    format_specific: Dict[str, Any] = field(default_factory=dict)  # Format-specific output options


@dataclass
class PipelineConfig:
    """Main configuration for the improvement pipeline."""
    
    # Core configuration sections
    dataset: DatasetConfig
    improvement: ImprovementConfig = field(default_factory=ImprovementConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Pipeline behavior
    continue_on_error: bool = False             # Fail fast on any strategy error (changed from True)
    parallel_strategies: bool = False           # Run strategies in parallel
    validate_tools: bool = True                 # Validate tools after improvement
    
    # Custom extensions
    custom_config: Dict[str, Any] = field(default_factory=dict)  # For custom extensions
    keep_filename: bool = False                # Keep the filename of the input dataset
    
    @classmethod
    def from_yaml(cls, yaml_path: str, args: argparse.Namespace) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        yaml_path = os.path.expanduser(yaml_path)
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if args.dataset:
            data['dataset']['tools_path'] = args.dataset
            data['dataset']['params']['server_name'] = "StableToolBench"
        
        if args.output:
            data['output']['base_dir'] = args.output

        if args.eval_results_path:
            data['improvement']['turn_2']['params']['eval_results_path'] = args.eval_results_path

        # Thread OpenAI API key and model name into turn params when provided via CLI
        openai_api_key = getattr(args, 'openai_api_key', None)
        cli_model_name = getattr(args, 'model_name', None)
        for turn_key in ('turn_1', 'turn_2'):
            turn_data = data.get('improvement', {}).get(turn_key)
            if turn_data and 'params' in turn_data:
                if openai_api_key:
                    turn_data['params']['openai_api_key'] = openai_api_key
                if cli_model_name:
                    turn_data['params']['model_name'] = cli_model_name

        data["keep_filename"] = args.keep_filename

        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        # Extract and convert nested configurations
        dataset_data = data.get('dataset', {})
        dataset_config = DatasetConfig(**dataset_data)
        
        improvement_data = data.get('improvement', {})
        
        # Handle turn-based configuration
        turn_1_data = improvement_data.get('turn_1')
        turn_1_config = None
        if turn_1_data:
            turn_1_config = ImprovementTurnConfig(**turn_1_data)
        
        turn_2_data = improvement_data.get('turn_2')
        turn_2_config = None
        if turn_2_data:
            turn_2_config = ImprovementTurnConfig(**turn_2_data)
        
        improvement_config = ImprovementConfig(
            turn_1=turn_1_config,
            turn_2=turn_2_config,
            guidelines_provider=improvement_data.get('guidelines_provider', 'file_based'),
            guidelines_path=improvement_data.get('guidelines_path', ''),
            guidelines_file=improvement_data.get('guidelines_file', ''),
            pattern_analyzer=improvement_data.get('pattern_analyzer', 'basic'),
            max_turns=improvement_data.get('max_turns', 2)
        )
        
        evaluation_data = data.get('evaluation', {})
        evaluation_config = EvaluationConfig(**evaluation_data)
        
        output_data = data.get('output', {})
        output_config = OutputConfig(**output_data)
        
        # Create main config
        config = cls(
            dataset=dataset_config,
            improvement=improvement_config,
            evaluation=evaluation_config,
            output=output_config,
            continue_on_error=data.get('continue_on_error', True),
            parallel_strategies=data.get('parallel_strategies', False),
            validate_tools=data.get('validate_tools', True),
            custom_config=data.get('custom_config', {}),
            keep_filename=data.get('keep_filename', False)
        )
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'dataset': {
                'name': self.dataset.name,
                'tools_path': self.dataset.tools_path,
                'queries_path': self.dataset.queries_path,
                'params': self.dataset.params
            },
            'improvement': {
                'turn_1': {
                    'strategy': self.improvement.turn_1.strategy,
                    'stage': self.improvement.turn_1.stage,
                    'params': self.improvement.turn_1.params
                } if self.improvement.turn_1 else None,
                'turn_2': {
                    'strategy': self.improvement.turn_2.strategy,
                    'stage': self.improvement.turn_2.stage,
                    'params': self.improvement.turn_2.params
                } if self.improvement.turn_2 else None,
                'guidelines_provider': self.improvement.guidelines_provider,
                'guidelines_path': self.improvement.guidelines_path,
                'guidelines_file': self.improvement.guidelines_file,
                'pattern_analyzer': self.improvement.pattern_analyzer,
                'max_turns': self.improvement.max_turns
            },
            'evaluation': {
                'enabled': self.evaluation.enabled,
                'evaluator': self.evaluation.evaluator,
                'eval_config_template': self.evaluation.eval_config_template,
                'dataset_for_eval': self.evaluation.dataset_for_eval,
                'params': self.evaluation.params
            },
            'output': {
                'base_dir': self.output.base_dir,
                'version_format': self.output.version_format,
                'save_intermediate': self.output.save_intermediate,
                'save_metadata': self.output.save_metadata,
                'format_specific': self.output.format_specific
            },
            'continue_on_error': self.continue_on_error,
            'parallel_strategies': self.parallel_strategies,
            'validate_tools': self.validate_tools,
            'custom_config': self.custom_config,
            'keep_filename': self.keep_filename
        }
    
    def save_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate dataset configuration
        if not self.dataset.name:
            errors.append("Dataset name is required")
        if not self.dataset.tools_path:
            errors.append("Dataset tools_path is required")
        if not os.path.exists(os.path.expanduser(self.dataset.tools_path)):
            errors.append(f"Tools file not found: {self.dataset.tools_path}")
        
        # Validate improvement configuration
        if not self.improvement.turn_1 and not self.improvement.turn_2:
            errors.append("At least one improvement turn (turn_1 or turn_2) is required")
        
        # Validate evaluation configuration if enabled
        if self.evaluation.enabled:
            if not self.evaluation.evaluator:
                errors.append("Evaluator is required when evaluation is enabled")
            if not self.evaluation.dataset_for_eval:
                errors.append("Dataset for evaluation is required when evaluation is enabled")
        
        # Validate output configuration
        if not self.output.base_dir:
            errors.append("Output base_dir is required")
        
        return errors
    
    def get_output_path(self, turn: int, filename: str = None) -> str:
        """Get output path for a specific turn."""
        version_name = self.output.version_format.format(iteration=turn)
        if filename:
            return os.path.join(self.output.base_dir, version_name, filename)
        else:
            return os.path.join(self.output.base_dir, f"{version_name}.yaml")
