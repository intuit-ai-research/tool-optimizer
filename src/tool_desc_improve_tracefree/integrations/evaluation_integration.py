"""
Integration layer connecting description improvement to evaluation abstraction.
"""

import os
import subprocess
import json
import tempfile
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..interfaces.dataset_interface import DatasetQuery, EvaluationInterface
from ..interfaces.improvement_interface import ImprovementContext


class AbstractionEvaluationAdapter(EvaluationInterface):
    """
    Adapter to connect description improvement system with evaluation abstraction.
    This allows any evaluation system that follows the abstraction pattern to be used.
    """
    
    def __init__(self, eval_config_template: str, dataset_name: str = "dbqa", 
                 debug_count: Optional[int] = None, **kwargs):
        """
        Initialize evaluation adapter.
        
        Args:
            eval_config_template: Path to evaluation config template YAML
            dataset_name: Dataset identifier for evaluation
            debug_count: Number of examples for debug mode (None for full)
            **kwargs: Additional parameters for specific evaluators
        """
        # Resolve eval_config_template to absolute path (repo-root relative allowed)
        repo_root = '/home/sagemaker-user/user-default-efs/FunctionWrapper'
        candidate_path = eval_config_template
        if not os.path.isabs(candidate_path):
            candidate_path = os.path.join(repo_root, eval_config_template)
        self.eval_config_template = candidate_path
        self.dataset_name = dataset_name
        self.debug_count = debug_count
        self.kwargs = kwargs
        
        # Verify template exists
        if not os.path.exists(self.eval_config_template):
            raise FileNotFoundError(f"Evaluation config template not found: {self.eval_config_template}")
    
    def evaluate_tools(self, tools_config_path: str, queries: List[DatasetQuery]) -> Dict[str, float]:
        """
        Evaluate tools using the evaluation abstraction system.
        
        Args:
            tools_config_path: Path to MCP config with tools to evaluate
            queries: List of queries (not used directly, loaded by eval system)
            
        Returns:
            Dictionary with evaluation metrics (accuracy, coverage, etc.)
        """
        # Create temporary evaluation config
        temp_config = self._create_evaluation_config(tools_config_path)
        
        try:
            # Run evaluation using subprocess (keeps systems separate)
            result = subprocess.run([
                'python', 
                '/home/sagemaker-user/user-default-efs/FunctionWrapper/eval/abstraction/reproduce_experiment.py',
                '--dataset', self.dataset_name,
                '--config', temp_config,
                '--full' if self.debug_count is None else '--debug'
            ], 
            capture_output=True, 
            text=True,
            cwd='/home/sagemaker-user/user-default-efs/FunctionWrapper'
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Evaluation failed: {result.stderr}")
            
            # Parse results from stdout or result files
            metrics = self._parse_evaluation_results(result.stdout, temp_config)
            return metrics
            
        finally:
            # Cleanup temporary config
            if os.path.exists(temp_config):
                os.remove(temp_config)
    
    def extract_usage_patterns(self, evaluation_results: Dict) -> Dict[str, Any]:
        """
        Extract tool usage patterns from evaluation results including MCP call history.
        
        Args:
            evaluation_results: Results from evaluate_tools with output_directory
            
        Returns:
            Dictionary with usage patterns for each tool
        """
        # Enhanced to work with MCP pattern analyzer
        output_dir = evaluation_results.get('output_directory')
        if output_dir and os.path.exists(output_dir):
            # Return the evaluation results with output directory for MCP analyzer
            return {
                'output_directory': output_dir,
                'evaluation_metrics': {
                    'accuracy': evaluation_results.get('accuracy', 0.0),
                    'coverage': evaluation_results.get('coverage', 0.0),
                    'precision': evaluation_results.get('precision', 0.0),
                    'total_tool_calls': evaluation_results.get('total_tool_calls', 0),
                    'successful_calls': evaluation_results.get('successful_calls', 0)
                },
                'mcp_data_available': self._check_mcp_files_exist(output_dir)
            }
        
        # Fallback to basic patterns if no detailed output available
        return {
            'total_calls': evaluation_results.get('total_tool_calls', 0),
            'successful_calls': evaluation_results.get('successful_calls', 0),
            'tool_frequency': evaluation_results.get('tool_usage', {}),
            'common_parameters': {},
            'error_patterns': {},
            'mcp_data_available': False
        }
    
    def _create_evaluation_config(self, mcp_config_path: str) -> str:
        """Create temporary evaluation config with specified MCP path."""
        # Load template config
        with open(self.eval_config_template, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update MCP path
        config_data['agent']['params']['mcp_yaml_path'] = mcp_config_path
        config_data['paths']['mcp_yaml_path'] = mcp_config_path
        
        # Set debug count if specified
        if self.debug_count is not None:
            config_data['evaluation']['debug_count'] = self.debug_count
        
        # Apply any additional parameters
        for key, value in self.kwargs.items():
            if '.' in key:
                # Handle nested keys like 'agent.params.llm'
                keys = key.split('.')
                current = config_data
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = value
            else:
                config_data[key] = value
        
        # Save to temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='eval_config_')
        try:
            with os.fdopen(temp_fd, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        except:
            os.close(temp_fd)
            raise
        
        return temp_path
    
    def _parse_evaluation_results(self, stdout: str, config_path: str) -> Dict[str, float]:
        """Parse evaluation results from stdout or result files."""
        metrics = {
            'accuracy': 0.0,
            'coverage': 0.0,
            'precision': 0.0,
            'total_tool_calls': 0,
            'successful_calls': 0
        }
        
        # Try to parse from stdout first
        lines = stdout.split('\n')
        for line in lines:
            if 'accuracy:' in line.lower():
                try:
                    accuracy = float(line.split(':')[-1].strip())
                    metrics['accuracy'] = accuracy
                except ValueError:
                    pass
            elif 'coverage:' in line.lower():
                try:
                    coverage = float(line.split(':')[-1].strip())
                    metrics['coverage'] = coverage
                except ValueError:
                    pass
            elif 'Results saved to:' in line:
                # Extract output directory from evaluation output
                output_dir = line.split('Results saved to:')[-1].strip()
                if os.path.exists(output_dir):
                    metrics['output_directory'] = output_dir
        
        # Try to find result files in output directory
        # The evaluation system saves results to specific directories
        # This would need to be implemented based on actual output structure
        
        return metrics
    
    def _check_mcp_files_exist(self, output_dir: str) -> bool:
        """Check if MCP call history files exist in output directory."""
        output_path = Path(output_dir)
        mcp_calls_file = output_path / "mcp_calls.jsonl"
        steps_file = output_path / "steps.jsonl"
        
        return mcp_calls_file.exists() and steps_file.exists()


class GenericEvaluationAdapter(EvaluationInterface):
    """
    Generic adapter that can work with any evaluation system via configuration.
    """
    
    def __init__(self, command_template: List[str], result_parser: str = "json", **kwargs):
        """
        Initialize generic evaluation adapter.
        
        Args:
            command_template: Command template with placeholders like {mcp_config}
            result_parser: How to parse results ("json", "stdout", "file")
            **kwargs: Additional parameters
        """
        self.command_template = command_template
        self.result_parser = result_parser
        self.kwargs = kwargs
    
    def evaluate_tools(self, tools_config_path: str, queries: List[DatasetQuery]) -> Dict[str, float]:
        """Evaluate using generic command template."""
        # Substitute placeholders in command template
        command = []
        for part in self.command_template:
            if '{mcp_config}' in part:
                command.append(part.replace('{mcp_config}', tools_config_path))
            elif '{dataset}' in part:
                command.append(part.replace('{dataset}', self.kwargs.get('dataset', 'default')))
            else:
                command.append(part)
        
        # Run command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Evaluation command failed: {result.stderr}")
        
        # Parse results based on parser type
        if self.result_parser == "json":
            return json.loads(result.stdout)
        elif self.result_parser == "stdout":
            return self._parse_stdout_metrics(result.stdout)
        else:
            raise ValueError(f"Unknown result parser: {self.result_parser}")
    
    def extract_usage_patterns(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Extract patterns from generic evaluation results."""
        return evaluation_results.get('usage_patterns', {})
    
    def _parse_stdout_metrics(self, stdout: str) -> Dict[str, float]:
        """Parse metrics from stdout text."""
        metrics = {}
        lines = stdout.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    try:
                        key = parts[0].strip().lower()
                        value = float(parts[1].strip())
                        metrics[key] = value
                    except ValueError:
                        continue
        return metrics


class MockEvaluationAdapter(EvaluationInterface):
    """
    Mock evaluation adapter for testing and development.
    """
    
    def __init__(self, mock_accuracy: float = 0.75, **kwargs):
        """Initialize with mock values."""
        self.mock_accuracy = mock_accuracy
        self.kwargs = kwargs
    
    def evaluate_tools(self, tools_config_path: str, queries: List[DatasetQuery]) -> Dict[str, float]:
        """Return mock evaluation results."""
        import random
        
        # Simulate some variance
        variance = random.uniform(-0.1, 0.1)
        accuracy = max(0.0, min(1.0, self.mock_accuracy + variance))
        
        return {
            'accuracy': accuracy,
            'coverage': accuracy * 0.9,  # Slightly lower coverage
            'precision': accuracy * 1.05,  # Slightly higher precision
            'total_tool_calls': random.randint(10, 50),
            'successful_calls': random.randint(8, 45)
        }
    
    def extract_usage_patterns(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Return mock usage patterns."""
        return {
            'tool_frequency': {
                'query_clinvar': 15,
                'query_ensembl': 10,
                'blast_sequence': 8
            },
            'common_parameters': {
                'max_results': [3, 5, 10],
                'verbose': [True]
            }
        }


# Factory function for easy creation
def create_evaluation_adapter(adapter_type: str, **kwargs) -> EvaluationInterface:
    """
    Factory function to create evaluation adapters.
    
    Args:
        adapter_type: Type of adapter ("abstraction", "generic", "mock")
        **kwargs: Parameters for the specific adapter
        
    Returns:
        EvaluationInterface implementation
    """
    if adapter_type == "abstraction":
        return AbstractionEvaluationAdapter(**kwargs)
    elif adapter_type == "generic":
        return GenericEvaluationAdapter(**kwargs)
    elif adapter_type == "mock":
        return MockEvaluationAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown evaluation adapter type: {adapter_type}")
