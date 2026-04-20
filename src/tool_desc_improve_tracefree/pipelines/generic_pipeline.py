"""
Generalized improvement pipeline that works with any dataset and improvement method.
"""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from tqdm import tqdm as _tqdm
except Exception:
    def _tqdm(iterable, total=None, desc=None, unit=None):
        return iterable

from ..config.config import PipelineConfig
from ..core.registries import (
    DatasetRegistry, ImprovementStrategyRegistry, EvaluationRegistry,
    GuidelinesRegistry, PatternAnalyzerRegistry
)
from ..integrations import evaluation_integration  # Load evaluation adapters
from ..interfaces.dataset_interface import StandardizedTool, DatasetQuery
from ..interfaces.improvement_interface import ImprovementContext, ImprovementStage
from .. import datasets  # Load dataset implementations
from .. import strategies  # Load strategy implementations
from ..formatters.mcp_yaml_formatter import save_as_mcp_yaml


logger = logging.getLogger(__name__)


class GenericImprovementPipeline:
    """
    Generic pipeline that can handle any dataset type and improvement method.
    """
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.dataset = None
        self.evaluator = None
        self.guidelines_provider = None
        self.pattern_analyzer = None
        self.strategies = []
        self.run_output_dir = None  # Will be set when run() is called
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components based on configuration."""
        logger.info(f"Initializing pipeline for dataset: {self.config.dataset.name}")
        
        # Initialize dataset
        self.dataset = DatasetRegistry.create(
            self.config.dataset.name,
            **self.config.dataset.params
        )
        logger.info(f"Dataset initialized: {self.dataset.name}")
        
        # Initialize evaluator if enabled
        if self.config.evaluation.enabled:
            # Merge required evaluation args with optional params
            eval_kwargs = dict(self.config.evaluation.params or {})
            if getattr(self.config.evaluation, 'eval_config_template', None):
                eval_kwargs['eval_config_template'] = self.config.evaluation.eval_config_template
            if getattr(self.config.evaluation, 'dataset_for_eval', None):
                # Adapter expects 'dataset_name'
                eval_kwargs['dataset_name'] = self.config.evaluation.dataset_for_eval

            self.evaluator = EvaluationRegistry.create(
                self.config.evaluation.evaluator,
                **eval_kwargs
            )
            logger.info(f"Evaluator initialized: {self.config.evaluation.evaluator}")
        
        # Initialize guidelines provider
        self.guidelines_provider = GuidelinesRegistry.create(
            self.config.improvement.guidelines_provider
        )
        
        # Initialize pattern analyzer (only if needed for data-dependent stages)
        needs_pattern_analyzer = (
            (hasattr(self.config.improvement, 'turn_2') and self.config.improvement.turn_2 is not None) or
            (hasattr(self.config.improvement, 'strategies') and 
             any(s.get('stage') == 'data_dependent' for s in (self.config.improvement.strategies or [])))
        )
        
        if needs_pattern_analyzer:
            self.pattern_analyzer = PatternAnalyzerRegistry.create(
                self.config.improvement.pattern_analyzer
            )
        else:
            self.pattern_analyzer = None
            logger.info("Skipping pattern analyzer - only data-independent improvement configured")
        
        # Initialize improvement strategies (handle both strategies list and turn-based config)
        if hasattr(self.config.improvement, 'strategies') and self.config.improvement.strategies:
            # Strategy-based configuration
            for strategy_config in self.config.improvement.strategies:
                strategy = ImprovementStrategyRegistry.create(
                    strategy_config['name'],
                    **strategy_config.get('params', {})
                )
                self.strategies.append({
                    'strategy': strategy,
                    'stage': strategy_config.get('stage', 'data_independent'),
                    'config': strategy_config
                })
        else:
            # Turn-based configuration - strategies are created dynamically during turns
            logger.info("Using turn-based configuration - strategies will be created per turn")
        
        logger.info(f"Initialized {len(self.strategies)} improvement strategies")
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete improvement pipeline.
        
        Returns:
            Dictionary containing results from all iterations.
        """
        from datetime import datetime
        
        logger.info("Starting improvement pipeline")
        
        # Create timestamped output directory for this run
        # timestamp can be provided in the config or Command Line Arguments
        self.run_output_dir = Path(self.config.output.base_dir)
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created run output directory: {self.run_output_dir}")
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
        
        # Load initial data
        tools, server_config = self.dataset.load_tools(self.config.dataset.tools_path)
        queries = []
        if self.config.dataset.queries_path:
            queries = self.dataset.load_queries(self.config.dataset.queries_path)
        
        logger.info(f"Loaded {len(tools)} tools and {len(queries)} queries")
        
        # Load guidelines
        guidelines = []
        if self.config.improvement.guidelines_path:
            # Pass guidelines_file in context if specified
            context = {}
            if self.config.improvement.guidelines_file:
                context['guidelines_file'] = self.config.improvement.guidelines_file
            
            guidelines = self.guidelines_provider.load_guidelines(
                self.config.improvement.guidelines_path,
                context=context
            )
        
        # Run improvement turns (D0->D1->D2)
        results = {'turns': []}
        current_tools = tools  # D0
        
        # Turn 1: D0 -> D1 (data independent)
        if self.config.improvement.has_turn(1):
            logger.info(f"\n--- Turn 1: D0 -> D1 (Data Independent) ---")
            
            turn_result = self._run_turn(1, current_tools, queries, guidelines, server_config)
            results['turns'].append(turn_result)
            current_tools = turn_result['improved_tools']  # Now D1
            
            # Save D1 in MCP YAML format (universal format) in timestamped directory
            if self.config.keep_filename:
                mcp_d1_path = self.run_output_dir / self.config.dataset.tools_path.split("/")[-1]
            else:
                mcp_d1_path = self.run_output_dir / "D1_mcp.yaml"
            # Get dataset name from turn config params or fall back to dataset loader name
            turn_1_params = self.config.improvement.turn_1.params if isinstance(self.config.improvement.turn_1.params, dict) else {}
            dataset_name = turn_1_params.get('dataset_name', self.dataset.name)
            save_as_mcp_yaml(server_config, current_tools, str(mcp_d1_path), dataset_name, "D1")
            logger.info(f"Saved D1 results to: {mcp_d1_path}")
            mcp_final_path = mcp_d1_path
        
        # Turn 2: D1 -> D2 (data dependent, building on D1)
        if self.config.improvement.has_turn(2):
            logger.info(f"\n--- Turn 2: D1 -> D2 (Data Dependent) ---")
            
            # For turn 2, we need usage patterns from D1 evaluation
            turn_result = self._run_turn(2, current_tools, queries, guidelines, server_config)
            results['turns'].append(turn_result)
            current_tools = turn_result['improved_tools']  # Now D2
            
            # Save D2 in MCP YAML format (universal format) in timestamped directory
            if self.config.keep_filename:
                mcp_d2_path = self.run_output_dir / self.config.dataset.tools_path.split("/")[-1]
            else:
                mcp_d2_path = self.run_output_dir / "D2_mcp.yaml"
            # Get dataset name from turn config params or fall back to dataset loader name
            turn_2_params = self.config.improvement.turn_2.params if isinstance(self.config.improvement.turn_2.params, dict) else {}
            dataset_name = turn_2_params.get('dataset_name', self.dataset.name)
            save_as_mcp_yaml(server_config, current_tools, str(mcp_d2_path), dataset_name, "D2")
            logger.info(f"Saved D2 results to: {mcp_d2_path}")
            mcp_final_path = mcp_d2_path
            # Save patterns/guidelines to separate file if available
            if 'improvement_metadata' in turn_result and 'tool_patterns' in turn_result.get('improvement_metadata', {}):
                self._save_patterns_file(turn_result['improvement_metadata']['tool_patterns'], mcp_d2_path)
        
        # Update final path to point to MCP YAML format in timestamped directory
        results['final_tools_path'] = str(mcp_final_path)
        results['run_output_dir'] = str(self.run_output_dir)
        results['config'] = self.config.to_dict()
        
        # Save the complete configuration used for this run
        self._save_config_file(results)
        
        logger.info("Pipeline completed successfully")
        return results
    
    def _run_turn(self, turn_number: int, tools: List[StandardizedTool], 
                  queries: List[DatasetQuery], guidelines: List[str], server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single improvement turn (D0->D1 or D1->D2)."""
        turn_config = self.config.improvement.get_turn_config(turn_number)
        if not turn_config:
            raise ValueError(f"No configuration found for turn {turn_number}")
        
        turn_result = {
            'turn_number': turn_number,
            'stage': turn_config.stage,
            'strategy_applied': turn_config.strategy,
            'evaluation_results': {},
            'improved_tools': tools,
            'errors': []
        }
        
        # Get the strategy for this turn
        try:
            # Add run output directory to strategy params for saving reasoning JSONs
            strategy_params = dict(turn_config.params or {})
            if 'output_dir' not in strategy_params:
                strategy_params['output_dir'] = str(self.run_output_dir)

            # Pass run timestamp (extracted from folder name) for consistent metadata
            strategy_params['run_timestamp'] = self.run_output_dir.name

            # Fix for GenericLLMStrategy: construct prompt_template_path from prompt_file and dataset_name
            if turn_config.strategy == "generic_llm_guidelines" and 'prompt_file' in strategy_params and 'dataset_name' in strategy_params:
                from pathlib import Path
                dataset_name = strategy_params['dataset_name']
                prompt_file = strategy_params['prompt_file']
                prompt_template_path = Path(f"prompts/datasets/{dataset_name}/{prompt_file}")
                strategy_params['prompt_template_path'] = str(prompt_template_path)
                # Remove prompt_file since it's now converted to prompt_template_path
                del strategy_params['prompt_file']

            # Pass guidelines file path for metadata
            if turn_config.strategy == "generic_llm_guidelines":
                guidelines_path = Path(self.config.improvement.guidelines_path)
                guidelines_file = self.config.improvement.guidelines_file or 'generic_v1.txt'
                guidelines_file_path = guidelines_path / guidelines_file
                strategy_params['guidelines_file_path'] = str(guidelines_file_path)

            strategy = ImprovementStrategyRegistry.create(
                turn_config.strategy,
                **strategy_params
            )
            
        except Exception as e:
            error_msg = f"Failed to create strategy {turn_config.strategy}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(f"Strategy creation failed: {error_msg}")
        
        # Prepare context for this turn
        context = self._prepare_context(
            turn_config.stage, tools, queries, guidelines, turn_number, 
            server_config
        )
        
        # For turn 2 (data dependent), add usage patterns from previous evaluation
        if turn_number == 2 and turn_config.stage == "data_dependent":
            # Get usage patterns from D1 evaluation (this would be implemented)
            context.usage_patterns = self._get_usage_patterns_from_previous_turn()
        
        # For turn 2 (data dependent), if a custom MCP logs dir is provided,
        # attach it directly as evaluation_results for the strategy to consume, and skip pre-eval.
        used_custom_mcp = False
        custom_mcp_dir = None
        if turn_number == 2 and turn_config.stage == "data_dependent":
            # Check both turn_config.params and evaluation.params for the path
            try:
                # First check turn_config.params (preferred location)
                turn_params = turn_config.params if isinstance(turn_config.params, dict) else {}
                custom_mcp_dir = turn_params.get('eval_results_path') or turn_params.get('custom_mcp_output_dir')
                
                # Fallback to evaluation.params
                if not custom_mcp_dir:
                    custom_mcp_dir = (self.config.evaluation.params or {}).get('custom_mcp_output_dir')
            except Exception:
                custom_mcp_dir = None
            
            if custom_mcp_dir and os.path.isdir(os.path.expanduser(custom_mcp_dir)):
                context.evaluation_results = {
                    'output_directory': os.path.expanduser(custom_mcp_dir)
                }
                used_custom_mcp = True
                logger.info(f"Using provided MCP logs directory for contrastive patterns: {custom_mcp_dir}")

        # For turn 2 (data dependent), pre-evaluate current tools to provide MCP data to strategy
        if (
            turn_number == 2 and turn_config.stage == "data_dependent" and
            self.config.evaluation.enabled and self.evaluator and not used_custom_mcp
        ):
            try:
                temp_path_pre = self.config.get_output_path(turn_number, f"temp_turn_{turn_number}_pre.yaml")
                self.dataset.save_tools(tools, temp_path_pre)
                eval_results_pre = self.evaluator.evaluate_tools(temp_path_pre, queries)
                # Attach evaluation results to context for strategy consumption
                context.evaluation_results = dict(eval_results_pre or {})
                # Provide alias expected by some strategies
                if (
                    isinstance(eval_results_pre, dict) and
                    'output_directory' in eval_results_pre and
                    'mcp_output_dir' not in eval_results_pre
                ):
                    context.evaluation_results['mcp_output_dir'] = eval_results_pre['output_directory']
                logger.info(f"Pre-evaluation for turn {turn_number} produced: {eval_results_pre}")
            except Exception as e:
                logger.warning(f"Pre-evaluation for turn {turn_number} failed: {e}")

        # Apply the strategy
        result = strategy.improve_descriptions(context)
        
        if result.success:
            turn_result['improved_tools'] = result.improved_tools
            turn_result['improvement_metadata'] = result.improvement_metadata
            logger.info(f"Turn {turn_number} completed successfully")
        else:
            error_msg = f"Strategy failed: {result.error_message}"
            logger.error(error_msg)
            raise RuntimeError(f"Strategy {turn_config.strategy} failed: {result.error_message}")
        
        # Run evaluation if enabled
        if self.config.evaluation.enabled and self.evaluator:
            try:
                # Save tools for evaluation
                temp_path = self.config.get_output_path(turn_number, f"temp_turn_{turn_number}.yaml")
                self.dataset.save_tools(turn_result['improved_tools'], temp_path)
                
                # Run evaluation
                eval_results = self.evaluator.evaluate_tools(temp_path, queries)
                turn_result['evaluation_results'] = eval_results
                
                logger.info(f"Turn {turn_number} evaluation results: {eval_results}")
                
            except Exception as e:
                error_msg = f"Evaluation failed: {str(e)}"
                logger.error(error_msg)
                turn_result['errors'].append(error_msg)
                
                if not self.config.continue_on_error:
                    raise
        
        return turn_result
    
    def _save_config_file(self, results: Dict[str, Any]) -> None:
        """Save the complete configuration used for this run."""
        import json
        from datetime import datetime
        
        try:
            # Save config to the timestamped run output directory
            config_path = self.run_output_dir / "config.json"
            
            # Prepare config data with additional metadata
            config_data = {
                "pipeline_run_info": {
                    "timestamp": datetime.now().isoformat(),
                    "pipeline_version": "1.0",
                    "dataset_name": self.dataset.name,
                    "total_turns": len(results.get('turns', [])),
                    "final_tools_path": results.get('final_tools_path', ''),
                },
                "configuration": self.config.to_dict(),
                "run_results_summary": {
                    "turns_completed": len(results.get('turns', [])),
                    "tools_processed": len(results['turns'][0]['improved_tools']) if results.get('turns') else 0,
                    "errors": [turn.get('errors', []) for turn in results.get('turns', [])],
                }
            }
            
            # Save config to JSON file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Saved complete configuration to: {config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save config file: {e}")
    
    def _get_usage_patterns_from_previous_turn(self) -> Dict[str, Any]:
        """Get usage patterns from previous turn's evaluation."""
        # This would analyze the evaluation results from D1 to extract usage patterns
        # For now, return empty dict as placeholder
        return {}
    
    def _prepare_context(self, stage_name: str, tools: List[StandardizedTool],
                        queries: List[DatasetQuery], guidelines: List[str],
                        iteration: int, server_config: Dict[str, Any]) -> ImprovementContext:
        """Prepare improvement context for a stage."""
        
        # Get stage enum
        stage_enum = ImprovementStage.DATA_INDEPENDENT
        if stage_name == "data_dependent":
            stage_enum = ImprovementStage.DATA_DEPENDENT
        elif stage_name == "iterative":
            stage_enum = ImprovementStage.ITERATIVE
        elif stage_name == "feedback_based":
            stage_enum = ImprovementStage.FEEDBACK_BASED
        
        context = ImprovementContext(
            tools=tools,
            queries=queries,
            guidelines=guidelines,
            iteration_number=iteration,
            stage=stage_enum,
            server_config=server_config
        )
        
        # Add usage patterns for data-dependent stages
        if stage_enum == ImprovementStage.DATA_DEPENDENT and iteration > 1:
            # Analyze patterns from previous evaluation
            # This would be implemented based on specific pattern analyzer
            context.usage_patterns = {}
        
        return context
    
    def _apply_strategies_sequential(self, strategies: List[Dict], 
                                   context: ImprovementContext,
                                   iteration_result: Dict) -> List[StandardizedTool]:
        """Apply strategies sequentially."""
        current_tools = context.tools
        
        for strategy_info in _tqdm(strategies, total=len(strategies), desc="Applying strategies", unit="strategy"):
            strategy = strategy_info['strategy']
            strategy_name = strategy.name
            
            logger.info(f"Applying strategy: {strategy_name}")
            
            try:
                # Update context with current tools
                updated_context = ImprovementContext(
                    tools=current_tools,
                    queries=context.queries,
                    guidelines=context.guidelines,
                    usage_patterns=context.usage_patterns,
                    evaluation_results=context.evaluation_results,
                    iteration_number=context.iteration_number,
                    stage=context.stage
                )
                
                # Apply strategy
                result = strategy.improve_descriptions(updated_context)
                
                if result.success:
                    current_tools = result.improved_tools
                    iteration_result['strategies_applied'].append({
                        'name': strategy_name,
                        'success': True,
                        'metadata': result.improvement_metadata
                    })
                    logger.info(f"Strategy {strategy_name} applied successfully")
                else:
                    error_msg = f"Strategy {strategy_name} failed: {result.error_message}"
                    logger.error(error_msg)
                    iteration_result['errors'].append(error_msg)
                    
                    if not self.config.continue_on_error:
                        raise RuntimeError(error_msg)
                
            except Exception as e:
                error_msg = f"Strategy {strategy_name} raised exception: {str(e)}"
                logger.error(error_msg)
                iteration_result['errors'].append(error_msg)
                
                if not self.config.continue_on_error:
                    raise
        
        return current_tools
    
    def _apply_strategies_parallel(self, strategies: List[Dict],
                                  context: ImprovementContext,
                                  iteration_result: Dict) -> List[StandardizedTool]:
        """Apply strategies in parallel (for independent strategies)."""
        # For parallel execution, we apply each strategy to the original tools
        # and then merge the results
        
        futures = []
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            for strategy_info in strategies:
                future = executor.submit(
                    self._apply_single_strategy, 
                    strategy_info['strategy'], 
                    context
                )
                futures.append((future, strategy_info))
        
        # Collect results
        strategy_results = []
        for future, strategy_info in _tqdm(futures, total=len(futures), desc="Waiting strategies", unit="future"):
            try:
                result = future.result()
                strategy_results.append((strategy_info, result))
            except Exception as e:
                error_msg = f"Parallel strategy {strategy_info['strategy'].name} failed: {str(e)}"
                logger.error(error_msg)
                iteration_result['errors'].append(error_msg)
                
                if not self.config.continue_on_error:
                    raise
        
        # Merge results (this would be implemented based on specific needs)
        # For now, use the last successful result
        final_tools = context.tools
        for strategy_info, result in strategy_results:
            if result.success:
                final_tools = result.improved_tools
                iteration_result['strategies_applied'].append({
                    'name': strategy_info['strategy'].name,
                    'success': True,
                    'metadata': result.improvement_metadata
                })
        
        return final_tools
    
    def _apply_single_strategy(self, strategy, context):
        """Apply a single strategy (for parallel execution)."""
        return strategy.improve_descriptions(context)
    
    def _save_patterns_file(self, tool_patterns: Dict[str, Any], tools_path: str) -> None:
        """Save extracted patterns/guidelines to a separate JSON file.
        
        Args:
            tool_patterns: Dictionary mapping tool names to their patterns/guidelines
            tools_path: Path to the tools file (patterns will be saved alongside)
        """
        import json
        from datetime import datetime
        
        # Generate patterns file path
        patterns_path = str(tools_path).replace('_mcp.yaml', '_patterns.json').replace('.yaml', '_patterns.json')
        
        # Create comprehensive patterns file
        patterns_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "tools_file": str(tools_path),
                "total_tools_with_patterns": len(tool_patterns),
                "description": "Extracted patterns and guidelines used for contrastive tool description improvement"
            },
            "tool_patterns": tool_patterns
        }
        
        # Save to JSON
        patterns_dir = os.path.dirname(patterns_path)
        if patterns_dir:
            os.makedirs(patterns_dir, exist_ok=True)
        with open(patterns_path, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        logger.info(f"💾 Saved patterns/guidelines to: {patterns_path}")
        logger.info(f"   Total tools with patterns: {len(tool_patterns)}")
