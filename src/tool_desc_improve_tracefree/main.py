#!/usr/bin/env python3
"""
Description Improvement Pipeline.

Usage:
    python tool_desc_improve_tracefree/main.py --dataset path/to/mcp.yaml --strategy generic_llm_guidelines
    python tool_desc_improve_tracefree/main.py --config config.yaml
    python tool_desc_improve_tracefree/main.py --list
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_desc_improve_tracefree.config.config import PipelineConfig
from tool_desc_improve_tracefree.pipelines.generic_pipeline import GenericImprovementPipeline
from tool_desc_improve_tracefree.core.registries import ImprovementStrategyRegistry



def create_simple_config(strategy: str, dataset: str, platform: str = 'tmdb', **kwargs) -> dict:
    """Create a minimal configuration from arguments with proper strategy parameters."""

    # Default strategy parameters based on platform and strategy
    strategy_params = {
        'model_name': 'gpt-41-2025-04-14',
        'temperature': 0.3,
        'max_tokens': 2048,
        'dataset_name': platform,  # tmdb, spotify, etc.
        'save_reasoning': True,
        'improve_parameters': False,
    }

    # For generic_llm_guidelines strategy, add prompt_file
    if strategy == 'generic_llm_guidelines':
        # Use available prompt files based on platform
        if platform == 'spotify':
            strategy_params['prompt_file'] = 'data_indep_v1_precise_types.txt'
        else:  # tmdb and others
            strategy_params['prompt_file'] = 'data_indep_v1.txt'

    # Override with any provided kwargs
    strategy_params.update(kwargs)

    # Validate guidelines path exists
    guidelines_path = f'guidelines/datasets/{platform}'
    if not Path(guidelines_path).exists():
        print(f"⚠️  Warning: Guidelines directory does not exist: {guidelines_path}")
        print(f"    This may cause issues with guideline loading")

    return {
        'dataset': {
            'name': 'tool_manager',
            'tools_path': dataset,
            'params': {
                'platform': platform,
                'verbose': False
            }
        },
        'improvement': {
            'turn_1': {
                'strategy': strategy,
                'stage': 'data_independent',  # Changed from data_dependent
                'params': strategy_params
            },
            'guidelines_provider': 'file_based',
            'guidelines_path': f'guidelines/datasets/{platform}',
            'guidelines_file': 'generic_v1.txt',
            'pattern_analyzer': 'basic',
            'max_turns': 1
        },
        'output': {
            'base_dir': f'results/{strategy}_{platform}',
            'save_metadata': True
        }
    }


def run_pipeline(config: PipelineConfig) -> bool:
    """Run the improvement pipeline with given configuration."""
    pipeline = GenericImprovementPipeline(config)
    
    # Print selected strategy if available (handle configs without turn_1)
    selected_strategy = None
    if getattr(config.improvement, 'turn_1', None) and getattr(config.improvement.turn_1, 'strategy', None):
        selected_strategy = config.improvement.turn_1.strategy
    elif getattr(config.improvement, 'turn_2', None) and getattr(config.improvement.turn_2, 'strategy', None):
        selected_strategy = config.improvement.turn_2.strategy
    elif hasattr(config.improvement, 'strategies') and getattr(config.improvement, 'strategies', None):
        try:
            selected_strategy = config.improvement.strategies[0].get('name')
        except Exception:
            selected_strategy = None
    print(f"🚀 Running {selected_strategy or 'improvement pipeline'}")
    print(f"📁 Dataset: {config.dataset.tools_path}")
    print(f"💾 Output: {config.output.base_dir}")
    
    results = pipeline.run()
    
    print(f"✅ Complete! Results in: {results.get('final_tools_path', 'results/')}")
    return True


    
def main(args):
    # List strategies
    if args.list:
        strategies = ImprovementStrategyRegistry.list_strategies()
        print("Available strategies:")
        for strategy in strategies:
            print(f"  - {strategy}")
        return 0
    
    # Load configuration
    config = None
    if args.config:
        # Load from YAML config file
        print(f"🔧 Loading configuration from: {args.config}")
        try:
            config = PipelineConfig.from_yaml(args.config, args)
        except FileNotFoundError:
            print(f"❌ Error: Config file not found: {args.config}")
            return 1
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return 1
    elif args.dataset:
        # Create config from dataset + strategy arguments (ToolManager mode)
        print("🚀 Using ToolManager with dataset + strategy arguments")
        platform = 'tmdb'  # Default platform

        # Detect platform from dataset path if possible
        if 'spotify' in args.dataset.lower():
            platform = 'spotify'
        elif 'tmdb' in args.dataset.lower():
            platform = 'tmdb'

        # Use a default strategy if none provided
        strategy = args.strategy or 'generic_llm_guidelines'

        # Create config using the original logic but with ToolManager-loaded data
        # First, detect the server name from the YAML file
        try:
            with open(args.dataset, 'r') as f:
                mcp_config = yaml.safe_load(f)
                server_name = list(mcp_config['mcp_servers'].keys())[0]  # Get actual server name
        except FileNotFoundError:
            print(f"❌ Error: Dataset file not found: {args.dataset}")
            return 1
        except yaml.YAMLError as e:
            print(f"❌ Error: Invalid YAML file: {e}")
            return 1
        except KeyError as e:
            print(f"❌ Error: Invalid MCP YAML structure - missing {e}")
            return 1

        config_dict = create_simple_config(
            strategy,
            args.dataset,
            platform=platform,  # Pass the detected platform
            extract_guidelines=getattr(args, 'extract_guidelines', False)
        )
        # Add the server name to the dataset params (merge, don't replace)
        config_dict['dataset']['params']['server_name'] = server_name
        config = PipelineConfig.from_dict(config_dict)
    else:
        print("\nError: Provide either --config or --dataset")
        return 1

    # Run the improvement pipeline
    print(f"\n🔧 Running description improvement pipeline...")
    success = run_pipeline(config)
    return 0 if success else 1


if __name__ == "__main__":
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Description Improvement Pipeline')
    
    # Primary options
    parser.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--strategy', help='Strategy name (e.g., mcp_contrastive_strategy)')
    parser.add_argument('--dataset', help='Path to tools file')
    parser.add_argument('--output', help='Path to output directory')
    parser.add_argument('--keep_filename', action='store_true', help='Keep original filenames when saving the results')
    
    # Strategy parameters
    parser.add_argument('--extract-guidelines', action='store_true', 
                       help='Enable guideline extraction (two-stage approach)')
    
    # Utility
    parser.add_argument('--list', action='store_true', help='List available strategies')
    
    args = parser.parse_args()
    main(args)
