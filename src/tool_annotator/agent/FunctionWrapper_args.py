import os
import yaml

# Resolve StableToolBench's tooleval directory relative to this file so cwd doesn't matter.
# Layout: <repo>/src/tool_annotator/agent/FunctionWrapper_args.py → <repo>/src/submodules/StableToolBench/toolbench/tooleval
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_AGENT_DIR, "..", ".."))
TOOLEVAL_DIR = os.path.join(_SRC_DIR, "submodules", "StableToolBench", "toolbench", "tooleval")

def add_FunctionWrapper_args(parser):
    # First, parse just the config file argument
    parser.add_argument("--config", type=str, 
                        default=f"../../tool_exec_tracer/tmdb/configs/tmdb_base.yaml",
                       help="Config file path")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Run in debug mode (limited queries for quick testing)")
    parser.add_argument("--full", action="store_true", default=False,
                       help="Run full evaluation (overrides debug)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset to evaluate (overrides config)")
    parser.add_argument("--tool_root_dir", type=str, default=False,
                       help="Tool root directory for StableToolBench. Will use it to call APIs.")

    parser.add_argument("--mcp_yaml_path", type=str, nargs="+", default=None,
                       help="MCP YAML path (overrides config). Can be a list of paths, the very first one is with the highest priority, overriding the others.")
    parser.add_argument("--decompo_mcp_yaml_path", type=str, default=None,
                       help="Decomposition MCP YAML path (overrides config)")

    parser.add_argument("--seed", type=int, default=None,
                       help="Seed for random number generator (overrides config)")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Temperature for task decomposition (overrides config)")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p for task decomposition (overrides config)")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Max tokens for task decomposition (overrides config)")
    parser.add_argument("--model_name", type=str, default="openai:gpt-4.1-2025-04-14",
                       help="Model name for LLM, e.g. 'openai:gpt-4.1-2025-04-14' or 'vllm:Qwen/Qwen2.5-7B-Instruct' (overrides config)")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")

    parser.add_argument("--max_queries", type=int, default=None,
                       help="Max queries to evaluate (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (overrides default path)")
    parser.add_argument("--runs_per_scenario", type=int, default=None,
                       help="Runs per scenario (overrides config)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers for query processing (default: 1)")
    
    parser.add_argument("--task_decomp_prompt_version", type=str, default=None,
                       help="Task decomposition prompt version (overrides config)")
    parser.add_argument("--param_gen_prompt_version", type=str, default=None,
                       help="Parameter generation prompt version (overrides config)")

    parser.add_argument("--expand_same_category", action="store_true", default=False,
                       help="Expand tools in the same category for StableToolBench")
    parser.add_argument("--BM25_threshold", type=float, default=float('inf'),
                       help="Threshold for BM25 similarity score")
    parser.add_argument("--retrieval_sample_size", type=int, default=20,
                       help="Sample size for api providers retrieval")
    parser.add_argument("--smolagents_demo", action="store_true", default=False,
                       help="Run a quick ToolCallingAgent demo with YAML tools")
    parser.add_argument("--demo_task", type=str, default=None,
                       help="Override demo task for ToolCallingAgent")
    return parser


def impute_functionwrapper_args(args):
    # Load YAML config to get defaults
    print(f"📄 Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    # Apply YAML defaults, then override with CLI args if provided
    # Dataset
    if args.dataset is None:
        args.dataset = "tmdb"  # fallback default
    
    # MCP paths - use config's mcp_yaml_path if CLI not provided
    # if args.mcp_yaml_path is None:
    #     args.mcp_yaml_path = config_data.get('agent', {}).get('params', {}).get('mcp_yaml_path', 
    #                             f"{FunctionWrapper_DIR}/eval/tmdb/desc_mcp_yaml/tmdb_d0_mcp.yaml")
    
    if args.decompo_mcp_yaml_path is None:
        # Use same as mcp_yaml_path if not specified
        args.decompo_mcp_yaml_path = args.mcp_yaml_path
    
    # LLM parameters from config
    llm_config = config_data.get('agent', {}).get('llm_config', {})
    
    if args.seed is None:
        args.seed = llm_config.get('seed', 42)
    
    if args.temperature is None:
        args.temperature = llm_config.get('temperature', 0.2)
    
    if args.top_p is None:
        args.top_p = llm_config.get('top_p', 1.0)
    
    if args.max_tokens is None:
        args.max_tokens = llm_config.get('max_tokens', 32768)
    
    if args.model_name is None:
        args.model_name = llm_config.get('model', 'openai:gpt-4.1-2025-04-14')
    
    # Evaluation parameters
    if args.max_queries is None:
        args.max_queries = config_data.get('dataset', {}).get('params', {}).get('max_queries', 100)
    
    if args.runs_per_scenario is None:
        args.runs_per_scenario = config_data.get('agent', {}).get('params', {}).get('runs_per_scenario', 1)
    
    # Semantic matching mode
    args.semantic_matching_mode = config_data.get('agent', {}).get('params', {}).get('semantic_matching_mode', 'keyword')
    
    # Prompt versions
    if args.task_decomp_prompt_version is None:
        args.task_decomp_prompt_version = config_data.get('agent', {}).get('params', {}).get('task_decomp_prompt_version', 'v3')
    
    if args.param_gen_prompt_version is None:
        args.param_gen_prompt_version = config_data.get('agent', {}).get('params', {}).get('param_gen_prompt_version', 'v3')
    
    # print(f"✅ Configuration loaded:")
    # print(f"   Dataset: {args.dataset}")
    # print(f"   Model: {args.model_name}")
    # print(f"   Temperature: {args.temperature}")
    # print(f"   Seed: {args.seed}")
    # print(f"   Max queries: {args.max_queries}")
    # print(f"   Runs per scenario: {args.runs_per_scenario}")
    # print(f"   Semantic matching mode: {args.semantic_matching_mode}")
    # print(f"   Parallel workers: {args.workers}")
    # print(f"   MCP YAML path: {args.mcp_yaml_path}")
    # print(f"   Task decomposition prompt: {args.task_decomp_prompt_version}")
    # print(f"   Parameter generation prompt: {args.param_gen_prompt_version}")
    # print(f"   API selection prompt: api_selection_single")

    # Override debug if --full is specified
    if args.full:
        debug = False
    else:
        debug = args.debug

    """
    Run step-wise evaluation with specified config file.

    Args:
        debug: Whether to run in debug mode (limited examples)
        config_path: Path to config file
    """

    if debug:
        # print("\n🚀 QUICK TEST MODE enabled for Step-wise Evaluation")
        max_queries = 2
        runs_per_scenario = 1
        output_suffix = '_step_wise_quick_test'
    else:
        max_queries = args.max_queries
        runs_per_scenario = args.runs_per_scenario
        output_suffix = '_step_wise_eval'

    # print(f"\n🎯 Starting STEP-WISE EVALUATION:")
    # print(f"   📊 Max queries: {max_queries}")
    # print(f"   🔄 Runs per scenario: {runs_per_scenario}")
    # print(f"   🚀 Mode: Step-wise evaluation with golden API log entries and API selection accuracy")
    # print(f"   📋 Prompt Collection: ENABLED (saving queries, contexts, and selected APIs to prompt.json)")

    # load tmdb or spotify queries
    if args.dataset == "tmdb":
        queries_path = f"../../tool_exec_tracer/tmdb/data/tmdb.json"
    elif args.dataset == "tmdb_0802_syn":
        queries_path = f"../../tool_exec_tracer/tmdb/data/tmdb_0802_syn.json"
    elif args.dataset == "tmdb_0709_syn":
        queries_path = f"../../tool_exec_tracer/tmdb/data/tmdb_0709_syn.json"
    elif args.dataset == "spotify":
        queries_path = f"../../tool_exec_tracer/tmdb/data/spotify.json"
    elif args.dataset == "spotify_1007_syn":
        queries_path = f"../../tool_exec_tracer/tmdb/data/spotify_1007_syn.json"
    elif args.dataset == "spotify_1021_syn":
        queries_path = f"../../tool_exec_tracer/tmdb/data/spotify_1021_syn.json"
    else:
        queries_path = None

    # Pass platform based on dataset - extract base platform name from synthetic dataset names
    # e.g., "spotify_1007_syn" -> "spotify", "tmdb_0802_syn" -> "tmdb"
    if args.dataset.startswith("spotify"):
        platform = "spotify"
    elif args.dataset.startswith("tmdb"):
        platform = "tmdb"
    else:
        platform = args.dataset  # fallback to original
    
    return args, queries_path, platform, output_suffix, max_queries, runs_per_scenario
