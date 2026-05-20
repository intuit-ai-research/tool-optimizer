import argparse
import sys
import os
import inspect
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
import numpy as np
from tqdm import tqdm
import copy
from collections import defaultdict

from typing import Dict, List
import random

from dotenv import load_dotenv
load_dotenv()

_PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])  # tool_exec_tracer/
ROOT_DIR = os.path.expanduser(os.environ.get("ROOT_DIR", _PACKAGE_ROOT))

# StableToolBench lives at <repo>/src/submodules/StableToolBench (a git submodule, not a subpackage of
# tool_exec_tracer). Add it to sys.path so its internal `toolbench.*` imports resolve.
_STB_DIR = str(Path(__file__).resolve().parents[3] / "submodules" / "StableToolBench")
sys.path.insert(0, _STB_DIR)

from toolbench.inference.Downstream_tasks.rapidapi_multithread import rapidapi_wrapper, pipeline_runner
from toolbench.utils import standardize, change_name
from tool_exec_tracer.utils.exp_meta import build_reproducibility_log
import logging
import json
import yaml

from tool_exec_tracer.tmdb.utils.task_decomp import task_decompose
from tool_exec_tracer.tmdb.utils.task_decomp import task_topology
from tool_exec_tracer.tmdb.utils.task_decomp import identify_api_requiring_subtasks

from tool_exec_tracer.tmdb.utils.step_wise_eval import evaluate_query_with_step_wise_eval

from tool_exec_tracer.tmdb.utils.tool_manager import ToolManager
from tool_exec_tracer.tmdb.utils.mcp_call_logger import initialize_global_logger, get_global_logger
from tool_exec_tracer.tmdb.utils.llm_call_logger import initialize_global_llm_logger, get_global_llm_logger
from tool_exec_tracer.tmdb.utils.evaluation_statistics import EvaluationStatistics

from tool_exec_tracer.tmdb.utils.prompt_loader import PromptLoader


from joblib import Memory
# Specify a directory for the cache
location = 'cache'
MEMORY = Memory(location, verbose=0) # verbose=0 suppresses output

# @MEMORY.cache
def load_mcp_yaml_func(mcp_tool_path , old_version=False):
    path = Path(mcp_tool_path)
    # load the yaml file into a dictionary "category" --> "tool_name" --> yaml
    cate_dict = defaultdict(dict)
    for category_folder in (progress_bar := tqdm(list(path.iterdir()))):
        # tqdm with category as progress bar
        progress_bar.set_description(f"Loading {category_folder.name} MCP YAML files")

        # check if the category folder is a folder
        if not category_folder.is_dir():
            continue
        for file in category_folder.iterdir():
            # check if the file is a yaml file
            if not str(file).endswith(".yaml"):
                continue
            with open(file, "r") as f:
                yaml_data = yaml.safe_load(f)
                tool_name = list(yaml_data["mcp_servers"].keys())[0]
                cate_name = yaml_data["mcp_servers"][tool_name]["category"]

                if old_version:
                    cate_dict[cate_name][standardize(tool_name)] = yaml_data
                else:
                    if standardize(tool_name) in cate_dict[cate_name]:
                        # Merge tools from both YAML files instead of overwriting
                        existing_yaml = cate_dict[cate_name][standardize(tool_name)]
                        existing_tool_key = list(existing_yaml["mcp_servers"].keys())[0]
                        existing_tools = existing_yaml["mcp_servers"][existing_tool_key]["tools"]
                        new_tools = yaml_data["mcp_servers"][tool_name]["tools"]
                        existing_tools.extend(new_tools)
                    else:
                        cate_dict[cate_name][standardize(tool_name)] = yaml_data
    return cate_dict

def get_tool_manager_and_solution(env, all_mcp_yaml, data_dict):
    # find the mcp yaml file for the tool
    cate_names = env.cate_names
    tool_names = env.tool_names
    functions = env.functions
    tools = []
    relevant_apis = {(tool_name_golden, change_name(standardize(api_name_golden))): True for tool_name_golden, api_name_golden in data_dict["relevant APIs"]}
    solution = []
    endpoint_mapping = {}
    # map name in yaml to the name to use in rapidapi_wrapper
    rapidapi_wrapper_name_mapping = {}
    for cate, tool1, function1 in zip(cate_names, tool_names, functions):
        # function1: {'type': 'function', 'function': {'name': 'transactions_for_orderful', 'description': 'This is the subfunction for tool "orderful", you can use this tool.The description of this function is: "Get Transaction by ID"', 'parameters': {...}}}
        cate_yaml = all_mcp_yaml[cate]
        tool_yaml = cate_yaml[standardize(tool1)]
        
        tool_provider_name = list(tool_yaml['mcp_servers'].keys())[0]

        api_name_in_query = env.api_name_reflect[function1["function"]["name"]]
        FOUND=False
        for api_in_yaml in list(tool_yaml['mcp_servers'].values())[0]['tools']:
            # api_in_yaml: {'tool_name': 'Create Transaction', 'description': 'Creates an Orderful Transaction', '_metadata': {'endpoint': '/transactions', 'method': 'POST'}}
            if change_name(standardize(api_in_yaml['tool_name'])) == api_name_in_query:
                FOUND=True

                api_in_yaml["tool_provider"] = standardize(tool_provider_name)
                api_in_yaml["tool_name"] = standardize(api_in_yaml['tool_name'])

                tools.append(api_in_yaml) # endpoint can be found here

                rapidapi_wrapper_name_mapping[f"{api_in_yaml['tool_provider']}::{api_in_yaml['tool_name']}"] = function1["function"]["name"]

                # request_method = api_in_yaml['_metadata']['method']
                # endpoint = api_in_yaml['_metadata']['endpoint']
                # endpoint_mapping[endpoint] = api_in_yaml['tool_name']

                break
        if not FOUND:
            raise ValueError(f"API {api_name_in_query} not found in {tool_yaml}")

        if (tool_provider_name, api_name_in_query) in relevant_apis:
            # solution.append(f"{request_method} {endpoint}") # GET/POST /endpoint
            solution.append([standardize(tool_provider_name), standardize(api_name_in_query)])
        
    if len(solution) != len(data_dict["relevant APIs"]):
        logging.info(f"Contain invalid APIs in the solution. Skip this query.")
        return None
    tool_manager = ToolManager(tools, platform="StableToolBench", verbose=False)

    # change the endpoint mapping of tool_manager to the endpoint mapping of env
    # tool_manager.endpoint_mapping = endpoint_mapping
    return tool_manager, solution, rapidapi_wrapper_name_mapping

def load_all_json(tool_root_dir):
    all_json = defaultdict(dict) # category_name --> standardize(change_name(tool_name)) --> json
    for cate in (progress_bar := tqdm(os.listdir(tool_root_dir))):
        if not os.path.isdir(os.path.join(tool_root_dir,cate)):
            continue
        progress_bar.set_description(f"Loading {cate} JSON files")
        for file in os.listdir(os.path.join(tool_root_dir,cate)):
            if not file.endswith(".json"):
                continue
            # :-5 to remove the .json extension
            json_file = json.load(open(os.path.join(tool_root_dir, cate, file), "r"))
            json_file["white_list"] = {"description": json_file["tool_description"], "standard_tool_name": standardize(json_file["tool_name"])}# file.split(".")[0]}
            all_json[cate][standardize(json_file["tool_name"])] = json_file
    return all_json


def prepare_data_from_StableToolBench(args_StableToolBench, mcp_tool_path, decompo_mcp_tool_path, param_yaml_paths=None, load_mcp_yaml_func=load_mcp_yaml_func, load_all_json_func=load_all_json, load_mcp_old_version=False) -> list[tuple[dict, tuple[dict, dict, dict, rapidapi_wrapper]]]:
    """
    Use the MCP YAML files to create the tool managers, which will be used for LLM to select and generate parameters.
    However, the original json file of StableToolBench is used to call the APIs.

    mcp_tool_path: 
        when mcp_tool_path: Path, it is the folder path of the MCP YAML files.
        when mcp_tool_path: List[Path], it is the list of the MCP YAML files. The very first one is with the highest priority, overriding the others.

    """
    
    all_json = load_all_json_func(args_StableToolBench.tool_root_dir)
    StableToolBench_pipeline = pipeline_runner(args_StableToolBench, all_json=all_json)
    task_list = StableToolBench_pipeline.task_list
    # method, backbone_model, query_id, data_dict, args, answer_dir, tool_des

    # override descriptions
    
    if isinstance(mcp_tool_path, list):
        all_mcp_yaml = defaultdict(dict)
        for mcp_tool_path_i in mcp_tool_path[::-1]:
            new_mcp_yaml = load_mcp_yaml_func(mcp_tool_path_i, old_version=load_mcp_old_version)
            for cate in new_mcp_yaml:
                for tool_name in new_mcp_yaml[cate]:
                    all_mcp_yaml[cate][tool_name] = new_mcp_yaml[cate][tool_name]

    else:
        all_mcp_yaml = load_mcp_yaml_func(mcp_tool_path, old_version=load_mcp_old_version)

    # override the mcp yaml's parameters with the param yaml's parameters
    if param_yaml_paths:
        for param_yaml_path_i in param_yaml_paths[::-1]:
            param_yaml = load_mcp_yaml_func(param_yaml_path_i, old_version=load_mcp_old_version)
            for cate in param_yaml:
                for tool_name in param_yaml[cate]:
                    for name, info in param_yaml[cate][tool_name]["mcp_servers"].items():
                        for api_idx, api in enumerate(info['tools']):
                            if 'parameters' not in api:
                                continue
                            
                            new_parameters = api['parameters']

                            # override the parameters with the new parameters
                            for old_api in all_mcp_yaml[cate][tool_name]["mcp_servers"][name]['tools']:
                                if old_api['tool_name'] == api['tool_name']:
                                    old_parameters = old_api.get('parameters', {}) #allow old_api does not have parameters
                                    old_api['parameters'] = new_parameters
                                    print(f"Overriding {cate}/{tool_name}/{name}/{api['tool_name']} from {old_parameters} to {new_parameters}")
                                    break
    else:
        logging.info("No parameter YAML paths provided. Using the default parameters.")


    # Build decomposition YAML map independently so decomposition can use a different description set.
    if decompo_mcp_tool_path is None:
        all_decompo_mcp_yaml = all_mcp_yaml
    elif isinstance(decompo_mcp_tool_path, list):
        all_decompo_mcp_yaml = defaultdict(dict)
        for decompo_mcp_tool_path_i in decompo_mcp_tool_path[::-1]:
            new_decompo_yaml = load_mcp_yaml_func(decompo_mcp_tool_path_i, old_version=load_mcp_old_version)
            for cate in new_decompo_yaml:
                for tool_name in new_decompo_yaml[cate]:
                    all_decompo_mcp_yaml[cate][tool_name] = new_decompo_yaml[cate][tool_name]
    else:
        all_decompo_mcp_yaml = load_mcp_yaml_func(decompo_mcp_tool_path, old_version=load_mcp_old_version)

    data_list = []

    for task in tqdm(task_list, desc="Build tool managers and solutions for StableToolBench"):
        method, backbone_model, query_id, data_dict, args, answer_dir, tool_des = task
        env = rapidapi_wrapper(data_dict, tool_des, None, args, process_id=0, all_json=all_json)
        env.functions = env.functions[:-1] # remove the finish function
        tool_manager_and_solution = get_tool_manager_and_solution(env, all_mcp_yaml, data_dict)
        
        if tool_manager_and_solution is None:
            continue
        else:
            tool_manager, solution, rapidapi_wrapper_name_mapping = tool_manager_and_solution
        env.rapidapi_wrapper_name_mapping = rapidapi_wrapper_name_mapping

        if all_decompo_mcp_yaml is all_mcp_yaml:
            decompo_tool_manager = tool_manager
        else:
            decompo_tool_manager_and_solution = get_tool_manager_and_solution(env, all_decompo_mcp_yaml, data_dict)
            if decompo_tool_manager_and_solution is None:
                logging.warning("Decomposition YAML does not cover all relevant APIs for query_id=%s. Falling back to main tool manager.", query_id)
                decompo_tool_manager = tool_manager
            else:
                decompo_tool_manager, _, _ = decompo_tool_manager_and_solution

        data_dict["solution"] = solution
        data_list.append((data_dict, tool_manager, decompo_tool_manager, env))

    return data_list

def create_output_directory(args, queries_path, output_suffix=''):
    """
    Create output directory following the format:
    /base_path/dataset_name/model_name/mcp_tool_decompo_tool/timestamp/
    """
    # Extract dataset name from queries_path (remove .json extension)
    dataset_filename = os.path.basename(queries_path)
    dataset_name = os.path.splitext(dataset_filename)[0]  # Remove .json extension
    
    # Extract tool names from MCP YAML paths
    # e.g., "tmdb_d1_1007_mcp.yaml" -> "tmdb_d1_1007_mcp"
    if isinstance(args.mcp_yaml_path, list):
        mcp_yaml_path = args.mcp_yaml_path[0]
    else:
        mcp_yaml_path = args.mcp_yaml_path
    mcp_tool_name = os.path.basename(mcp_yaml_path).replace('.yaml', '').replace('.yml', '')
    if isinstance(args.decompo_mcp_yaml_path, list):
        decompo_mcp_yaml_path = args.decompo_mcp_yaml_path[0]
    else:
        decompo_mcp_yaml_path = args.decompo_mcp_yaml_path
    decompo_tool_name = os.path.basename(decompo_mcp_yaml_path).replace('.yaml', '').replace('.yml', '')
    
    # Combine mcp tool name with decompo tool name
    # If they're the same, only use one; otherwise combine both
    if mcp_tool_name == decompo_tool_name:
        description_name = mcp_tool_name
    else:
        description_name = f"{mcp_tool_name}_{decompo_tool_name}"
    
    # Generate timestamp with optional suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_suffix:
        timestamp = f"{timestamp}{output_suffix}"
    
    # Build the output directory path
    output_dir = os.path.join(
        ROOT_DIR,
        "tmdb_results",
        dataset_name,
        args.model_name,
        description_name,
        timestamp
    )
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory created: {output_dir}")
    
    return output_dir


def save_queries_snapshot(output_dir, queries_path, requested_max_queries, queries, include_queries, sampling_meta=None):
    """
    Save metadata (and optionally the used queries) to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    meta = {
        "queries_path": queries_path,
        "requested_max_queries": requested_max_queries,
        "actual_queries_used": len(queries),
    }
    if sampling_meta:
        meta.update(sampling_meta)
    meta_path = os.path.join(output_dir, "queries_used_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if include_queries:
        queries_path_out = os.path.join(output_dir, "queries_used.json")
        with open(queries_path_out, "w") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run TMDB step-wise evaluation")

    # First, parse just the config file argument
    parser.add_argument("--config", type=str, 
                        default=f"{ROOT_DIR}/tmdb/configs/tmdb_d0.yaml",
                       help="Config file path")
    parser.add_argument("--debug", action="store_true", default=False,
                       help="Run in debug mode (limited queries for quick testing)")
    parser.add_argument("--full", action="store_true", default=False,
                       help="Run full evaluation (overrides debug)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset to evaluate (overrides config)")
    parser.add_argument("--tool_root_dir", type=str, default="StableToolBench/toolenv2404_filtered/",
                       help="Tool root directory for StableToolBench. Will use it to call APIs.")

    parser.add_argument("--mcp_yaml_path", type=str, nargs="+", default=None,
                       help="MCP YAML path (overrides config). Can be a list of paths, the very first one is with the highest priority, overriding the others.")
    parser.add_argument("--decompo_mcp_yaml_path", type=str, nargs="+", default=None,
                       help="Decomposition MCP YAML path (overrides config). Can be a list of paths, the very first one is with the highest priority.")
    parser.add_argument("--param_yaml_path", type=str, nargs="+", default=None,
                       help="Parameter YAML path (overrides config). Can be a list of paths, the very first one is with the highest priority, overriding the others.")
    parser.add_argument("--load_mcp_old_version", action="store_true", default=False,
                       help="Use old MCP YAML loading behavior (overwrite tools instead of merging).")

    parser.add_argument("--seed", type=int, default=None,
                       help="Seed for random number generator (overrides config)")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Temperature for task decomposition (overrides config)")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p for task decomposition (overrides config)")
    parser.add_argument("--max_tokens", type=int, default=None,
                       help="Max tokens for task decomposition (overrides config)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name for LLM, e.g. 'gpt-4.1-2025-04-14' or 'vllm:Qwen/Qwen2.5-7B-Instruct' (overrides config)")
    parser.add_argument("--openai_api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")

    parser.add_argument("--max_queries", type=int, default=None,
                       help="Max queries to evaluate (overrides config)")
    parser.add_argument("--sample_queries", action="store_true", default=False,
                       help="Randomly sample max_queries queries instead of taking the first K")
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

    parser.add_argument(
        "--desc_in_task_decomp",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=None,
        help="Whether to include API descriptions in task decomposition prompt (overrides config)")

    # parser.add_argument(
    #             "--context_source",
    #             type=str,
    #             choices=["golden", "selected"],
    #             default=None,
    #             help="Context source for dependent subtasks: golden or selected"
    #         )

    parser.add_argument("--consider_provider_in_exact_match", action="store_true", default=False,
                       help="Consider provider in exact match accuracy calculation")

    # Mixed-category expansion via precomputed cache
    parser.add_argument("--candidate_api_cache", type=str, default=None,
                       help="Path to precomputed candidate API cache JSON "
                            "(generated by precompute_candidate_apis.py). "
                            "When provided, enables mixed-category expansion "
                            "and takes precedence over --expand_same_category.")
    parser.add_argument("--extra_api_count", type=int, default=20,
                       help="N: total number of extra distractor APIs to add (default: 20)")
    parser.add_argument("--same_category_ratio", type=float, nargs="+", default=[0.4],
                       help="A: per-bucket same-category BM25 ratio(s). Provide one value per "
                            "bucket (e.g., 0.05 0.04 0.03 0.033) or a single value to broadcast "
                            "to all buckets. (default: 0.4)")
    parser.add_argument("--cross_category_ratio", type=float, nargs="+", default=[0.3],
                       help="B: per-bucket cross-category BM25 ratio(s). Provide one value per "
                            "bucket (e.g., 0.05 0.06 0.07 0.067) or a single value to broadcast "
                            "to all buckets. (default: 0.3)")
    parser.add_argument("--bucket_targets", type=str, default=None,
                       help="Comma-separated cumulative bucket targets for v1_bucket cache "
                            "(e.g., '50,75,100,120'). Each value is the cumulative extra API "
                            "count after that bucket. Required when using v1_bucket cache format.")


    args = parser.parse_args()

    # Set OpenAI API key for utils.llm
    if args.openai_api_key:
        from tool_exec_tracer.utils.llm import set_openai_api_key
        set_openai_api_key(args.openai_api_key)

    reproducibility_log = build_reproducibility_log(args)

    # Load YAML config to get defaults
    print(f"📄 Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Apply YAML defaults, then override with CLI args if provided
    # Dataset
    if args.dataset is None:
        args.dataset = "tmdb"  # fallback default
    
    # MCP paths - use config's mcp_yaml_path if CLI not provided
    if args.mcp_yaml_path is None:
        args.mcp_yaml_path = config_data.get('agent', {}).get('params', {}).get('mcp_yaml_path', 
                                f"{ROOT_DIR}/tmdb/desc_mcp_yaml/tmdb_d0_mcp.yaml")
    
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
        args.model_name = llm_config.get('model', 'gpt-4.1-2025-04-14')
    
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

    # Simple API instruction mode
    args.simple_api_instruction = config_data.get('agent', {}).get('params', {}).get('simple_api_instruction', True)
    if args.desc_in_task_decomp is None:
        args.desc_in_task_decomp = config_data.get('agent', {}).get('params', {}).get('desc_in_task_decomp', True)

    # if args.context_source is None:
    #     args.context_source = config_data.get("agent", {}).get("params", {}).get("context_source", "golden")

    print(f"✅ Configuration loaded:")
    print(f"   Simple API Instruction: {args.simple_api_instruction}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Model: {args.model_name}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Seed: {args.seed}")
    print(f"   Max queries: {args.max_queries}")
    print(f"   Runs per scenario: {args.runs_per_scenario}")
    print(f"   Semantic matching mode: {args.semantic_matching_mode}")
    print(f"   Parallel workers: {args.workers}")
    print(f"   MCP YAML path: {args.mcp_yaml_path}")
    print(f"   Task decomposition prompt: {args.task_decomp_prompt_version}")
    print(f"   Parameter generation prompt: {args.param_gen_prompt_version}")
    print(f"   API selection prompt: api_selection_single")
    print(f"   Use API descriptions in task decomposition: {args.desc_in_task_decomp}")


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
        print("\n🚀 QUICK TEST MODE enabled for Step-wise Evaluation")
        max_queries = 2
        runs_per_scenario = 1
        output_suffix = '_step_wise_quick_test'
    else:
        max_queries = args.max_queries
        runs_per_scenario = args.runs_per_scenario
        output_suffix = '_step_wise_eval'

    print(f"\n🎯 Starting STEP-WISE EVALUATION:")
    print(f"   📊 Max queries: {max_queries}")
    print(f"   🔄 Runs per scenario: {runs_per_scenario}")
    print(f"   🚀 Mode: Step-wise evaluation with golden API log entries and API selection accuracy")
    print(f"   📋 Prompt Collection: ENABLED (saving queries, contexts, and selected APIs to prompt.json)")

    # load tmdb or spotify queries
    if args.dataset == "tmdb":
        queries_path = f"{ROOT_DIR}/DRAFT/tmdb.json"
    elif args.dataset == "tmdb_0802_syn":
        queries_path = f"{ROOT_DIR}/DRAFT/tmdb_0802_syn.json"
    elif args.dataset == "tmdb_0709_syn":
        queries_path = f"{ROOT_DIR}/DRAFT/tmdb_0709_syn.json"
    elif args.dataset == "spotify":
        queries_path = f"{ROOT_DIR}/DRAFT/spotify.json"
    elif args.dataset == "spotify_1007_syn":
        queries_path = f"{ROOT_DIR}/DRAFT/spotify_1007_syn.json"
    elif args.dataset == "spotify_1021_syn":
        queries_path = f"{ROOT_DIR}/DRAFT/spotify_1021_syn.json"
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

    # Create output directory with hierarchical structure
    if args.output_dir:
        # Use custom output directory if provided
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Using custom output directory: {output_dir}")
    else:
        # Use default hierarchical structure
        output_dir = create_output_directory(args, queries_path, output_suffix)
    
    if queries_path is None:
        queries_path = args.dataset
        args_StableToolBench = argparse.Namespace(
            tool_root_dir=args.tool_root_dir,
            method="",
            input_query_file=queries_path,
            output_answer_file=output_dir,
            backbone_model="",
            toolbench_key=os.environ["TOOLBENCH_KEY"],
            rapidapi_key="",
            use_rapidapi_key=False,
            api_customization="",
            max_observation_length=999999,
            observ_compress_method="",
            retrieved_api_nums=10,
            expand_same_category=args.expand_same_category,
            BM25_threshold=args.BM25_threshold,
            retrieval_sample_size=args.retrieval_sample_size,
            seed=args.seed,
            
            candidate_api_cache=args.candidate_api_cache, # scaling experiments
            extra_api_count=args.extra_api_count,
            same_category_ratio=args.same_category_ratio,
            cross_category_ratio=args.cross_category_ratio,
            bucket_targets=[int(x) for x in args.bucket_targets.split(",")] if args.bucket_targets else None,
        )



        queries = prepare_data_from_StableToolBench(args_StableToolBench, 
        mcp_tool_path=args.mcp_yaml_path, decompo_mcp_tool_path=args.decompo_mcp_yaml_path, param_yaml_paths=args.param_yaml_path, load_mcp_old_version=args.load_mcp_old_version) # list[tuple[dict, tuple[query, tool_manager, decompo_tool_manager, env]]]
    else:
        queries = load_queries(queries_path)
        # load tools from mcp config
        tools = load_tools(args.mcp_yaml_path)
        decompo_tools = load_tools(args.decompo_mcp_yaml_path)

        tool_manager = ToolManager(tools, platform=platform)
        decompo_tool_manager = ToolManager(decompo_tools, platform=platform)
        queries = [(query, tool_manager, decompo_tool_manager, None) for query in queries]
    
    # queries is a list of tuples: (query, tool_manager, decompo_tool_manager, None (for RestBench) or rapidapi_wrapper (for StableToolBench))
    # Limit queries based on max_queries (use all if fewer than requested)
    sampling_meta = None
    if max_queries is None or len(queries) <= max_queries:
        queries = queries
    else:
        if args.sample_queries:
            rng = random.Random(args.seed)
            queries = rng.sample(queries, k=max_queries)
            sampling_meta = {
                "sampling": "random",
                "sampling_seed": args.seed,
            }
        else:
            queries = queries[:max_queries]
            sampling_meta = {
                "sampling": "first_k",
            }
    # Save query snapshot/metadata (only save full queries for JSON input datasets)
    include_queries = queries_path is not None and os.path.isfile(queries_path)
    save_queries_snapshot(
        output_dir,
        queries_path,
        max_queries,
        [q[0] for q in queries] if include_queries else queries,
        include_queries,
        sampling_meta,
    )
    print(f"📊 Processing {len(queries)} queries (max_queries={max_queries})")

    with open(os.path.join(output_dir, "reproducibility_log.txt"), "w") as f:
        f.write(reproducibility_log)

    # Initialize MCP call logger
    mcp_logger = initialize_global_logger(output_dir=output_dir, enabled=True)
    print(f"📝 MCP call logging enabled - logs will be saved to: {output_dir}")
    
    # Initialize LLM call logger for parameter generation
    llm_logger = initialize_global_llm_logger(output_dir=output_dir, enabled=True)
    print(f"📝 LLM parameter generation logging enabled - logs will be saved to: {output_dir}")

    prompts_dir = f"{ROOT_DIR}/tmdb/prompts"
    prompt_loader = PromptLoader(prompts_dir)

    # Pre-allocate list to preserve query order in parallel processing
    query_results = [None] * len(queries)
    
    overall_statistics = {
        "total_scenarios": 0,
        "total_runs_all_scenarios": 0,
        "total_api_requiring_runs": 0,  # NEW: Runs that require API calls (excludes processing steps)
        "total_correct_exact_matches": 0,
        "overall_exact_match_accuracy": [],
        # Query-level perfect execution tracking
        "query_perfect_api_selection": [],
        "query_perfect_parameter": [],
        "query_perfect_api_success": [],
        "query_perfect_sel_acc_and_api_success": [],
        "query_perfect_sel_acc_and_param_valid": [],
        "agent_query_accuracy": [],
        "agent_query_recall": [],
        "agent_query_accuracy_and_select": [],
        "agent_query_recall_and_select": [],
        "agent_query_accuracy_and_select_and_success": [],
        "agent_query_recall_and_select_and_success": [],
        
        # Subtask-level sel_acc_and_param_valid tracking
        "sel_acc_and_param_valid": [],
        "sel_acc_and_api_success": []
        }
    
    # Thread-safe lock for updating statistics and results
    stats_lock = threading.Lock()

    # Define output file paths
    results_file = os.path.join(output_dir, "step_wise_eval_results.json")
    statistics_file = os.path.join(output_dir, "step_wise_eval_overall_statistics.json")
    run_params_file = os.path.join(output_dir, "run_parameters.json")
    success_file = os.path.join(output_dir, "success")
    
    # ========================================
    # SAVE METADATA FILES AT START (not at end)
    # This ensures we have config/params even if evaluation is interrupted
    # ========================================
    
    # Save run parameters (save immediately at start)
    # Note: We save run_parameters.json instead of copying config.yaml because
    # run_parameters.json contains the ACTUAL parameters used (including CLI overrides),
    # while config.yaml would only show the base config without overrides.
    # Paths are stored as absolute so downstream stages (e.g. tool_desc_atomic/main.py)
    # can read run_parameters.json from any cwd without breaking.
    def _abs(p):
        if p is None:
            return None
        if isinstance(p, (list, tuple)):
            return [os.path.abspath(x) if x is not None else None for x in p]
        return os.path.abspath(p)

    run_params = {
        "config": _abs(args.config),
        "dataset": _abs(args.dataset),
        "mcp_yaml_path": _abs(args.mcp_yaml_path),
        "decompo_mcp_yaml_path": _abs(args.decompo_mcp_yaml_path),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "model_name": args.model_name,
        "max_queries": max_queries,
        "runs_per_scenario": runs_per_scenario,
        "workers": args.workers,
        "desc_in_task_decomp": args.desc_in_task_decomp,
        "debug_mode": debug,
        "queries_path": _abs(queries_path),
        "prompts": {
            "task_decomposition": args.task_decomp_prompt_version,
            "parameter_generation": args.param_gen_prompt_version,
            "api_selection": "api_selection_single"
        }
    }
    with open(run_params_file, "w") as f:
        json.dump(run_params, f, indent=2, ensure_ascii=False)
    print(f"💾 Run parameters saved to: {run_params_file}")

    # ========================================
    # HELPER FUNCTION FOR PROCESSING SINGLE QUERY WITH INDEX
    # ========================================
    
    def process_query_wrapper(idx, query_tuple):
        """Wrapper function to process a single query with thread-safe statistics updates."""
        query, tool_manager, decompo_tool_manager, rapidapi_wrapper = query_tuple
        query_id = query.get("query_id", f"query_{idx}")
        query_result = process_single_query(query, args, tool_manager, decompo_tool_manager, prompt_loader, query_id, rapidapi_wrapper)
        
        # Thread-safe statistics update
        with stats_lock:
            # Use idx-1 because enumerate starts at 1 but list indices start at 0
            query_results[idx - 1] = query_result

            overall_statistics["total_scenarios"] += query_result["overall_statistics"]["total_scenarios"]
            overall_statistics["total_runs_all_scenarios"] += query_result["overall_statistics"]["total_runs_all_scenarios"]
            overall_statistics["total_correct_exact_matches"] += query_result["overall_statistics"]["total_correct_exact_matches"]

            overall_statistics["overall_exact_match_accuracy"].extend(query_result["overall_statistics"]["overall_exact_match_accuracy"]["raw_data"])
            
            # Aggregate query-level perfect execution metrics
            overall_statistics["query_perfect_api_selection"].extend(
                query_result["overall_statistics"].get("query_perfect_api_selection_raw", []))
            overall_statistics["query_perfect_parameter"].extend(
                query_result["overall_statistics"].get("query_perfect_parameter_raw", []))
            overall_statistics["query_perfect_api_success"].extend(
                query_result["overall_statistics"].get("query_perfect_api_success_raw", []))
            overall_statistics["query_perfect_sel_acc_and_param_valid"].extend(
                query_result["overall_statistics"].get("query_perfect_sel_acc_and_param_valid_raw", []))
            overall_statistics["query_perfect_sel_acc_and_api_success"].extend(
                query_result["overall_statistics"].get("query_perfect_sel_acc_and_api_success_raw", []))
            overall_statistics["agent_query_accuracy"].append(
                query_result["overall_statistics"].get("agent_query_accuracy", 0.0))
            overall_statistics["agent_query_recall"].append(
                query_result["overall_statistics"].get("agent_query_recall", 0.0))
            overall_statistics["agent_query_accuracy_and_select"].append(
                query_result["overall_statistics"].get("agent_query_accuracy_and_select", 0.0))
            overall_statistics["agent_query_recall_and_select"].append(
                query_result["overall_statistics"].get("agent_query_recall_and_select", 0.0))
            overall_statistics["agent_query_accuracy_and_select_and_success"].append(
                query_result["overall_statistics"].get("agent_query_accuracy_and_select_and_success", 0.0))
            overall_statistics["agent_query_recall_and_select_and_success"].append(
                query_result["overall_statistics"].get("agent_query_recall_and_select_and_success", 0.0))
            
            # Aggregate subtask-level metrics
            overall_statistics["sel_acc_and_param_valid"].extend(
                query_result["overall_statistics"].get("sel_acc_and_param_valid_raw", []))
            overall_statistics["sel_acc_and_api_success"].extend(
                query_result["overall_statistics"].get("sel_acc_and_api_success_raw", []))
            
            # Calculate total API-requiring runs (length of exact_match_accuracy array)
            overall_statistics["total_api_requiring_runs"] = len(overall_statistics["overall_exact_match_accuracy"])
            
            overall_statistics["overall_exact_match_accuracy_mean"] = np.mean(overall_statistics["overall_exact_match_accuracy"])
            overall_statistics["overall_exact_match_accuracy_std"] = np.std(overall_statistics["overall_exact_match_accuracy"])
            
            # Calculate query-level perfect execution rates
            if overall_statistics["query_perfect_api_selection"]:
                overall_statistics["query_perfect_api_selection_rate"] = np.mean(overall_statistics["query_perfect_api_selection"])
                overall_statistics["query_perfect_parameter_rate"] = np.mean(overall_statistics["query_perfect_parameter"])
                overall_statistics["query_perfect_api_success_rate"] = np.mean(overall_statistics["query_perfect_api_success"])
                overall_statistics["query_perfect_sel_acc_and_param_valid_rate"] = np.mean(overall_statistics["query_perfect_sel_acc_and_param_valid"])
                overall_statistics["query_perfect_sel_acc_and_api_success_rate"] = np.mean(overall_statistics["query_perfect_sel_acc_and_api_success"])
            if overall_statistics["agent_query_accuracy"]:
                overall_statistics["agent_query_accuracy_mean"] = np.mean(overall_statistics["agent_query_accuracy"])
                overall_statistics["agent_query_accuracy_std"] = np.std(overall_statistics["agent_query_accuracy"])
                overall_statistics["agent_query_recall_mean"] = np.mean(overall_statistics["agent_query_recall"])
                overall_statistics["agent_query_recall_std"] = np.std(overall_statistics["agent_query_recall"])
                overall_statistics["agent_query_accuracy_and_select_mean"] = np.mean(overall_statistics["agent_query_accuracy_and_select"])
                overall_statistics["agent_query_accuracy_and_select_std"] = np.std(overall_statistics["agent_query_accuracy_and_select"])
                overall_statistics["agent_query_recall_and_select_mean"] = np.mean(overall_statistics["agent_query_recall_and_select"])
                overall_statistics["agent_query_recall_and_select_std"] = np.std(overall_statistics["agent_query_recall_and_select"])
                overall_statistics["agent_query_accuracy_and_select_and_success_mean"] = np.mean(overall_statistics["agent_query_accuracy_and_select_and_success"])
                overall_statistics["agent_query_accuracy_and_select_and_success_std"] = np.std(overall_statistics["agent_query_accuracy_and_select_and_success"])
                overall_statistics["agent_query_recall_and_select_and_success_mean"] = np.mean(overall_statistics["agent_query_recall_and_select_and_success"])
                overall_statistics["agent_query_recall_and_select_and_success_std"] = np.std(overall_statistics["agent_query_recall_and_select_and_success"])
            
            # Calculate subtask-level sel_acc_and_param_valid mean/std
            if overall_statistics["sel_acc_and_param_valid"]:
                overall_statistics["sel_acc_and_param_valid_mean"] = np.mean(overall_statistics["sel_acc_and_param_valid"])
                overall_statistics["sel_acc_and_param_valid_std"] = np.std(overall_statistics["sel_acc_and_param_valid"])
            
            # Calculate subtask-level sel_acc_and_api_success mean/std
            if overall_statistics["sel_acc_and_api_success"]:
                overall_statistics["sel_acc_and_api_success_mean"] = np.mean(overall_statistics["sel_acc_and_api_success"])
                overall_statistics["sel_acc_and_api_success_std"] = np.std(overall_statistics["sel_acc_and_api_success"])

            # Save results incrementally after each query (filter out None for failed queries)
            with open(results_file, "w") as f:
                json.dump([r for r in query_results if r is not None], f, indent=2, ensure_ascii=False)
            
            with open(statistics_file, "w") as f:
                json.dump(overall_statistics, f, indent=2, ensure_ascii=False)
        
        return idx
    
    # ========================================
    # MAIN EVALUATION LOOP WITH TRY/FINALLY
    # Ensures summaries are saved even if interrupted
    # ========================================
    
    try:
        if args.workers > 1:
            # Parallel processing with ThreadPoolExecutor
            print(f"🚀 Using {args.workers} parallel workers for query processing")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all queries for processing
                future_to_idx = {
                    executor.submit(process_query_wrapper, idx, query): idx
                    for idx, query in enumerate(queries, 1)
                }
                
                # Process completed futures with progress bar
                with tqdm(total=len(queries), desc="Processing queries", unit="query") as pbar:
                    for future in as_completed(future_to_idx):
                        # try:
                        idx = future.result()
                        pbar.update(1)
                        pbar.set_postfix({"completed": idx})
                        # except Exception as e:
                        #     query_idx = future_to_idx[future]
                        #     print(f"\n⚠️  Error processing query {query_idx}: {e}")
                        #     pbar.update(1)
        else:
            # Sequential processing (original behavior)
            print(f"🔄 Using sequential processing (1 worker)")
            for idx, query in enumerate(tqdm(queries, desc="Processing queries", unit="query"), 1):
                # try:
                process_query_wrapper(idx, query)
                # except Exception as e:
                #     print(f"\n⚠️  Error processing query {idx}: {e}")

        # Final save with completion message (filter out None for failed queries)
        with open(results_file, "w") as f:
            json.dump([r for r in query_results if r is not None], f, indent=2, ensure_ascii=False)
        print(f"✅ Final results saved to: {results_file}")

        # touch a file called 'success' in the output directory
        with open(success_file, "w") as f:
            f.write("success")
    
    finally:
        # ========================================
        # ALWAYS SAVE SUMMARIES (even if interrupted)
        # This ensures we don't lose MCP/LLM logs
        # ========================================
        
        print("\n💾 Saving final summaries...")
        
        # Add MCP call statistics to overall statistics
        mcp_logger = get_global_logger()
        if mcp_logger:
            mcp_stats = mcp_logger.get_stats()
            overall_statistics["mcp_call_statistics"] = {
                "total_mcp_calls": mcp_stats["total_calls"],
                "successful_mcp_calls": mcp_stats["successful_calls"],
                "failed_mcp_calls": mcp_stats["failed_calls"],
                "mcp_success_rate": mcp_stats["success_rate"],
                "mcp_error_analysis": mcp_stats["error_analysis"]
            }
        
        # Save updated statistics with MCP data
        with open(statistics_file, "w") as f:
            json.dump(overall_statistics, f, indent=2, ensure_ascii=False)
        print(f"✅ Final statistics saved to: {statistics_file}")
        
        # Save MCP call summary (this creates mcp_call_summary.json)
        if mcp_logger:
            mcp_logger.save_summary()
            print(f"📊 MCP Call Statistics:")
            print(f"   Total Calls: {mcp_stats['total_calls']}")
            print(f"   Successful: {mcp_stats['successful_calls']}")
            print(f"   Failed: {mcp_stats['failed_calls']}")
            print(f"   Success Rate: {mcp_stats['success_rate']:.2%}")
        
        # Save LLM parameter generation history (this creates llm_parameter_generation_summary.json)
        llm_logger = get_global_llm_logger()
        llm_stats = None
        if llm_logger:
            llm_logger.save_summary()
            llm_stats = llm_logger.get_stats()
        
        # Save unified evaluation statistics (combines MCP + LLM stats + step-wise eval stats)
        if mcp_logger and llm_logger:
            eval_stats = EvaluationStatistics(output_dir=output_dir)
            eval_stats.save_statistics(mcp_stats, llm_stats, overall_statistics)
            
            print(f"📊 Unified Evaluation Statistics:")
            print(f"   Total API Calls: {mcp_stats['total_calls']}")
            print(f"   ")
            print(f"   API Execution:")
            print(f"      Successful API Calls: {mcp_stats['successful_calls']}")
            print(f"      Failed API Calls: {mcp_stats['failed_calls']}")
            print(f"      API Success Rate: {mcp_stats['success_rate']:.2%}")
            print(f"   ")
            print(f"   Parameter Validation:")
            print(f"      Valid Parameters: {llm_stats['valid_parameters']}")
            print(f"      Invalid Parameters: {llm_stats['invalid_parameters']}")
            print(f"      Parameter Validation Rate: {llm_stats['parameter_validation_rate']:.2%}")
            print(f"   ")
            print(f"   Step-wise Evaluation:")
            print(f"      Total Scenarios: {overall_statistics.get('total_scenarios', 0)}")
            print(f"      Total API-Requiring Runs: {overall_statistics.get('total_api_requiring_runs', 0)}")
            print(f"      Exact Match Accuracy: {overall_statistics.get('overall_exact_match_accuracy_mean', 0.0):.2%} ± {overall_statistics.get('overall_exact_match_accuracy_std', 0.0):.4f}")
            print(f"   ")
            print(f"   Query-Level Perfect Execution:")
            print(f"      Perfect API Selection (all correct): {overall_statistics.get('query_perfect_api_selection_rate', 0.0):.2%}")
            print(f"      Perfect Parameters (all valid): {overall_statistics.get('query_perfect_parameter_rate', 0.0):.2%}")
            print(f"      Perfect API Success (all succeed): {overall_statistics.get('query_perfect_api_success_rate', 0.0):.2%}")
            print(f"      Perfect Sel Acc & Param Valid: {overall_statistics.get('query_perfect_sel_acc_and_param_valid_rate', 0.0):.2%}")
            print(f"      Perfect Sel Acc & API Success: {overall_statistics.get('query_perfect_sel_acc_and_api_success_rate', 0.0):.2%}")
            print(f"      Agent Query Accuracy: {overall_statistics.get('agent_query_accuracy_mean', 0.0):.2%} ± {overall_statistics.get('agent_query_accuracy_std', 0.0):.4f}")
            print(f"   ")
            print(f"   Subtask-Level Metrics:")
            print(f"      Sel Acc & Param Valid: {overall_statistics.get('sel_acc_and_param_valid_mean', 0.0):.2%} ± {overall_statistics.get('sel_acc_and_param_valid_std', 0.0):.4f}")
            print(f"      Sel Acc & API Success: {overall_statistics.get('sel_acc_and_api_success_mean', 0.0):.2%} ± {overall_statistics.get('sel_acc_and_api_success_std', 0.0):.4f}")
            if llm_stats.get('error_categories'):
                print(f"   ")
                print(f"   Error Categories:")
                for cat, count in sorted(llm_stats['error_categories'].items(), key=lambda x: -x[1]):
                    print(f"      {cat}: {count}")
            
            # Print error analysis if available
            if mcp_logger:
                error_analysis = mcp_stats.get('error_analysis', {})
                if error_analysis.get('total_errors', 0) > 0:
                    print(f"\n📊 Error Analysis:")
                    print(f"   Total Errors: {error_analysis['total_errors']}")
                    
                    if error_analysis.get('error_types'):
                        print(f"\n   Error Types:")
                        for error_type, count in sorted(error_analysis['error_types'].items(), key=lambda x: x[1], reverse=True):
                            percentage = (count / error_analysis['total_errors']) * 100
                            print(f"      {error_type}: {count} ({percentage:.1f}%)")
                    
                    if error_analysis.get('http_status_codes'):
                        print(f"\n   HTTP Status Codes:")
                        for status, count in sorted(error_analysis['http_status_codes'].items(), key=lambda x: x[1], reverse=True):
                            percentage = (count / error_analysis['total_errors']) * 100
                            print(f"      {status}: {count} ({percentage:.1f}%)")
                    
                    if error_analysis.get('errors_by_api'):
                        print(f"\n   Top APIs with Errors:")
                        top_apis = sorted(error_analysis['errors_by_api'].items(), key=lambda x: x[1], reverse=True)[:5]
                        for api_name, count in top_apis:
                            print(f"      {api_name}: {count} errors")

def process_single_query(query, args, tool_manager, decompo_tool_manager, prompt_loader, query_id=None, rapidapi_wrapper=None):

    # Extract base platform name from dataset (e.g., "spotify_1007_syn" -> "spotify")
    if args.dataset.startswith("spotify"):
        platform = "spotify"
    elif args.dataset.startswith("tmdb"):
        platform = "tmdb"
    else:
        platform = "StableToolBench"

    task_decomp_kwargs = {
        "model_name": args.model_name,
        "prompt_version": args.task_decomp_prompt_version,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "ROOT_DIR": ROOT_DIR,
        "seed": args.seed,
    }
    # Backward compatible with older task_decompose signatures.
    if "desc_in_task_decomp" in inspect.signature(task_decompose).parameters:
        task_decomp_kwargs["desc_in_task_decomp"] = args.desc_in_task_decomp

    subtasks = task_decompose(query, decompo_tool_manager, **task_decomp_kwargs)

    print(subtasks)

    task_ls = []
    for t in range(len(subtasks.get('Tasks', []))):
        task_ls.append({"task": subtasks['Tasks'][t], "id": t + 1})

    task_topology_result = task_topology(
        query, task_ls, model_name=args.model_name, 
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)

    print(task_topology_result)

    solution = query["solution"]

    # golden_apis = extract_golden_apis(decompo_tool_manager, solution)

    golden_apis : List[List[str]] = solution
    print(f"🎯 Ground truth APIs: {golden_apis}")

    # subtask indices that require API calls + semantic mapping
    # Note: Mode is configured in the YAML config file (semantic_matching_mode parameter)
    api_requiring_indices, processing_indices, subtask_to_api = identify_api_requiring_subtasks(
        task_topology_result, 
        golden_apis,
        tool_manager=decompo_tool_manager,
        query_text=query.get("query", ""),
        mode=args.semantic_matching_mode,  # Read from config file
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed
    )
    
    print(f"📋 Subtask to API mapping: {subtask_to_api}")
    
    subtasks = convert_task_topology_to_subtasks(
        task_topology_result, 
        subtask_to_api  # Pass the mapping dict instead of api_requiring_indices + golden_apis
    )

    converted_query_data = query.copy()
    converted_query_data["subtasks"] = subtasks

    # Use the new step-wise evaluation method
    query_result = evaluate_query_with_step_wise_eval(
        tool_manager=tool_manager,
        query_data=converted_query_data,
        golden_apis=golden_apis,
        api_requiring_indices=api_requiring_indices,
        processing_indices=processing_indices,
        subtask_to_api=subtask_to_api,  # Pass the semantic mapping
        runs_per_scenario=args.runs_per_scenario,
        seed=args.seed,
        prompt_loader=prompt_loader,
        query_id=query_id,
        dataset=platform,
        parameter_generation_prompt_version=args.param_gen_prompt_version,
        simple_api_instruction=args.simple_api_instruction,
        rapidapi_wrapper=rapidapi_wrapper,
        model_name=args.model_name,
        consider_provider_in_exact_match=args.consider_provider_in_exact_match
    )

    return query_result


def convert_task_topology_to_subtasks(
    task_topology_result: List[Dict], 
    subtask_to_api: Dict[int, str]) -> List[Dict]:
    """
    Convert task topology result to subtasks format using semantic mapping.
    
    Args:
        task_topology_result: Task topology result from LLM decomposition
        subtask_to_api: Dictionary mapping subtask_id → golden_api_name
                        (from semantic or heuristic matching)
    Returns:
        List of subtasks with golden API assignments
    """
    subtasks = []

    for task in task_topology_result:
        task_id = task["id"]
        task_description = task["task"]

        if "depend" in task:
            task_dependencies = task["depend"]
        else:
            print(f"Task {task_id} → No dependencies")
            print(f"{task}")
            task_dependencies = []

        # Look up golden API from semantic mapping
        golden_api = subtask_to_api.get(task_id, "")
        
        if golden_api:
            print(f"🎯 Task {task_id} → Golden API: {golden_api}")
        else:
            print(f"⚙️  Task {task_id} → Processing step (no API)")

        dependencies = [dep for dep in task_dependencies if dep != -1]
        subtask = {
            "input": task_description,
            "dependencies": dependencies,
            "golden_api": golden_api
        }
        subtasks.append(subtask)

    return subtasks


def load_tools(mcp_config_path):
    with open(mcp_config_path, "r") as f:
        mcp_config = yaml.safe_load(f)
        servers = mcp_config["mcp_servers"]
        server_name = list(servers.keys())[0] # only one server is supported
        server_config = servers[server_name]
        
        command = server_config["command"]
        tools = server_config["tools"]

        # tool_name = mcp_config_path.split("/")[-1].split(".")[0]

    return tools

def load_queries(queries_path):
    with open(queries_path, "r") as f:
        data = json.load(f)
    
    # Handle both formats:
    # 1. Direct list: [query1, query2, ...]
    # 2. Dict with generated_questions: {"generation_info": {...}, "generated_questions": [...]}
    if isinstance(data, list):
        queries = data
    elif isinstance(data, dict) and 'generated_questions' in data:
        queries = data['generated_questions']
    else:
        raise ValueError(f"Invalid queries file format in {queries_path}")
    
    # Convert synthetic format to standard format if needed
    # Synthetic format has: {generated_question, api_sequence, chain_info, ...}
    # Standard format has: {query, solution, ...}
    converted_queries = []
    for q in queries:
        if 'query' in q and 'solution' in q:
            # Already in standard format
            converted_queries.append(q)
        elif 'generated_question' in q and 'api_sequence' in q:
            # Synthetic format (tmdb_0709_syn) - single question per item
            converted_q = {
                'query': q['generated_question'],
                'solution': [{'API': api_name} for api_name in q['api_sequence']]
            }
            # Preserve other fields if needed
            if 'chain_info' in q:
                converted_q['chain_info'] = q['chain_info']
            converted_queries.append(converted_q)
        elif 'generated_questions' in q and 'api_sequence' in q:
            # Synthetic format (tmdb_0802_syn) - multiple questions per item
            # Expand each item into multiple query entries (one per question)
            questions_list = q['generated_questions']
            if isinstance(questions_list, list):
                for question_text in questions_list:
                    converted_q = {
                        'query': question_text,
                        'solution': [{'API': api_name} for api_name in q['api_sequence']]
                    }
                    # Preserve other fields if needed
                    if 'chain_info' in q:
                        converted_q['chain_info'] = q['chain_info']
                    converted_queries.append(converted_q)
            else:
                raise ValueError(f"Expected 'generated_questions' to be a list, got {type(questions_list)}")
        else:
            raise ValueError(f"Unknown query format: {list(q.keys())}")
    
    return converted_queries

    # load tools from mcp config
if __name__ == "__main__":
    main()