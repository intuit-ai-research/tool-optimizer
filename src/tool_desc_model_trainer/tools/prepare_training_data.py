#!/usr/bin/env python3
"""
Unified data preparation for tool-level SFT training.

This script handles the complete data preparation pipeline:
1. Load input manifest (baseline YAMLs, ground truth YAMLs, eval dirs)
2. Flatten evaluation results
3. Group queries by tool
4. Prepare training dataset with prompts
5. Convert to train/val parquet files

Usage:
    python prepare_training_data.py \
        --input-manifest inputs.json \
        --output-dir /path/to/experiment/data \
        --config-file config.yaml \
        [additional options]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from tqdm import tqdm

from convert_tmdb_eval_format import convert_new_to_legacy_format

from group_by_tool import load_baseline_descriptions, group_by_tool, create_tool_groups

from data_process_tool_level import process_tool_level_data
from mcp_utils import load_mcp_yaml_files
from jsonl_to_parquet_split import convert_jsonl_to_parquet

from StableToolBench.toolbench.utils import standardize


def run_command(cmd: List[str], description: str) -> None:
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    print(f"   Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"   ✅ {description} complete")


def discover_yaml_files(path: Path) -> List[str]:
    """
    Discover YAML files in a directory or return the path if it's a file.
    
    Args:
        path: Path to a file or directory
    
    Returns:
        List of YAML file paths as strings
    """
    return sorted(list(path.rglob("*.yaml")))

def normalize_path_list(paths: List[str]) -> List[str]:
    """Normalize and sort paths for consistent hashing."""
    normalized = []
    for p in paths:
        path = Path(p).resolve()
        normalized.append(str(path))
    return sorted(normalized)


def compute_data_config_hash(
    eval_dirs: List[str],
    baseline_desc_files: List[str],
    ground_truth_desc_files: List[str],
    min_queries_per_tool: int,
    train_query_ratio: float,
    num_query_examples: int,
    max_samples_per_tool: int,
    ground_truth_strategy: str,
    query_selection_strategy: str
) -> tuple:
    """
    Compute a deterministic hash from data preparation configuration.
    
    Returns:
        (hash_string, config_dict)
    """
    # Normalize paths for consistent hashing
    eval_dirs_normalized = normalize_path_list(eval_dirs)
    baseline_normalized = normalize_path_list(baseline_desc_files)
    ground_truth_normalized = normalize_path_list(ground_truth_desc_files)
    
    # Create config dictionary
    config = {
        "eval_dirs": eval_dirs_normalized,
        "baseline_desc_files": baseline_normalized,
        "ground_truth_desc_files": ground_truth_normalized,
        "min_queries_per_tool": min_queries_per_tool,
        "train_query_ratio": train_query_ratio,
        "num_query_examples": num_query_examples,
        "max_samples_per_tool": max_samples_per_tool,
        "ground_truth_strategy": ground_truth_strategy,
        "query_selection_strategy": query_selection_strategy,
    }
    
    # Compute hash from JSON representation
    config_json = json.dumps(config, sort_keys=True)
    hash_obj = hashlib.sha256(config_json.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:16]  # Use first 16 chars for readability
    
    return hash_hex, config


def check_cached_data(cache_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Check if cached data exists and is valid.
    
    Returns:
        Summary dict if cache is valid, None otherwise
    """
    summary_file = cache_dir / "data_preparation_summary.json"
    if not summary_file.exists():
        return None
    
    try:
        with summary_file.open() as f:
            summary = json.load(f)
        
        # Verify all required files exist
        required_files = [
            summary.get('train_parquet'),
            summary.get('val_parquet'),
            summary.get('tool_grouped_file'),
            summary.get('dataset_file')
        ]
        
        for file_path in required_files:
            if not file_path or not Path(file_path).exists():
                return None
        
        return summary
    except Exception as e:
        print(f"   ⚠️  Failed to load cached data: {e}")
        return None


def auto_detect_tool_desc_file(eval_dirs: List[str]) -> Optional[str]:
    """
    Auto-detect tool description file from run_parameters.json in eval directories.
    
    Returns path to tool description file, or None if not found.
    """
    for eval_dir in eval_dirs:
        run_params_file = Path(eval_dir) / "run_parameters.json"
        if not run_params_file.exists():
            continue
        
        try:
            with run_params_file.open() as f:
                params = json.load(f)
            
            mcp_path = params.get('mcp_tool_path', '')
            if not mcp_path:
                continue
            
            # Convert relative path to absolute
            if not mcp_path.startswith('/'):
                mcp_path = '/home/sagemaker-user/user-default-efs/FunctionWrapper/' + mcp_path
            
            # Check if file exists
            if Path(mcp_path).exists():
                print(f"🔍 Auto-detected tool description file: {mcp_path}")
                return mcp_path
        except Exception as e:
            print(f"   ⚠️  Failed to read {run_params_file}: {e}")
            continue
    
    return None


def discover_eval_result_dirs(parent_dir: Path) -> List[Path]:
    """
    Recursively discover directories containing step_wise_eval_results.json.
    
    Args:
        parent_dir: Parent directory to search in
    
    Returns list of directories containing step_wise_eval_results.json
    """
    result_dirs = []
    
    for item in parent_dir.rglob("step_wise_eval_results.json"):
        if item.is_file():
            result_dirs.append(item.parent)
    
    return sorted(result_dirs)


def flatten_eval_results(
    eval_dirs: List[str],
    output_dir: Path,
    baseline_desc_file: str,
    root_dir: Path
) -> List[Path]:
    """
    Step 1: Convert evaluation results to flattened format.
    
    Args:
        eval_dirs: List of evaluation directories (can be parent dirs or exact dirs)
        output_dir: Output directory for flattened files
        baseline_desc_file: Baseline description file (used for baseline_desc in flattened records)
        root_dir: Root directory of the project
    
    Returns list of flattened JSON files.
    """
    print("\n📊 Step 1: Converting evaluation results to flattened format")
    
    # First, expand eval_dirs by discovering subdirectories with step_wise_eval_results.json
    actual_eval_dirs = []
    for eval_dir in eval_dirs:

        eval_dir = Path(eval_dir)

        actual_eval_dirs = discover_eval_result_dirs(eval_dir)

        for actual_eval_dir in actual_eval_dirs:

            eval_results_file = actual_eval_dir / "step_wise_eval_results.json"
            if eval_results_file.exists():
                print(f"   ✓ Found eval results in: {actual_eval_dir.name}")
            else:
                raise FileNotFoundError(
                        f"No step_wise_eval_results.json found in {eval_dir} or its subdirectories"
                    )
    
    print(f"\n   Processing {len(actual_eval_dirs)} evaluation directories")

    all_records = []    
    
    for i, eval_dir_path in enumerate(tqdm(actual_eval_dirs, desc="Flattening eval results", unit="dir")):
        eval_results_file = eval_dir_path / "step_wise_eval_results.json"

        with open(eval_results_file, 'r') as f:
            eval_results = json.load(f)
        
        with open(eval_dir_path / "run_parameters.json", 'r') as f:
            run_parameters = json.load(f)
        
        print("run_parameters: ", run_parameters)

        legacy_data = convert_new_to_legacy_format(eval_results, run_parameters)

        print(legacy_data['metadata']['total_records'])

        # Extract the actual records from the legacy_data dict and add to flat list
        all_records.extend(legacy_data['records'])

    print(f"   Total flattened records: {len(all_records)}")
    return all_records

def extract_valid_categories(baseline_path: Path) -> Set[str]:
    """
    Extract valid category names from the baseline YAML folder structure.
    
    Categories can have underscores (e.g., 'Video_Images', 'Artificial_Intelligence_Machine_Learning').
    We normalize them to lowercase with underscores.
    
    Args:
        baseline_path: Path to the baseline descriptions directory
    
    Returns:
        Set of valid category names (lowercase with underscores)
    """
    valid_categories = set()
    
    if baseline_path.is_dir():
        for item in baseline_path.iterdir():
            if item.is_dir():
                # Normalize: lowercase and keep underscores
                category_name = item.name.lower()
                valid_categories.add(category_name)
    
    print(f"   📁 Found {len(valid_categories)} valid categories from folder structure")
    return valid_categories


def group_queries_by_tool(
    records: List[Dict[str, Any]],
    args: argparse.Namespace) -> Tuple[Dict, List, Dict]:
    
    """
    Step 2: Group queries by tool with train/held-out split.

    Args:
        records: List of records to group
        args: Arguments namespace

    Returns:
        Tuple of (tools_to_queries, tool_groups, baseline_mcp_tools)
    """
    print("\n🔨 Step 2: Grouping queries by tool")

    output_dir = args.output_dir
    min_queries = args.min_queries_per_tool
    train_ratio = args.train_query_ratio
    root_dir = args.root_dir
    seed = args.seed
    w1 = args.w1
    w2 = args.w2

    # Extract valid categories from baseline folder structure BEFORE grouping
    baseline_desc_files_path = Path(args.baseline_desc_files)
    valid_categories = extract_valid_categories(baseline_desc_files_path)
    
    tool_to_queries = group_by_tool(records, include_selected_api=True, valid_categories=valid_categories)

    baseline_desc_files_path = Path(args.baseline_desc_files)

    if baseline_desc_files_path.is_file():
        baseline_desc_files = [baseline_desc_files_path]
    else:
        baseline_desc_files = discover_yaml_files(baseline_desc_files_path)


    # keyed by (server_name, tool_name) tuple
    # server name is standardized
    baseline_descs, metadata_dict, parameters_dict = load_baseline_descriptions(baseline_desc_files) 
    
    cnt = 0
    for tool_key, tool_group in tool_to_queries.items():
        if tool_key not in baseline_descs:
            cnt += 1
            print("tool_key: ", tool_key)
    
    # print(metadata_dict.keys())
    print(f"   ⚠️  {cnt} tools not found in metadata_dict")
    ratio = cnt / len(tool_to_queries)
    print(f"   ⚠️  {ratio} tools not found in metadata_dict")

    tool_groups = create_tool_groups(
        tool_to_queries, min_queries, train_ratio, seed, 
        baseline_descs, metadata_dict, parameters_dict, w1, w2)

    # Build mcp_tools dict for parameter extraction (reuse already-loaded data)
    baseline_mcp_tools = {
        key: {'parameters': parameters_dict.get(key, {}), '_metadata': metadata_dict.get(key, {})}
        for key in set(parameters_dict.keys()) | set(metadata_dict.keys())
    }

    return tool_to_queries, tool_groups, baseline_mcp_tools

def hash_strings(strings):
    # Example: order matters; join with a sentinel and encode
    payload = "\n".join(strings).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def build_tool_to_yaml_index(yaml_files: List[str], output_dir: Path, file_type: str = "baseline") -> Dict[tuple, str]:
    """
    Build an index mapping (server_name, tool_name) tuples to YAML file paths.
    Uses cache if available and valid.
    
    Args:
        yaml_files: List of all YAML file paths
        cache_file: Path to cache file
        file_type: "baseline" or "ground_truth" (for logging)
    
    Returns:
        Dict mapping (server_name, tool_name) tuple -> yaml_file_path
    """

    cache_file = output_dir / f"tool_to_yaml_index_{hash_strings(str(yaml_files))}_{file_type}.json"

    print("cache_file: ", cache_file)
    
    # Check if cache exists and is valid
    if cache_file.exists():
        with cache_file.open() as f:
            cache_data = json.load(f)
            tool_to_yaml_raw = cache_data['tool_to_yaml']
            tool_to_yaml = {}
            for tool_tuple_str, yaml_path in tool_to_yaml_raw.items():
                server_name, tool_name = tool_tuple_str.split('|||')
                server_name = standardize(server_name)
                tool_to_yaml[(server_name, tool_name)] = Path(yaml_path)
            return tool_to_yaml
        
    # Build index from scratch
    print(f"   🔨 Building tool→YAML index from {len(yaml_files)} files...")
    tool_to_yaml = {}
    
    for yaml_file in tqdm(yaml_files, desc=f"Indexing {file_type} YAMLs", unit="file", leave=False):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if data and 'mcp_servers' in data:
                for server_name, server_info in data['mcp_servers'].items():

                    # Standardize server name for consistent matching
                    server_name = standardize(server_name)

                    if "spotify" in server_name:
                        server_name = "spotify"
                    elif "tmdb" in server_name:
                        server_name = "tmdb"

                    for tool in server_info.get('tools', []):
                        tool_name = tool.get('tool_name', '')
                        if tool_name:
                            # Store with (standardized server_name, tool_name) as key
                            tool_to_yaml[(server_name, tool_name)] = yaml_file
    
    # Save cache (convert tuple keys to strings for JSON)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # Convert tuple keys to "server_name|||tool_name" strings for JSON
    tool_to_yaml_serializable = {}
    for (server, tool), yaml_path in tool_to_yaml.items():
        key_str = f"{server or 'None'}|||{tool}"
        tool_to_yaml_serializable[key_str] = str(yaml_path)
    
    cache_data_serializable = {
        'yaml_files': [str(yaml_file) for yaml_file in yaml_files],
        'tool_to_yaml': tool_to_yaml_serializable,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with cache_file.open('w') as f:
        json.dump(cache_data_serializable, f, indent=2)
    print(f"   💾 Saved index cache to {cache_file.name} ({len(tool_to_yaml)} (server, tool) mappings)")
    
    return tool_to_yaml


def filter_yaml_files_by_tools(
    yaml_files_path: Path,
    tools_to_queries: Dict[Tuple[str, str], List[Dict[str, Any]]],
    file_type: str = "baseline",
) -> List[str]:
    """
    Filter YAML files to only those needed for the tools in the evaluation data.
    This avoids loading 10,000+ YAMLs when we only need a few hundred.
    
    IMPORTANT: Reads the flattened eval files (NOT tool_grouped.json) to get ALL tools
    from the evaluation data, including those with < min_queries that were filtered out
    during the grouping step.
    
    Uses a cached index (tool_name → yaml_file) for fast lookups on subsequent runs.
    
    Args:
        yaml_files: List of all discovered YAML files
        tool_grouped_file: Path to tool_grouped.json (used to locate flattened files)
        file_type: "baseline" or "ground_truth" (for logging)
        needed_tools_cache: Optional pre-computed set of needed tools (avoids re-scanning flattened files)
    
    Returns:
        Tuple of (filtered_yaml_files, needed_tools_set)
    """

    
    # Get output directory (where flattened files and cache are stored)
    # tool_grouped.json is in output_dir/tool_grouped/, so we need parent.parent
    output_dir = Path("DRAFT/policy_learn/cache")
    
    # Check if we can reuse cached needed_tools

    needed_tools = set()
        
    for tool_tuple, tool_group in tools_to_queries.items():
        server_name, tool_name = tool_tuple
        needed_tools.add((server_name, tool_name))

    print(f"   🔍 Need {file_type} YAMLs for {len(needed_tools)} unique (server, tool) pairs (from tools_to_queries)")
    
    # Fast lookup: find YAMLs containing needed tools
    relevant_yamls = set()
    tools_found = set()
    tools_not_found = set()

    # Build or load tool→YAML index (cached for performance)

    # search for all yaml files in yaml_files_path
    if yaml_files_path.is_file():
        yaml_files = [yaml_files_path]
    else:
        yaml_files = discover_yaml_files(yaml_files_path)
    
    tool_to_yaml = build_tool_to_yaml_index(yaml_files, output_dir, file_type)

    # print("tool_to_yaml: ", tool_to_yaml)
    
    for tool_key in needed_tools:
        # tool_key is (server_name, tool_name) tuple
        yaml_file = tool_to_yaml.get(tool_key, None)

        if yaml_file:
            relevant_yamls.add(yaml_file)
            tools_found.add(tool_key)
        else:
            tools_not_found.add(tool_key)
    
    print(f"   ✅ Filtered to {len(relevant_yamls)} relevant {file_type} YAML files (from {len(yaml_files)})")
    print(f"   📊 Found {len(tools_found)}/{len(needed_tools)} (server, tool) pairs in {file_type} YAMLs")
    
    if tools_not_found:
        # Show a sample of missing tools (not all, to avoid spam)
        sample_missing = [f"{s}/{t}" if s else t for s, t in list(tools_not_found)[:5]]
        print(f"   ⚠️  {len(tools_not_found)} out of {len(needed_tools)} (server, tool) pairs not found in {file_type} YAMLs (sample: {sample_missing})")
    
    return list(relevant_yamls)


def prepare_training_dataset(
    tools_to_queries: Dict[Tuple[str, str], List[Dict[str, Any]]],
    tool_groups: List[Dict[str, Any]],
    args: argparse.Namespace,
    no_eval_mode: bool = False,
    baseline_mcp_tools: Dict[tuple, Dict[str, Any]] = None,
    refine_desc: bool = False,
    start_token: str = "<output>",
    end_token: str = "</output>",
) -> Path:
    """
    Step 3: Prepare tool-level training dataset with prompts.
    
    Args:
        tools_to_queries: Dict mapping (server_name, tool_name) to queries
        tool_groups: List of tool group dicts
        args: Command line arguments
        no_eval_mode: Whether running without eval records
        baseline_mcp_tools: Pre-loaded MCP tools dict (reused from earlier loading)
    
    Returns path to training dataset JSON file.
    """
    print("\n📝 Step 3: Preparing training dataset")

    output_dir = args.output_dir
    prompt_path = args.prompt_path
    num_query_examples = args.num_query_examples
    ground_truth_strategy = args.ground_truth_strategy
    max_samples_per_tool = args.max_samples_per_tool
    model_name = args.model_name
    parameter_generation_prompt_version = args.parameter_generation_prompt_version
    w1 = args.w1
    w2 = args.w2
    root_dir = args.root_dir
    query_selection_strategy = args.query_selection_strategy
    refine_desc = args.refine_desc

    baseline_desc_files = Path(args.baseline_desc_files)
    ground_truth_desc_files = None
    if args.ground_truth_desc_files:
        ground_truth_desc_files = [Path(p) for p in args.ground_truth_desc_files]
    
    fixed_param_path = Path(args.fixed_param_path) if args.fixed_param_path else None

    # only tools with queries will be used
    baseline_desc_files = filter_yaml_files_by_tools(
        baseline_desc_files,
        tools_to_queries,
        file_type="baseline")
    
    # Only filter fixed_param_path if it's provided
    if fixed_param_path:
        fixed_param_path = filter_yaml_files_by_tools(
            fixed_param_path,
            tools_to_queries,
            file_type="fixed_param")
    else:
        fixed_param_path = None

    if ground_truth_desc_files:
        filtered_ground_truth_desc_files = []
        for ground_truth_desc_file in ground_truth_desc_files:
            # TODO: override later ones with earlier ones
            filtered_ground_truth_desc_files_ = filter_yaml_files_by_tools(
                ground_truth_desc_file,
                tools_to_queries,
                file_type="ground_truth")
            filtered_ground_truth_desc_files.extend(filtered_ground_truth_desc_files_)
    else:
        filtered_ground_truth_desc_files = None

    ground_truth_desc_files = filtered_ground_truth_desc_files
    
    # Create dataset subdirectory for organized output
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # prepare_script = root_dir / "DRAFT/policy_learn/prepare_pl_dataset.py"
    dataset_path = dataset_dir / "tool_level.json"
    
    # Save file lists to JSON file (for debugging/reproducibility)
    desc_files_config = output_dir / "desc_files_config.json"
    config_data = {
        "baseline_desc_files": [str(desc_file) for desc_file in baseline_desc_files] if baseline_desc_files else [],
        "ground_truth_desc_files": [str(desc_file) for desc_file in ground_truth_desc_files] if ground_truth_desc_files else [],
        "fixed_param_path": [str(p) for p in fixed_param_path] if fixed_param_path else []
    }
    with desc_files_config.open('w') as f:
        json.dump(config_data, f, indent=2)
    
    # Validate baseline_mcp_tools was passed (should be pre-loaded by caller)
    if not baseline_mcp_tools:
        raise ValueError("baseline_mcp_tools must be provided (pre-loaded from group_queries_by_tool or load_tools_without_eval)")
    
    print(f"   ✅ Using pre-loaded parameter schemas for {len(baseline_mcp_tools)} tools")
    
    # Load fixed_mcp_tools if fixed_param_path is provided
    fixed_mcp_tools = None
    if fixed_param_path:
        fixed_mcp_tools = load_mcp_yaml_files([str(p) for p in fixed_param_path])
        print(f"   ✅ Loaded fixed parameter schemas for {len(fixed_mcp_tools)} tools from {len(fixed_param_path)} files")

    dataset_path = process_tool_level_data(
        tool_groups=tool_groups,
        output_file=dataset_path,
        prompt_template_path=prompt_path,
        no_eval_mode=no_eval_mode,
        ground_truth_desc_files=ground_truth_desc_files,
        num_query_examples=num_query_examples,
        query_selection_strategy=query_selection_strategy,
        random_seed=args.seed,
        model_name=model_name,
        parameter_generation_prompt_version=parameter_generation_prompt_version,
        ground_truth_strategy=ground_truth_strategy,
        max_samples_per_tool=max_samples_per_tool,
        w1=w1,
        w2=w2,
        baseline_mcp_tools=baseline_mcp_tools,
        fixed_mcp_tools=fixed_mcp_tools,
        refine_desc=refine_desc,
        start_token=start_token,
        end_token=end_token,
    )
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not created: {dataset_path}")
    
    print(f"✅ Training dataset: {dataset_path}")
    return dataset_path


# New function to load tools without eval records:
def load_tools_without_eval(args: argparse.Namespace) -> Tuple[Dict, List, Dict]:
    """
    Load tools directly from baseline YAML files without eval records.
    Creates empty query lists for each tool.
    
    Returns:
        Tuple of (tools_to_queries, tool_groups, baseline_mcp_tools)
    """
    
    baseline_path = Path(args.baseline_desc_files)
    if baseline_path.is_file():
        baseline_files = [baseline_path]
    else:
        baseline_files = discover_yaml_files(baseline_path)
    
    print(f"   Loading {len(baseline_files)} baseline YAML files...")
    baseline_descs, metadata_dict, parameters_dict = load_baseline_descriptions(baseline_files)
    
    # Create tool_to_queries with empty queries
    tools_to_queries = {}
    tool_groups = []
    
    for (server_name, tool_name), description in baseline_descs.items():
        tool_key = (server_name, tool_name)
        tools_to_queries[tool_key] = []  # Empty queries

        tool_metadata = metadata_dict.get(tool_key, {})
        tool_parameters = parameters_dict.get(tool_key, {})

        tool_group = {
            "tool_name": tool_name,
            "server_name": server_name,
            "num_queries": 0,
            "num_train_queries": 0,
            "num_heldout_queries": 0,
            "original_description": description,
            "baseline_metrics": {
                "avg_combined_score": 0.0,
                "avg_api_selection_accuracy": 0.0,
                "avg_api_success_rate": 0.0,
            },
            "train_queries": [],
            "heldout_queries": [],
            "all_queries": [],
            "metadata": tool_metadata,
            "parameters": tool_parameters,
            "w1": args.w1,
            "w2": args.w2,
        }

        tool_groups.append(tool_group)
    
    # Build mcp_tools dict for parameter extraction (reuse already-loaded data)
    baseline_mcp_tools = {
        key: {'parameters': parameters_dict.get(key, {}), '_metadata': metadata_dict.get(key, {})}
        for key in set(parameters_dict.keys()) | set(metadata_dict.keys())
    }
    
    print(f"   ✅ Loaded {len(tool_groups)} tools from baseline YAMLs")
    return tools_to_queries, tool_groups, baseline_mcp_tools

def convert_to_parquet(
    dataset_path: Path,
    output_dir: Path,
    dataset_name: str,
    val_ratio: float,
    root_dir: Path,
    debug: bool = False,
    train_parquet_name: Optional[str] = None,
    val_parquet_name: Optional[str] = None
) -> tuple[Path, Path]:
    """
    Step 4: Convert JSON dataset to train/val parquet files.
    
    Returns (train_parquet_path, val_parquet_path).
    """
    print("\n🔁 Step 4: Converting to Parquet format")
    
    # Use dataset subdirectory for organized output
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_parquet = dataset_dir / (train_parquet_name or "train.parquet")
    val_parquet = dataset_dir / (val_parquet_name or "val.parquet")
    
    # Call the conversion function directly (no subprocess)
    train_path, val_path, train_rows, val_rows = convert_jsonl_to_parquet(
        input_jsonl=str(dataset_path),
        train_parquet=str(train_parquet),
        val_parquet=str(val_parquet),
        prompt_key="prompt",
        completion_key="ground_truth",
        filter_positive=False,
        val_ratio=val_ratio,
        debug=debug,
    )
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Parquet conversion failed")
    
    print(f"✅ Parquet ready: train={train_rows} samples, val={val_rows} samples")
    return train_path, val_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified data preparation for tool-level SFT training"
    )
    
    # Input sources
    parser.add_argument(
        "--eval-dirs",
        nargs='+',
        dest='eval_dirs',
        help="Evaluation directories (alternative to manifest)"
    )
    parser.add_argument(
        "--baseline-desc-files",
        dest='baseline_desc_files',
        help="Comma-separated baseline description files (alternative to manifest)"
    )
    parser.add_argument(
        "--ground-truth-desc-files",
        nargs='+',
        dest='ground_truth_desc_files',
        help="Comma-separated ground truth description files (optional, can be multiple folders)"
    )

    parser.add_argument(
        "--fixed-param-path",
        # nargs='+',
        dest='fixed_param_path',
        help="Fixed parameter path"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for prepared data"
    )
    parser.add_argument(
        "--dataset-name",
        default="tool_level_dataset",
        help="Base name for dataset files"
    )
    
    # Tool grouping parameters
    parser.add_argument(
        "--min-queries-per-tool",
        type=int,
        default=2,
        help="Minimum queries required per tool"
    )
    parser.add_argument(
        "--train-query-ratio",
        type=float,
        default=0.9,
        help="Ratio of queries for training vs held-out"
    )
    
    # Training dataset parameters
    parser.add_argument(
        "--prompt-path",
        required=True,
        help="Path to prompt template file"
    )

    parser.add_argument(
        "--num-query-examples",
        type=int,
        default=5,
        help="Number of query examples in each prompt"
    )
    parser.add_argument(
        "--query-selection-strategy",
        default="random",
        choices=["random"],
        help="Strategy for selecting example queries"
    )
    parser.add_argument(
        "--ground-truth-strategy",
        default="select_best",
        choices=["select_best", "use_all_variants"],
        help="Strategy for handling multiple ground truth variants"
    )
    parser.add_argument(
        "--max-samples-per-tool",
        type=int,
        default=5,
        help="Maximum training samples per tool"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--model-name",
        default="gpt-41-2025-04-14",
        help="Model name for reward evaluation"
    )
    parser.add_argument(
        "--parameter-generation-prompt-version",
        default="v3",
        help="Parameter generation prompt version"
    )
    parser.add_argument(
        "--w1",
        type=float,
        default=0.5,
        help="Weight for success rate in reward"
    )
    parser.add_argument(
        "--w2",
        type=float,
        default=0.5,
        help="Weight for parameter quality in reward"
    )
    
    # Parquet conversion
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--train-parquet-name",
        help="Name for train parquet file (default: {dataset_name}_train.parquet)"
    )
    parser.add_argument(
        "--val-parquet-name",
        help="Name for val parquet file (default: {dataset_name}_val.parquet)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode (limit data size)"
    )
    

    # Paths
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent.resolve(),  # tools -> policy_learn -> DRAFT -> FunctionWrapper_PL
        help="Root directory of the project"
    )

    parser.add_argument(
        "--start-token",
        default="<|extra_0|>",
        help="Start token for ground truth"
    )
    parser.add_argument(
        "--end-token",
        default="<|extra_1|>",
        help="End token for ground truth"
    )

    parser.add_argument(
        "--refine-desc",
        action="store_true",
        help="Refine description using GPT"
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.time()

    no_eval_mode = not args.eval_dirs or all(not d.strip() for d in args.eval_dirs)
    
    print("🚀 Unified Tool-Level Data Preparation")
    print(f"   Output directory: {args.output_dir}")
    print("")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save all args to data_config.json
    args_dict = vars(args).copy()
    args_dict['no_eval_mode'] = no_eval_mode
    
    # Convert Path objects to strings for JSON serialization
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
    
    data_config_path = args.output_dir / "data_config.json"
    with data_config_path.open('w') as f:
        json.dump(args_dict, f, indent=2)
    print(f"✅ Saved args to {data_config_path}")

    # Step 1 read eval_dirs and tool_desc_file
    eval_dirs = args.eval_dirs
    # tool_desc_files = args.tool_desc_files

    if not no_eval_mode:

        records = flatten_eval_results(
            eval_dirs, args.output_dir, args.baseline_desc_files, args.root_dir)
        
        tools_to_queries, tool_groups, baseline_mcp_tools = group_queries_by_tool(
            records, args)
    
    else:

        tools_to_queries, tool_groups, baseline_mcp_tools = load_tools_without_eval(args)

    start_token = args.start_token
    end_token = args.end_token
    
    dataset_path = prepare_training_dataset(
        tools_to_queries,
        tool_groups,
        args,
        no_eval_mode,
        baseline_mcp_tools,
        refine_desc=args.refine_desc,
        start_token=start_token,
        end_token=end_token,
    )

    #save tool_groups to json file
    tool_groups_dir =args.output_dir / "dataset" / "tool_grouped"
    tool_groups_dir.mkdir(parents=True, exist_ok=True)
    tool_groups_path = tool_groups_dir / "tool_groups.json"
    with tool_groups_path.open('w') as f:
        json.dump(tool_groups, f, indent=2)
        print(f"✅ Tool groups saved to {tool_groups_path}")

    train_parquet, val_parquet = convert_to_parquet(
        dataset_path, 
        args.output_dir, args.dataset_name, args.val_ratio, args.root_dir, args.debug, 
        args.train_parquet_name, args.val_parquet_name)

    print(f"✅ Dataset: {dataset_path}")
    print(f"✅ Train parquet: {train_parquet}")
    print(f"✅ Val parquet: {val_parquet}")

    
if __name__ == "__main__":
    main()