#!/usr/bin/env python3
"""
Group evaluation data by tool/API.

For each tool, collect all queries where the tool appears in either:
- golden_api (expected/correct API)
- selected_api (actually selected API, may be correct or incorrect)

This creates tool-centric training data where:
- State: (tool_name, [query_examples])
- Action: Tool description
- Reward: Average success across all queries using this tool
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Set, Optional, Tuple
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm

from StableToolBench.toolbench.utils import standardize


def _process_single_baseline_file(baseline_desc_file: Path) -> Tuple[Dict[tuple, str], Dict[tuple, Dict], Dict[tuple, Dict]]:
    """
    Process a single baseline description file. Helper for parallel processing.
    
    Args:
        baseline_desc_file: Path to a D0 MCP YAML file
        
    Returns:
        Tuple of (baseline_descs, metadata_dict, parameters_dict) for this file
    """
    baseline_descs = {}
    metadata_dict = {}
    parameters_dict = {}
    
    try:
        with open(baseline_desc_file, 'r') as f:
            d0_data = yaml.safe_load(f)
        
        # MCP YAML structure: mcp_servers -> {server_name} -> tools -> [tool_list]
        for server_name, server_config in d0_data['mcp_servers'].items():
            # Normalize server name
            if "spotify" in server_name:
                server_name = "spotify"
            elif "tmdb" in server_name:
                server_name = "tmdb"
            else:
                server_name = standardize(server_name)
            
            tools = server_config['tools']
            for tool in tools:
                tool_name = tool['tool_name']
                description = tool['description']
                metadata = tool['_metadata']

                if 'parameters' in tool:
                    parameters = tool['parameters']
                else:
                    parameters = {}
                
                if tool_name:
                    key = (server_name, tool_name)
                    if description:
                        baseline_descs[key] = description
                    metadata_dict[key] = metadata
                    parameters_dict[key] = parameters
                    
    except Exception as e:
        print(f"⚠️  Failed to load baseline descriptions from {baseline_desc_file}: {e}")
    
    return baseline_descs, metadata_dict, parameters_dict



def parse_args():
    p = argparse.ArgumentParser(
        description="Group evaluation data by tool for tool-level policy learning"
    )
    p.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="Input JSON files (desc_eval_res_step_wise_eval_flattened.json format)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for grouped data",
    )
    p.add_argument(
        "--output-name",
        default="tool_grouped",
        help="Base name for output file (default: tool_grouped)",
    )
    p.add_argument(
        "--min-queries-per-tool",
        type=int,
        default=3,
        help="Minimum number of queries required to include a tool (default: 3)",
    )
    p.add_argument(
        "--include-selected-api",
        action="store_true",
        default=True,
        help="Include tools from selected_api field (not just golden_api)",
    )
    p.add_argument(
        "--train-query-ratio",
        type=float,
        default=0.8,
        help="Ratio of queries to use for training (rest for held-out eval)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split",
    )
    p.add_argument(
        "--baseline-desc-file",
        type=str,
        required=True,
        help="Path to baseline description file (D0 YAML) for D0→D2 training mode. Required.",
    )
    return p.parse_args()


# def load_baseline_descriptions(baseline_desc_file: str) -> Dict[str, str]:
#     """
#     Load baseline descriptions from D0 YAML file.
    
#     Args:
#         baseline_desc_file: Path to D0 MCP YAML file
        
#     Returns:
#         Dictionary mapping tool_name to baseline description
#     """
#     # if not baseline_desc_file or not os.path.exists(baseline_desc_file):
#     #     return {}
    
#     try:
#         with open(baseline_desc_file, 'r') as f:
#             d0_data = yaml.safe_load(f)
        
#         baseline_descs = {}
#         # MCP YAML structure: mcp_servers -> {server_name} -> tools -> [tool_list]
#         for server_name, server_config in d0_data.get('mcp_servers', {}).items():
#             tools = server_config.get('tools', [])
#             for tool in tools:
#                 tool_name = tool.get('tool_name', '')
#                 description = tool.get('description', '')
#                 if tool_name and description:
#                     baseline_descs[tool_name] = description
        
#         print(f"Loaded {len(baseline_descs)} baseline descriptions")
#         return baseline_descs
#     except Exception as e:
#         print(f"⚠️  Failed to load baseline descriptions: {e}")
#         return {}


def load_baseline_descriptions(baseline_desc_files: List[Path], max_workers: int = None) -> tuple[Dict[tuple, str], Dict[tuple, Dict], Dict[tuple, Dict]]:
    """
    Load baseline descriptions from D0 YAML files using parallel processing.
    
    Args:
        baseline_desc_files: List of paths to D0 MCP YAML files
        max_workers: Maximum number of parallel workers (defaults to CPU count)
        
    Returns:
        Tuple of:
        - baseline_descs: Dict mapping (server_name, tool_name) to description
        - metadata_dict: Dict mapping (server_name, tool_name) to _metadata
        - parameters_dict: Dict mapping (server_name, tool_name) to parameters
    """
    baseline_descs = {}
    metadata_dict = {}
    parameters_dict = {}
    
    num_files = len(baseline_desc_files)
    if num_files == 0:
        return baseline_descs, metadata_dict, parameters_dict
    
    # Use ProcessPoolExecutor for parallel YAML parsing (CPU-bound)
    # Limit workers to avoid overhead for small file counts
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, num_files)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all files for parallel processing
        future_to_file = {
            executor.submit(_process_single_baseline_file, f): f 
            for f in baseline_desc_files
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_file), 
                          total=num_files, 
                          desc="Loading baseline descriptions"):
            file_descs, file_metadata, file_params = future.result()
            # Merge results
            baseline_descs.update(file_descs)
            metadata_dict.update(file_metadata)
            parameters_dict.update(file_params)
    
    print(f"Loaded {len(baseline_descs)} baseline descriptions from {len(baseline_desc_files)} files")
    return baseline_descs, metadata_dict, parameters_dict


def extract_tool_name(api_string: str) -> str:
    """
    Extract tool name from API string.
    
    Handles various formats:
    - "get_movie_details" -> "get_movie_details"
    - "tmdb.get_movie_details" -> "get_movie_details"
    - "API.method_name" -> "method_name"
    """
    if not api_string:
        return ""
    
    # Remove any module/class prefix
    if "." in api_string:
        return api_string.split(".")[-1]
    
    return api_string


def load_records(input_files: List[str]) -> List[Dict[str, Any]]:
    """Load and merge records from multiple input files."""
    all_records = []
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and "records" in data:
            records = data["records"]
        else:
            print(f"⚠️  Unexpected format in {file_path}")
            continue
        
        all_records.extend(records)
    
    print(f"Loaded {len(all_records)} records from {len(input_files)} file(s)")
    return all_records


def extract_category_and_server(file_name: str, valid_categories: Set[str]) -> tuple[str, str]:
    """
    Extract category and server name from a file/directory name.
    
    Categories can have underscores (e.g., 'video_images', 'artificial_intelligence_machine_learning'),
    so we need to find the longest matching category prefix.
    
    Args:
        file_name: The file/directory name (e.g., 'travel_airbnb_search', 'video_images_flaticon')
        valid_categories: Set of valid category names (lowercase with underscores)
    
    Returns:
        Tuple of (category_name, server_name)
    """
    # Sort by length descending to match longest category first
    sorted_categories = sorted(valid_categories, key=len, reverse=True)
    
    for category in sorted_categories:
        prefix = category + '_'
        if file_name.startswith(prefix):
            server_name = file_name[len(prefix):]
            return category, server_name
    
    # Fallback: assume first underscore-separated part is category
    parts = file_name.split('_')
    if len(parts) >= 2:
        return parts[0], '_'.join(parts[1:])
    return file_name, ''


def group_by_tool(
    records: List[Dict[str, Any]],
    include_selected_api: bool = True,
    valid_categories: Set[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group records by tool.
    
    For each tool, collect all queries where the tool appears in:
    - golden_api (expected API)
    - selected_api (actually selected API, if include_selected_api=True)
    
    Args:
        records: List of evaluation records
        include_selected_api: Whether to include selected_api field
        valid_categories: Set of valid category names (lowercase with underscores).
                         Used to correctly parse directory names like 'video_images_flaticon'.
    
    Returns:
        Dict mapping (server_name, tool_name) -> list of records
    """
    tool_to_queries = defaultdict(list)
    
    if valid_categories is None:
        raise ValueError("❌ Error: valid_categories is required")
    
    for record in records:
        tools_in_record: Set[Tuple[str, str]] = set()
        
        # Add golden API (expected/correct) with its server name
        golden_api = record.get("expected_golden_api", record.get("golden_api", ""))
        server_name_golden = record['server_name_golden_api']
        if golden_api and server_name_golden:
            tool_name = extract_tool_name(golden_api)
            # Standardize server name for consistent matching with YAML files
            server_name_golden = standardize(server_name_golden)
            if tool_name:
                tools_in_record.add((server_name_golden, tool_name))
        
        # Optionally add selected API (what was actually chosen) with its server name
        if include_selected_api:
            selected_api = record["selected_api_name"]
            server_name_selected = record['server_name_selected_api']
            if selected_api and server_name_selected:
                tool_name = extract_tool_name(selected_api)
                # Standardize server name for consistent matching with YAML files
                server_name_selected = standardize(server_name_selected)
                if tool_name:
                    tools_in_record.add((server_name_selected, tool_name))
        
        # Add this record to all relevant tools
        for server_name, tool_name in tools_in_record:
            # Mark whether this tool was the golden API for this query
            record_copy = record.copy()
            record_copy["is_golden"] = (extract_tool_name(golden_api) == tool_name)
            tool_to_queries[(server_name, tool_name)].append(record_copy)
    
    return tool_to_queries


def calculate_baseline_metrics(queries: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate baseline metrics for a tool across its queries."""
    
    # Collect metrics from queries where this tool was the golden API
    golden_queries = [q for q in queries if q.get("is_golden", False)]
    
    if not golden_queries:
        return {
            "avg_api_selection_accuracy": 0.0,
            "avg_api_success_rate": 0.0,
            "avg_combined_score": 0.0,
            "num_golden_queries": 0,
            "num_total_queries": len(queries),
        }
    
    # Extract metrics
    api_selection_scores = []
    api_success_scores = []
    combined_scores = []
    
    for query in golden_queries:
        # API selection accuracy
        sel_acc = query.get("avg_api_selection_accuracy")
        if sel_acc is not None:
            api_selection_scores.append(float(sel_acc))
        
        # API success rate
        api_sr = query.get("avg_api_success_rate")
        if api_sr is not None:
            api_success_scores.append(float(api_sr))
        
        # Combined/linear combination score
        combined = query.get("linear_combination")
        if combined is not None:
            combined_scores.append(float(combined))
        elif sel_acc is not None and api_sr is not None:
            # Compute if not present (w1=0.5, w2=0.5)
            combined_scores.append(0.5 * float(sel_acc) + 0.5 * float(api_sr))
    
    return {
        "avg_api_selection_accuracy": np.mean(api_selection_scores) if api_selection_scores else 0.0,
        "avg_api_success_rate": np.mean(api_success_scores) if api_success_scores else 0.0,
        "avg_combined_score": np.mean(combined_scores) if combined_scores else 0.0,
        "num_golden_queries": len(golden_queries),
        "num_total_queries": len(queries),
    }


def split_queries(
    queries: List[Dict[str, Any]],
    train_ratio: float,
    seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split queries into train and held-out sets."""
    np.random.seed(seed)
    
    # Shuffle queries
    shuffled = queries.copy()
    np.random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_queries = shuffled[:split_idx]
    heldout_queries = shuffled[split_idx:]
    
    return train_queries, heldout_queries


def extract_tool_description(
    queries: List[Dict[str, Any]],
    server_name: str,
    tool_name: str,
    baseline_descs: Dict[str, str]
) -> Optional[str]:
    """
    Extract the D0 baseline tool description for this tool.

    Args:
        queries: List of query records for this tool (unused, kept for compatibility)
        tool_name: Name of the tool
        baseline_descs: Dictionary of D0 baseline descriptions from YAML file
                       (keyed by (server_name, tool_name) tuples)

    Returns:
        D0 baseline description if found, None otherwise

    Note: Only supports D0→D2 training mode. Tools without D0 baselines are skipped.
    """
    # First try direct lookup (in case keys are strings)
    # if tool_name in baseline_descs:
    #     return baseline_descs[tool_name]
    
    # Otherwise search for matching tuple key (server_name, tool_name)
    # for key in baseline_descs:
    #     if isinstance(key, tuple) and len(key) == 2 and key[0] == server_name and key[1] == tool_name:
    #         return baseline_descs[key]

    key = (server_name, tool_name)
    return baseline_descs.get(key, None)


def create_tool_groups(
    tool_to_queries: Dict[str, List[Dict[str, Any]]],
    min_queries: int,
    train_ratio: float,
    seed: int,
    baseline_descs: Dict[str, str],
    metadata_dict: Dict[str, Any],
    parameters_dict: Dict[str, Any],
    w1: float,
    w2: float
) -> List[Dict[str, Any]]:
    """
    Create tool-centric training data for D0→D2 improvement task.

    Args:
        tool_to_queries: Dictionary mapping tool names to query lists
        min_queries: Minimum queries required per tool
        train_ratio: Ratio for train/test split
        seed: Random seed
        baseline_descs: D0 baseline descriptions (required)
        metadata_dict: Dictionary containing tool metadata
        parameters_dict: Dictionary containing tool parameters
        w1: Weight parameter 1
        w2: Weight parameter 2

    Returns:
        List of tool groups, each containing:
        - tool_name
        - all_queries (full list)
        - train_queries (for reward computation)
        - heldout_queries (for generalization eval)
        - original_description (D0 baseline description)
        - baseline_metrics
    """
    tool_groups = []
    skipped = []
    
    for (server_name, tool_name), queries in sorted(tool_to_queries.items()):
        # Filter tools with too few queries
        if len(queries) < min_queries:
            skipped.append(f"{server_name}.{tool_name}({len(queries)} queries)")
            continue
        
        # Split into train/heldout
        train_queries, heldout_queries = split_queries(queries, train_ratio, seed)
        
        # CRITICAL: Filter out tools where split results in no training queries
        # These tools cannot be used for training and should be excluded

        # TODO: handle tools with no train queries
        if not train_queries or len(train_queries) == 0:
            skipped.append(f"{tool_name}(0 train queries after split)")
            continue
        
        # Calculate baseline metrics
        baseline_metrics = calculate_baseline_metrics(queries)
        
        # Extract D0 baseline description
        original_description = extract_tool_description(queries, server_name, tool_name, baseline_descs)
        if not original_description:
            skipped.append(f"{server_name}.{tool_name}(no baseline description)")
            continue
        
        # Get tool-specific metadata and parameters
        # Note: metadata_dict and parameters_dict are keyed by (server_name, tool_name) tuples
        # We need to find the matching entry by tool_name (second element of tuple)

        tool_metadata = {}
        tool_parameters = {}

        tool_metadata = metadata_dict.get((server_name, tool_name), {})
        tool_parameters = parameters_dict.get((server_name, tool_name), {})

        # for key in metadata_dict:
        #     if isinstance(key, tuple) and len(key) == 2 and key[1] == tool_name:
        #         tool_metadata = metadata_dict.get(key, {})
        #         tool_parameters = parameters_dict.get(key, {})
        #         break
        
        tool_group = {
            "tool_name": tool_name,
            "server_name": server_name,
            "num_queries": len(queries),
            "num_train_queries": len(train_queries),
            "num_heldout_queries": len(heldout_queries),
            "original_description": original_description,
            "baseline_metrics": baseline_metrics,
            "train_queries": train_queries,
            "heldout_queries": heldout_queries,
            "all_queries": queries,
            "metadata": tool_metadata,
            "parameters": tool_parameters,
            "w1": w1,
            "w2": w2,
        }
        
        tool_groups.append(tool_group)
    
    if skipped:
        print(f"Skipped {len(skipped)} tools (< {min_queries} queries): {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")
    print(f"Created {len(tool_groups)} tool groups")
    
    return tool_groups


def save_tool_groups(
    tool_groups: List[Dict[str, Any]],
    output_dir: str,
    output_name: str
) -> str:
    """Save tool groups to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{output_name}.json")
    
    with open(output_path, "w") as f:
        json.dump(tool_groups, f, indent=2)
    
    # Also save a summary
    summary_path = os.path.join(output_dir, f"{output_name}_summary.json")
    summary = {
        "num_tools": len(tool_groups),
        "total_queries": sum(tg["num_queries"] for tg in tool_groups),
        "total_train_queries": sum(tg["num_train_queries"] for tg in tool_groups),
        "total_heldout_queries": sum(tg["num_heldout_queries"] for tg in tool_groups),
        "avg_queries_per_tool": np.mean([tg["num_queries"] for tg in tool_groups]),
        "avg_baseline_reward": np.mean([tg["baseline_metrics"]["avg_combined_score"] for tg in tool_groups]),
        "tools": [
            {
                "name": tg["tool_name"],
                "num_queries": tg["num_queries"],
                "baseline_reward": tg["baseline_metrics"]["avg_combined_score"],
            }
            for tg in tool_groups
        ],
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return output_path


# def main():
#     args = parse_args()
    
#     print(f"🔧 Tool-Level Data Grouping | Min queries: {args.min_queries_per_tool} | Train ratio: {args.train_query_ratio:.0%}")
    
#     # Load baseline descriptions (required for D0→D2 training)
#     if not args.baseline_desc_file:
#         raise ValueError("❌ Error: --baseline-desc-file is required for D0→D2 training mode")

#     baseline_descs = load_baseline_descriptions(args.baseline_desc_file)
#     if not baseline_descs:
#         raise ValueError("❌ Error: Failed to load baseline descriptions from file")

#     print(f"   ✅ Using D0 baselines as prompts → D2 as targets ({len(baseline_descs)} tools)")
    
#     # Load records
#     records = load_records(args.input_files)
#     if not records:
#         raise ValueError("❌ No records loaded!")
    
#     # Group by tool
#     tool_to_queries = group_by_tool(records, args.include_selected_api)
#     print(f"Found {len(tool_to_queries)} unique tools")
    
#     # Create tool groups with train/heldout split
#     tool_groups = create_tool_groups(
#         tool_to_queries,
#         args.min_queries_per_tool,
#         args.train_query_ratio,
#         args.seed,
#         baseline_descs
#     )
    
#     # Save
#     output_path = save_tool_groups(tool_groups, args.output_dir, args.output_name)
    
#     # Summary
#     total_queries = sum(tg['num_queries'] for tg in tool_groups)
#     avg_queries = np.mean([tg['num_queries'] for tg in tool_groups])
#     avg_reward = np.mean([tg['baseline_metrics']['avg_combined_score'] for tg in tool_groups])
    
#     print(f"✅ Saved to: {output_path}")
#     print(f"   {len(tool_groups)} tools | {total_queries} queries (avg {avg_queries:.1f}/tool) | baseline reward: {avg_reward:.3f}")
    
#     # Show top 3 tools
#     sorted_tools = sorted(tool_groups, key=lambda x: x["num_queries"], reverse=True)
#     top_tools = ', '.join([f"{tg['tool_name']}({tg['num_queries']})" for tg in sorted_tools[:3]])
#     print(f"   Top tools: {top_tools}")


# if __name__ == "__main__":
#     main()

