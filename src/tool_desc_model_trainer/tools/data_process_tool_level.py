import json
import os
import sys
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import yaml
# from prepare_pl_dataset import generate_compatible_ground_truth, extract_parameter_json, load_mcp_yaml_files

from mcp_utils import extract_parameter_json, generate_compatible_ground_truth
from ground_truth_evaluator import GroundTruthEvaluator

from StableToolBench.server.utils import standardize
from utils.llm import llm_call

# =============================================================================
# Helper Functions for Ground Truth Processing
# =============================================================================

def _select_variants(
    tool_name: str,
    all_variants: List[Dict[str, Any]],
    train_queries: List[Dict[str, Any]],
    original_desc: str,
    strategy: str,
    w1: float,
    w2: float,
    evaluator: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Select ground truth variants based on strategy.
    
    Args:
        tool_name: Name of the tool
        all_variants: List of variant dicts with 'source' and 'description'
        train_queries: Training queries for evaluation
        original_desc: Original/baseline description
        strategy: One of 'select_best', 'use_all_variants', 'use_first'
        w1, w2: Weights for evaluation
        evaluator: GroundTruthEvaluator instance (required for select_best)
    
    Returns:
        Tuple of (variants_list, strategy_message)
    """

    # TODO: implement select_best strategy correctly

    # if strategy == "select_best" and len(all_variants) > 1:
    #     print(f"   🧪 Evaluating {len(all_variants)} ground truth variants for {tool_name}...")
    #     try:
    #         best, details = evaluator.select_best_ground_truth(
    #             tool_name, all_variants, train_queries, original_desc
    #         )
    #         print(f"   🏆 Selected best variant: {best['source']}")
    #         score = details['best_variant']['aggregated_score']
    #         return [best], f"best of {len(all_variants)}: {best['source']} (score: {score:.4f})"

    #     except Exception as e:
    #         print(f"   ❌ Ground truth evaluation failed: {e}")
    #         return [all_variants[0]], f"first variant (fallback): {all_variants[0]['source']}"
    
    # elif strategy == "use_all_variants":
    #     sources = ', '.join(v['source'] for v in all_variants)
    #     return all_variants, f"all {len(all_variants)} variants: {sources}"
    
    # else:  # "use_first" or single variant
    #     return [all_variants[0]], f"first variant: {all_variants[0]['source']}"

    return [all_variants[0]], f"first variant: {all_variants[0]['source']}"


def _compute_samples_per_variant(
    strategy: str,
    max_samples: int,
    num_variants: int,
    num_queries: int,
    num_examples: int
) -> int:
    """Compute number of samples to generate per variant."""
    if strategy == "use_all_variants":
        return max(1, min(2, max_samples // num_variants))
    return max(1, min(max_samples, num_queries // num_examples))


def _select_sample_queries(
    train_queries: List[Dict[str, Any]],
    num_examples: int,
    strategy: str,
    seed: int,
    sample_idx: int
) -> List[Dict[str, Any]]:
    """Select query subset for a training sample."""
    random.seed(seed)
    n = min(num_examples, len(train_queries))
    
    if strategy == "random":
        return random.sample(train_queries, n)
    
    if len(train_queries) <= num_examples:
        return train_queries
    
    # Rotating window for diversity
    start_idx = (sample_idx * num_examples) % len(train_queries)
    result = train_queries[start_idx:start_idx + num_examples]
    if len(result) < num_examples:
        result += train_queries[:num_examples - len(result)]
    return result


# =============================================================================
# Main Processing Function
# =============================================================================

def process_tool_level_data(
    # tool_grouped_file: str,
    tool_groups: List[Dict[str, Any]],
    output_file: str,
    prompt_template_path: str,
    no_eval_mode: bool = False,
    num_query_examples: int = 5,
    query_selection_strategy: str = "random",
    random_seed: int = 42,
    model_name: str = "gpt-41-2025-04-14",
    parameter_generation_prompt_version: str = "v2",
    use_delta_reward: bool = False,
    w1: float = 0.5,
    w2: float = 0.5,
    ground_truth_desc_files: List[str] = None,
    ground_truth_strategy: str = "select_best",
    max_samples_per_tool: int = 5,
    baseline_mcp_tools: Dict[tuple, Dict[str, Any]] = None,
    fixed_mcp_tools: Dict[tuple, Dict[str, Any]] = None,
    start_token: str = "",
    end_token: str = "",
    refine_desc: bool = False,
) -> str:
    """
    Process tool-grouped data for tool-level policy learning.

    Args:
        tool_groups: List of tool groups
        output_file: Path for output dataset file
        prompt_template_path: Path to tool-level prompt template
        num_query_examples: Number of example queries to show in prompt
        query_selection_strategy: How to select example queries (random, first, diverse, balanced)
        random_seed: Random seed for reproducibility
        model_name: Model name for reward evaluation
        parameter_generation_prompt_version: Prompt version
        use_delta_reward: If True, compute delta from baseline
        w1, w2: Weights for API selection and parameter generation
        ground_truth_desc_files: List of YAML file paths with D2 ground truth descriptions (optional)
        ground_truth_strategy: Strategy for handling multiple variants (select_best|use_all_variants|use_first)
        max_samples_per_tool: Maximum number of training samples to generate per tool (default: 5)
        baseline_mcp_tools: Pre-loaded MCP tools dict mapping (server_name, tool_name) to tool info
        fixed_mcp_tools: Pre-loaded fixed parameter MCP tools dict (optional)

    Returns:
        str: Path to created dataset file
    """
    # Import tool-level utilities
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
    from tool_level_prompt_utils import create_tool_level_prompt

    # Validate baseline_mcp_tools (must be pre-loaded by caller)
    if not baseline_mcp_tools:
        raise ValueError("baseline_mcp_tools must be provided (pre-loaded MCP tools dict)")
    
    print(f"   ✅ Using pre-loaded parameter schemas for {len(baseline_mcp_tools)} tools")
    
    if fixed_mcp_tools:
        print(f"   ✅ Using pre-loaded fixed parameter schemas for {len(fixed_mcp_tools)} tools")
    else:
        fixed_mcp_tools = {}
        print("   ℹ️  No fixed_mcp_tools provided, using baseline parameters only")

    # Validate ground truth strategy
    if ground_truth_strategy not in ['select_best', 'use_all_variants', 'use_first']:
        raise ValueError(f"Invalid ground_truth_strategy: {ground_truth_strategy}. Must be one of: select_best, use_all_variants, use_first")

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"🔄 Processing tool-level data for policy learning")
    # print(f"📦 Tool grouped file: {tool_grouped_file}")
    print(f"📝 Prompt template: {prompt_template_path}")
    print(f"🎲 Query examples per tool: {num_query_examples}")
    print(f"🎯 Selection strategy: {query_selection_strategy}")

    if ground_truth_desc_files:
        print(f"📋 Ground truth descriptions: {len(ground_truth_desc_files)} files")
    else:
        print("   ℹ️  No ground truth descriptions provided, using baseline descriptions only")

    # Load ground truth descriptions if provided (supports multiple files)
    # Store as list of variants per tool for multiple samples

    if ground_truth_desc_files:
        print("🔍 Loading external ground truth descriptions...")

    def _load_ground_truth_variants(ground_truth_files: List[str]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Load ground truth variants from a list of YAML file paths.
        
        Returns dict keyed by (server_name, tool_name) to avoid cross-server collisions.
        """
        
        ground_truth_variants = {}
        
        if not ground_truth_files:
            return ground_truth_variants
        
        # Convert Path objects to strings if needed
        files = [str(p) for p in ground_truth_files]

        total_loaded = 0
        for file_path in files:
            if not file_path or not os.path.exists(file_path):
                print(f"   ⚠️ Ground truth file not found: {file_path}")
                continue


            with open(file_path, 'r') as f:
                gt_data = yaml.safe_load(f)

            # Extract descriptions from MCP YAML format
            file_count = 0
            source_name = file_path
            for server_name, server_config in gt_data.get('mcp_servers', {}).items():
                # Standardize server name to match baseline lookup
                # NOTE: baseline loader normalizes tmdb/spotify to short names.
                if "spotify" in server_name:
                    server_name_std = "spotify"
                elif "tmdb" in server_name:
                    server_name_std = "tmdb"
                else:
                    server_name_std = standardize(server_name)
                
                tools = server_config.get('tools', [])
                for tool in tools:
                    tool_name = tool.get('tool_name', '')
                    description = tool.get('description', '')
                    reasoning = tool.get('_metadata', {}).get('reasoning', '') or ""
                    if tool_name and description:
                        # Use (server_name, tool_name) as key to avoid cross-server collisions
                        key = (server_name_std, tool_name)
                        if key not in ground_truth_variants:
                            ground_truth_variants[key] = []

                        ground_truth_variants[key].append({
                            'source': source_name,
                            'description': description,
                            'reasoning': reasoning
                        })
                        file_count += 1

            print(f"   ✅ Loaded {file_count} descriptions from {source_name}")
            total_loaded += file_count

        # Show variant statistics
        total_variants = sum(len(variants) for variants in ground_truth_variants.values())
        tools_with_variants = len([t for t, v in ground_truth_variants.items() if len(v) > 1])
        print(f"   📊 Total loaded: {total_loaded} descriptions from {len(files)} file(s)")
        print(f"   🎯 Tools with multiple variants: {tools_with_variants}")
        print(f"   📈 Total training variants: {total_variants}")

        return ground_truth_variants

    if ground_truth_desc_files:
        ground_truth_variants = _load_ground_truth_variants(ground_truth_desc_files) # (server_name, tool_name) -> [{'source': file, 'description': desc}, ...]
    else:
        ground_truth_variants = {}
        print("   ℹ️  No ground truth descriptions provided, using baseline descriptions only")
    
    # Create dataset entries
    dataset = []
    skipped_tools = []
    
    for tool_group in tool_groups:
        tool_name = tool_group["tool_name"]
        server_name = tool_group["server_name"]
        train_queries = tool_group["train_queries"]
        heldout_queries = tool_group["heldout_queries"]
        original_description = tool_group["original_description"]
        baseline_metrics = tool_group["baseline_metrics"]

        MAX_DESC_CHARS = 10000
        if len(original_description) > MAX_DESC_CHARS:
            print(f"   ⚠️  Truncating original_description for {tool_name}: {len(original_description)} -> {MAX_DESC_CHARS} chars")
            original_description = original_description[:MAX_DESC_CHARS] + "\n... (truncated)"
        
        # CRITICAL: Skip tools with no training queries
        # Without training queries, we cannot compute rewards for training
        # and the model should not be updated on these tools

        no_train_queries = not train_queries or len(train_queries) == 0

        if no_train_queries and not no_eval_mode:
            skipped_tools.append(f"{tool_name} (0 train queries)")
            print(f"   ⚠️  Skipping {tool_name}: No training queries available")
            continue
        
        # Ground truth variant handling - use (server_name, tool_name) key
        all_tool_variants = ground_truth_variants.get((server_name, tool_name), [])

        # Early exit: skip tools without ground truth
        if not all_tool_variants:
            skipped_tools.append(f"{tool_name} (no ground truth)")
            # print(f"   ⚠️  Skipping {tool_name}: No ground truth description available")
            continue

        # Pre-validate parameter JSON (fail fast, before processing variants)
        # if not baseline_mcp_tools:
        #     raise ValueError("No MCP tools loaded")
            
        baseline_parameter_json = extract_parameter_json(
            server_name, tool_name, baseline_mcp_tools)
        
        # Override with fixed parameter json only if fixed_mcp_tools has this tool
        # (extract_parameter_json always returns a JSON string, so check the dict directly)
        if fixed_mcp_tools and (server_name, tool_name) in fixed_mcp_tools:
            parameter_json = extract_parameter_json(server_name, tool_name, fixed_mcp_tools)
        else:
            parameter_json = baseline_parameter_json

        MAX_PARAM_JSON_CHARS = 10000
        if len(parameter_json) > MAX_PARAM_JSON_CHARS:
            print(f"   ⚠️  Truncating parameter_json for {tool_name}: {len(parameter_json)} -> {MAX_PARAM_JSON_CHARS} chars")
            parameter_json = parameter_json[:MAX_PARAM_JSON_CHARS] + "\n... (truncated)"

        if not parameter_json:
            raise ValueError(f"No parameter JSON found for {tool_name}")

        # Select variants based on strategy
        evaluator = None
        if len(all_tool_variants) > 1:
            
            evaluator = GroundTruthEvaluator(
                aggregation_method="mean", variance_penalty=0.0, w1=w1, w2=w2,
                max_workers=1, cache_dir=None
            )
        
        tool_variants_to_process, strategy_msg = _select_variants(
            tool_name, all_tool_variants, train_queries, original_description,
            ground_truth_strategy, w1, w2, evaluator
        )
        print(f"   🎯 Strategy '{ground_truth_strategy}': {strategy_msg}")

        # Generate training samples for each variant
        def _refine_desc(improved_description: str,
                         refine_desc_prompt_path: str = "/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/DRAFT/policy_learn/prompts/refine_desc_v0.txt") -> str:
            # Load the refinement prompt and fill required placeholders from refine_desc_v0.txt
            with open(refine_desc_prompt_path, 'r') as f:
                refine_desc_prompt = f.read()

            prompt = refine_desc_prompt.format(
                tool_name=tool_name,
                server_name=server_name,
                parameter_json=parameter_json,
                original_description=improved_description,
                start_token=start_token,
                end_token=end_token,
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant that refines API descriptions."},
                {"role": "user", "content": prompt}
            ]

            try:
                response = llm_call(messages, model="gpt-5-mini-2025-08-07", temperature=0.0, max_tokens=32768)
                improved_description = response.choices[0].message.content
                return improved_description
            except Exception as exc:
                print(f"   ⚠️  refine_desc failed for {tool_name}: {exc}; using original description")
                return improved_description

        def _build_sample_entry(variant_idx: int, variant: Dict[str, Any], sample_idx: int, samples_per_variant: int):
            sample_seed = random_seed + variant_idx * 100 + sample_idx
            sample_queries = _select_sample_queries(
                train_queries, num_query_examples, query_selection_strategy,
                sample_seed, sample_idx
            )

            prompt = create_tool_level_prompt(
                tool_name=tool_name,
                queries=sample_queries,
                original_description=original_description,
                prompt_template_path=prompt_template_path,
                num_examples=num_query_examples,
                selection_strategy=query_selection_strategy,
                seed=sample_seed,
                parameter_json=parameter_json,
                server_name=server_name,
                start_token=start_token,
                end_token=end_token,
            )

            improved_description = variant['description']
            ground_truth_source = variant['source']
            print(f"   📋 Using ground truth for {tool_name} from {variant['source']} (variant {variant_idx + 1}/{len(tool_variants_to_process)})")

            if refine_desc:
                improved_description = _refine_desc(improved_description)

            ground_truth = generate_compatible_ground_truth(
                improved_description,
                prompt_template_path,
                start_token=start_token,
                end_token=end_token,
                reasoning=variant.get("reasoning", "") if ground_truth_desc_files else None,
            )

            extra_info = {
                # Tool identification
                "tool_name": tool_name,
                "server_name": server_name,
                "sample_idx": sample_idx,
                "variant_idx": variant_idx,

                # Ground truth variant information
                "ground_truth_source": ground_truth_source,
                "ground_truth_strategy": ground_truth_strategy,
                "variant_info": f"{variant_idx + 1}/{len(tool_variants_to_process)}",

                # Queries for reward computation (train/heldout/all)
                "train_queries": train_queries,
                "heldout_queries": heldout_queries,
                "sample_queries_used": [q.get("query_text", "") for q in sample_queries],

                # Baseline for comparison
                "baseline_reward": baseline_metrics["avg_combined_score"],
                "baseline_api_selection": baseline_metrics["avg_api_selection_accuracy"],
                "baseline_api_success": baseline_metrics["avg_api_success_rate"],

                # Evaluation config (passed to reward function)
                "model_name": model_name,
                "param_gen_prompt_version": parameter_generation_prompt_version,
                "use_delta_reward": use_delta_reward,
                "w1": w1,
                "w2": w2,

                # Tool-level specific
                "tool_level_mode": True,
                "num_train_queries": len(train_queries),
                "num_heldout_queries": len(heldout_queries),
                "samples_per_variant": samples_per_variant,
            }

            entry = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "data_source": "tmdb",
                "ability": "tool_description_generation",
                "extra_info": json.dumps(extra_info),  # Serialize for storage
                "tool_name": tool_name,  # For debugging/analysis
                "sample_idx": f"{variant_idx}_{sample_idx}",  # Track variant and sample
                "variant_idx": variant_idx,  # Track which variant this is
            }
            return variant_idx, sample_idx, entry

        # Parallelize sample creation across variants/samples
        futures = []
        max_workers = min(4, max(1, (os.cpu_count() or 1) * 2))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for variant_idx, variant in enumerate(tool_variants_to_process):
                samples_per_variant = _compute_samples_per_variant(
                    ground_truth_strategy, max_samples_per_tool,
                    len(tool_variants_to_process), len(train_queries), num_query_examples)
                for sample_idx in range(samples_per_variant):
                    futures.append(executor.submit(_build_sample_entry, variant_idx, variant, sample_idx, samples_per_variant))

        # Preserve deterministic ordering by sorting results
        results = [f.result() for f in tqdm(futures, desc=f"Building samples for {tool_name}", leave=False)]
        for _, _, entry in sorted(results, key=lambda x: (x[0], x[1])):
            dataset.append(entry)
        
        # Calculate total samples created for this tool (across all variants)
        total_samples_this_tool = len(results)
        print(f"   ✓ {tool_name}: {total_samples_this_tool} samples created across variants")
    
    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples created: {len(dataset)}")
    print(f"Tools processed: {len(tool_groups) - len(skipped_tools)}/{len(tool_groups)}")
    if skipped_tools:
        print(f"Tools skipped: {len(skipped_tools)}")
        if len(skipped_tools) <= 5:
            for skipped in skipped_tools:
                print(f"   - {skipped}")
        else:
            for skipped in skipped_tools[:3]:
                print(f"   - {skipped}")
            print(f"   ... and {len(skipped_tools) - 3} more")
    if len(dataset) > 0:
        tools_with_samples = len(tool_groups) - len(skipped_tools)
        print(f"Avg samples per tool: {len(dataset) / max(1, tools_with_samples):.1f}")
    print()
    print("💡 Note: Train/val split happens at SAMPLE level, not tool level")
    print("   All tools will appear in both train and validation sets")
    print()
    
    # Save dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    
    # Save metadata
    metadata_file = str(output_file).replace(".json", "_metadata.json")
    metadata = {
        "mode": "tool_level_query_split",  # Updated mode name
        "num_samples": len(dataset),
        "num_unique_tools": len(tool_groups),
        "avg_samples_per_tool": len(dataset) / len(tool_groups),
        "num_query_examples": num_query_examples,
        "query_selection_strategy": query_selection_strategy,
        "random_seed": random_seed,
        "prompt_template": prompt_template_path,
        "num_baseline_tools": len(baseline_mcp_tools) if baseline_mcp_tools else 0,
        "model_name": model_name,
        "use_delta_reward": use_delta_reward,
        "w1": w1,
        "w2": w2,
        "split_strategy": "query_level",  # NEW: indicate split strategy
        "tools": [
            {
                "name": tg["tool_name"],
                "num_queries": len(tg["all_queries"]),
                "num_samples_created": max(1, min(5, len(tg["all_queries"]) // num_query_examples)),
                "baseline_reward": tg["baseline_metrics"]["avg_combined_score"],
            }
            for tg in tool_groups
        ],
    }
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print(f"✅ Created tool-level dataset:")
    print(f"   📄 Dataset: {output_file}")
    print(f"   📊 Metadata: {metadata_file}")
    print(f"   📦 Total samples: {len(dataset)}")
    print(f"   🔧 Unique tools: {len(tool_groups)}")
    print(f"   📝 Avg samples per tool: {len(dataset) / len(tool_groups):.1f}")
    
    return output_file