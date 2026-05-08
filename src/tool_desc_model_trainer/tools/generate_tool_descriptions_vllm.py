#!/usr/bin/env python3
"""
Generate improved tool descriptions using a trained model with vLLM for fast batch inference.

This script:
1. Loads a trained model checkpoint with vLLM
2. For each tool in the tool-grouped data, generates an improved description (batched)
3. Saves the descriptions in MCP YAML format for evaluation

Usage:
    python generate_tool_descriptions_vllm.py \
        --model-checkpoint /path/to/checkpoint \
        --tool-grouped-file /path/to/tool_grouped.json \
        --output-mcp-yaml /path/to/output_mcp.yaml \
        --base-mcp-yaml /path/to/base_mcp.yaml \
        --prompt-path /path/to/policy_prompt_tool_level_v2.txt
"""

# import ast
import shutil

import argparse
import copy
import json
import os
import sys
import subprocess
import yaml
import random
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

# Add path to access parameter extraction functions
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from prepare_pl_dataset import load_prompt_template
from mcp_utils import load_mcp_yaml_files, extract_parameter_json

from output_parser import extract_improved_description, extract_json_description_and_reasoning

from StableToolBench.server.utils import standardize


# Fix for PyTorch 2.6+ weights_only=True compatibility with vLLM
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that forces weights_only=False for compatibility with omegaconf checkpoints"""
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = _patched_torch_load


class LoRAConfigError(Exception):
    """Custom exception for LoRA configuration issues"""
    pass


def set_random_seed(seed: Optional[int]) -> None:
    """Set RNG seeds for reproducibility when a seed is provided."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_lora_adapter(checkpoint_path: str) -> tuple[bool, str, int]:
    """
    Detect if checkpoint contains LoRA adapter and get configuration.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        tuple: (has_lora, lora_path, lora_rank)
            - has_lora: bool - Whether LoRA adapter exists
            - lora_path: str - Path to LoRA adapter directory (empty if no LoRA)
            - lora_rank: int - LoRA rank from adapter_config.json (0 if no LoRA)
    """
    checkpoint_path = os.path.expanduser(os.path.abspath(checkpoint_path))

    # Look for LoRA adapter in the standard location
    lora_adapter_path = os.path.join(checkpoint_path, 'lora_adapter')

    if not os.path.isdir(lora_adapter_path):
        print(f"🔍 No LoRA adapter found at: {lora_adapter_path}")
        return False, "", 0

    # Check for required LoRA files
    adapter_config_path = os.path.join(lora_adapter_path, 'adapter_config.json')
    adapter_model_path = os.path.join(lora_adapter_path, 'adapter_model.safetensors')

    if not os.path.exists(adapter_config_path):
        print(f"⚠️  LoRA directory found but missing adapter_config.json: {adapter_config_path}")
        return False, "", 0

    if not os.path.exists(adapter_model_path):
        print(f"⚠️  LoRA directory found but missing adapter_model.safetensors: {adapter_model_path}")
        return False, "", 0

    # Parse LoRA configuration
    try:
        with open(adapter_config_path, 'r') as f:
            lora_config = json.load(f)

        # Validate it's actually a LoRA config
        if lora_config.get('peft_type') != 'LORA':
            print(f"⚠️  Adapter config found but peft_type is not LORA: {lora_config.get('peft_type')}")
            return False, "", 0

        # Get LoRA rank
        lora_rank = lora_config.get('r', 0)
        if lora_rank <= 0:
            raise LoRAConfigError(f"Invalid LoRA rank in config: {lora_rank}")

        print(f"✅ LoRA adapter detected:")
        print(f"   Path: {lora_adapter_path}")
        print(f"   Rank: {lora_rank}")
        print(f"   Target modules: {lora_config.get('target_modules', [])}")

        return True, lora_adapter_path, lora_rank

    except (json.JSONDecodeError, KeyError, IOError) as e:
        raise LoRAConfigError(f"Failed to parse LoRA adapter config: {e}")


def prepare_model_for_vllm(model_path: str) -> tuple[str, dict]:
    """
    Prepare model for VLLM loading. If it's a VERL checkpoint, convert it first.

    Returns:
        tuple: (vllm_model_path, lora_config)
            - vllm_model_path: str - Path to use for vLLM model loading
            - lora_config: dict - LoRA configuration with keys:
                - has_lora: bool
                - lora_path: str (empty if no LoRA)
                - lora_rank: int (0 if no LoRA)
    """
    model_path = os.path.expanduser(model_path)

    # Check if this is a HuggingFace model ID (contains '/' and is not an existing directory)
    if '/' in model_path and not os.path.exists(model_path):
        print(f"✅ Detected HuggingFace model ID, using directly: {model_path}")
        # HuggingFace model IDs don't have LoRA adapters in this context
        return model_path, {"has_lora": False, "lora_path": "", "lora_rank": 0}
    
    # If the caller passed the inner 'huggingface' directory and it already contains
    # a complete HF model, reuse it directly.
    if os.path.isdir(model_path) and os.path.basename(os.path.normpath(model_path)) == 'huggingface':
        hf_files_direct = set(os.listdir(model_path))
        has_config_direct = 'config.json' in hf_files_direct
        has_weights_direct = any(
            name.startswith('pytorch_model') or name.startswith('model.safetensors')
            for name in hf_files_direct
        )
        if has_config_direct and has_weights_direct:
            print(f"✅ Using provided HuggingFace directory directly: {model_path}")
            # Detect LoRA adapter in this directory
            has_lora, lora_path, lora_rank = detect_lora_adapter(model_path)
            lora_config = {"has_lora": has_lora, "lora_path": lora_path, "lora_rank": lora_rank}
            return model_path, lora_config
    
    # Normalize input path: if a user passes the inner 'huggingface' directory,
    # resolve to the checkpoint root so the VERL merger can find fsdp_config.json
    normalized_model_path = os.path.abspath(model_path)
    if os.path.basename(os.path.normpath(normalized_model_path)) == 'huggingface':
        normalized_model_path = os.path.dirname(normalized_model_path)
        print(f"ℹ️  Adjusted model path from '{model_path}' to checkpoint root '{normalized_model_path}'")
    else:
        print(f"🔍 Checking model format: {normalized_model_path}")
    
    # Reuse an existing HuggingFace conversion if complete
    if os.path.isdir(normalized_model_path):
        root_hf = os.path.join(normalized_model_path, 'huggingface')
        actor_hf = os.path.join(normalized_model_path, 'actor', 'huggingface')
        for candidate in (actor_hf, root_hf):
            if os.path.isdir(candidate):
                hf_files = set(os.listdir(candidate))
                has_config = 'config.json' in hf_files
                has_weights = any(
                    name.startswith('pytorch_model') or name.startswith('model.safetensors')
                    for name in hf_files
                )
                if has_config and has_weights:
                    print(f"✅ Found complete HuggingFace conversion, reusing: {candidate}")
                    # Detect LoRA adapter in this directory
                    has_lora, lora_path, lora_rank = detect_lora_adapter(candidate)
                    lora_config = {"has_lora": has_lora, "lora_path": lora_path, "lora_rank": lora_rank}
                    return candidate, lora_config
    
    # If no HF conversion exists, try to convert VERL checkpoint
    # print(f"🔄 Converting VERL checkpoint to HuggingFace format...")
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # converter_script = os.path.join(script_dir, 'convert_fsdp_checkpoint.py')
    
    # if not os.path.exists(converter_script):
    #     raise FileNotFoundError(f"❌ Converter script not found: {converter_script}")
    
    # # Try to convert
    # try:
    #     subprocess.run(
    #         ['python', converter_script, normalized_model_path],
    #         check=True,
    #         capture_output=False
    #     )
    # except subprocess.CalledProcessError as e:
    #     raise RuntimeError(f"❌ Failed to convert checkpoint: {e}")
    
    # Return the HuggingFace directory
    hf_dir = os.path.join(normalized_model_path, 'huggingface')
    if os.path.exists(hf_dir):
        print(f"✅ Conversion successful: {hf_dir}")
        # Detect LoRA adapter in the converted directory
        has_lora, lora_path, lora_rank = detect_lora_adapter(hf_dir)
        lora_config = {"has_lora": has_lora, "lora_path": lora_path, "lora_rank": lora_rank}
        return hf_dir, lora_config

    raise RuntimeError(f"❌ Conversion completed but HuggingFace directory not found: {hf_dir}")


def load_model_vllm(
    model_path: str,
    max_model_len: int = 8192,
    tensor_parallel_size: int = 0,
) -> tuple[LLM, dict]:
    """
    Load model with vLLM for fast batch inference.

    Returns:
        tuple: (llm, lora_config)
            - llm: vLLM LLM instance
            - lora_config: LoRA configuration dict
    """
    print("=" * 70)
    print("📦 Loading Model with vLLM")
    print("=" * 70)
    print(f"   Model checkpoint: {model_path}")

    # Prepare model for vLLM and get LoRA configuration
    vllm_model_path, lora_config = prepare_model_for_vllm(model_path)

    # Configure vLLM parameters based on LoRA detection
    if tensor_parallel_size <= 0:
        tensor_parallel_size = max(torch.cuda.device_count(), 1)
    llm_kwargs = {
        "model": vllm_model_path,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "enforce_eager": True,  # Avoid compilation issues with converted checkpoints
        "tensor_parallel_size": tensor_parallel_size,
    }

    # Add LoRA parameters if LoRA adapter is detected
    if lora_config["has_lora"]:
        print(f"   🎯 LoRA adapter detected! Configuring vLLM for LoRA support...")
        print(f"      LoRA rank: {lora_config['lora_rank']}")
        print(f"      LoRA path: {lora_config['lora_path']}")

        llm_kwargs.update({
            "enable_lora": True,
            "max_lora_rank": lora_config["lora_rank"]  # Critical: match checkpoint's LoRA rank
        })
    else:
        print(f"   📋 No LoRA adapter detected, loading as standard model")

    print(f"   Loading with vLLM...")
    try:
        llm = LLM(**llm_kwargs)
        if lora_config["has_lora"]:
            print(f"   ✅ Model loaded successfully with LoRA support (rank={lora_config['lora_rank']})")
        else:
            print(f"   ✅ Model loaded successfully with vLLM")
        return llm, lora_config
    except Exception as e:
        if lora_config["has_lora"] and "max_lora_rank" in str(e):
            print(f"   ❌ LoRA loading failed - likely rank mismatch: {e}")
            print(f"   💡 Try adjusting max_lora_rank parameter or check LoRA configuration")
        raise


def load_prompts_from_parquet(parquet_path: str, all_api_json: dict = None) -> tuple[List[str], List[Dict]]:
    """Load prompts directly from a parquet file.
    
    The parquet file should have:
    - 'prompt': The full prompt text
    - 'extra_info': JSON string containing 'tool_name' and other metadata
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        tuple: (prompts, tool_metadata)
            - prompts: List of prompt strings
            - tool_metadata: List of metadata dicts for each prompt
    """
    print(f"📂 Loading prompts from parquet: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"   ✅ Loaded {len(df)} rows from parquet")
    
    prompts = []
    extra_infos = []
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        
        # Parse extra_info to get tool_name and other metadata
        extra_info = {}
        if 'extra_info' in row and row['extra_info']:
            if isinstance(row['extra_info'], str):
                extra_info = json.loads(row['extra_info'])
            elif isinstance(row['extra_info'], dict):
                extra_info = row['extra_info']
        
        tool_name = extra_info['tool_name']
        server_name = extra_info['server_name']

        key = (standardize(server_name), tool_name)

        if all_api_json:
            if key not in all_api_json:
                print(f"   ⚠️  Tool {key} not found in all api.json")
                continue
        
        # original description is already in the prompt but not in extra_info

        # Get original description if available (from ground_truth or extra_info)
        # original_description = extra_info['original_description']
        
        # Get train queries if available
        train_queries = extra_info['train_queries']
        
        prompts.append(prompt)

        extra_infos.append({
            'tool_name': tool_name,
            'server_name': server_name,
            # 'original_description': original_description,
            'num_examples': len(train_queries) if train_queries else 0,
            'train_queries': train_queries,
            'extra_info': extra_info  # Keep full extra_info for debugging
        })
    
    print(f"   ✅ Prepared {len(prompts)} prompts with metadata")
    return prompts, extra_infos


def prepare_prompts_for_tools(
    tools_data: List[Dict],
    prompt_template: str,
    num_examples: int = 5,
    mcp_tools: Dict[str, Dict] = None,
    start_token: str = "<|extra_0|>",
    end_token: str = "<|extra_1|>",
) -> tuple[List[str], List[Dict]]:
    """Prepare prompts for all tools in batch.
    
    Returns:
        tuple: (prompts, tool_metadata)
            - prompts: List of formatted prompts
            - tool_metadata: List of metadata dicts for each tool
    """
    prompts = []
    metadata = []
    
    for tool_data in tools_data:
        tool_name = tool_data['tool_name']
        server_name = tool_data['server_name']
        train_queries = tool_data['train_queries']
        original_description = tool_data['original_description']
        
        # Select example queries
        if len(train_queries) <= num_examples:
            example_queries = train_queries
        else:
            example_queries = random.sample(train_queries, num_examples)
        
        # Format query examples
        query_examples_text = "\n".join([
            f"{i+1}. Query: {q['query_text']}\n   Subtask: {q['subtask_text']}"
            for i, q in enumerate(example_queries)
        ])

        # Extract parameter JSON for v3 templates (if MCP YAML files provided)
        parameter_json = ""
        if mcp_tools:
            parameter_json = extract_parameter_json(server_name, tool_name, mcp_tools)

        # Create prompt (handle both v1/v2 and v3 templates)
        template_values = {
            "tool_name": tool_name,
            "query_examples": query_examples_text,
            "original_description": original_description,
            "start_token": start_token,
            "end_token": end_token,
        }

        # Add parameter JSON for v3 templates
        if parameter_json:
            template_values["parameter_json"] = parameter_json

        # Format template with only required fields to avoid KeyError
        try:
            prompt = prompt_template.format(**template_values)
        except KeyError as e:
            # Fallback for templates that might need parameter_json but it's empty
            template_values["parameter_json"] = "{}"
            prompt = prompt_template.format(**template_values)
        
        prompts.append(prompt)
        metadata.append({
            'tool_name': tool_name,
            'original_description': original_description,
            'num_examples': len(example_queries),
            'train_queries': train_queries
        })
    
    return prompts, metadata


def generate_raw_outputs_batch(
    llm: LLM,
    prompts: List[str],
    lora_config: dict = None,
    temperature: float = 0.3,
    top_p: float = 0.8,
    top_k: int = 0,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 512,
    num_samples: int = 1,
    seed: Optional[int] = None,
) -> List[List[str]]:
    """Generate raw outputs for all tools in batch using vLLM.

    Args:
        llm: vLLM LLM instance
        prompts: List of formatted prompts
        lora_config: LoRA configuration dict (optional)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_new_tokens: Maximum new tokens to generate
        num_samples: Number of samples per prompt (SamplingParams.n)
        seed: Random seed for vLLM sampling

    Returns:
        List of samples, each containing a list of raw outputs per prompt.
    """
    print("")
    print("=" * 70)
    print("🚀 Generating Raw Outputs (Batched with vLLM)")
    print("=" * 70)
    print(f"   Total tools: {len(prompts)}")
    print(f"   Temperature: {temperature}")
    print(f"   Top-p: {top_p}")
    print(f"   Top-k: {top_k}")
    print(f"   Repetition penalty: {repetition_penalty}")
    print(f"   Max new tokens: {max_new_tokens}")
    print("")
    
    # Create sampling params
    # Note: Don't set stop_token_ids explicitly - let vLLM use tokenizer's EOS token
    # The model's generation_config.json has eos_token_id=[151645, 151643] but vLLM
    # handles this correctly by default (uses 151645, not 151643 which is BOS/PAD)
    sampling_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        # Don't override stop_token_ids - let vLLM handle it automatically
    }
    # Only add top_k if it's > 0 (0 means disabled)
    if top_k > 0:
        sampling_kwargs["top_k"] = top_k
    if num_samples > 1:
        sampling_kwargs["n"] = num_samples
    if seed is not None:
        sampling_kwargs["seed"] = seed
    sampling_params = SamplingParams(**sampling_kwargs)
    
    # Generate all descriptions in batch
    print("   Generating descriptions...")
    if lora_config and lora_config.get("has_lora", False):
        # Create LoRA request for generation
        lora_request = LoRARequest(
            lora_name="policy_lora",
            lora_int_id=1,
            lora_path=lora_config["lora_path"]
        )
        print(f"   🎯 Using LoRA adapter: {lora_config['lora_path']}")
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        print("   📋 Using base model (no LoRA)")
        outputs = llm.generate(prompts, sampling_params)
    
    if not outputs:
        return []

    min_samples = min(len(output.outputs) for output in outputs)
    if min_samples < num_samples:
        print(f"   ⚠️  Requested {num_samples} samples but got {min_samples}; using {min_samples}")
        num_samples = min_samples

    raw_outputs_per_sample = []
    for sample_idx in range(num_samples):
        raw_outputs = [output.outputs[sample_idx].text for output in outputs]
        raw_outputs_per_sample.append(raw_outputs)

    return raw_outputs_per_sample


def process_raw_outputs(
    raw_outputs: List[str],
    extra_infos: List[Dict],
    debug_log_dir: str = None,
    max_workers: int = 4,
    start_token: str = "<|extra_0|>",
    end_token: str = "<|extra_1|>",
    use_llm_extraction: bool = False,
    force_replace: bool = False,
) -> tuple[List[str], List[Optional[str]], Dict]:
    """Process raw model outputs into descriptions/reasonings with extraction."""
    descriptions = [None] * len(raw_outputs)
    reasonings = [None] * len(raw_outputs)
    consolidated_debug = {}
    failed_tools = []

    if debug_log_dir:
        os.makedirs(debug_log_dir, exist_ok=True)

    def process_single_output(args: Tuple[int, str, Dict]) -> Tuple[int, str, str, Dict, bool]:
        idx, raw_output, extra_info = args
        tool_name = extra_info['tool_name']

        json_desc, json_reasoning = extract_json_description_and_reasoning(raw_output)

        # Extract improved description from structured output
        try:
            extracted_desc = extract_improved_description(
                raw_output,
                start_token=start_token,
                end_token=end_token,
                use_llm_extraction=use_llm_extraction,
            )
        except Exception:
            extracted_desc = None

        extracted_reasoning = None
        if json_reasoning and json_desc and extracted_desc:
            if json_desc.strip() == extracted_desc.strip():
                extracted_reasoning = json_reasoning

        is_success = extracted_desc and len(extracted_desc.strip()) >= 10
        used_force_replace = False

        # If extraction failed but force_replace is enabled, use raw output
        if not is_success and force_replace and raw_output and len(raw_output.strip()) > 0:
            extracted_desc = raw_output.strip()
            is_success = True
            used_force_replace = True

        debug_info = {
            'tool_name': tool_name,
            'server_name': extra_info.get('server_name', ''),
            'original_description': extra_info.get('original_description', ''),
            'raw_model_output': raw_output,
            'num_train_queries': len(extra_info.get('train_queries', [])),
            'extracted_desc': extracted_desc if is_success else None,
            'extracted_reasoning': extracted_reasoning,
        }

        if debug_log_dir:
            safe_tool_name = tool_name.replace("/", "_").replace("\\", "_")
            if is_success:
                debug_log = os.path.join(debug_log_dir, f"{safe_tool_name}_extracted.txt")
                with open(debug_log, 'w') as f:
                    f.write(extracted_desc)
            else:
                failure_log = os.path.join(debug_log_dir, f"{safe_tool_name}_GENERATION_FAILED.txt")
                with open(failure_log, 'w') as f:
                    f.write(f"GENERATION FAILED FOR: {tool_name}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Raw output:\n{raw_output}\n\n")
                    f.write(f"Extracted: {extracted_desc}\n")

        return (idx, tool_name, extracted_desc, extracted_reasoning, debug_info, is_success, used_force_replace)

    process_args = [(i, raw_output, extra_info) for i, (raw_output, extra_info) in enumerate(zip(raw_outputs, extra_infos))]
    max_workers = min(max_workers, len(raw_outputs)) if raw_outputs else 1
    print(f"   🔄 Processing {len(raw_outputs)} outputs with {max_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(process_single_output, args): args[0] for args in process_args}
        completed = 0
        for future in as_completed(future_to_idx):
            completed += 1
            try:
                idx, tool_name, extracted_desc, extracted_reasoning, debug_info, is_success, used_force_replace = future.result()

                if is_success:
                    descriptions[idx] = extracted_desc
                    reasonings[idx] = extracted_reasoning
                    consolidated_debug[tool_name] = {
                        'tool_name': tool_name,
                        'server_name': debug_info['server_name'],
                        'original_description': debug_info['original_description'],
                        'generated_description': extracted_desc,
                        'generated_reasoning': extracted_reasoning,
                        'raw_model_output': debug_info['raw_model_output'],
                        'num_train_queries': debug_info['num_train_queries'],
                        'used_force_replace': used_force_replace
                    }
                    if used_force_replace:
                        print(f"   ⚡ [{completed}/{len(raw_outputs)}] {tool_name}: Used raw output (force_replace) {len(extracted_desc)} chars")
                    else:
                        print(f"   ✅ [{completed}/{len(raw_outputs)}] {tool_name}: Generated {len(extracted_desc)} chars")
                else:
                    descriptions[idx] = None
                    reasonings[idx] = None
                    failed_tools.append(tool_name)
                    raw_len = len(debug_info['raw_model_output'])
                    ext_len = len(extracted_desc) if extracted_desc else 0
                    print(f"   ⚠️  [{completed}/{len(raw_outputs)}] {tool_name}: Generation failed or too short")
                    print(f"        Raw output length: {raw_len} chars")
                    print(f"        Extracted length: {ext_len} chars")
                    print(f"        Raw output preview: {debug_info['raw_model_output'][:200]}...")
            except Exception as e:
                idx = future_to_idx[future]
                tool_name = extra_infos[idx]['tool_name']
                print(f"   ❌ [{completed}/{len(raw_outputs)}] {tool_name}: Exception during processing: {e}")
                descriptions[idx] = None
                reasonings[idx] = None
                failed_tools.append(tool_name)

    print("")
    print(f"   ✅ Batch processing complete")
    print(f"   Successful: {len(descriptions) - len(failed_tools)}/{len(descriptions)}")
    print(f"   Failed: {len(failed_tools)}/{len(descriptions)}")

    return descriptions, reasonings, consolidated_debug


def merge_single_target_mcp_stb(
    target_mcp: Dict,
    generated_descriptions: Dict[str, str],
    generated_reasonings: Dict[str, str],
    output_path: str
):
    """Merge generated descriptions with single target MCP YAML for STB format."""
    updated_count = 0
    
    new_mcp = copy.deepcopy(target_mcp)

    server_name = list(new_mcp['mcp_servers'].keys())[0]
    server_name_standardized = standardize(server_name)

    # Tools are nested under mcp_servers -> server_name -> tools
    server_data = new_mcp['mcp_servers'][server_name]
    
    for tool in server_data['tools']:
        tool_name = tool['tool_name']
        metadata = tool['_metadata']

        if (server_name_standardized, tool_name) in generated_descriptions:
            tool['description'] = generated_descriptions[(server_name_standardized, tool_name)]
            metadata['guidelines_applied'] = None
            metadata['improvement_method'] = "inference"
            metadata['improvement_source'] = None
            metadata['improvement_stage'] = "inference"
            metadata['improvement_timestamp'] = None
            if (server_name_standardized, tool_name) in generated_reasonings:
                metadata['reasoning'] = generated_reasonings[(server_name_standardized, tool_name)]
            else:
                metadata['reasoning'] = metadata.get('reasoning', "")

            updated_count += 1

    # save the new mcp to the output path if there are any updates
    if updated_count > 0:
        with open(output_path, 'w') as f:
            yaml.dump(new_mcp, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return new_mcp


#TODO: fix these two functions to work with TMDB and Spotify formats
def merge_single_target_mcp_tmdb(
    target_mcp: Dict,
    generated_descriptions: Dict[str, str],
    generated_reasonings: Dict[str, str],
    output_path: str
):
    """Merge generated descriptions with TMDB-format MCP YAML.
    
    Note: For TMDB, server_name in parquet may differ from YAML (e.g., 'tmdb' vs 'tmdb_d1').
    We match by tool_name only since TMDB has a single server.
    """
    updated_count = 0
    new_mcp = copy.deepcopy(target_mcp)

    # Build a tool_name -> description mapping (ignoring server_name for TMDB)
    tool_name_to_desc = {}
    tool_name_to_reasoning = {}
    for (server_name, tool_name), desc in generated_descriptions.items():
        tool_name_to_desc[tool_name] = desc
        if (server_name, tool_name) in generated_reasonings:
            tool_name_to_reasoning[tool_name] = generated_reasonings[(server_name, tool_name)]

    for server_name, server_data in new_mcp.get('mcp_servers', {}).items():
        tools = server_data.get('tools', [])

        for tool in tools:
            tool_name = tool.get('tool_name')
            if tool_name in tool_name_to_desc:
                tool['description'] = tool_name_to_desc[tool_name]

                # Ensure metadata reflects inference update while preserving TMDB-specific fields
                metadata = tool.get('_metadata', {})
                # Preserve TMDB-specific fields: endpoint, method, category, dataset
                # Update inference-related fields
                metadata['improvement_method'] = "inference"
                metadata['improvement_stage'] = "inference"
                metadata['improvement_source'] = None
                metadata['improvement_timestamp'] = None
                metadata['guidelines_applied'] = None
                metadata['reasoning_file'] = None
                if tool_name in tool_name_to_reasoning:
                    metadata['reasoning'] = tool_name_to_reasoning[tool_name]
                tool['_metadata'] = metadata

                updated_count += 1

    if updated_count > 0:
        # Override the output path: name YAML file after the folder for exp tracking
        # e.g., /path/to/folder_name/tmdb_d1.yaml -> /path/to/folder_name/folder_name.yaml
        output_dir = os.path.dirname(output_path)
        folder_name = os.path.basename(output_dir)
        output_path = os.path.join(output_dir, f"{folder_name}.yaml")
        with open(output_path, 'w') as f:
            yaml.dump(new_mcp, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"   💾 Saved YAML: {output_path}")

    return new_mcp


def merge_single_target_mcp(
    target_mcp: Dict,
    generated_descriptions: Dict[str, str],
    generated_reasonings: Dict[str, str],
    output_path: str,
    dataset_name: str = "STB"
):
    if dataset_name == "STB":
        return merge_single_target_mcp_stb(target_mcp, generated_descriptions, generated_reasonings, output_path)
    elif dataset_name in ["TMDB", "spotify"]:
        return merge_single_target_mcp_tmdb(target_mcp, generated_descriptions, generated_reasonings, output_path)


def merge_with_target_mcp(
    target_mcp_path: str,
    generated_descriptions: Dict[str, str],
    generated_reasonings: Dict[str, str],
    output_path: str,
    max_workers: int = 8,
    dataset_name: str = "STB"
):
    """Merge generated descriptions with base MCP YAML."""
    print("")
    print("=" * 70)
    print("🔀 Merging with Base MCP YAML")
    print("=" * 70)
    print(f"   Merge target: {target_mcp_path}")
    print(f"   Generated descriptions: {len(generated_descriptions)}")

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # check if target_mcp_path is a file or a directory
    if os.path.isfile(target_mcp_path):
        target_mcp = yaml.safe_load(open(target_mcp_path, 'r'))
        new_mcp_path = os.path.join(output_path, os.path.basename(target_mcp_path))
        merge_single_target_mcp(target_mcp, generated_descriptions, generated_reasonings, new_mcp_path, dataset_name=dataset_name)
    
    else:
        # Collect all YAML files to process
        yaml_files_to_process = []
        for folder in os.listdir(target_mcp_path):
            folder_path = os.path.join(target_mcp_path, folder)
            if not os.path.isdir(folder_path):
                continue
            for file in os.listdir(folder_path):
                if file.endswith('.yaml'):
                    yaml_files_to_process.append((folder, file))
        
        print(f"   🔄 Processing {len(yaml_files_to_process)} YAML files with {min(max_workers, len(yaml_files_to_process))} parallel workers...")
        
        def process_single_yaml(args: Tuple[str, str]) -> Tuple[str, str, bool]:
            """Process a single YAML file."""
            folder, file = args
            try:
                source_path = os.path.join(target_mcp_path, folder, file)
                target_mcp = yaml.safe_load(open(source_path, 'r'))
                new_mcp_path = os.path.join(output_path, folder, file)

                # Create output folder if needed
                output_folder = os.path.join(output_path, folder)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder, exist_ok=True)

                # copy the source mcp to the output path first
                shutil.copy(source_path, new_mcp_path)
                
                merge_single_target_mcp(target_mcp, generated_descriptions, generated_reasonings, new_mcp_path, dataset_name=dataset_name)
                return (folder, file, True)
            except Exception as e:
                print(f"   ❌ Failed to process {folder}/{file}: {e}")
                return (folder, file, False)
        
        # Process YAML files in parallel
        successful = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=min(max_workers, len(yaml_files_to_process))) as executor:
            futures = {executor.submit(process_single_yaml, args): args for args in yaml_files_to_process}
            for future in as_completed(futures):
                folder, file, success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
        
        print(f"   ✅ Merged {successful} YAML files successfully" + (f", {failed} failed" if failed > 0 else ""))
    
    
def main():
    parser = argparse.ArgumentParser(description="Generate tool descriptions with vLLM")
    parser.add_argument("--model-checkpoint", required=True, help="Path to model checkpoint")
    # parser.add_argument("--tool-grouped-file", default=None, help="Path to tool_grouped.json (not needed if using --prompts-parquet)")
    parser.add_argument("--prompts-parquet", default=None, help="Path to parquet file with pre-built prompts (alternative to tool-grouped-file)")
    
    parser.add_argument("--merge-target-yaml", default=None, help="Path to YAML to merge into (if different from base-mcp-yaml)")
    parser.add_argument("--output-mcp-yaml", required=True, help="Path to output MCP YAML")

    # Debugging arguments
    # parser.add_argument("--debug-log-dir", default=None, help="Directory to save debug logs")

    # parser.add_argument("--prompt-path", default=None, help="Path to prompt template (not needed if using --prompts-parquet)")
    # parser.add_argument("--base-mcp-yaml", required=True, help="Path to base MCP YAML (default merge target)")
    parser.add_argument("--dataset-name", default="STB", help="Name of the dataset")
    parser.add_argument("--all-api-json", default=None, help="Path to all api.json file")

    parser.add_argument("--max-model-len", type=int, default=8192, help="Max model length for vLLM")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="Tensor parallel size for vLLM (0 = use all visible GPUs)",
    )
    parser.add_argument("--num-examples", type=int, default=5, help="Number of example queries per tool")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 to disable)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens to generate")
    parser.add_argument("--mcp-yaml-files", default="", help="Path(s) to MCP YAML files for parameter extraction (v3 prompts)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    parser.add_argument("--max-workers", type=int, default=4, help="Max number of workers for parallel processing")
    parser.add_argument("--start-token", default="<|extra_0|>", help="Start token wrapping the model output")
    parser.add_argument("--end-token", default="<|extra_1|>", help="End token wrapping the model output")
    parser.add_argument("--use-llm-extraction", action="store_true", help="Use LLM to extract description (slower but more robust)")
    parser.add_argument("--force-replace", action="store_true", help="When extraction fails, use raw generated text as description instead of marking as failed")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples per prompt (vLLM SamplingParams.n)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for vLLM sampling (set to -1 to disable)")

    args = parser.parse_args()
    
    # Validate arguments
    use_parquet = args.prompts_parquet is not None
    if use_parquet:
        if not os.path.exists(args.prompts_parquet):
            print(f"❌ Error: Parquet file not found: {args.prompts_parquet}")
            sys.exit(1)
    if args.num_samples < 1:
        print("❌ Error: --num-samples must be >= 1")
        sys.exit(1)
    if args.seed is not None and args.seed < 0:
        args.seed = None

    print("")
    print("=" * 70)
    print("🚀 Tool Description Generation Pipeline (vLLM)")
    print("=" * 70)
    print(f"Model checkpoint: {args.model_checkpoint}")
    if use_parquet:
        print(f"Prompts parquet: {args.prompts_parquet}")
    else:
        print(f"Tool grouped file: {args.tool_grouped_file}")
        print(f"Prompt template: {args.prompt_path}")
    

    if args.merge_target_yaml:
        print(f"Merge target YAML: {args.merge_target_yaml} (overriding base)")
    print(f"Output MCP YAML: {args.output_mcp_yaml}")
    if args.mcp_yaml_files:
        print(f"MCP YAML files: {args.mcp_yaml_files}")
    print("")

    # Load model first (same for both modes)
    llm, lora_config = load_model_vllm(
        args.model_checkpoint,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    if args.all_api_json:
        all_api_json_raw = json.load(open(args.all_api_json, 'r'))
        all_api_json = {}
        for key, value in all_api_json_raw.items():
            # Parse tuple string like "(a, b)" manually since values aren't quoted
            key_stripped = key.strip()[1:-1]  # Remove parentheses
            parts = [p.strip() for p in key_stripped.split(',', 1)]  # Split by first comma
            all_api_json[tuple(parts)] = value
    else:
        all_api_json = None


    if use_parquet:
        # Load prompts directly from parquet
        prompts, extra_infos = load_prompts_from_parquet(args.prompts_parquet, all_api_json)
        tools_data = None  # Not needed in parquet mode

    else:
        raise ValueError("Parquet file is required for tool description generation")

    if args.debug:
        prompts = prompts[:5]
        extra_infos = extra_infos[:5]
        print(f"Prompts: {prompts}")
        print(f"Extra infos: {extra_infos}")
        base_output_path = args.output_mcp_yaml + "_debug"
    else:
        base_output_path = args.output_mcp_yaml

    set_random_seed(args.seed)

    raw_outputs_per_sample = generate_raw_outputs_batch(
        llm,
        prompts,
        lora_config=lora_config,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    effective_num_samples = len(raw_outputs_per_sample)
    if effective_num_samples == 0:
        print("❌ Error: No outputs generated")
        sys.exit(1)

    merge_target = args.merge_target_yaml
    sample_success_counts = []

    for sample_idx, raw_outputs in enumerate(raw_outputs_per_sample, start=1):
        if effective_num_samples > 1:
            sample_output_path = os.path.join(base_output_path, f"sample_{sample_idx:02d}")
        else:
            sample_output_path = base_output_path
        os.makedirs(sample_output_path, exist_ok=True)

        inference_config_path = os.path.join(sample_output_path, "inference_config.json")
        inference_config = vars(args).copy()
        inference_config["output_mcp_yaml"] = sample_output_path
        inference_config["sample_index"] = sample_idx
        inference_config["num_samples_effective"] = effective_num_samples
        with open(inference_config_path, 'w') as f:
            json.dump(inference_config, f, indent=2)
        print(f"   💾 Saved inference config: {inference_config_path}")

        debug_log_dir = os.path.join(sample_output_path, "debug")

        print("")
        print("=" * 70)
        print(f"📦 Processing sample {sample_idx}/{effective_num_samples}")
        print("=" * 70)

        descriptions, reasonings, consolidated_debug = process_raw_outputs(
            raw_outputs,
            extra_infos,
            debug_log_dir=debug_log_dir,
            max_workers=args.max_workers,
            start_token=args.start_token,
            end_token=args.end_token,
            use_llm_extraction=args.use_llm_extraction,
            force_replace=args.force_replace,
        )

        generated_descriptions = {}
        generated_reasonings = {}
        for desc, reasoning, extra_info in zip(descriptions, reasonings, extra_infos):
            if desc is not None:
                key = (standardize(extra_info['server_name']), extra_info['tool_name'])
                generated_descriptions[key] = desc
                if reasoning:
                    generated_reasonings[key] = reasoning

        if args.debug:
            consolidated_path = os.path.join(sample_output_path, "consolidated_debug.json")
            with open(consolidated_path, 'w') as f:
                json.dump(consolidated_debug, f, indent=2)
            print(f"   💾 Saved consolidated debug: {consolidated_path} in {sample_output_path}")

        merge_with_target_mcp(
            merge_target,
            generated_descriptions,
            generated_reasonings,
            output_path=sample_output_path,
            max_workers=args.max_workers,
            dataset_name=args.dataset_name
        )

        sample_success_counts.append(len(generated_descriptions))
    
    print("")
    print("=" * 70)
    print("✅ Generation Complete")
    print("=" * 70)
    print(f"Output MCP YAML: {base_output_path}")
    total_count = len(tools_data) if tools_data else len(extra_infos)
    if effective_num_samples > 1:
        for sample_idx, success_count in enumerate(sample_success_counts, start=1):
            print(f"Sample {sample_idx:02d}: {success_count}/{total_count}")
    else:
        print(f"Successful: {sample_success_counts[0]}/{total_count}")
    print("")


if __name__ == "__main__":
    main()

