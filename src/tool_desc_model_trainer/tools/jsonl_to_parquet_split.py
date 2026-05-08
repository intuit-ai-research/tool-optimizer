#!/usr/bin/env python3
"""
Convert JSON/JSONL with prompt/completion to Parquet and split train/val.

Can be used as a module or CLI tool.
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def convert_jsonl_to_parquet(
    input_jsonl: str,
    train_parquet: str,
    val_parquet: str,
    prompt_key: str = "prompt",
    completion_key: str = "ground_truth",
    label_key: str = "label",
    filter_positive: bool = False,
    val_ratio: float = 0.2,
    seed: int = 42,
    debug: bool = False,
) -> Tuple[Path, Path, int, int]:
    """
    Convert JSON/JSONL file to train/val parquet files.
    
    Args:
        input_jsonl: Path to input JSON/JSONL file
        train_parquet: Output path for train parquet
        val_parquet: Output path for val parquet
        prompt_key: Key name for prompt in JSON objects
        completion_key: Key name for completion in JSON objects
        label_key: Key name for label in JSON objects
        filter_positive: If True, only keep samples with positive labels
        val_ratio: Validation split ratio (default 0.2)
        seed: Random seed for shuffling
        debug: If True, limit number of records for quick checks
    
    Returns:
        Tuple of (train_parquet_path, val_parquet_path, train_count, val_count)
    """
    if not os.path.isfile(input_jsonl):
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    records = []
    total_records = 0
    filtered_records = 0
    
    # Try to detect format: JSON array vs JSONL
    with open(input_jsonl, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # JSON array format (used by tool-level dataset)
            print("📋 Detected JSON array format")
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array format but got different type")
            
            for obj in data:
                total_records += 1
                
                # Check if we should filter based on label
                if filter_positive and label_key in obj:
                    label = obj[label_key]
                    if not label:  # Skip samples with label=False
                        filtered_records += 1
                        continue
                
                # Extract prompt and completion
                prompt = obj.get(prompt_key, "")
                completion = obj.get(completion_key, "")
                
                # Skip if either is empty
                if not prompt or not completion:
                    filtered_records += 1
                    continue
                
                # Extract additional fields for VERL
                data_source = obj.get('data_source', '')
                ability = obj.get('ability', '')
                reward_model = obj.get('reward_model', {})
                extra_info = obj.get('extra_info', {})
                parameters = obj.get('parameters', {})
                
                # Serialize dicts to JSON strings to avoid PyArrow empty struct issues
                if isinstance(reward_model, dict):
                    reward_model = json.dumps(reward_model) if reward_model else ""
                if isinstance(extra_info, dict):
                    extra_info = json.dumps(extra_info) if extra_info else ""
                if isinstance(parameters, dict):
                    parameters = json.dumps(parameters) if parameters else ""
                
                # Keep the original completion key name (e.g., 'ground_truth') for VERL
                records.append({
                    "prompt": prompt, 
                    "ground_truth": completion,
                    "data_source": data_source,
                    "ability": ability,
                    "reward_model": reward_model,
                    "extra_info": extra_info,
                    "parameters": parameters,
                })
                
                if debug and len(records) >= 50:
                    break
        else:
            # JSONL format (original behavior)
            print("📋 Detected JSONL format")
            for line in f:
                if not line.strip():
                    continue
                    
                obj = json.loads(line)
                total_records += 1
                
                # Check if we should filter based on label
                if filter_positive and label_key in obj:
                    label = obj[label_key]
                    if not label:  # Skip samples with label=False
                        filtered_records += 1
                        continue
                
                # Extract fields
                prompt = obj.get(prompt_key, "")
                ground_truth = obj.get(completion_key, "")
                
                # Skip if either is empty
                if not prompt or not ground_truth:
                    filtered_records += 1
                    continue
                
                data_source = obj.get('data_source', '')
                ability = obj.get('ability', '')
                reward_model = obj.get('reward_model', {})
                extra_info = obj.get('extra_info', {})
                parameters = obj.get('parameters', {})
                
                # Serialize dicts to JSON strings to avoid PyArrow empty struct issues
                if isinstance(reward_model, dict):
                    reward_model = json.dumps(reward_model) if reward_model else ""
                if isinstance(extra_info, dict):
                    extra_info = json.dumps(extra_info) if extra_info else ""
                if isinstance(parameters, dict):
                    parameters = json.dumps(parameters) if parameters else ""
                
                records.append({
                    "prompt": prompt, 
                    "ground_truth": ground_truth,  # Keep as 'ground_truth' for VERL compatibility
                    "data_source": data_source, 
                    "ability": ability,
                    "reward_model": reward_model,
                    "extra_info": extra_info,
                    "parameters": parameters,
                })
                
                if debug and len(records) >= 50:
                    break

    if not records:
        raise RuntimeError("No records found in JSONL")
    
    # Print filtering statistics
    if filter_positive:
        kept_records = total_records - filtered_records
        print(f"📊 Label filtering: kept {kept_records}/{total_records} positive samples ({kept_records/total_records*100:.1f}%)")
        print(f"   Filtered out {filtered_records} negative samples for SFT training")
    else:
        print(f"📊 No label filtering: using all {total_records} samples for SFT training")

    random.seed(seed)
    random.shuffle(records)

    if debug:
        records = records[:200]

    if val_ratio <= 0.0:
        # Entire dataset goes to train; keep an empty validation set
        train_records = records
        val_records = []
    else:
        split_idx = max(1, int(len(records) * (1.0 - val_ratio)))
        train_records = records[:split_idx]
        val_records = records[split_idx:]
        # For small datasets, ensure at least one validation sample when val_ratio > 0
        if not val_records:
            val_records = train_records[-1:]
            train_records = train_records[:-1]

    def to_parquet(path: str, rows, schema: Optional[pa.Schema] = None) -> pa.Schema:
        """
        Write rows to Parquet. If rows is empty, use the provided schema to
        produce an empty table with the correct columns.
        """
        if rows:
            table = pa.Table.from_pylist(rows, schema=schema)
            schema = table.schema
        else:
            if schema is None:
                raise ValueError("Schema is required to write an empty Parquet file")
            arrays = [pa.array([], type=field.type) for field in schema]
            table = pa.Table.from_arrays(arrays, schema=schema)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(table, path)
        return schema

    schema = to_parquet(train_parquet, train_records)

    try:
        to_parquet(val_parquet, val_records, schema=schema)
    except Exception as e:
        print(f"Error writing val parquet: {e}")
        print(f"Val records: {val_records}")
        print(f"Schema: {schema}")
        raise e

    print(f"Wrote train: {train_parquet} ({len(train_records)})")
    print(f"Wrote val:   {val_parquet} ({len(val_records)})")
    
    return Path(train_parquet), Path(val_parquet), len(train_records), len(val_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL with prompt/completion to Parquet and split train/val.")
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file with keys: prompt, completion")
    parser.add_argument("--train-parquet", required=True, help="Output train parquet path")
    parser.add_argument("--val-parquet", required=True, help="Output val parquet path")
    parser.add_argument("--prompt-key", default="prompt", help="Key name for prompt in JSON objects")
    parser.add_argument("--completion-key", default="ground_truth", help="Key name for completion in JSON objects")
    parser.add_argument("--label-key", default="label", help="Key name for label in JSON objects")
    parser.add_argument("--filter-positive-only", action="store_true", default=True, help="Only keep samples with positive labels for SFT")
    parser.add_argument("--no-filter-positive", action="store_true", help="Disable positive label filtering (keep all samples)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--debug", action="store_true", help="If set, limit number of records for quick checks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Determine if we should filter to positive labels only
    filter_positive = args.filter_positive_only and not args.no_filter_positive
    
    convert_jsonl_to_parquet(
        input_jsonl=args.input_jsonl,
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        prompt_key=args.prompt_key,
        completion_key=args.completion_key,
        label_key=args.label_key,
        filter_positive=filter_positive,
        val_ratio=args.val_ratio,
        seed=args.seed,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
