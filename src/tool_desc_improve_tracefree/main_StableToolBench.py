#!/usr/bin/env python3
"""
Description Improvement Pipeline.

Usage:
    python main.py --dataset path/to/mcp.yaml --strategy generic_llm_guidelines
    python main.py --config config.yaml
    python main.py --list


D0 --> D1:
python main_StableToolBench.py --mcp_yaml_path eval/StableToolBench  --output_path results/StableToolBench_D1/ --config config/StableToolBench.yaml

D1 --> D2:
python main_StableToolBench.py --mcp_yaml_path results/StableToolBench_D1/  --output_path results/StableToolBench_D2/ --config config/StableToolBench_D2.yaml --eval_results_dir <eval_results_path>
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime
from termcolor import cprint
import json
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import main as main_main
import random
from collections import defaultdict
from tqdm import tqdm
from typing import Tuple

from joblib import Memory
# Specify a directory for the cache
location = 'cache'
MEMORY = Memory(location, verbose=0) # verbose=0 suppresses output

@MEMORY.cache
def load_mcp_yaml(mcp_tool_path):
    path = Path(mcp_tool_path)
    # load the yaml file into a dictionary "category" --> "tool_name" --> yaml
    cate_dict = defaultdict[Tuple[str, str], dict](dict)
    for category_folder in (progress_bar := tqdm(list(path.iterdir()), desc="Loading MCP YAML files")):
        # tqdm with category as progress bar
        for file in category_folder.iterdir():
            # check if the file is a yaml file
            if not str(file).endswith(".yaml"):
                continue
            with open(file, "r") as f:
                yaml_data = yaml.safe_load(f)
            tool_name = list(yaml_data["mcp_servers"].keys())[0]
            cate_name = yaml_data["mcp_servers"][tool_name]["category"]
            cate_dict[(cate_name, tool_name)] = file
    return cate_dict

def list_eval_results_folders(eval_results_dir):
    path = Path(eval_results_dir)
    eval_results_list = []
    for folder in path.iterdir():
        # if is dir, then it is a api provider
        if folder.is_dir():
            eval_results_list.append(folder)
    return eval_results_list

def is_valid_eval_result(eval_result_path):
    # check if file "evaluation_statistics.json" exists
    if not (Path(eval_result_path) / "evaluation_statistics.json").exists():
        return False
    # if summary->total_api_calls is 0, then it is not a valid eval result
    with open(Path(eval_result_path) / "evaluation_statistics.json", "r") as f:
        eval_results = json.load(f)
    if eval_results["summary"]["total_api_calls"] == 0:
        return False
    return True

def find_corresponding_yaml_file(eval_result_path):
    # read "run_parameters.json"
    with open(Path(eval_result_path) / "run_parameters.json", "r") as f:
        run_parameters = json.load(f)
    # use the dataset to find the corresponding query file
    queries_path = Path(run_parameters["dataset"])
    # read the query file
    with open(queries_path, "r") as f:
        queries = json.load(f)
    # find the corresponding query
    query1 = queries[0]
    api1 = query1["api_list"][0]
    category_name = api1["category_name"]
    tool_name = api1["tool_name"]
    return (category_name, tool_name)
    

def main_StableToolBench_D2(args):
    # 1. read all yaml files
    # 2. read the eval results folder
    # 2.1 for each query, find the corresponding yaml file
    # 2.1.1 use the (category_name, tool_name) to find the corresponding yaml file
    # 2.2 if the yaml file is already in the output folder, skip it
    # 2.3 if the yaml file is not in the output folder, run the D2 pipeline


    mcp_yaml_path = Path(args.mcp_yaml_path)
    output_path = Path(args.output_path)

    # read yaml
    all_mcp_yaml = load_mcp_yaml(mcp_yaml_path)
    # read eval results folders
    eval_results_folders = list_eval_results_folders(args.eval_results_dir)
    random.shuffle(eval_results_folders)
    for eval_results_folder in eval_results_folders:
        if is_valid_eval_result(eval_results_folder):
            (category_name, tool_name) = find_corresponding_yaml_file(eval_results_folder)
            if (category_name, tool_name) in all_mcp_yaml:
                yaml_file = all_mcp_yaml[(category_name, tool_name)]
                if (output_path / category_name / yaml_file.name).exists():
                    cprint(f"Skipping {yaml_file.name} because it already exists.", 'yellow')
                    continue
                new_args = argparse.Namespace(
                    config=args.config,
                    strategy=None,
                    dataset=str(yaml_file),
                    output=str(output_path / category_name),
                    keep_filename=True,
                    extract_guidelines=False,
                    list=False,
                    eval_results_path=str(eval_results_folder),
                    openai_api_key=args.openai_api_key,
                    model_name=args.model_name,
                )
                main_main(new_args)
                cprint(f"Processed {(output_path / category_name / yaml_file.name)}.", 'green')


def main_StableToolBench_D1(args):

    mcp_yaml_path = Path(args.mcp_yaml_path)
    output_path = Path(args.output_path)

    for category in mcp_yaml_path.iterdir():
        if category.is_dir():
            # get all yaml files in the category
            yaml_files = list(category.glob('*.yaml'))
            if len(yaml_files) == 0:
                continue
            # shuffle the yaml files
            random.shuffle(yaml_files)
            for yaml_file in yaml_files:
                # check if the yaml file is already in the output folder
                if (output_path / category.name / yaml_file.name).exists():
                    cprint(f"Skipping {yaml_file.name} because it already exists.", 'yellow')
                    continue
                else:
                    new_args = argparse.Namespace(
                        config=args.config,
                        strategy=None,
                        dataset=str(yaml_file),
                        output=str(output_path / category.name),
                        keep_filename=True,
                        extract_guidelines=False,
                        list=False,
                        eval_results_path=None,
                        openai_api_key=args.openai_api_key,
                        model_name=args.model_name,
                    )
                    main_main(new_args)
                    cprint(f"Processed {yaml_file.name}.", 'green')
if __name__ == "__main__":
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Description Improvement Pipeline')

    # StableToolBench parameters
    parser.add_argument('--mcp_yaml_path', type=str, required=True, default=None, help='Path to MCP YAML folder')
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--output_path', type=str, default=f"../../data/StableToolBench/tools_mcp_yaml_desc_improve_tracefree_{date_time}", help='Path to output folder (default: ../../data/StableToolBench/tools_mcp_yaml_desc_improve_tracefree_<timestamp>)')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--eval_results_dir', help='Path to eval results folder for D2')
    parser.add_argument('--model_name', type=str, default="openai:gpt-4.1-2025-04-14",
                        help="Model name for LLM, e.g. 'openai:gpt-4.1-2025-04-14' or 'vllm:Qwen/Qwen2.5-7B-Instruct' (overrides config)")
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    args = parser.parse_args()
    if args.eval_results_dir:
        # when eval_results_path is provided, we are running D2.
        # For D2, we expect the eval_results_dir is the root folder of several eval results folders.
        # In each eval results folder, it is only for one api provider. TODO: assert the api provider.
        # For each query, we first find the corresponding yaml file. Then we run the D2 pipeline.
        # For the rest of mcp servers, we just copy the yaml file to the output folder.
        main_StableToolBench_D2(args)
    else:
        main_StableToolBench_D1(args)

