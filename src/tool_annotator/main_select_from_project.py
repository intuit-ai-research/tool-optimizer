"""
python main_select.py --eval_results_folders ../FunctionWrapper/experiments_track/D0_agent_20251121_193228/* --mcp_yaml_path ../FunctionWrapper/eval/StableToolBench_D0_by_agent ../FunctionWrapper/eval/StableToolBench --tool_root_dir ../FunctionWrapper/StableToolBench/data/toolenv/tools/ --output_folder ../FunctionWrapper/tmp --debug

This script is used to select the health tools.
"""
import argparse
import os, sys
from datetime import datetime
from dotenv import load_dotenv
import json
from collections import defaultdict
from typing import Any, Tuple, List, Dict
from tqdm import tqdm
import yaml
import tempfile
import random
load_dotenv()

from pathlib import Path
import termcolor

import smolagents.agents
smolagents.agents.validate_tool_arguments = lambda tool, arguments: None
from smolagents import ToolCallingAgent, populate_template

print("Successfully bypassed smolagents.agents.validate_tool_arguments")
from tool_annotator.agent.select_tool import (
    AnnotationFinalAnswer,
    AnnotateHealth,
    AnnotateExample
)
from tool_annotator.agent.models import get_model
from tool_annotator.agent.phoenix_utils import PhoenixUtils
from phoenix.otel import register
from openinference.instrumentation import capture_span_context
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

from tool_annotator.agent.FunctionWrapper_args import FunctionWrapper_DIR, TOOLEVAL_DIR, add_FunctionWrapper_args, impute_functionwrapper_args
sys.path.append(FunctionWrapper_DIR)
sys.path.append(TOOLEVAL_DIR)
from tool_annotator.agent.api_tool import build_tools_from_yaml_tools
from utils.exp_meta import build_reproducibility_log
from eval.tmdb.examples.main_tmdb import prepare_data_from_StableToolBench, load_queries, load_tools, ToolManager, create_output_directory, standardize
from evaluators import load_registered_automatic_evaluator

from tool_annotator.main_rewrite import load_mcp_yaml_paths, load_json_paths, load_mcp_logs, prune_tools_in_query
import pandas as pd

from joblib import Memory
# Specify a directory for the cache
location = 'cache'
MEMORY = Memory(location, verbose=0) # verbose=0 suppresses output

def build_mcp_log(trace_df: pd.DataFrame) -> List[Dict[str, Any]]:
    # get tool rows
    tool_rows = trace_df[(trace_df["span_kind"] == "TOOL") & ("FinalAnswerTool" !=trace_df["name"])]
    mcp_log = []
    for index, row in tool_rows.iterrows():
        mcp_log.append({
            "name": row["attributes.tool.name"],
            "arguments": row["attributes.input.value"],
            "response": row["attributes.output.value"],
        })
    return mcp_log

def main():
    parser = argparse.ArgumentParser(description="Run TMDB step-wise evaluation")

    parser = add_FunctionWrapper_args(parser)
    parser.add_argument("--eval_project_name", type=str, help="the project name of the eval result")
    parser.add_argument("--output_folder", type=str, help="the folder to save the output")
    # mcp_yaml_path will be overridden
    args = parser.parse_args()

    # by default, tool call by prompt
    args.tool_call_by_prompt = True
    
    args, queries_path, platform, output_suffix, max_queries, runs_per_scenario = impute_functionwrapper_args(args)
    
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # find all mcp yaml file paths
    all_mcp_yaml_paths = load_mcp_yaml_paths(args.mcp_yaml_path) # dict[(category_name, tool_name)] -> yaml_file
    all_json_paths = load_json_paths(args.tool_root_dir) # dict[(category_name, tool_name)] -> json_file

    # Per each mcp yaml file, we first build a dummy query file such that the tool manager and env can be built
    # Enable Logging and Instrumentation
    if args.debug:
        project_name = f"debug"
    else:
        project_name = f"select_{date_time}"
    log_path = Path("log") / Path(project_name)
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "reproducibility_log.txt", "w") as f:
        f.write(build_reproducibility_log(args))
    with open(log_path / "project_info.json", "w") as f:
        json.dump({"project_name": project_name}, f, indent=2, ensure_ascii=False)
    register(project_name=project_name)
    SmolagentsInstrumentor().instrument()
    phx_utils = PhoenixUtils()
    client = phx_utils.get_client()

    if queries_path is None:
        queries_path = args.dataset
        args_StableToolBench = argparse.Namespace(
            tool_root_dir=args.tool_root_dir,
            method="",
            input_query_file=queries_path,
            output_answer_file=create_output_directory(args, queries_path, output_suffix),
            backbone_model="",
            toolbench_key="lIoNrcOLqpVjwugdHm8B1XR9fnZtzaySbMi0Qk2TUPYCGKeD4E",
            rapidapi_key="",
            use_rapidapi_key=False,
            api_customization="",
            max_observation_length=999999,
            observ_compress_method="",
            retrieved_api_nums=10,
            expand_same_category=args.expand_same_category,
            BM25_threshold=args.BM25_threshold,
            retrieval_sample_size=args.retrieval_sample_size,
        )
        queries = prepare_data_from_StableToolBench(args_StableToolBench, mcp_tool_path=args.mcp_yaml_path, decompo_mcp_tool_path=args.decompo_mcp_yaml_path) # list[tuple[dict, tuple[query, tool_manager, decompo_tool_manager, env]]]
    else:
        queries = load_queries(queries_path)
        # load tools from mcp config
        tools = load_tools(args.mcp_yaml_path)

        decompo_tools = load_tools(args.decompo_mcp_yaml_path)
        tool_manager = ToolManager(tools, platform=platform)
        decompo_tool_manager = ToolManager(decompo_tools, platform=platform)
        queries = [(query, tool_manager, decompo_tool_manager, None) for query in queries]
    query_to_trace_id = phx_utils.extract_query_from_project(args.eval_project_name)

    @MEMORY.cache
    def load_mcp_logs(eval_project_name: str):
        # read all eval result folders and collect all mcp logs, with the corresponding mcp yaml file path
        mcp_logs = defaultdict(list)
        for data_dict, tool_manager, decompo_tool_manager, env in queries:
            if data_dict["query"] not in query_to_trace_id:
                print(termcolor.colored(f"Query {data_dict['query']} not found in the eval project, skipping", "red"))
                continue
            category_name = data_dict['api_list'][0]['category_name']
            tool_name = data_dict['api_list'][0]['tool_name']
            assert len(set([(api['category_name'], api['tool_name']) for api in data_dict['api_list']])) == 1, "Assume one category and one tool for each query"
            trace_id = query_to_trace_id[data_dict["query"]]
            trace_df = phx_utils.collect_trace_df(trace_id, args.eval_project_name)
            mcp_log = build_mcp_log(trace_df)
            mcp_logs[(category_name, tool_name)].append(mcp_log)
        return mcp_logs
    mcp_logs = load_mcp_logs(args.eval_project_name)

    LLM_model = get_model(model_id=args.model_name, tool_call_by_prompt=args.tool_call_by_prompt, api_key=args.openai_api_key)
    for (category_name, tool_name) in tqdm(mcp_logs, desc="Processing mcp yamls"):
        dummy_query = prune_tools_in_query(category_name, tool_name, all_json_paths)
        mcp_log_for_one_api_provider = mcp_logs[(category_name, tool_name)]
        single_mcp_yaml_path = all_mcp_yaml_paths[(category_name, tool_name)]
        # get the folder and file name from the single_mcp_yaml_path
        folder_name = single_mcp_yaml_path.parent.name
        file_name = single_mcp_yaml_path.name
        schema_file_path = Path(args.output_folder) / folder_name / file_name

        if schema_file_path.exists():
            print(termcolor.colored(f"Schema file {schema_file_path} already exists, skipping", "yellow"))
            continue

        queries_path = tempfile.NamedTemporaryFile(delete=False).name
        with open(queries_path, "w") as f:
            json.dump([dummy_query], f, indent=2, ensure_ascii=False)
        args_StableToolBench = argparse.Namespace(
            tool_root_dir=args.tool_root_dir,
            method="",
            input_query_file=queries_path,
            output_answer_file=create_output_directory(args, queries_path, output_suffix),
            backbone_model="",
            toolbench_key="lIoNrcOLqpVjwugdHm8B1XR9fnZtzaySbMi0Qk2TUPYCGKeD4E",
            rapidapi_key="",
            use_rapidapi_key=False,
            api_customization="",
            max_observation_length=999999,
            observ_compress_method="",
            retrieved_api_nums=10,
            expand_same_category=args.expand_same_category,
            BM25_threshold=args.BM25_threshold,
            retrieval_sample_size=args.retrieval_sample_size,
        )
        def load_mcp_yaml_func(mcp_yaml_path):
            with open(mcp_yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
                tool_name = list(yaml_data["mcp_servers"].keys())[0]
                cate_name = yaml_data["mcp_servers"][tool_name]["category"]
            return {cate_name:{standardize(tool_name):yaml_data}}
        
        queries = prepare_data_from_StableToolBench(args_StableToolBench, mcp_tool_path=single_mcp_yaml_path, decompo_mcp_tool_path=single_mcp_yaml_path, load_mcp_yaml_func=load_mcp_yaml_func) # list[tuple[dict, tuple[query, tool_manager, decompo_tool_manager, env]]]

        data_dict, tool_manager, decompo_tool_manager, env = queries[0] # only one query in the list
        # Build smolagents Tool objects from YAML tools (adapter)
        smolagents_tools = build_tools_from_yaml_tools(tool_manager, env, tool_call_by_prompt=args.tool_call_by_prompt)
        number_of_APIs = len(tool_manager.tools)

        # name_to_tool = {tool.name: tool for tool in smolagents_tools}
        # name_to_tool['remove_albums_user']({"ids": "5JNrPPT60TzrqBxsY3hn0A"})
        with open(single_mcp_yaml_path, "r") as f:
            # this schema is mutable and will be modified in place by other tools.
            schema = yaml.safe_load(f)
        with open(Path("prompts") / "select_agent.yaml", "r") as f:
            prompt_template = yaml.safe_load(f)
        
        atomic_tools = [
            AnnotateHealth(schema=schema),
            AnnotateExample(schema=schema)
        ]
        smolagents_tools.extend(atomic_tools)
        max_steps = 1 + 1 + number_of_APIs*4 + 1 + 20 # 1 for AnnotateHealth, 1 for AnnotateExample, number_of_APIs*2 for API calls, 1 for AnnotationFinalAnswer, 20 for the maximum number of steps
        agent = ToolCallingAgent(tools=smolagents_tools, model=LLM_model, prompt_templates=prompt_template, max_steps=max_steps)

        agent.tools["final_answer"] = AnnotationFinalAnswer(schema=schema, schema_file_path=schema_file_path)
        with capture_span_context() as capture:
            agent.run(populate_template(prompt_template["user_prompt"], variables={"schema": json.dumps(schema), "history": mcp_log_for_one_api_provider}))
            first_span_id = capture.get_first_span_id()
        # if first_span_id:
        #     # LLM-as-judge: build payload from spans and annotate PassRate
        #     task_description, answer = phx_utils.build_tooleval_payload(
        #         project_name=project_name, root_span_id=first_span_id, tools=smolagents_tools
        #     )
        print(termcolor.colored(f"Tool calling agent finished for {single_mcp_yaml_path}\nSchema file saved to {schema_file_path}", "green"))
        # break


    print(termcolor.colored(f"Finish", "green"))

if __name__ == "__main__":
    main()
