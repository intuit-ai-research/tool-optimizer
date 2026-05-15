"""
python main_rewrite.py --tool_usage_path ../FunctionWrapper/experiments_track/20251115_031551/StableToolBench/* --mcp_yaml_path ../FunctionWrapper/eval/StableToolBench --tool_root_dir ../FunctionWrapper/StableToolBench/data/toolenv/tools/ --output_folder ../FunctionWrapper/outputs
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
load_dotenv()

from pathlib import Path
from typing import Union
import termcolor

import smolagents.agents
smolagents.agents.validate_tool_arguments = lambda tool, arguments: None
from smolagents import ToolCallingAgent, populate_template

print("Successfully bypassed smolagents.agents.validate_tool_arguments")
from tool_annotator.agent.rewrite_tool import (
    SchemaRewriteFinalAnswer,
    TakeNotes,
    UpdateProviderDescription,
    UpdateAPIDescription,
    UpdateAPIParameters,
    RemoveAPIParameters,
    PrintSchema
)
from tool_annotator.agent.models import get_model
from tool_annotator.agent.phoenix_utils import PhoenixUtils
from phoenix.otel import register
from openinference.instrumentation import capture_span_context
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

from tool_annotator.agent.FunctionWrapper_args import TOOLEVAL_DIR, SUBMODULES_DIR, add_FunctionWrapper_args, impute_functionwrapper_args
sys.path.append(TOOLEVAL_DIR)
from tool_annotator.agent.api_tool import build_tools_from_yaml_tools
from tool_exec_tracer.utils.exp_meta import build_reproducibility_log
from tool_exec_tracer.tmdb.examples.main_tmdb import prepare_data_from_StableToolBench, load_queries, load_tools, ToolManager, create_output_directory, standardize
from evaluators import load_registered_automatic_evaluator

def load_json_paths(json_root_dir):
    json_paths = dict[Tuple[str, str], Path]()
    for category_folder in (progress_bar := tqdm(list(Path(json_root_dir).iterdir()), desc="Loading JSON files")):
        if not category_folder.is_dir():
            continue
        for file in category_folder.iterdir():
            if not str(file).endswith(".json"):
                continue
            with open(file, "r") as f:
                json_data = json.load(f)
            tool_name = json_data["tool_name"]
            cate_name = category_folder.name
            json_paths[(cate_name, tool_name)] = file
    return json_paths

def load_mcp_yaml_paths_helper(mcp_tool_path):
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

def load_mcp_yaml_paths(mcp_tool_path: Union[str, List[str]]):
    if isinstance(mcp_tool_path, list):
        all_mcp_yaml = defaultdict(dict)
        for mcp_tool_path_i in mcp_tool_path[::-1]:
            new_mcp_yaml = load_mcp_yaml_paths_helper(mcp_tool_path_i)
            for key in new_mcp_yaml:
                all_mcp_yaml[key] = new_mcp_yaml[key]
    else:
        all_mcp_yaml = load_mcp_yaml_paths_helper(mcp_tool_path)
    return all_mcp_yaml

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


def load_mcp_logs_under_one_folder(tool_usage_path: str, all_json_paths: Dict[Tuple[str, str], Path]):
    # read logs, along with its query id
    mcp_log_path = Path(tool_usage_path) / "mcp_call_log.jsonl"
    logs_under_one_folder = []
    mcp_logs_under_one_folder = defaultdict[Tuple[str, str], List[Dict]](list)
    with open(mcp_log_path, "r") as f:
        for line in f:
            log = json.loads(line)
            # log: {"timestamp": "2025-11-14T21:45:43.616052", "query_id": 394, "subtask_id": 4, "call_signature": {"api_name": "Video", "parameters": {}, "endpoint": "/video", "method": "GET", "platform": "StableToolBench"}, "response": {"success": false, "error": "Function executing from toolenv.tools.Data.simple_youtube_search.api import video error...\nvideo() missing 1 required positional argument: 'search'", "response": ""}, "metadata": {"response_size": 0, "has_error": true}}
            logs_under_one_folder.append(log)
    # read the queries, and find the query with the same query id. this leads to the category and tool name
    # read "run_parameters.json"
    with open(Path(tool_usage_path) / "run_parameters.json", "r") as f:
        run_parameters = json.load(f)
    # use the dataset to find the corresponding query file
    queries_path = Path(run_parameters["dataset"])
    if not queries_path.is_absolute() and not queries_path.exists():
        # Legacy run_parameters.json files store `dataset` as a path relative to the old
        # FunctionWrapper checkout root (e.g. "StableToolBench/solvable_queries/...").
        # After the repo merge, StableToolBench lives under <repo>/src/submodules/.
        queries_path = Path(SUBMODULES_DIR) / queries_path
    # read the query file
    with open(queries_path, "r") as f:
        queries = json.load(f)
    queries = {query["query_id"]: query for query in queries}
    for log in logs_under_one_folder:
        query_id = log["query_id"]
        api_name = log["call_signature"]["api_name"]
        query = queries[query_id]
        # in the api list, find the api with the same api name
        for api in query["api_list"]:
            if api["api_name"] == api_name:
                (category_name, tool_name) = (api["category_name"], api["tool_name"])
                break
        else:
            raise ValueError(f"API {api_name} not found in query {query_id}")
        mcp_logs_under_one_folder[(category_name, tool_name)].append(log)
    return mcp_logs_under_one_folder


def prune_tools_in_query(category_name: str, tool_name: str, all_json_paths: Dict[Tuple[str, str], Path]):
    # remove the tools and relevant APIs in the query
    json_path = all_json_paths[(category_name, tool_name)]
    with open(json_path, "r") as f:
        json_data = json.load(f)
    MAPPING_FROM_JSON_TO_QUERY = {
        "name": "api_name",
        "description": "api_description",
    }
    api_list = []
    for each_api in json_data["api_list"]:
        each_api_copy = {
            "category_name": category_name,
            "tool_name": tool_name,
            "api_name": each_api["name"],
            "api_description": each_api["description"],
        }
        api_list.append(each_api_copy)
    query_copy = {
        "query_id": -1, 
        "query": "", 
        "api_list": api_list, 
        "relevant APIs": []
    }
    return query_copy


def load_mcp_logs(tool_usage_paths: List[str], all_json_paths: Dict[Tuple[str, str], Path]) -> Dict[Tuple[str, str], List[Dict]]:
    mcp_logs = defaultdict[Tuple[str, str], List[Dict]](list)
    if tool_usage_paths is None:
        return mcp_logs
    # If a single directory is passed, expand it into its subdirectories
    if len(tool_usage_paths) == 1 and Path(tool_usage_paths[0]).is_dir():
        parent = Path(tool_usage_paths[0])
        if not (parent / "evaluation_statistics.json").exists():
            tool_usage_paths = sorted(str(p) for p in parent.iterdir() if p.is_dir())
    for trace_path in tool_usage_paths:
        if is_valid_eval_result(trace_path):
            mcp_logs_under_one_folder = load_mcp_logs_under_one_folder(trace_path, all_json_paths)
            for (category_name, tool_name), logs in mcp_logs_under_one_folder.items():
                mcp_logs[(category_name, tool_name)].extend(logs)
    return mcp_logs
            

def main():
    parser = argparse.ArgumentParser(description="Run TMDB step-wise evaluation")

    parser = add_FunctionWrapper_args(parser)
    parser.add_argument("--tool_usage_path", nargs="+", type=str, help="Directory path to tools usage traces, including which APIs were called, which succeeded (i.e. healthy) and which threw errors (i.e. unhealthy).")
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
    # read all eval result folders and collect all mcp logs, with the corresponding mcp yaml file path
    # output: dict[mcp_yaml_path] -> list[mcp_log]
    mcp_logs = load_mcp_logs(args.tool_usage_path, all_json_paths)
    # Rebuild the mcp_logs,only collect logs with failed api calls
    for key, logs in mcp_logs.items():
        logs_failed = [log for log in logs if log["response"]["success"] == False]
        mcp_logs[key] = logs_failed
    print(termcolor.colored(f"Loaded {len(mcp_logs)} mcp logs by mcp yaml file.", "green"))

    # Per each mcp yaml file, we first build a dummy query file such that the tool manager and env can be built
    # Enable Logging and Instrumentation
    if args.debug:
        project_name = f"debug"
    else:
        project_name = f"rewrite_{date_time}"
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

    LLM_model = get_model(model_id=args.model_name, tool_call_by_prompt=args.tool_call_by_prompt, api_key=args.openai_api_key)
    queries_and_tools = []

    for (category_name, tool_name) in tqdm(all_mcp_yaml_paths.keys(), desc="Processing mcp logs"):
        dummy_query = prune_tools_in_query(category_name, tool_name, all_json_paths)
        mcp_log_for_one_api_provider = mcp_logs.get((category_name, tool_name), [])
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
        )
        def load_mcp_yaml_func(mcp_yaml_path, old_version=False):
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
        with open(Path("prompts") / "D2_agent.yaml", "r") as f:
            prompt_template = yaml.safe_load(f)
        
        atomic_tools = [
            TakeNotes(),
            UpdateProviderDescription(schema=schema),
            UpdateAPIDescription(schema=schema),
            UpdateAPIParameters(schema=schema),
            RemoveAPIParameters(schema=schema),
            PrintSchema(schema=schema)
        ]
        smolagents_tools.extend(atomic_tools)
        max_steps = 1 + 1 + number_of_APIs*4 + 1 + 20 # 1 for TakeNotes, 1 for UpdateProviderDescription, number_of_APIs*4 for API calls, UpdateAPIDescription, UpdateAPIParameters, RemoveAPIParameters, PrintSchema, 1 for SchemaRewriteFinalAnswer, 20 for the maximum number of steps
        agent = ToolCallingAgent(tools=smolagents_tools, model=LLM_model, prompt_templates=prompt_template, max_steps=max_steps)

        agent.tools["final_answer"] = SchemaRewriteFinalAnswer(schema=schema, schema_file_path=schema_file_path)
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
