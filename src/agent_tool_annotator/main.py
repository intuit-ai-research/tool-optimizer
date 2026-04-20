"""
python main.py --config ../FunctionWrapper/eval/tmdb/configs/tmdb_base.yaml --dataset ../FunctionWrapper/StableToolBench/solvable_queries/test_instruction/test.json --mcp_yaml_path ../FunctionWrapper/eval/StableToolBench/ --tool_root_dir ../FunctionWrapper/StableToolBench/data/toolenv/tools/ --debug
"""

import argparse
import os, sys
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import termcolor
import json
import random
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, VLLMModel
from agent_tool_annotator.agent.models import get_model
from agent_tool_annotator.agent.phoenix_utils import PhoenixUtils
from phoenix.otel import register
from openinference.instrumentation import capture_span_context
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

from agent_tool_annotator.agent.FunctionWrapper_args import FunctionWrapper_DIR, TOOLEVAL_DIR, add_FunctionWrapper_args, impute_functionwrapper_args
sys.path.append(FunctionWrapper_DIR)
sys.path.append(TOOLEVAL_DIR)
from agent_tool_annotator.agent.api_tool import build_tools_from_yaml_tools
from utils.exp_meta import build_reproducibility_log
from eval.tmdb.examples.main_tmdb import prepare_data_from_StableToolBench, load_queries, load_tools, ToolManager, create_output_directory
from evaluators import load_registered_automatic_evaluator

def main():
    parser = argparse.ArgumentParser(description="Run TMDB step-wise evaluation")

    parser = add_FunctionWrapper_args(parser)

    parser.add_argument("--tool_call_by_prompt", action="store_true", default=False,
                       help="Tool call by prompt")
    parser.add_argument("--evaluator_name", type=str, default="tooleval_gpt-3.5-turbo_default", help="name of the LLM-as-Judge")
    parser.add_argument("--project_name", type=str, default=None,
                       help="Project name for the evaluation. If provided, will continue the evaluation from the project name.")
    parser.add_argument("--shuffle_query", action="store_true", default=False,
                       help="Shuffle the query list")

    args = parser.parse_args()
    
    args, queries_path, platform, output_suffix, max_queries, runs_per_scenario = impute_functionwrapper_args(args)
    

    dataset_filename = os.path.basename(args.dataset)
    dataset_name = os.path.splitext(dataset_filename)[0]
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")


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

    # Enable Logging and Instrumentation
    if args.project_name is not None:
        project_name = args.project_name
    else:
        project_name = f"{dataset_name}_{date_time}_{args.model_name}"
    log_path = Path("log") / Path(f"{project_name}_{date_time}")
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "reproducibility_log.txt", "w") as f:
        f.write(build_reproducibility_log(args))
    with open(log_path / "project_info.json", "w") as f:
        json.dump({"project_name": project_name}, f, indent=2, ensure_ascii=False)
    register(project_name=project_name)
    SmolagentsInstrumentor().instrument()
    phx_utils = PhoenixUtils()
    processed_queries = phx_utils.extract_query_from_project(project_name)
    client = phx_utils.get_client()
    evaluator = load_registered_automatic_evaluator(
        evaluator_name=args.evaluator_name,
        evaluators_cfg_path=f"{FunctionWrapper_DIR}/StableToolBench/toolbench/tooleval/evaluators",
    )

    LLM_model = get_model(model_id=args.model_name, tool_call_by_prompt=args.tool_call_by_prompt, api_key=args.openai_api_key)
    queries_and_tools = []
    if args.shuffle_query:
        random.shuffle(queries) # shuffle the query list
    for data_dict, tool_manager, decompo_tool_manager, env in queries:
        if data_dict["query"] in processed_queries:
            print(termcolor.colored(f"Query {data_dict['query']} already processed, skipping", "yellow"))
            continue
        # Build smolagents Tool objects from YAML tools (adapter)
        smolagents_tools = build_tools_from_yaml_tools(tool_manager, env, tool_call_by_prompt=args.tool_call_by_prompt)
        queries_and_tools.append((data_dict, smolagents_tools))

        # name_to_tool = {tool.name: tool for tool in smolagents_tools}
        # name_to_tool['remove_albums_user']({"ids": "5JNrPPT60TzrqBxsY3hn0A"})

        agent = ToolCallingAgent(tools=smolagents_tools, model=LLM_model)
        # Swap in a prompt-aware FinalAnswer tool that renders JSON schema in the system prompt
        if args.tool_call_by_prompt and "final_answer" in agent.tools:
            from agent.rewrite_tool import PromptFinalAnswerTool
            agent.tools["final_answer"] = PromptFinalAnswerTool()
        with capture_span_context() as capture:
            agent.run(data_dict["query"])
            first_span_id = capture.get_first_span_id()
        if first_span_id:
            # LLM-as-judge: build payload from spans and annotate PassRate
            task_description, answer = phx_utils.build_tooleval_payload(
                project_name=project_name, root_span_id=first_span_id, tools=smolagents_tools
            )
            answer_status, _reason = evaluator.check_is_solved(
                task_description, answer, return_reason=True
            )
            label = answer_status.name  # "Solved" | "Unsure" | "Unsolved"
            client.spans.add_span_annotation(
                annotation_name="PassRate",
                annotator_kind="LLM",
                span_id=first_span_id,
                label=label,
            )

            # Metrics: tool selection recall and success call rate
            gt_tools = set([tool_manager.extract_api_name_from_endpoint(step) for step in data_dict['solution']])
            sanity_name2raw_name = {tool.name: tool.raw_name for tool in smolagents_tools}
            used_tool_names_agent, success_count, total_count = phx_utils.extract_tool_calls(
                project_name=project_name, root_span_id=first_span_id
            )
            used_tool_names = set([sanity_name2raw_name[name] for name in used_tool_names_agent])
            # Recall of ground-truth tools
            tool_selection = 0.0
            if len(gt_tools) > 0:
                tool_selection = len(used_tool_names.intersection(gt_tools)) / float(len(gt_tools))
            # Success rate of tool invocations
            success_call = (success_count / float(total_count)) if total_count > 0 else 0.0

            client.spans.add_span_annotation(
                annotation_name="ToolSelectionRecall",
                annotator_kind="LLM",
                span_id=first_span_id,
                score=f"{tool_selection:.3f}",
                explanation=f"Used tools: {used_tool_names}, Ground truth tools: {gt_tools}"
            )
            client.spans.add_span_annotation(
                annotation_name="ToolSuccessRate",
                annotator_kind="LLM",
                span_id=first_span_id,
                score=f"{success_call:.3f}",
            )
        print(termcolor.colored(f"Tool calling agent finished for query: {data_dict['query']}", "green"))
        # break


    print(termcolor.colored(f"Finish", "green"))

if __name__ == "__main__":
    main()