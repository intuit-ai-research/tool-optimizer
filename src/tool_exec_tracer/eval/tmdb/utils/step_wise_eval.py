from typing import Dict, Optional
from typing import List
from typing import Any, Tuple

from tool_exec_tracer.eval.tmdb.utils.task_decomp import identify_api_requiring_subtasks
from tool_exec_tracer.eval.tmdb.utils.tool_manager import ToolManager
from tool_exec_tracer.eval.tmdb.config.tmdb_endpoint_mapping import TMDB_ENDPOINT_MAPPING

# Use generic API caller that supports both TMDB and Spotify
from tool_exec_tracer.eval.tmdb.utils.api_caller import call_api_mcp
# Backwards compatibility - keep the old import working
from tool_exec_tracer.eval.tmdb.utils.mcp import call_tmdb_api_mcp

# Parameter validation utility
from tool_exec_tracer.eval.tmdb.utils.parameter_validator import evaluate_parameter_generation_quality, validate_json

import json

from tool_exec_tracer.DRAFT.Inference_DFSDT import answer_generation_depend, truncate_content
# from DRAFT.desc_eval_new.main_evaluator import choose_parameter_depend_with_first_attempt_tracking
# from DRAFT.desc_eval_new.cached_tmdb_api_caller import call_tmdb_api

import random
import numpy as np
from datetime import datetime
from tool_exec_tracer.eval.tmdb.utils.prompt_loader import PromptLoader
from tool_exec_tracer.utils.llm import llm_call
import os

from collections import Counter




ROOT_DIR=os.environ.get("ROOT_DIR", "/home/sagemaker-user/user-default-efs/FunctionWrapper")
def generate_scenarios_from_golden_apis_for_step_wise_eval(query_data: Dict, 
            tool_manager: ToolManager, 
            api_requiring_indices: List[int] = None, 
            processing_indices: List[int] = None,
            subtask_to_api: Dict[int, Any] = None) -> List[Dict]:

    """
    Generate evaluation scenarios for step-wise evaluation using golden APIs.
    
    Args:
        query_data: Query data with subtasks
        tool_manager: Tool manager for API descriptions
        api_requiring_indices: List of 1-based subtask IDs that need APIs
        processing_indices: List of 1-based subtask IDs that are processing steps
        subtask_to_api: Dict mapping subtask_id (1-based) → golden_api_name
    """
    scenarios = []
    
    # Get the subtasks structure
    subtasks = query_data.get("subtasks", [])
    if not subtasks:
        raise ValueError("No subtasks found in query data")
    
    print(f"📋 Found {len(subtasks)} subtasks")
    print(f"🔍 Identified {len(api_requiring_indices)} API-requiring subtasks and {len(processing_indices)} processing subtasks")
    
    # Get API-specific descriptions from tool manager
    if tool_manager is None:
        raise ValueError("tool_manager is required for API description retrieval but was not initialized")
    
    # Generate scenarios for API-requiring subtasks using semantic mapping
    for task_id in api_requiring_indices:
        # task_id is 1-based, convert to 0-based for array access
        subtask_idx = task_id - 1
        if subtask_idx >= len(subtasks):
            print(f"⚠️  Warning: task_id {task_id} is out of bounds for {len(subtasks)} subtasks")
            continue
        
        subtask = subtasks[subtask_idx]
        
        # Get golden API from semantic mapping
        if subtask_to_api and task_id in subtask_to_api:
            tool_provider, golden_api_name = subtask_to_api[task_id]
        else:
            # Fallback: get from subtask's golden_api field
            # golden_api_name = subtask.get("golden_api", "")
            # if not golden_api_name:
            #     print(f"⚠️  Warning: No golden API found for subtask {task_id}, skipping")
            #     continue
            raise ValueError(f"No subtask to api mapping found for subtask {task_id}")
        
        print(f"🎯 Subtask {task_id} → Golden API: {tool_provider}::{golden_api_name}")
        
        # Get descriptions for this specific API

        for tool in tool_manager.tools:
            if tool["tool_provider"] == tool_provider and tool["tool_name"] == golden_api_name:
                description = tool["description"]
                parameters = tool.get("parameters", {})
                metadata = tool["_metadata"]
                break
        
        else:
            raise ValueError(f"Description not found for {golden_api_name}")
        
        # Use original dependencies from task decomposition
        original_dependencies = subtask.get("dependencies", [])

        scenario = {
                "target_subtask_id": task_id,  # Use 1-based task_id
                "selected_api_name": golden_api_name,
                "selected_description": description,
                "selected_parameters": parameters,
                "selected_metadata": metadata,
                "subtask_input": subtask.get("input", ""),
                "dependencies": original_dependencies,
                "is_processing_step": False
            }
        scenarios.append(scenario)
        
    # Generate scenarios for processing subtasks (NO API CALLS but still need evaluation for context passing)
    for task_id in processing_indices:
        # task_id is 1-based, convert to 0-based for array access
        subtask_idx = task_id - 1
        if subtask_idx >= len(subtasks):
            print(f"⚠️  Warning: task_id {task_id} is out of bounds for {len(subtasks)} subtasks")
            continue
        
        proc_subtask = subtasks[subtask_idx]
        
        print(f"⚙️  Processing Subtask {task_id} → No API needed")
        
        # Use original dependencies from task decomposition
        original_dependencies = proc_subtask.get("dependencies", [])
        
        scenario = {
            "target_subtask_id": task_id,  # Use 1-based task_id
            "selected_api_name": "",  # No API for processing steps
            "selected_description": "",  # No API description needed
            "selected_parameters": {},
            "selected_metadata": {},
            "subtask_input": proc_subtask.get("input", ""),
            "dependencies": original_dependencies,
            "is_processing_step": True
        }
        scenarios.append(scenario)
        print(f"   Subtask {task_id}: {proc_subtask.get('input', '')[:100]}...")
    
    # Sort scenarios by subtask_id to ensure proper execution order
    scenarios.sort(key=lambda x: x["target_subtask_id"])
    
    print(f"📊 Generated {len(scenarios)} evaluation scenarios for step-wise evaluation")
    return scenarios


def evaluate_query_with_step_wise_eval(query_data: Dict, golden_apis: List[str], tool_manager: ToolManager = None,
                                          runs_per_scenario: int = 5, seed: int = 42,
                                          api_requiring_indices: List[int] = None, 
                                          processing_indices: List[int] = None,
                                          subtask_to_api: Dict[int, Any] = None,
                                          prompt_loader: PromptLoader = None, 
                                          query_id: str = None, dataset: str = None,
                                          parameter_generation_prompt_version: str = "v3", 
                                          simple_api_instruction: bool = True, 
                                          rapidapi_wrapper=None, 
                                          model_name: str = "gpt-4.1-2025-04-14",
                                          context_source: str = "golden",
                                          consider_provider_in_exact_match: bool = False) -> Dict:
    """
    Step-wise evaluation: Execute subtasks sequentially using golden APIs to build log entries,
    then evaluate API selection accuracy for dependent subtasks
    
    Args:
        golden_apis: List of ground truth APIs in order (0-based indexing)
        api_requiring_indices: List of subtask IDs that require APIs (1-based indexing)
        subtask_to_api: Dictionary mapping subtask_id → golden_api_name (from semantic matching)
    """
    
    # Use the provided subtask_to_api mapping if available, otherwise create position-based mapping
    if subtask_to_api is not None:
        golden_apis_by_step = subtask_to_api
    else:
        # Fallback: Convert golden_apis list to dict mapping subtask_id → api_name
        # api_requiring_indices are 1-based, golden_apis is 0-based
        # golden_apis_by_step = {}
        # for idx, task_id in enumerate(sorted(api_requiring_indices)):
        #     if idx < len(golden_apis):
        #         golden_apis_by_step[task_id] = golden_apis[idx]

        raise ValueError("subtask_to_api is required for golden API mapping")
    
    # Extract original query for logging
    original_query = query_data.get("query", "")
    
    # Step 1: Generate API scenarios using golden APIs from solution for step-wise evaluation
    golden_api_scenarios = generate_scenarios_from_golden_apis_for_step_wise_eval(
        query_data, tool_manager, api_requiring_indices, processing_indices, subtask_to_api)
    
    if not golden_api_scenarios:
        raise ValueError("No golden API scenarios found for this query")
    
    # Sort scenarios by subtask_id to process earlier subtasks first
    golden_api_scenarios.sort(key=lambda x: x["target_subtask_id"])
    
    print(f"📊 Step-wise evaluation for {len(golden_api_scenarios)} scenarios × {runs_per_scenario} runs")
        
    # Step 2: Execute step-wise evaluation
    all_step_results = []
        
    # Process subtasks in order to build golden log entries
    golden_log_entries = {}  # {subtask_id: log_entry}
    
    for idx, scenario in enumerate(golden_api_scenarios):

        subtask_id = scenario["target_subtask_id"]
        print(f"\n🔍 Processing Subtask {subtask_id} for golden log entry creation...")
        
        # Initialize deterministic seed for this subtask
        import random
        import numpy as np
        deterministic_seed = seed + idx
        random.seed(deterministic_seed)
        np.random.seed(deterministic_seed)
                
        # Check if this is a processing step first
        is_processing_step = scenario.get("is_processing_step", False)
        
        # Get the expected golden API for this subtask
        if is_processing_step:
            expected_golden_api_for_log = ""  # Processing steps should never have golden APIs
            print(f"   ⚙️  Processing step detected - no golden API expected")
        else:
            # Use direct mapping to get golden API for this subtask
            expected_golden_api_for_log = golden_apis_by_step.get(subtask_id, "")
            print(f"   🎯 Expected golden API: {expected_golden_api_for_log}")
        
        # Execute with golden API to create log entry

            golden_result = evaluate_single_subtask_with_cache(
                scenario, golden_log_entries, tool_manager,
                expected_golden_api=expected_golden_api_for_log,
                llm_seed=deterministic_seed,
                simple_api_instruction=simple_api_instruction,
                query_id=query_id,
                original_query=original_query,
                dataset=dataset,
                rapidapi_wrapper=rapidapi_wrapper,
                model_name="gpt-4.1-2025-04-14",
            )
            
            if golden_result.get("api_success", False) or is_processing_step:
                golden_log_entries[subtask_id] = {
                    "subtask_id": subtask_id,
                    "subtask_input": golden_result.get("subtask_input", ""),
                    "subtask_output": golden_result.get("subtask_output", ""),
                    "expected_golden_api": expected_golden_api_for_log
                    # "api_response": golden_result.get("api_response", {})  # Removed: already cached
                }
                if is_processing_step:
                    print(f"   ✅ Created golden log entry for processing subtask {subtask_id}")
                else:
                    print(f"   ✅ Created golden log entry for subtask {subtask_id}")
            else:
                # Create fallback golden log entry even when subtask fails
                # This ensures dependent subtasks have some context to work with
                fallback_output = f"Subtask failed: {scenario.get('subtask_input', 'Unknown task')}"
                if golden_result.get("error"):
                    fallback_output += f" (Error: {golden_result.get('error', 'Unknown error')})"
                
                golden_log_entries[subtask_id] = {
                    "subtask_id": subtask_id,
                    "subtask_input": golden_result.get("subtask_input", scenario.get("subtask_input", "")),
                    "subtask_output": fallback_output,
                    "expected_golden_api": expected_golden_api_for_log
                }
                print(f"   ⚠️  Created fallback golden log entry for failed subtask {subtask_id}")
                
        # except Exception as e:
        #     # Create fallback golden log entry even when there's an exception
        #     # This ensures dependent subtasks have some context to work with
        #     fallback_output = f"Subtask execution failed: {scenario.get('subtask_input', 'Unknown task')} (Exception: {str(e)})"
            
        #     golden_log_entries[subtask_id] = {
        #         "subtask_id": subtask_id,
        #         "subtask_input": scenario.get("subtask_input", ""),
        #         "subtask_output": fallback_output,
        #         "expected_golden_api": expected_golden_api_for_log if not is_processing_step else ""
        #     }
        #     print(f"   💥 Error creating golden log entry for subtask {subtask_id}: {e}")
        #     print(f"   ⚠️  Created fallback golden log entry for subtask {subtask_id}")
    
    # Step 4: Now evaluate API selection for each subtask using golden log entries
    for scenario in golden_api_scenarios:
        subtask_id = scenario["target_subtask_id"]
        scenario_results = []
        
        print(f"\n🎯 Evaluating API Selection for Subtask {subtask_id}")
        print(f"   📊 Testing scenario: {scenario.get('selected_api_name', 'N/A')}")
        
        # Check if this is a processing step first
        is_processing_step = scenario.get("is_processing_step", False)
        
        # Get expected golden API for this subtask
        if is_processing_step:
            expected_golden_api = ""  # Processing steps should never have golden APIs
            print(f"   ⚙️  Processing subtask {subtask_id} - no API expected")
        else:
            expected_golden_api = golden_apis_by_step.get(subtask_id, "")
            if expected_golden_api == "":
                print(f"   ❌ No expected golden API found for subtask {subtask_id}")
            else:
                print(f"   🎯 Expected golden API: {expected_golden_api}")
        
        for run_idx in range(runs_per_scenario):
            print(f"     🏃 Run {run_idx + 1}/{runs_per_scenario}")
            
            # Reset random seed for independent sampling (same as golden API testing)
            import random
            import numpy as np
            deterministic_seed = seed + run_idx
            random.seed(deterministic_seed)
            np.random.seed(deterministic_seed)
            
            try:
                # Step 4a: Test API selection for this subtask
                api_selection_result = evaluate_api_selection_for_subtask(
                    scenario, 
                    golden_log_entries, 
                    tool_manager, 
                    expected_golden_api, 
                    query_data, 
                    llm_seed=deterministic_seed, 
                    prompt_loader=prompt_loader,
                    model_name=model_name,
                    consider_provider_in_exact_match=consider_provider_in_exact_match
                )

                if api_selection_result["exact_match_accuracy"] == 1.0 and api_selection_result.get("expected_golden_api"):
                    # Skip strict tuple assertions for processing subtasks where expected_golden_api is empty.
                    if consider_provider_in_exact_match:
                        assert api_selection_result["selected_provider"] == api_selection_result["expected_golden_api"][0]
                    assert api_selection_result["selected_api"] == api_selection_result["expected_golden_api"][1]
                
                # Step 4b: Execute the subtask with selected API      
                execution_result = evaluate_single_subtask_with_cache(
                    scenario, golden_log_entries, tool_manager,
                    expected_golden_api=expected_golden_api,
                    llm_seed=deterministic_seed,
                    simple_api_instruction=simple_api_instruction,
                    query_id=query_id,
                    original_query=original_query,
                                        dataset=dataset,
                                        parameter_generation_prompt_version=parameter_generation_prompt_version,
                                        rapidapi_wrapper=rapidapi_wrapper, 
                                        model_name=model_name,
                )
                
                # Combine results
                run_result = {
                    **execution_result,
                    "selected_api": api_selection_result["selected_api"],
                    "expected_golden_api": expected_golden_api,
                    "api_selection_reasoning": api_selection_result.get("reasoning", ""),
                    "exact_match_accuracy": api_selection_result["exact_match_accuracy"]
                }
                
                scenario_results.append(run_result)
                
                # Print run summary
                api_correct = "✅" if run_result["exact_match_accuracy"] == 1.0 else "❌"
                subtask_success = "✅" if run_result.get("subtask_success", False) else "❌"
                print(f"       API Selection: {api_correct} | Subtask Success: {subtask_success}")
                
            except Exception as e:
                print(f"     💥 Run {run_idx + 1} failed: {e}")
                error_result = {
                    "error": str(e),
                    "run_failed": True,
                    "subtask_success": False,
                    "api_success": False,
                    "exact_match_accuracy": 0.0
                }
                scenario_results.append(error_result)
        
        # Calculate scenario statistics
        scenario_stats = calculate_step_wise_scenario_statistics(scenario_results)
        
        step_result = {
            "subtask_id": subtask_id,
            "query_id": query_id,  # Add query_id for matching with LLM logs
            "scenario": scenario,
            "expected_golden_api": expected_golden_api,
            "runs": scenario_results,
            "statistics": scenario_stats
        }
        
        all_step_results.append(step_result)
        
        # Print scenario summary
        print(f"     📊 Scenario Summary:")
        print(f"       Exact Match Accuracy: {safe_mean(scenario_stats['exact_match_accuracy']):.3f}")
        print(f"       Total Runs: {scenario_stats['total_runs']}")
        print(f"       Correct Exact Matches: {scenario_stats['correct_exact_matches']}")
    
    # Step 5: Calculate overall statistics
    overall_stats = calculate_step_wise_overall_statistics(all_step_results)
    
    # Step 6: Calculate query-level perfect execution metrics
    # For each run across all scenarios, check if ALL steps succeeded
    query_level_stats = calculate_query_level_statistics(all_step_results, runs_per_scenario)
    query_level_stats["number_of_api"] = len(tool_manager.tools)
    # overall_stats.update(query_level_stats)

    # Step 7: Calculate task decomposition aware query-level accuracy
    agent_query_accuracy = 1.0
    agent_query_recall = 1.0

    # TODO: now consider whether each api is used the right number of times
    # since ground truth decomposition may not be perfect and the order of golden apis is not given

    mapped_apis = list(golden_apis_by_step.values())

    mapped_api_strs = []
    for tool_name, api_name in mapped_apis:
        api_str = f"{tool_name}::{api_name}"
        mapped_api_strs.append(api_str)
    
    golden_api_strs = []
    for tool_name, api_name in golden_apis:
        api_str = f"{tool_name}::{api_name}"
        golden_api_strs.append(api_str)

    golden_api_counter = Counter(golden_api_strs)
    mapped_api_counter = Counter(mapped_api_strs)

    for api_str, count in golden_api_counter.items():

        if mapped_api_counter[api_str] < count:
            agent_query_recall = 0.0

        if mapped_api_counter[api_str] != count:
            agent_query_accuracy = 0.0
    
    # if we covered all golden apis, whether all tool selection are correct
    query_level_stats["agent_query_accuracy"] = agent_query_accuracy
    query_level_stats["agent_query_recall"] = agent_query_recall

    query_level_stats["agent_query_accuracy_and_select"] = agent_query_accuracy * query_level_stats["query_perfect_api_selection_rate"]
    query_level_stats["agent_query_recall_and_select"] = agent_query_recall * query_level_stats["query_perfect_api_selection_rate"]

    query_level_stats["agent_query_accuracy_and_select_and_success"] = agent_query_accuracy * query_level_stats["query_perfect_api_selection_rate"] * query_level_stats["query_perfect_api_success_rate"]
    query_level_stats["agent_query_recall_and_select_and_success"] = agent_query_recall * query_level_stats["query_perfect_api_selection_rate"] * query_level_stats["query_perfect_api_success_rate"]

    print("agent_query_accuracy", agent_query_accuracy)
    print("agent_query_recall", query_level_stats["agent_query_recall"])
    print("agent_query_accuracy_and_select", query_level_stats["agent_query_accuracy_and_select"])
    print("agent_query_recall_and_select", query_level_stats["agent_query_recall_and_select"])
    print("agent_query_accuracy_and_select_and_success", query_level_stats["agent_query_accuracy_and_select_and_success"])
    print("agent_query_recall_and_select_and_success", query_level_stats["agent_query_recall_and_select_and_success"])
    print("query_level_stats['query_perfect_api_selection_rate']", query_level_stats["query_perfect_api_selection_rate"])

    overall_stats.update(query_level_stats)

    # Print overall summary
    print(f"\n📊 Step-wise Evaluation Overall Statistics:")
    print(f"   📊 Total Scenarios: {overall_stats['total_scenarios']}")
    print(f"   🎯 Overall Exact Match Accuracy: {overall_stats['overall_exact_match_accuracy']['mean']:.3f}")
    print(f"   ✅ Total Correct Exact Matches: {overall_stats['total_correct_exact_matches']}")
    print(f"   🏃 Total Runs: {overall_stats['total_runs_all_scenarios']}")
    print(f"\n📊 Query-Level Perfect Execution:")
    print(f"   🎯 Perfect API Selection Rate: {query_level_stats['query_perfect_api_selection_rate']:.2%}")
    # print(f"   ✅ Perfect Parameter Rate: {query_level_stats['query_perfect_parameter_rate']:.2%}")
    print(f"   🚀 Perfect API Success Rate: {query_level_stats['query_perfect_api_success_rate']:.2%}")
    # print(f"   🎯✅ Perfect Sel Acc & Param Valid Rate: {query_level_stats['query_perfect_sel_acc_and_param_valid_rate']:.2%}")
    print(f"   🎯 Agent Query Accuracy: {query_level_stats['agent_query_accuracy']:.2%}")
    print(f"   🎯 Agent Query Recall: {query_level_stats['agent_query_recall']:.2%}")
    print(f"   🎯 Agent Query Accuracy & Select: {query_level_stats['agent_query_accuracy_and_select']:.2%}")
    print(f"   🎯 Agent Query Recall & Select: {query_level_stats['agent_query_recall_and_select']:.2%}")
    print(f"   🎯 Agent Query Accuracy & Select & Success: {query_level_stats['agent_query_accuracy_and_select_and_success']:.2%}")
    print(f"   🎯 Agent Query Recall & Select & Success: {query_level_stats['agent_query_recall_and_select_and_success']:.2%}")
    # Return results
    result = {
        "step_wise_results": all_step_results,
        "golden_log_entries": golden_log_entries,
        "golden_apis_by_step": golden_apis_by_step,
        "total_scenarios": len(all_step_results),
        "runs_per_scenario": runs_per_scenario,
        "evaluation_type": "step_wise_evaluation",
    }
    
    return {
        "query_id": query_id,
        "query_data": query_data,
        "main_results": result,
        "overall_statistics": overall_stats
    }


def calculate_step_wise_overall_statistics(all_step_results: List[Dict]) -> Dict:
    """Calculate simplified overall statistics across all step-wise results"""
    
    if not all_step_results:
        raise ValueError("No step-wise results found")
    
    # Aggregate data from all scenarios
    all_exact_match_accuracy = []
    
    total_scenarios = len(all_step_results)
    total_runs_all_scenarios = 0
    total_correct_exact_matches = 0
    
    for step_result in all_step_results:
        stats = step_result["statistics"]
        
        # Collect raw data from simplified structure (arrays instead of nested dicts)
        all_exact_match_accuracy.extend(stats.get("exact_match_accuracy", []))
        
        # Aggregate counts
        total_runs_all_scenarios += stats["total_runs"]
        total_correct_exact_matches += stats.get("correct_exact_matches", 0)

    
    return {
        "total_scenarios": total_scenarios,
        "total_runs_all_scenarios": total_runs_all_scenarios,
        "total_correct_exact_matches": total_correct_exact_matches,
        "overall_exact_match_accuracy": {
            "mean": safe_mean(all_exact_match_accuracy),
            "std": safe_std(all_exact_match_accuracy),
            "raw_data": all_exact_match_accuracy,
            "count": len(all_exact_match_accuracy)
        }
    }


def calculate_step_wise_scenario_statistics(
    scenario_results: List[Dict]) -> Dict:

    """Calculate simplified statistics for step-wise scenario - only raw data, no expensive calculations"""
    
    if not scenario_results:
        raise ValueError("No scenario results found")
    
    # Calculate basic counts
    total_runs = len(scenario_results)
    
    # Only count API calls for exact match metrics (exclude processing steps)
    api_runs = [r for r in scenario_results if r.get("expected_golden_api", "") != ""]
    correct_exact_matches = sum(1 for r in api_runs if r.get("exact_match_accuracy", 0.0) == 1.0)

    # Only include actual API calls in exact match accuracy (exclude processing steps)
    exact_match_accuracy_data = [r.get("exact_match_accuracy", 0.0) for r in api_runs]
    
    return {
        "total_runs": total_runs,
        "correct_exact_matches": correct_exact_matches,
        "exact_match_accuracy": exact_match_accuracy_data
    }


def calculate_query_level_statistics(all_step_results: List[Dict], runs_per_scenario: int) -> Dict:
    """
    Calculate query-level perfect execution statistics.
    
    A query execution is "perfect" when ALL API-requiring subtasks in that run succeed.
    We track four independent metrics:
    1. Perfect API Selection: All APIs selected match golden APIs exactly
    2. Perfect Parameters: All parameter generations are valid
    3. Perfect API Success: All API calls succeed
    4. Perfect Sel Acc and Param Valid: All APIs selected correctly AND all parameters valid
    
    Args:
        all_step_results: Results from all scenarios (subtasks)
        runs_per_scenario: Number of runs per scenario
    
    Returns:
        Dictionary with query-level statistics
    """
    from tool_exec_tracer.eval.tmdb.utils.llm_call_logger import get_global_llm_logger
    
    if not all_step_results:
        return {
            "query_perfect_api_selection_rate": 0.0,
            "query_perfect_parameter_rate": 0.0,
            "query_perfect_api_success_rate": 0.0,
            "query_perfect_sel_acc_and_param_valid_rate": 0.0,
            "query_perfect_api_selection_raw": [],
            "query_perfect_parameter_raw": [],
            "query_perfect_api_success_raw": [],
            "query_perfect_sel_acc_and_param_valid_raw": []
        }
    
    # Get LLM logger to access parameter validation data
    llm_logger = get_global_llm_logger()
    
    # Organize results by run index across all scenarios
    # Each run should execute all scenarios in sequence
    query_level_results = []
    
    # Also collect subtask-level metrics for all runs
    all_sel_acc_and_api_success = []
    all_sel_acc_and_param_valid = []
    
    for run_idx in range(runs_per_scenario):
        # Collect results for this specific run across all scenarios
        run_exact_matches = []
        run_params_valid = []
        run_api_success = []

        run_sel_acc_and_api_success = []
        run_sel_acc_and_param_valid = []
        
        for step_result in all_step_results:
            runs = step_result.get("runs", [])
            if run_idx < len(runs):
                run_data = runs[run_idx]
                
                # Skip processing steps (those without golden APIs)
                expected_api = step_result.get("expected_golden_api", "")
                if not expected_api:
                    continue
                
                # Track exact match accuracy
                exact_match = run_data.get("exact_match_accuracy", 0.0)
                run_exact_matches.append(exact_match)
                
                # Track API success
                api_success = 1.0 if run_data.get("api_success", False) else 0.0
                run_api_success.append(api_success)
                
                # Track parameter validity from LLM logger
                # We need to match this run to the logged parameter generation
                # Use query_id and subtask_id to find the corresponding log entry
                params_valid = 0.0  # Default to invalid if we can't find the log
                
                if llm_logger:
                    query_id = step_result.get("query_id")
                    subtask_id = step_result.get("subtask_id")
                    
                    # Search through call history for matching entry
                    for log_entry in llm_logger.call_history:
                        if (log_entry.get("query_id") == query_id and 
                            log_entry.get("subtask_id") == subtask_id):
                            param_eval = log_entry.get("parameter_quality_evaluation", {})
                            params_valid = 1.0 if param_eval.get("params_valid", False) else 0.0
                            break
                
                run_params_valid.append(params_valid)
                
                # NEW: Compute sel_acc_and_param_valid = exact_match AND params_valid


                sel_acc_and_api_success = 1.0 if (exact_match == 1.0 and api_success == 1.0) else 0.0
                sel_acc_and_param_valid = 1.0 if (exact_match == 1.0 and params_valid == 1.0) else 0.0

                run_sel_acc_and_api_success.append(sel_acc_and_api_success)
                run_sel_acc_and_param_valid.append(sel_acc_and_param_valid)
                all_sel_acc_and_api_success.append(sel_acc_and_api_success)
                all_sel_acc_and_param_valid.append(sel_acc_and_param_valid)
        
        # A query run is "perfect" if ALL its API-requiring subtasks succeeded
        if run_exact_matches:  # Only if there were API-requiring subtasks
            perfect_api_selection = 1.0 if all(x == 1.0 for x in run_exact_matches) else 0.0
            perfect_params = 1.0 if all(x == 1.0 for x in run_params_valid) else 0.0
            perfect_api_success = 1.0 if all(x == 1.0 for x in run_api_success) else 0.0
            perfect_sel_acc_and_api_success = 1.0 if all(x == 1.0 for x in run_sel_acc_and_api_success) else 0.0
            # NEW: Perfect sel_acc_and_param_valid = all subtasks have both correct API and valid params
            perfect_sel_acc_and_param_valid = 1.0 if all(x == 1.0 for x in run_sel_acc_and_param_valid) else 0.0
            
            query_level_results.append({
                "perfect_api_selection": perfect_api_selection,
                "perfect_params": perfect_params,
                "perfect_api_success": perfect_api_success,
                "perfect_sel_acc_and_api_success": perfect_sel_acc_and_api_success,
                "perfect_sel_acc_and_param_valid": perfect_sel_acc_and_param_valid
            })
    
    # Calculate aggregate statistics
    if not query_level_results:
        return {
            "query_perfect_api_selection_rate": 0.0,
            "query_perfect_parameter_rate": 0.0,
            "query_perfect_api_success_rate": 0.0,
            "query_perfect_sel_acc_and_param_valid_rate": 0.0,
            "query_perfect_api_selection_raw": [],
            "query_perfect_parameter_raw": [],
            "query_perfect_api_success_raw": [],
            "query_perfect_sel_acc_and_api_success_raw": [],
            "query_perfect_sel_acc_and_param_valid_raw": [],
            "sel_acc_and_api_success_raw": [],
            "sel_acc_and_param_valid_raw": []
        }
    
    perfect_api_selection_raw = [r["perfect_api_selection"] for r in query_level_results]
    perfect_params_raw = [r["perfect_params"] for r in query_level_results]
    perfect_api_success_raw = [r["perfect_api_success"] for r in query_level_results]

    perfect_sel_acc_and_api_success_raw = [r["perfect_sel_acc_and_api_success"] for r in query_level_results]
    perfect_sel_acc_and_param_valid_raw = [r["perfect_sel_acc_and_param_valid"] for r in query_level_results]
    
    return {
        "query_perfect_api_selection_rate": safe_mean(perfect_api_selection_raw),
        "query_perfect_parameter_rate": safe_mean(perfect_params_raw),
        "query_perfect_api_success_rate": safe_mean(perfect_api_success_raw),

        "query_perfect_sel_acc_and_api_success_rate": safe_mean(perfect_sel_acc_and_api_success_raw),
        "query_perfect_sel_acc_and_param_valid_rate": safe_mean(perfect_sel_acc_and_param_valid_raw),

        "query_perfect_api_selection_raw": perfect_api_selection_raw,
        "query_perfect_parameter_raw": perfect_params_raw,
        "query_perfect_api_success_raw": perfect_api_success_raw,
        "query_perfect_sel_acc_and_api_success_raw": perfect_sel_acc_and_api_success_raw,
        "query_perfect_sel_acc_and_param_valid_raw": perfect_sel_acc_and_param_valid_raw,
        "sel_acc_and_api_success_raw": all_sel_acc_and_api_success,
        "sel_acc_and_param_valid_raw": all_sel_acc_and_param_valid
    }

def select_single_api_with_model(query: str, available_tools: List[Dict], 
                                 previous_log=None,
                                 llm_seed: int | None = None,
                                 prompt_loader=None,
                                 model_name: str = "gpt-4.1-2025-04-14",
                                 temperature: float = 0.2,
                                 max_tokens: int = 32768,
                                 query_id: str = None,
                                 subtask_id: int = None, 
                                 original_query: str = "",
                                 expected_golden_api: str = "",
                                 consider_provider_in_exact_match: bool = False):
    """Let the model select exactly ONE API for a specific subtask in step-wise evaluation"""

    # Prepare tool information for the model
    tools_info = []
    for tool in available_tools:
        if consider_provider_in_exact_match:
            tool_info = {
                "provider_name": tool["tool_provider"],
                "tool_name": tool["tool_name"],
                "tool_description": tool["description"],
            }
        else:
            tool_info = {
                "tool_name": tool["tool_name"],
                "tool_description": tool["description"],
            }
        tools_info.append(tool_info)

    # Format previous context for the prompt
    formatted_previous_context = format_previous_context(previous_log) if previous_log else None

    # Use prompt loader to get single API selection prompt
    prompt = prompt_loader.get_single_api_selection_prompt(
        query=query,
        tools_info=json.dumps(tools_info, indent=2),
        previous_context=formatted_previous_context,
        consider_provider_in_exact_match=consider_provider_in_exact_match
    )

    # Construct full prompt for logging
    system_message = "You are an expert API selector that chooses exactly one most appropriate API for each subtask."
    full_prompt = f"System: {system_message}\n\nUser: {prompt}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    # Initialize variables for logging
    selected_api = None
    selected_provider = None
    llm_response_raw = None
    selection_result = {}
    error_type = None
    error_message = None
    confidence = None

    try:
        print(f"\n🤖 Asking model to select ONE API for subtask...")
        # NOTE: Select the API for the subtask, relying on the LLM's Tool Use Ability.
        response = llm_call(messages, model=model_name, temperature=temperature, max_tokens=max_tokens, max_retries=6, base_delay=10.0, seed=llm_seed, step_name="api_selection")
        if response is None or response.choices is None:
            error_type = "NO_RESPONSE"
            error_message = "No response from model"
            print("💥 Error: No response from model")
        else:
            content = response.choices[0].message.content
            llm_response_raw = content

            # Check for context window exceeded error (simple string check)
            if isinstance(content, str) and ("CONTEXT_WINDOW_EXCEEDED" in content or "maximum context length" in str(content)):
                error_type = "CONTEXT_WINDOW_EXCEEDED"
                error_message = "Context window exceeded during single API selection"
                print("⚠️  Context window exceeded during single API selection")
                selected_api = "CONTEXT_WINDOW_EXCEEDED_ERROR"
            else:
                print(f"📝 Model Single API Selection Response: {content[:300]}...")

                # Parse JSON response
                if content.startswith('```json'):
                    content = content.replace('```json\n', '').replace('\n```', '')
                elif content.startswith('```'):
                    content = content.replace('```\n', '').replace('\n```', '')

                try:
                    selection_result = json.loads(content)
                    if "api_name" in selection_result:
                        selected_api_obj = selection_result
                    else:
                        selected_api_obj = selection_result.get("selected_api", {})
                    # Only extract confidence if it exists in the response
                    confidence = selection_result.get("confidence") if "confidence" in selection_result else None

                    if selected_api_obj and "api_name" in selected_api_obj:
                        selected_api = selected_api_obj["api_name"]
                        selected_provider = selected_api_obj.get("provider_name")
                        print(f"🎯 Model selected API: {selected_api}")
                        if selected_provider:
                            print(f"🏢 Model selected provider: {selected_provider}")
                        print(f"💭 Reasoning: {selected_api_obj.get('reasoning', 'N/A')}")
                        if confidence is not None:
                            print(f"🔍 Confidence: {confidence}")
                    else:
                        error_type = "NO_API_SELECTED"
                        error_message = "No valid API selected by model"
                        print("⚠️  No valid API selected")
                except json.JSONDecodeError as json_err:
                    error_type = "PARSING_ERROR"
                    error_message = f"Failed to parse JSON response: {json_err}"
                    print(f"💥 JSON parsing error: {json_err}")

    except Exception as e:
        error_msg = str(e)
        llm_response_raw = f"Exception: {error_msg}"
        # Check if the exception message contains context window exceeded error
        if "ContextWindowExceededError" in error_msg or "maximum context length" in error_msg:
            error_type = "CONTEXT_WINDOW_EXCEEDED"
            error_message = f"Context window exceeded during single API selection: {error_msg}"
            print(f"⚠️  Context window exceeded during single API selection: {error_msg}")
            selected_api = "CONTEXT_WINDOW_EXCEEDED_ERROR"
        else:
            error_type = "LLM_CALL_ERROR"
            error_message = f"Error in single API selection: {error_msg}"
            print(f"💥 Error in single API selection: {e}")

    # Log the API selection call if we have logging context
    if query_id is not None and subtask_id is not None:
        from tool_exec_tracer.eval.tmdb.utils.llm_call_logger import get_global_llm_logger
        llm_logger = get_global_llm_logger()

        if llm_logger:
            # Calculate accuracy using provider-aware mode when requested.
            valid_selection = selected_api not in [None, "CONTEXT_WINDOW_EXCEEDED_ERROR"]
            if consider_provider_in_exact_match:
                exact_match_accuracy = 1.0 if (
                    valid_selection and
                    selected_provider is not None and
                    selected_provider == expected_golden_api[0] and
                    selected_api == expected_golden_api[1]
                ) else 0.0
            else:
                exact_match_accuracy = 1.0 if (
                    valid_selection and selected_api == expected_golden_api[1]
                ) else 0.0

            # Create reasoning string
            if error_type:
                reasoning = f"Expected: {expected_golden_api} | Selected: ({selected_provider}, {selected_api}) | Error: {error_type}"
            elif exact_match_accuracy == 1.0:
                reasoning = f"Expected: {expected_golden_api} | Selected: ({selected_provider}, {selected_api}) | ✅ Exact Match"
            else:
                reasoning = f"Expected: {expected_golden_api} | Selected: ({selected_provider}, {selected_api}) | ❌ Incorrect"

            # Format previous context for logging
            previous_context_formatted = []
            if previous_log:
                for entry in previous_log:
                    if isinstance(entry, dict):
                        previous_context_formatted.append({
                            "subtask_id": entry.get("subtask_id", "unknown"),
                            "input": entry.get("subtask_input", "")[:100] + "..." if len(str(entry.get("subtask_input", ""))) > 100 else entry.get("subtask_input", ""),
                            "output": entry.get("subtask_output", "")[:100] + "..." if len(str(entry.get("subtask_output", ""))) > 100 else entry.get("subtask_output", ""),
                            "api_used": entry.get("api_name", "unknown")
                        })

            llm_logger.log_api_selection_call(
                query_id=query_id,
                subtask_id=subtask_id,
                subtask_input=query,
                original_query=original_query,
                llm_prompt=full_prompt,
                llm_response=selection_result if selection_result else llm_response_raw,
                available_tools=tools_info,
                selected_api=selected_api,
                selected_provider=selected_provider,
                expected_golden_api=expected_golden_api,
                exact_match_accuracy=exact_match_accuracy,
                reasoning=reasoning,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                llm_seed=llm_seed,
                previous_context=previous_context_formatted,
                selection_confidence=confidence,
                error_type=error_type,
                error_message=error_message
            )

    return selected_provider, selected_api



def evaluate_api_selection_for_subtask(scenario: Dict, 
                                        golden_log_entries: Dict, 
                                        tool_manager, expected_golden_api: str, 
                                        query_data: Dict = None, llm_seed: int | None = None,
                                        model_name: str = "gpt-4.1-2025-04-14",
                                        pmpt_without_api_res: bool = True,
                                        prompt_loader: PromptLoader = None,
                                        prompt_data_collection: List[Dict] = [],
                                        consider_provider_in_exact_match: bool = False) -> Dict:
    """Evaluate single API selection for a specific subtask - simplified step-wise evaluation"""
    

    
    subtask_id = scenario["target_subtask_id"]
    subtask_input = scenario["subtask_input"]
    dependencies = scenario.get("dependencies", [])
    # Normalize dependency IDs to integers to ensure proper lookup in golden_log_entries
    try:
        dependencies = [int(dep) for dep in dependencies]
    except Exception:
        # If conversion fails for any entry, keep original values; this will surface errors in lookups
        pass
    is_processing_step = scenario.get("is_processing_step", False)
    
    # Handle processing subtasks (no API selection needed)
    if is_processing_step or expected_golden_api == "":
        print(f"   ⚙️  Processing subtask {subtask_id} - skipping API selection (no API needed)")
        
        # Get task decomposition info for prompt data
        # _, actual_source, actual_file_path = get_tmdb_apis()
        
        # Collect prompt data for processing subtasks too
        prompt_data_entry = {
            "query_text": query_data.get("query", "") if query_data else "",
            "subtask_id": subtask_id,
            "subtask_input": subtask_input,
            "previous_context": [],  # Build below for consistency
            "available_tools": [],
            "selected_api": "",  # No API selected for processing
            "expected_golden_api": expected_golden_api,
            "dependencies": dependencies,
            "model_name": model_name,
            "pmpt_without_api_res": pmpt_without_api_res,
            # "task_decomposition_source": actual_source,
            # "task_decomposition_description": actual_file_path if actual_file_path else "evolution_history",
            "timestamp": datetime.now().isoformat(),
            "exact_match_accuracy": 1.0,
            "reasoning": "Processing subtask - no API selection required"
        }
        prompt_data_collection.append(prompt_data_entry)
        
        return {
            "selected_api": "",
            "exact_match_accuracy": 1.0,
            "reasoning": "Processing subtask - no API selection required",
            "subtask_id": subtask_id,
            "expected_golden_api": expected_golden_api
        }
    
    # Build previous log from golden log entries for dependencies
    previous_log = []
    for dep_id in dependencies:
        if dep_id in golden_log_entries:
            log_entry = golden_log_entries[dep_id].copy()
            
            # Note: api_response is no longer stored in log entries (already cached)
            # The pmpt_without_api_res functionality is automatically achieved
            if pmpt_without_api_res and "api_response" in log_entry:
                del log_entry["api_response"]
                
            previous_log.append(log_entry)
    
    # Get available tools for API selection
    # Get API guidelines for API selection (not tracking decomposition source here)
    # api_guidelines, actual_source, actual_file_path = get_tmdb_apis()
    available_tools = tool_manager.tools
    
    # Perform SINGLE API selection for step-wise evaluation
    selected_provider, selected_api = select_single_api_with_model(
        subtask_input, available_tools, previous_log, llm_seed=llm_seed,
        prompt_loader=prompt_loader, model_name=model_name,
        query_id=query_data.get("id") if query_data else None,
        subtask_id=subtask_id,
        original_query=query_data.get("query", "") if query_data else "",
        expected_golden_api=expected_golden_api,
        consider_provider_in_exact_match=consider_provider_in_exact_match)
    
    # Collect prompt data for step-wise evaluation
    prompt_data_entry = {
        "query_text": query_data.get("query", "") if query_data else "",
        "subtask_id": subtask_id,
        "subtask_input": subtask_input,
        "previous_context": previous_log.copy() if previous_log else [],
        "available_tools": available_tools,
        "selected_provider": selected_provider,
        "selected_api": selected_api,
        "expected_golden_api": expected_golden_api,
        "dependencies": dependencies,
        "model_name": model_name,
        "pmpt_without_api_res": pmpt_without_api_res,
        # "task_decomposition_source": actual_source,
        # "task_decomposition_description": actual_file_path if actual_file_path else "evolution_history",
        "timestamp": datetime.now().isoformat()
    }
    prompt_data_collection.append(prompt_data_entry)
    
    if selected_api == "CONTEXT_WINDOW_EXCEEDED_ERROR":
        return {
            "selected_provider": None,
            "selected_api": None,
            "exact_match_accuracy": 0.0,
            "error": "Context window exceeded"
        }
    
    if not selected_api:
        return {
            "selected_provider": None,
            "selected_api": None,
            "exact_match_accuracy": 0.0,
            "error": "No API selected"
        }
    
    # Enhanced comparison with semantic equivalence checking
    # api_selection_correct = check_api_semantic_equivalence(
    #     selected_api, expected_golden_api, scenario, golden_log_entries
    # )
    # api_selection_accuracy = 1.0 if api_selection_correct else 0.0
    
    # Track exact match accuracy
    # exact_match_accuracy = 1.0 if (selected_api == expected_golden_api) else 0.0

    if consider_provider_in_exact_match:
        exact_match_accuracy = 1.0 if (
            selected_provider is not None and
            selected_provider == expected_golden_api[0] and
            selected_api == expected_golden_api[1]
        ) else 0.0
    else:
        exact_match_accuracy = 1.0 if (selected_api == expected_golden_api[1]) else 0.0
    
    # Create detailed reasoning including semantic equivalence info
    if exact_match_accuracy == 1.0:
        analysis = f"Expected: {expected_golden_api} | Selected: ({selected_provider}, {selected_api}) | ✅ Exact Match"
    # elif api_selection_correct:
    #     reasoning = f"Expected: {expected_golden_api} | Selected: {selected_api} | ✅ Semantically Equivalent"
    else:
        analysis = f"Expected: {expected_golden_api} | Selected: ({selected_provider}, {selected_api}) | ❌ Incorrect"
    
    print(f"   🎯 API Selection: {analysis}")
    
    # Update the last prompt data entry with the results
    if prompt_data_collection:
        prompt_data_collection[-1].update({
            "exact_match_accuracy": exact_match_accuracy,
            "analysis": analysis
        })
    
    return {
        "selected_provider": selected_provider,
        "selected_api": selected_api,
        "exact_match_accuracy": exact_match_accuracy,
        "analysis": analysis,
        "subtask_id": subtask_id,
        "expected_golden_api": expected_golden_api
    }


def evaluate_single_subtask_with_cache(scenario: Dict, subtask_cache: Dict, tool_manager,
                                        expected_golden_api: str = "",
                                        llm_seed: int | None = None,
                                        model_name: str = "gpt-4.1-2025-04-14",
                                        temperature: float = 0.2,
                                        top_p: float = 1.0,
                                        max_tokens: int = 32768,
                                        parameter_generation_prompt_version: str = "v3",
                                        simple_api_instruction: bool = True,
                                        use_first_attempt_success: bool = False,
                                        query_id: str = None,
                                        original_query: str = "",
                                        dataset: str = None, rapidapi_wrapper=None) -> Dict:

    """Evaluate a single subtask in isolation, using cached results from earlier subtasks"""
    # Extract scenario information
    subtask_id = scenario.get("target_subtask_id")
    subtask_input = scenario.get("subtask_input", "")
    dependencies = scenario.get("dependencies", [])
    selected_api_name = scenario.get("selected_api_name")
    # selected_description_index = scenario.get("selected_description_index")
    selected_description = scenario.get("selected_description")

    print(f"\n🔍 Evaluating Subtask {subtask_id}: {selected_api_name}")
    print(f"   📝 Description: {selected_description[:100] if selected_description else 'None'}...")
    print(f"   📋 Input: {subtask_input[:200]}...")

    # Check if this subtask should execute an API based on expected_golden_api
    if expected_golden_api == "":
        # This is a processing step - no API execution needed
        should_execute_api = False
        api_used = None
        print(f"   ⏭️  Skipping API execution: This is a processing step (expected_golden_api is empty)")
    else:
        # This is an API step - execute the expected golden API
        should_execute_api = True
        api_used = expected_golden_api
        print(f"   🎯 Using expected golden API for execution: {expected_golden_api}")

    # Build previous_log from subtask_cache for dependency-aware functions
    previous_log = []
    
    # Filter out None keys before sorting to avoid TypeError
    valid_cached_ids = [k for k in subtask_cache.keys() if k is not None]
    for cached_subtask_id in sorted(valid_cached_ids):
        if cached_subtask_id < subtask_id:
            cached_result = subtask_cache[cached_subtask_id]
            if isinstance(cached_result, dict):
                log_entry = {
                    "subtask_id": cached_subtask_id,
                    "subtask_input": cached_result.get("subtask_input", ""),
                    "subtask_output": cached_result.get("subtask_output", ""),
                    "expected_golden_api": cached_result.get("expected_golden_api", "")
                    # "api_response": cached_result.get("api_response", {})  # Removed: already cached
                }
                previous_log.append(log_entry)

    # API execution logic
    if should_execute_api and api_used:
        # Generate parameters using the subtask input and the API to execute
        if tool_manager is None:
            raise ValueError("tool_manager is required for API execution but was not initialized")

        api_info = None
        for tool in tool_manager.tools:
            if tool["tool_provider"] == api_used[0] and tool["tool_name"] == api_used[1]:
                api_info = tool
                break
        
        if api_info is None:
            # print(f"   ❌ API '{api_used}' not found in tool manager")
            # return {
            #     "subtask_id": subtask_id,
            #     "subtask_input": subtask_input,
            #     "subtask_output": "",
            #     "expected_golden_api": expected_golden_api,
            #     "api_parameters": {},
            #     "api_success": False,
            #     "dependencies": dependencies,
            #     "parameter_generation_prompt_version": parameter_generation_prompt_version,
            #     "error": f"API '{api_used}' not found in tool manager"
            # }
            raise ValueError(f"API '{api_used}' not found in tool manager")
        
        required_parameters = {}
        optional_parameters = {}
        for k,v in api_info.get("parameters", {}).items():
            # Handle 'required' field that may be string 'true'/'false' or boolean True/False
            required_value = v.get("required", False)
            is_required = required_value if isinstance(required_value, bool) else (required_value == 'true' or required_value == True)
            
            if is_required:
                required_parameters[k] = v
            else:
                optional_parameters[k] = v
        
        # Use the description from scenario, but parameters from the API we're actually executing
        if api_used is not None:
            api_guidelines = {
                f"{api_used[0]}::{api_used[1]}": {
                    "description": selected_description,
                    "required_parameters": required_parameters,
                    "optional_parameters": optional_parameters,
                    "metadata": api_info["_metadata"]
                }
            }
        else:
            api_guidelines = {}
        
        try:
            # Prepare golden API information for logging
            golden_api_info = {
                "name": expected_golden_api,
                "description": selected_description,
                "parameters": {
                    "required_parameters": required_parameters,
                    "optional_parameters": optional_parameters
                }
            }
            
            # Get parameters with retry logic but track parameter generation success
            parameters_result, parameter_generation_success = choose_parameter_depend_with_first_attempt_tracking(
                api_guidelines, subtask_input, previous_log, model_name,
                simple_api_instruction=simple_api_instruction,
                parameter_generation_prompt_version=parameter_generation_prompt_version,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                n_attempts=5, use_first_attempt_success=use_first_attempt_success, llm_seed=llm_seed,
                query_id=query_id, subtask_id=subtask_id, original_query=original_query, golden_api=golden_api_info)
            print(f"   🎯 Parameters: {parameters_result}")
            print(f"   🎯 Parameter Generation Success: {'✅' if parameter_generation_success else '❌'} ({parameter_generation_success})")

            # Handle different parameter result formats and try API calls
            api_response = None
            final_api_success = False
            parameters_used = {}
            
            # Try different parameter formats/attempts
            # Track FIRST attempt separately for metrics (not retries)
            first_attempt_params = {}
            first_attempt_response = None
            first_attempt_success = False
            
            if isinstance(parameters_result, list):
                # Try each parameter set in the list
                for i, param_set in enumerate(parameters_result):
                    print(f"   🔄 Trying parameter set {i+1}: {param_set}")
                    api_response = call_api_mcp(api_used[1], param_set, api_info["_metadata"], 
                                               platform=dataset, query_id=query_id, subtask_id=subtask_id, rapidapi_wrapper=rapidapi_wrapper, tool_provider=api_used[0])
                    api_response = clean_api_response(api_response)
                    current_success = not bool(api_response.get("error", ""))
                    
                    # Record FIRST attempt for metrics
                    if i == 0:
                        first_attempt_params = param_set
                        first_attempt_response = api_response
                        first_attempt_success = current_success
                        print(f"   📊 [METRICS] First attempt recorded: success={current_success}")
                    
                    if current_success:
                        parameters_used = param_set
                        final_api_success = current_success
                        if i == 0:
                            print(f"   ✅ API call succeeded with parameter set {i+1} (first attempt)")
                        else:
                            print(f"   ✅ API call succeeded with parameter set {i+1} (retry for ground truth only)")
                        break
                    else:
                        print(f"   ❌ API call failed with parameter set {i+1}: {api_response.get('error', 'Unknown error')}")
                
                # If all attempts failed, use the last response for ground truth
                if not final_api_success:
                    parameters_used = first_attempt_params  # Use first attempt params
                    # api_response and final_api_success already set from last iteration
                    
            elif isinstance(parameters_result, dict):
                # Single parameter set
                parameters_used = parameters_result
                first_attempt_params = parameters_result
                
                if "uri" in parameters_used:
                    print(f"   🔄 Parameters: {parameters_used}")
                api_response = call_api_mcp(api_used[1], parameters_used, api_info["_metadata"],
                                           platform=dataset, query_id=query_id, subtask_id=subtask_id, rapidapi_wrapper=rapidapi_wrapper, tool_provider=api_used[0])
                api_response = clean_api_response(api_response)
                final_api_success = not bool(api_response.get("error", ""))
                first_attempt_response = api_response
                first_attempt_success = final_api_success
            else:
                # Empty or invalid parameters
                parameters_used = {}
                first_attempt_params = {}
                raw_api_response = call_api_mcp(api_used[1], parameters_used, api_info["_metadata"],
                                           platform=dataset, query_id=query_id, subtask_id=subtask_id, rapidapi_wrapper=rapidapi_wrapper, tool_provider=api_used[0])
                api_response = clean_api_response(raw_api_response)
                final_api_success = not bool(raw_api_response.get("error", ""))
                first_attempt_response = api_response
                first_attempt_success = final_api_success

            # For metrics: Use FIRST attempt only
            # For ground truth: Use final successful result
            api_success_for_metrics = first_attempt_success
            api_success = final_api_success  # Keep for ground truth collection
            
            if not final_api_success:
                print(f"   ❌ API Call Failed (all attempts): {api_response.get('error', 'Unknown error')}")
            parameters = parameters_used  # Final parameters used (for ground truth)
            print(f"   ✅ API Call Success (Final - Ground Truth): {'✅' if final_api_success else '❌'}")
            print(f"   📊 API Call Success (Metrics - First Attempt Only): {'✅' if first_attempt_success else '❌'}")
            print(f"   🎯 Parameter Generation Success Rate: {parameter_generation_success}")
            
            # Validate parameter generation quality separately from API execution
            # IMPORTANT: Use FIRST attempt only for metrics, not the successful retry
            param_quality_eval = evaluate_parameter_generation_quality(
                generated_params=first_attempt_params,  # Use first attempt
                required_parameters=required_parameters,
                optional_parameters=optional_parameters,
                api_success=first_attempt_success,  # Use first attempt
                api_error_message=first_attempt_response.get('error', '') if isinstance(first_attempt_response, dict) else '',
                api_response=first_attempt_response  # Use first attempt
            )
            
            # Print parameter validation results
            if param_quality_eval["params_valid"]:
                print(f"   ✅ Parameters: Valid")
            else:
                print(f"   ❌ Parameters: Invalid")
                for error in param_quality_eval["parameter_validation_errors"]:
                    print(f"      - {error}")
            
            if not first_attempt_success:  # Use first attempt for metrics
                print(f"   📋 API Error Category: {param_quality_eval['api_error_category']}")
            
            # Log the complete parameter generation + API execution result
            # IMPORTANT: Log FIRST attempt only for metrics tracking
            from tool_exec_tracer.eval.tmdb.utils.llm_call_logger import get_global_llm_logger
            llm_logger = get_global_llm_logger()
            if llm_logger and parameter_generation_success:
                # Get the prompt used (reconstruct it)
                prompt_file_path = f"{ROOT_DIR}/eval/tmdb/prompts/parameter_generation_{parameter_generation_prompt_version}.txt"
                try:
                    with open(prompt_file_path, 'r') as f:
                        prompt_template = f.read()
                    formatted_previous_log = format_previous_context(previous_log) if previous_log else "[]"
                    llm_prompt = prompt_template.format(
                        previous_log=formatted_previous_log,
                        api_instruction=api_guidelines,
                        question=subtask_input
                    )
                except:
                    llm_prompt = "[Prompt reconstruction failed]"
                
                # Extract API error message if present (from FIRST attempt)
                api_error_msg = ""
                if not first_attempt_success and first_attempt_response:
                    if isinstance(first_attempt_response, dict):
                        error_content = first_attempt_response.get("error", "")
                        if error_content:
                            # Format like "HTTP 400: {...}" if possible
                            api_error_msg = str(error_content)
                
                llm_logger.log_parameter_generation_call(
                    query_id=query_id,
                    subtask_id=subtask_id,
                    subtask_input=subtask_input,
                    original_query=original_query,
                    llm_prompt=llm_prompt,
                    llm_response={"Parameters": first_attempt_params},  # Use first attempt
                    golden_api=golden_api_info,
                    api_success=first_attempt_success,  # Use first attempt
                    api_response=first_attempt_response,  # Use first attempt
                    api_error_message=api_error_msg,  # From first attempt
                    parameter_quality_evaluation=param_quality_eval  # Already uses first attempt
                )
            
        except Exception as e:
            import traceback
            print(f"   💥 Error during API execution: {e}")
            print(f"   📍 Error type: {type(e).__name__}")
            print(f"   📍 Traceback:")
            traceback.print_exc()
            api_response = {"error": str(e), "response": ""}
            api_success = False
            parameters = {}
            
            # Log the error case
            from tool_exec_tracer.eval.tmdb.utils.llm_call_logger import get_global_llm_logger
            llm_logger = get_global_llm_logger()
            if llm_logger:
                llm_logger.log_parameter_generation_call(
                    query_id=query_id,
                    subtask_id=subtask_id,
                    subtask_input=subtask_input,
                    original_query=original_query,
                    llm_prompt="[Error occurred during execution]",
                    llm_response={},
                    golden_api=golden_api_info if 'golden_api_info' in locals() else {},
                    api_success=False,
                    api_response=api_response,
                    api_error_message=str(e)
                )
    else:
        # Skip API execution for processing steps or when no API specified
        api_response = {"error": "", "response": ""}
        api_success = True  # Processing steps are considered successful by default
        parameters = {}
        # api_used already set to "" above for processing steps
        print(f"   ✅ Processing Step: Skipped API execution")
        # a dummy one
        api_guidelines = {
            "dummy_api": {
                "description": "dummy_description",
                "required_parameters": {},
                "optional_parameters": {}
            }
        }
    if "vllm" in model_name:
        api_response_truncated, truncation_ratio = truncate_content(str(api_response))
    else:
        api_response_truncated = str(api_response)
        truncation_ratio = 1.0
    try:
        # Generate natural language answer
        # NOTE: Generate natural language answer for the subtask.
        subtask_output = answer_generation_depend(subtask_input, api_response_truncated, previous_log, model_name, 
                                                    temperature, top_p, max_tokens, seed=llm_seed)
        print(f"   📝 Generated Answer: {subtask_output[:200]}...")
        
        
        # Include parameter quality evaluation if available
        # IMPORTANT: Use first_attempt_success for metrics (api_success field)
        # Use final api_success for ground truth collection success
        result = {
            "subtask_id": subtask_id,
            "subtask_input": subtask_input,
            "subtask_output": subtask_output,
            # "subtask_answer": subtask_output,
            "expected_golden_api": expected_golden_api,
            "golden_api_guidelines": api_guidelines,
            "api_parameters": parameters,
            # "api_response": api_response,  # Removed: already cached
            "api_success": api_success_for_metrics if 'api_success_for_metrics' in locals() else api_success,  # Use first attempt for metrics
            "ground_truth_collected": api_success,  # Whether we successfully collected ground truth (final attempt)
            # "description_used": selected_description_index,
            "dependencies": dependencies,
            "parameter_generation_prompt_version": parameter_generation_prompt_version
        }
        
        # Add parameter quality evaluation if it was performed
        if 'param_quality_eval' in locals():
            result["parameter_quality_evaluation"] = param_quality_eval
        
        return result
        
    except Exception as e:
        print(f"   💥 Error in subtask evaluation: {e}")
        return {
            "subtask_id": subtask_id,
            "subtask_input": subtask_input,
            "subtask_output": "",
            "expected_golden_api": expected_golden_api,
            "golden_api_guidelines": api_guidelines,
            "api_parameters": {},
            "api_success": False,
            # "description_used": selected_description_index,
            "dependencies": dependencies,
            "parameter_generation_prompt_version": parameter_generation_prompt_version,
            "error": str(e)
        }


def safe_mean(lst: List[float]) -> float:
    """Calculate mean safely handling empty lists"""
    return sum(lst) / len(lst) if lst else 0


def safe_std(lst: List[float]) -> float:
    """Calculate standard deviation safely handling small lists"""
    if len(lst) < 2:
        return 0
    mean = safe_mean(lst)
    variance = sum((x - mean) ** 2 for x in lst) / (len(lst) - 1)
    return variance ** 0.5


def clean_api_response(api_response: Dict[str, Any]) -> Dict[str, Any]:
    """Clean API response to remove very large content"""
    if not api_response or "response" not in api_response:
        return api_response
    
    cleaned_response = api_response.copy()
    
    # Remove very large response content but keep structure for debugging
    if isinstance(cleaned_response.get("response"), (dict, list, str)):
        response_str = str(cleaned_response["response"])
        if len(response_str) > 10000:  # Only keep first 10k characters
            cleaned_response["response"] = response_str[:10000] + "... [TRUNCATED]"
    
    return cleaned_response


def format_previous_context(previous_context: List[Dict[str, Any]]) -> str:
    """Format previous context for the prompt."""
    if not previous_context:
        return "No previous context available"
    
    formatted_contexts = []
    for i, ctx in enumerate(previous_context):
        subtask_id = ctx.get("subtask_id", i + 1)
        subtask_input = ctx.get("subtask_input", "Unknown input")
        subtask_output = ctx.get("subtask_output", "Unknown output")
        expected_golden_api = ctx.get("expected_golden_api", "")
        
        context_str = f"Subtask {subtask_id}: {subtask_input[:100]}... → {subtask_output[:100]}..."
        if expected_golden_api:
            context_str += f" (API: {expected_golden_api})"
        
        formatted_contexts.append(context_str)
    
    return "\n".join(formatted_contexts)


def choose_parameter_depend_with_first_attempt_tracking(API_instruction, 
question, previous_log, model_name, simple_api_instruction=True,
parameter_generation_prompt_version="v0", temperature=0.2, top_p=1.0, max_tokens=32768, 
n_attempts=5, use_first_attempt_success=False, llm_seed: int | None = None, query_id: str = None, subtask_id: int = None, original_query: str = "", golden_api: Dict = None):
    """
    Modified version of choose_parameter_depend that tracks parameter generation success.
    Returns (parameters, parameter_generation_success) where parameter_generation_success indicates either:
    - first_attempt_success: whether the first parameter generation attempt was successful (if use_first_attempt_success=True)
    - average success rate: average success across all attempts (if use_first_attempt_success=False, default)
    
    Note: Success is determined by successful LLM call and JSON parsing. Empty parameters are 
    considered successful - only LLM call failures or JSON parsing errors count as failures.
    This function only handles parameter generation, not API execution.
    
    Args:
        query_id: ID of the query being processed (for logging)
        subtask_id: ID of the subtask (for logging)
        original_query: The original full query (for logging)
        golden_api: Expected golden API info with name, description, parameters (for logging)
    """
    from tool_exec_tracer.eval.tmdb.utils.llm_call_logger import get_global_llm_logger
    
    # template = "You are a helpful assistant that generates parameters for an API call."
    
    # Load prompt from file
    prompt_file_path = f"./eval/tmdb/prompts/parameter_generation_{parameter_generation_prompt_version}_user.txt"
    with open(prompt_file_path, 'r') as f:
        prompt_template_user = f.read()

    prompt_file_path = f"./eval/tmdb/prompts/parameter_generation_{parameter_generation_prompt_version}_system.txt"
    with open(prompt_file_path, 'r') as f:
        prompt_template_system = f.read()
    
    # Replace placeholders with actual values

    if not simple_api_instruction:
        API_instruction_processed = API_instruction
    else:
        API_instruction_processed = {}
        for key, value in API_instruction.items():
            API_instruction_processed[key] = {}
            API_instruction_processed[key]["description"] = value["description"]
            API_instruction_processed[key]["metadata"] = {}
            API_instruction_processed[key]["metadata"]["endpoint"] = value["metadata"]["endpoint"]
            API_instruction_processed[key]["metadata"]["method"] = value["metadata"]["method"]
            API_instruction_processed[key]["metadata"]["category"] = value["metadata"]["category"]

    user_prompt = prompt_template_user.format(
        previous_log=previous_log,
        api_instruction=API_instruction,
        question=question
    )
    messages = [
            {"role": "system", "content": prompt_template_system},
            {"role": "user", "content": user_prompt}
        ]
    
    first_attempt_success = False
    successful_attempts = 0
    total_attempts = 0
    n_attempts = n_attempts
    llm_logger = get_global_llm_logger()
    
    for attempt in range(n_attempts):  # Try up to n_attempts times
        total_attempts += 1
        try:
            print(f"[PROMPT] {user_prompt[:300]}{'...' if len(user_prompt) > 300 else ''}")
            # NOTE: Generate parameters for the API, using the selected API. (could be either golden or LLM-selected)
            result = llm_call(messages, model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=llm_seed, step_name="parameter_generation")
            print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
            
            # Extract content from the response
            if result and hasattr(result, 'choices') and result.choices:
                content = result.choices[0].message.content
            else:
                error_msg = f"Attempt {attempt + 1}: No valid response from LLM"
                print(f"⚠️  {error_msg}")
                
                # Don't log individual failed attempts - only log final result after API execution
                
                if attempt < n_attempts - 1:  # Continue if not the last attempt
                    continue
                else:
                    # Calculate final success rate
                    if use_first_attempt_success:
                        final_success_rate = first_attempt_success
                    else:
                        final_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
                    return {}, final_success_rate
            
            # Clean the content to extract JSON
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content_json = content.replace('```json\n', '').replace('\n```', '')
            elif content.startswith('```'):
                content_json = content.replace('```\n', '').replace('\n```', '')
            elif not content.strip().startswith('{'):
                    content_json = f"{{{content}}}"
            else:
                content_json = content

            is_valid, error = validate_json(content_json)
            if not is_valid:
                # parse the last part of the content, enclosed with '{' and '}'
                lastpart = content.split('{')[-1].split('}')[0]
                lastpart = f"{{{lastpart}}}"
                is_valid, error = validate_json(lastpart)
                if is_valid:
                    content_json = json.dumps({
                        "Reasoning": "",
                        "Parameters": json.loads(lastpart)
                    })
                elif "Expecting property name enclosed in double quotes" in error.args[0]:
                    # convert single quotes to double quotes
                    lastpart = lastpart.replace("'", '"')
                    is_valid, error = validate_json(lastpart)
                    if is_valid:
                        content_json = json.dumps({
                            "Reasoning": "",
                            "Parameters": json.loads(lastpart)
                        })
                    else:
                        print(f"⚠️  Attempt {attempt + 1}: Invalid JSON: {lastpart}")
                else:
                    print(f"⚠️  Attempt {attempt + 1}: Invalid JSON: {lastpart}")
            
            # Try to parse as JSON
            try:
                parsed_result = json.loads(content_json)
                # Validate that parsed_result is a dictionary
                if not isinstance(parsed_result, dict):
                    error_msg = f"Attempt {attempt + 1}: LLM returned non-dict JSON: {type(parsed_result)}"
                    print(f"⚠️  {error_msg}")
                    
                    # Don't log individual failed attempts - only log final result after API execution
                    
                    if attempt < n_attempts - 1:
                        continue
                    else:
                        if use_first_attempt_success:
                            final_success_rate = first_attempt_success
                        else:
                            final_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
                        return {}, final_success_rate
            except json.JSONDecodeError as e:
                error_msg = f"Attempt {attempt + 1}: JSON decode error: {e}"
                print(f"⚠️  {error_msg}")
                print(f"Raw content: {content}")
                
                # Don't log individual failed attempts - only log final result after API execution
                
                if attempt < n_attempts - 1:  # Continue if not the last attempt
                    continue
                else:
                    # Calculate final success rate
                    if use_first_attempt_success:
                        final_success_rate = first_attempt_success
                    else:
                        final_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
                    return {}, final_success_rate
            
            parameters = parsed_result.get("Parameters", {})
            
            # Handle case where LLM returns a list with single dict instead of just the dict
            if isinstance(parameters, list) and len(parameters) == 1 and isinstance(parameters[0], dict):
                parameters = parameters[0]
            
            # Parameter generation is successful if we got valid JSON response
            # Empty parameters are still considered successful - only LLM/JSON failures are not
            attempt_successful = True  # We reached here, so LLM call and JSON parsing succeeded
            
            # Track first attempt success for fair measurement
            if attempt == 0:
                first_attempt_success = attempt_successful
            
            # Track successful attempts for average calculation
            successful_attempts += 1
            
            # Log the result (empty parameters are OK)
            if parameters and parameters != {}:
                print(f"✅ Got parameters on attempt {attempt + 1}: {parameters}")
            else:
                print(f"✅ Got empty parameters on attempt {attempt + 1} (this is OK)")
            
            # Note: Successful attempts will be logged after API execution with complete info
            # We don't log successful parameter generation here to avoid duplicate entries
            
            # Calculate final success rate
            if use_first_attempt_success:
                final_success_rate = first_attempt_success
            else:
                final_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
            return parameters, final_success_rate
                
        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed with error: {e}"
            print(f"⚠️  {error_msg}")
            
            # Don't log individual failed attempts - only log final result after API execution
            
            if attempt < n_attempts - 1:  # Continue if not the last attempt
                continue
            else:
                print(f"❌ All {n_attempts} attempts failed due to errors")
                # Calculate final success rate
                if use_first_attempt_success:
                    final_success_rate = first_attempt_success
                else:
                    final_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
                return {}, final_success_rate
    
    # Calculate final success rate (fallback)
    if use_first_attempt_success:
        final_success_rate = first_attempt_success
    else:
        final_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
    return {}, final_success_rate
