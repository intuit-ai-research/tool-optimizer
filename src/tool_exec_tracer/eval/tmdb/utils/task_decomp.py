from tool_exec_tracer.utils.llm import llm_call
import json
from typing import List, Dict, Tuple

from pathlib import Path


def format_tools_for_decomposition(tools: List[Dict], 
                                    include_parameters: bool = False, 
                                    desc_in_task_decomp: bool = False ) -> str:
    """
    Format tools cleanly for task decomposition by removing metadata noise.
    
    Args:
        tools: Raw tools list from YAML/config
        include_parameters: If True, include simplified parameter info
        
    Returns:
        Clean formatted string of tools suitable for task decomposition
    """
    formatted_tools = []
    
    for idx, tool in enumerate(tools, 1):
        tool_name = tool.get('tool_name', '')
        description = tool.get('description', '')

        param_str = ""
        if include_parameters:
            # Extract only essential parameter info (names of required params)
            params = tool.get('parameters', {})
            required_params = [
                param_name for param_name, param_info in params.items() 
                if isinstance(param_info, dict) and (
                    param_info.get('required') == 'true' or 
                    param_info.get('required') == True
                )
            ]
            
            if required_params:
                param_str = f" (requires: {', '.join(required_params)})"
            else:
                param_str = ""
        
        if desc_in_task_decomp:
            formatted_tools.append(f"{idx}. {tool_name}: {description}{param_str}")
        else:
            formatted_tools.append(f"{idx}. {tool_name}{param_str}")
    
    return '\n'.join(formatted_tools)


def task_decompose(query, tool_manager, 
ROOT_DIR, 
model_name="gpt-4.1-2025-04-14", prompt_version="v3", temperature=0.2, top_p=1.0, max_tokens=32768, 
seed: int | None = None, include_parameters: bool = False, desc_in_task_decomp=False):
    """
    Decompose a complex question into simple subtasks.
    
    Args:
        question: The complex question to decompose
        tool_manager: Tool manager for extracting API names from endpoints
        ROOT_DIR: Root directory for loading prompts
        model_name: LLM model to use
        prompt_version: Version of the prompt template to use
        temperature: Temperature for LLM generation
        top_p: Top-p parameter for LLM generation
        max_tokens: Maximum tokens for LLM generation
        seed: Random seed for reproducibility
        include_parameters: If True, include required parameter info in tool formatting
        
    Returns:
        Dictionary with 'Tasks' key containing list of subtasks
    """
    model_name = "gpt-4.1-2025-04-14"
    tool_list = tool_manager.tools

    # Format tools cleanly (remove metadata noise)
    formatted_tools = format_tools_for_decomposition(tool_list, 
    include_parameters=include_parameters, 
    desc_in_task_decomp=desc_in_task_decomp)

    # Extract Solution
    solution = query["solution"]
    golden_apis : List[List[str]] = solution
    golden_apis_desc = ""
    counter=1
    for tool in tool_manager.tools:
        if tool["tool_name"] in golden_apis:
            golden_apis_desc += f"  - API {counter}: {tool['description']}\n"
            counter += 1
    
    placeholders = {
        "tools": formatted_tools,
        "question": query['query'],
        "golden_apis_names": golden_apis,
        "golden_apis_desc": golden_apis_desc,
    }
    
    prompt_file_path = Path(f"{ROOT_DIR}/eval/tmdb/prompts/task_decomposition_{prompt_version}")
    print(f"[PROMPT FILE PATH] {prompt_file_path}")
    # check if prompt path is a directory (prompts folder) or a file (txt)
    if prompt_file_path.is_dir():
        with open(prompt_file_path/'system_prompt.txt', 'r') as f:
            system_prompt = f.read()
        with open(prompt_file_path/'user_prompt.txt', 'r') as f:
            user_prompt = f.read()
        user_prompt = user_prompt.format(**placeholders)
    else:
        # with .txt extension
        prompt_file_path = prompt_file_path.with_suffix('.txt')
        system_prompt = "You are a task‑decomposer agent. Given available API tools and a complex user question, produce a sequence of simple, natural‑language subtasks so each can be executed by one API tool. Use the rules below and output only valid JSON."
        # Load prompt from file based on version
        with open(prompt_file_path, 'r') as f:
            prompt_template = f.read()
        
        # Replace placeholders with actual values
        user_prompt = prompt_template.format(**placeholders)
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    print(f"[PROMPT] {user_prompt[:300]}{'...' if len(user_prompt) > 300 else ''}")
    
    # Initial attempt with provided temperature
    # NOTE: Task decomposition step.
    max_retries = 5
    for retry_attempt in range(1, max_retries + 1):
        print(f"Retry attempt {retry_attempt}/{max_retries} with temperature={temperature}")
        result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, step_name="task_decomp")
        if result is not None:
            break
    print(f"[QUESTION] {query['query']}")
    print(f"[GOLDEN SOLUTION] {query['solution']}")
    
    # Extract content from the response
    if result and hasattr(result, 'choices') and result.choices:
        content = result.choices[0].message.content
        print(f"[SUBTASKS] {content}")
        print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
    else:
        print("Error: No valid response from LLM after all retries")
        return {"Tasks": []}
    
    # Clean the content to extract JSON
    content = content.strip()
    
    # Remove markdown code blocks if present
    if content.startswith('```json'):
        content = content.replace('```json\n', '').replace('\n```', '')
    elif content.startswith('```'):
        content = content.replace('```\n', '').replace('\n```', '')
    
    # Try to parse as JSON
    try:
        parsed_result = json.loads(content)
        
        # Check if the returned Tasks list is empty
        if not parsed_result.get('Tasks', []):
            print(f"Warning: Initial attempt returned empty subtask list. Retrying with temperature=1.0...")
            
            # Retry up to 5 times with temperature=1.0
            max_retries = 5
            for retry_attempt in range(1, max_retries + 1):
                print(f"Retry attempt {retry_attempt}/{max_retries} with temperature=1.0")
                
                retry_result = llm_call(messages, model=model_name, temperature=1.0, top_p=top_p, max_tokens=max_tokens, seed=seed, step_name="task_decomp_retry")
                print(f"[RETRY RESULT] {str(retry_result)[:100]}{'...' if len(str(retry_result)) > 100 else ''}")
                
                # Extract content from retry response
                if retry_result and hasattr(retry_result, 'choices') and retry_result.choices:
                    retry_content = retry_result.choices[0].message.content
                else:
                    print(f"Error: No valid response from LLM on retry {retry_attempt}")
                    continue
                
                # Clean the retry content
                retry_content = retry_content.strip()
                if retry_content.startswith('```json'):
                    retry_content = retry_content.replace('```json\n', '').replace('\n```', '')
                elif retry_content.startswith('```'):
                    retry_content = retry_content.replace('```\n', '').replace('\n```', '')
                
                # Try to parse retry result
                try:
                    retry_parsed_result = json.loads(retry_content)
                    
                    # Check if retry returned non-empty Tasks list
                    if retry_parsed_result.get('Tasks', []):
                        print(f"Success: Retry {retry_attempt} returned non-empty subtask list with {len(retry_parsed_result['Tasks'])} tasks")
                        return retry_parsed_result
                    else:
                        print(f"Retry {retry_attempt} still returned empty subtask list")
                        
                except json.JSONDecodeError as retry_e:
                    print(f"JSON decode error on retry {retry_attempt}: {retry_e}")
                    print(f"Raw retry content: {retry_content}")
            
            # If all retries failed, return the original empty result
            print(f"Warning: All {max_retries} retry attempts failed to return non-empty subtask list")
            return parsed_result
        
        # If initial result was non-empty, return it
        return parsed_result
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw content: {content}")
        # Return a default structure if parsing fails
        return {"Tasks": []}



def task_topology(question, task_ls, model_name, 
temperature=0.2, top_p=1.0, max_tokens=32768, seed: int | None = None, desc_in_task_decomp=False):
    """
    Determine task topology and dependencies.
    
    Args:
        question: Original question
        task_ls: List of tasks with 'task' and 'id' keys
        model_name: LLM model to use
        
    Returns:
        Updated task list with dependency information
    """
    template = "You are a helpful assistant."
    prompt = (
        "You should decide whether there are dependencies between the tasks. Dependencies means whether the execution of a task must rely on the output of another task. Please note that:\n"
        "1. If Task A needs data from Task B to complete, then Task A depends on Task B.\n"
        "2. Only consider direct dependencies, not transitive ones.\n"
        "3. Independent tasks can be executed in parallel.\n"
        f"Question: {question}\n"
        f"Tasks: {task_ls}\n"
        "Based on this question and tasks, analyze dependencies between tasks.\n"
        "You should output the result in the following JSON format:\n"
        "```json\n"
        "{\n"
        "  \"task_dependencies\": {\n"
        "    \"1\": [],\n"
        "    \"2\": [\"1\"],\n"
        "    \"3\": [\"1\", \"2\"]\n"
        "  }\n"
        "}\n"
        "```\n"
        "Where each task ID maps to a list of task IDs it depends on.\n"
        "Output:"
    )
    model_name = "gpt-4.1-2025-04-14"
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(f"[PROMPT] {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    # NOTE: Task topology step.
    result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, step_name="task_topology")
    print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
    
    # Extract content from the response
    if result and hasattr(result, 'choices') and result.choices:
        content = result.choices[0].message.content
    else:
        print("Error: No valid response from LLM")
        return task_ls
    
    # Clean the content to extract JSON
    content = content.strip()
    
    # Remove markdown code blocks if present
    if content.startswith('```json'):
        content = content.replace('```json\n', '').replace('\n```', '')
    elif content.startswith('```'):
        content = content.replace('```\n', '').replace('\n```', '')
    
    # Try to parse as JSON and add dependency information
    try:
        parsed_result = json.loads(content)
        dependencies = parsed_result.get('task_dependencies', {})
        
        # Add dependency information to task list
        for task in task_ls:
            task_id = str(task['id'])
            task['depend'] = dependencies.get(task_id, [])
            
        return task_ls
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw content: {content}")
        # Return original task list without dependency info
        for task in task_ls:
            task['depend'] = []
        return task_ls


# def extract_golden_apis(tool_manager, golden_solution: List) -> List[str]:
#     """
#     Extract golden APIs from ground truth solution as a simple list.
    
#     Returns APIs in the same order as they appear in the solution.
#     This is a POSITION-BASED list, not a semantic mapping to decomposed subtasks.
    
#     Args:
#         tool_manager: Tool manager for extracting API names from endpoints
#         golden_solution: Ground truth solution (list of API calls)
    
#     Returns:
#         List of API names in order: ["api1", "api2", ...]
#     """
#     if tool_manager is None:
#         raise ValueError("tool_manager is required for extracting API names from endpoints but was not initialized")
    
#     # golden_apis = []
    
#     # for step in golden_solution:
#         # if isinstance(step, str):
#         #     # Handle format like "GET /search/person" or "GET /person/{person_id}/movie_credits"
#         #     api_name = tool_manager.extract_api_name_from_endpoint(step)
#         #     if api_name:
#         #         golden_apis.append(api_name)
#         # elif isinstance(step, dict) and "API" in step:
#         #     # Handle format like {"API": "search_person"}
#         #     golden_apis.append(step["API"])

#     if not isinstance(golden_solution, list):
#         golden_apis = golden_solution
#     else:
#         raise ValueError("golden_solution is not a list")
    
#     return golden_apis


def identify_api_requiring_subtasks(
    subtasks: List[Dict], 
    golden_apis: List[str],
    tool_manager=None,
    query_text: str = "",
    mode: str = "semantic",
    model_name: str = "gpt-4.1-2025-04-14",
    temperature: float = 1.0,
    max_tokens: int = 4096,
    seed: int = 42) -> Tuple[List[int], List[int], Dict[int, str]]:
    """
    Identify which subtasks require API calls vs which are processing steps.
    
    Args:
        subtasks: List of subtasks from query_data
        golden_apis: List of golden APIs from solution
        tool_manager: Tool manager for semantic matching (required if mode="semantic")
        query_text: Original query text for semantic matching
        mode: "keyword" for heuristic-based, "semantic" for LLM-based matching
        model_name: LLM model name (from config, used for semantic matching)
        temperature: LLM temperature (from config, used for semantic matching)
        max_tokens: Max tokens (from config, used for semantic matching)
        seed: Random seed (from config, used for semantic matching)
        
    Returns:
        Tuple of (api_requiring_indices, processing_indices, subtask_to_api_mapping)
        - api_requiring_indices: 1-based list of subtask IDs that need APIs
        - processing_indices: 1-based list of subtask IDs that are processing steps
        - subtask_to_api_mapping: Dict mapping subtask_id → golden_api_name
    """
    num_subtasks = len(subtasks)
    num_golden_apis = len(golden_apis)
    
    # Use semantic matching if requested
    if mode == "semantic":
        if tool_manager is None:
            raise ValueError("tool_manager is required for semantic matching mode")
        
        subtask_to_api = identify_api_requiring_subtasks_with_semantic(
            subtasks, golden_apis, tool_manager, query_text,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )
        
        # Convert dict to two lists
        api_requiring_indices = sorted(subtask_to_api.keys())
        all_task_ids = set(range(1, num_subtasks + 1))
        processing_indices = sorted(all_task_ids - set(api_requiring_indices))
        
        return api_requiring_indices, processing_indices, subtask_to_api
    
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # # Fallback to heuristic-based matching (keyword or position-based)
    # if num_subtasks == num_golden_apis:
    #     # Perfect match - assume all subtasks need APIs (1-based indices)
    #     api_requiring_indices = list(range(1, num_subtasks + 1))
    #     # Create position-based mapping
    #     subtask_to_api = {i+1: golden_apis[i] for i in range(num_golden_apis)}
    #     return api_requiring_indices, [], subtask_to_api

    # elif num_subtasks > num_golden_apis:
    #     api_requiring_indices, processing_indices = identify_api_requiring_subtasks_with_keywords(
    #         subtasks, golden_apis)
    #     # Create position-based mapping for API-requiring subtasks
    #     subtask_to_api = {}
    #     for idx, task_id in enumerate(sorted(api_requiring_indices)):
    #         if idx < len(golden_apis):
    #             subtask_to_api[task_id] = golden_apis[idx]
    #     return api_requiring_indices, processing_indices, subtask_to_api

    # else:
    #     # Fewer subtasks than golden APIs - all subtasks likely need APIs (1-based indices)
    #     api_requiring_indices = list(range(1, num_subtasks + 1))
    #     # Create position-based mapping (may have more golden APIs than subtasks)
    #     subtask_to_api = {i+1: golden_apis[i] for i in range(min(num_subtasks, num_golden_apis))}
    #     return api_requiring_indices, [], subtask_to_api

def identify_api_requiring_subtasks_with_semantic(
    subtasks: List[Dict], 
    golden_apis: List[str],
    tool_manager,
    query_text: str = "",
    model_name: str = "gpt-4-turbo-2024-04-09",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    seed: int = 42) -> Dict[int, str]:
    """
    Use LLM to semantically match decomposed subtasks to ground truth APIs.
    
    Args:
        subtasks: List of decomposed subtasks with descriptions
        golden_apis: List of ground truth API names
        tool_manager: Tool manager to get API descriptions and parameters
        query_text: Original query text for context
        model_name: LLM model name (from config)
        temperature: LLM temperature (from config, recommend 0.0 for deterministic matching)
        max_tokens: Max tokens for LLM response (from config)
        seed: Random seed for reproducibility (from config)
    
    Returns:
        Dictionary mapping subtask_id (1-based) → golden_api_name
        Subtasks without API matches are not included in the dictionary
    """
    model_name = "gpt-4.1-2025-04-14"
    from tool_exec_tracer.utils.llm import llm_call
    import json
    import sys
    import os
    
    # Import prompt function
    from tool_exec_tracer.eval.tmdb.prompts.match_api_with_subtask import get_semantic_matching_prompt
    
    # Get API metadata (descriptions and parameters) from tool manager
    api_metadata = {}
    for golden_tool_name, golden_api_name in golden_apis:
        # Search for the API in the tools list
        api_info = None

        for tool in tool_manager.tools:
            # print(f"tool: {tool}")
            # print(f"api: {golden_tool_name, golden_api_name}")
            if tool["tool_provider"] == golden_tool_name and tool["tool_name"] == golden_api_name:
                api_info = tool
                break
        
        if api_info:
            api_metadata[golden_api_name] = {
                "tool_provider": golden_tool_name,
                "name": golden_api_name,
                "description": api_info.get("description", ""),
                "parameters": api_info.get("parameters", {})
            }
        else:
            # Fallback if API not found
            raise ValueError(f"API {golden_api_name} not found in tool manager")
    
    # Build API metadata section
    api_metadata_section = ""
    for i, (golden_tool_name, golden_api_name) in enumerate(golden_apis):
        metadata = api_metadata[golden_api_name]

        # api_metadata_section += f"\n{i+1}. Tool Provider: {metadata['tool_provider']}\n"
        api_metadata_section += f"\n{i+1}. API: {metadata['name']}\n"
        api_metadata_section += f"   Description: {metadata['description']}\n"
        if metadata['parameters']:
            api_metadata_section += f"   Parameters: {', '.join(metadata['parameters'].keys())}\n"
    
    # Build subtasks section
    subtasks_section = ""
    for i, subtask in enumerate(subtasks):
        task_id = i + 1
        task_desc = subtask.get("input", subtask.get("task", ""))
        dependencies = subtask.get("dependencies", [])
        dep_str = f" (depends on: {dependencies})" if dependencies else ""
        subtasks_section += f"\n{task_id}. {task_desc}{dep_str}\n"
    
    # Get prompt from template
    prompt = get_semantic_matching_prompt(query_text, api_metadata_section, subtasks_section)

    # Call LLM with config parameters
    # NOTE: Given the subtasks and the golden APIs, match the subtasks to the golden APIs.
    response = llm_call(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed, 
        step_name="semantic_matching"
    )
    
    # Parse response
    try:
        # Check if response is valid
        if response is None or not hasattr(response, 'choices') or not response.choices:
            print("💥 Error: No response from model for semantic matching")
            return {}
        
        # Extract content from response
        message = response.choices[0].message
        if not hasattr(message, 'content') or message.content is None:
            print("💥 Error: Response has no content")
            return {}
            
        response_text = message.content.strip()
        
        # Try to find JSON block
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            # Assume entire response is JSON
            json_str = response_text
        
        matching = json.loads(json_str)
        
        # Validate that matched APIs are in the golden_apis list
        subtask_to_api = {}
        print("Golden APIs: ", golden_apis)

        golden_api_name_to_tool_provider = {}
        for tool_provider, api_name in golden_apis:
            golden_api_name_to_tool_provider[api_name] = tool_provider

        for subtask_id_str, extracted_api_name in matching.items():
            subtask_id = int(subtask_id_str)

            if extracted_api_name in golden_api_name_to_tool_provider:
                subtask_to_api[subtask_id] = (golden_api_name_to_tool_provider[extracted_api_name], extracted_api_name)
            else:
                print(f"⚠️  Warning: LLM returned '{extracted_api_name}' which is not in golden_apis")

        
        print(f"✅ LLM Semantic Matching Result:")
        print(f"   Matched {len(subtask_to_api)} subtasks to APIs")
        print(f"   Processing steps: {len(subtasks) - len(subtask_to_api)}")
        for subtask_id, api_name in sorted(subtask_to_api.items()):
            print(f"   Subtask {subtask_id} → {api_name}")
        
        return subtask_to_api
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"⚠️  Failed to parse LLM response: {e}")
        print(f"   Response: {response}...")
        # Fallback: return empty dict (all processing steps)
        return {}

def identify_api_requiring_subtasks_with_keywords(subtasks: List[Dict], golden_apis: List[str]) -> Tuple[List[int], List[int]]:
    """
    Identify API-requiring vs processing subtasks using keyword analysis.
    
    Returns 1-based indices to match task IDs.
    """
    api_requiring_indices = []
    processing_indices = []
    
    # Keywords that suggest API calls are needed
    api_keywords = [
        "search for", "find", "get", "retrieve", "fetch", "obtain", "lookup", "query",
        "search", "call", "request", "api", "endpoint", "service", "discover", "locate"
    ]
    
    # Keywords that suggest data processing (no API needed)
    processing_keywords = [
        "count", "number of", "how many", "total", "sum", "calculate", "compute",
        "identify", "select", "determine", "choose", "pick", "filter", "extract",
        "compare", "versus", "vs", "which has", "which is", "difference", "between",
        "highest", "lowest", "best", "worst", "top", "bottom", "first", "last",
        "most", "least", "maximum", "minimum", "latest", "earliest", "recent",
        "rank", "order", "sort", "arrange", "list", "organize", "group",
        "analyze", "examine", "check", "review", "evaluate", "assess", "process",
        "from the list", "from the results", "using the", "based on the",
        "then", "finally", "lastly", "after that", "next"
    ]
    
    # First pass: identify obvious API-requiring vs processing subtasks
    for i, subtask in enumerate(subtasks):
        task_id = i + 1  # Convert to 1-based index
        text = subtask.get("input", "").lower()
        dependencies = subtask.get("dependencies", [])
        
        has_api_keywords = any(keyword in text for keyword in api_keywords)
        has_processing_keywords = any(keyword in text for keyword in processing_keywords)
        
        # Strong indicators of processing (even if API keywords present)
        strong_processing_indicators = [
            "select one", "choose one", "pick one", "from the list", "from the results",
            "select a", "choose a", "pick a", "select the", "choose the", "pick the"
        ]
        has_strong_processing = any(indicator in text for indicator in strong_processing_indicators)
        
        # If subtask has dependencies AND strong processing indicators, it's likely processing
        if dependencies and has_strong_processing:
            processing_indices.append(task_id)
        elif has_api_keywords and not has_processing_keywords:
            api_requiring_indices.append(task_id)
        elif has_processing_keywords and not has_api_keywords:
            processing_indices.append(task_id)
        elif has_strong_processing:
            # Strong processing indicators override mixed signals
            processing_indices.append(task_id)
        # If both or neither (and no strong processing), we'll decide later
    
    # If we haven't identified enough API-requiring subtasks, add the first ones
    remaining_needed = len(golden_apis) - len(api_requiring_indices)
    if remaining_needed > 0:
        # Add subtasks that aren't already classified as processing (use 1-based IDs)
        all_task_ids = list(range(1, len(subtasks) + 1))
        unclassified = [task_id for task_id in all_task_ids
                        if task_id not in api_requiring_indices and task_id not in processing_indices]
        # Prefer earlier subtasks (they're more likely to be API calls in linear workflows)
        for task_id in sorted(unclassified)[:remaining_needed]:
            api_requiring_indices.append(task_id)
    
    # Mark remaining unclassified as processing
    all_task_ids = list(range(1, len(subtasks) + 1))
    for task_id in all_task_ids:
        if task_id not in api_requiring_indices and task_id not in processing_indices:
            processing_indices.append(task_id)
            
    return sorted(api_requiring_indices), sorted(processing_indices)