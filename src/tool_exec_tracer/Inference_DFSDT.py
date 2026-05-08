"""
Cleaned version of Inference_DFSDT.py containing only functions used by the evaluation system.

This file contains:
- task_decompose: Used for breaking down complex queries into subtasks
- task_topology: Used for determining task dependencies  
- answer_generation_depend: Used for generating answers with context
- answer_check: Used for validating generated answers
- Global variables: temperature, top_p, max_tokens
"""

import json
import sys
import os
from pathlib import Path
from typing import Tuple
from tool_exec_tracer.utils.llm import llm_call

# Note: temperature, top_p, and max_tokens are now passed as function parameters
# from the argparse configuration in run_evaluation.py


def truncate_content(content: str, max_tokens: int = 2000) -> Tuple[str, float]:
    """
    Truncate content to specified number of tokens.
    
    Args:
        content: The content to truncate
        max_tokens: The maximum number of tokens to truncate to
        
    Returns:
        Tuple containing the truncated content and the truncation ratio
    """
    words = content.split()
    if len(words) > max_tokens:
        return ' '.join(words[:max_tokens]) + '... (truncated)', max_tokens / len(words)
    else:
        return content, 1.0

def task_decompose(question, Tool_dic, model_name="gpt-4.1-2025-04-14", prompt_version="v0", temperature=0.2, top_p=1.0, max_tokens=32768, seed: int | None = None):
    """
    Decompose a complex question into simple subtasks.
    
    Args:
        question: The complex question to decompose
        Tool_dic: Available tools/APIs
        model_name: LLM model to use
        prompt_version: Version of the prompt template to use
        temperature: Temperature for LLM generation
        top_p: Top-p parameter for LLM generation
        max_tokens: Maximum tokens for LLM generation
        
    Returns:
        Dictionary with 'Tasks' key containing list of subtasks
    """
    template = "You are a task‑decomposer agent. Given available TMDB tools and a complex user question, produce a sequence of simple, natural‑language subtasks so each can be executed by one TMDB tool. Use the rules below and output only valid JSON."
    
    # Load prompt from file based on version
    prompt_file_path = str(Path(__file__).parent / f"prompts/task_decomposition_{prompt_version}.txt")
    with open(prompt_file_path, 'r') as f:
        prompt_template = f.read()
    
    # Replace placeholders with actual values
    prompt = prompt_template.format(tools=Tool_dic, question=question)
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(f"[PROMPT] {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    
    # Initial attempt with provided temperature
    result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
    
    # Extract content from the response
    if result and hasattr(result, 'choices') and result.choices:
        content = result.choices[0].message.content
    else:
        print("Error: No valid response from LLM")
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
                
                retry_result = llm_call(messages, model=model_name, temperature=1.0, top_p=top_p, max_tokens=max_tokens, seed=seed)
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


def task_topology(question, task_ls, model_name, temperature=0.2, top_p=1.0, max_tokens=32768, seed: int | None = None):
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

    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(f"[PROMPT] {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
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


def answer_generation_depend(question, call_result, previous_log, model_name, temperature=0.2, top_p=1.0, max_tokens=32768, seed: int | None = None):
    """
    Generate an answer based on API response and previous context.
    
    Args:
        question: The question to answer
        call_result: Result from API call
        previous_log: Previous questions and answers for context
        model_name: LLM model to use
        
    Returns:
        Generated answer as string
    """
    template = "You are a helpful assistant."
    prompt = (
        "You should answer the question based on the response output by the API tool."
        "Please note that:\n"
        "1. Try to organize the response into a natural language answer.\n"
        "2. We will not show the API response to the user, thus you need to make full use of the response and give the information in the response that can satisfy the user's question in as much detail as possible.\n"
        "3. The question may have dependencies on answers of other questions, so we will provide logs of previous questions and answers.\n"
        f"There are logs of previous questions and answers: \n {previous_log}\n"
        f"This is the user's question: {question}\n"
        f"This is the response output by the API tool: \n{call_result}\n"
        "We will not show the API response to the user, "
        "thus you need to make full use of the response and give the information "
        "in the response that can satisfy the user's question in as much detail as possible.\n"
        "Output:"
    )

    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(f"[PROMPT] {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
    
    # Extract content from the response
    if result and hasattr(result, 'choices') and result.choices:
        content = result.choices[0].message.content
        return content
    elif "Error code: 400" in str(result):
        print(str(result))
        return "bad request"
    else:
        print("Error: No valid response from LLM")
        return "Error: No response received from the model."


def answer_check(question, api_name, parameters, answer, model_name, temperature=0.2, top_p=1.0, max_tokens=32768, seed: int | None = None):
    """
    Check if the generated answer adequately addresses the question.
    
    Args:
        question: Original question
        api_name: Name of the API used
        parameters: Parameters passed to the API
        answer: Generated answer to check
        model_name: LLM model to use
        
    Returns:
        Quality assessment result
    """
    template = "You are a helpful assistant."
    prompt = (
        "You need to judge whether the answer is good for the question. Please note that:\n"
        "1. The answer should directly address the question asked.\n"
        "2. The answer should be factually correct based on the API response.\n"
        "3. The answer should be complete and informative.\n"
        "4. Consider the context of the API used and parameters provided.\n"
        f"Question: {question}\n"
        f"API used: {api_name}\n"
        f"Parameters: {parameters}\n"
        f"Answer: {answer}\n"
        "Judge whether this answer adequately addresses the question.\n"
        "Output your assessment as a score from 1-5, where:\n"
        "1 = Very poor answer\n"
        "2 = Poor answer\n"
        "3 = Adequate answer\n"
        "4 = Good answer\n"
        "5 = Excellent answer\n"
        "Provide only the numeric score.\n"
        "Output:"
    )

    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(f"[PROMPT] {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
    print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
    
    # Extract content from the response
    if result and hasattr(result, 'choices') and result.choices:
        content = result.choices[0].message.content
        return content
    else:
        print("Error: No valid response from LLM")
        print(f"Raw content: {content}")
        return content


def answer_generation(question, call_result, model_name, temperature=0.2, top_p=1.0, max_tokens=32768):
    """
    Generate an answer based on API response (without previous context).
    """
    template = "You are a helpful assistant."
    prompt = (
        "You should answer the question based on the response output by the API tool."
        "Please note that:\n"
        "1. Answer the question in natural language based on the API response reasonably and effectively.\n"
        "2. The user cannot directly get API response, "
        "so you need to make full use of the response and give the information "
        "in the response that can satisfy the user's question in as much detail as possible.\n"
        f"This is the user's question:\n {question}\n"
        f"This is the API response:\n {call_result}\n"
        "Output:"
    )

    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    print(f"[PROMPT] {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
    result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
    
    # Extract content from the response
    if result and hasattr(result, 'choices') and result.choices:
        content = result.choices[0].message.content
        return content
    else:
        print("Error: No valid response from LLM")
        return "Error: No response received from the model."


def choose_parameter(API_instruction, question, model_name, temperature=0.2, top_p=1.0, max_tokens=32768):
    """
    Choose parameters for API call based on the question.
    """
    template = "You are a helpful assistant."
    prompt = (
        "Given a user's question, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
        "Please note that: \n"
        "1. The Example in the API tool documentation can help you better understand the use of the API.\n"
        "2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {\"Parameters\":{}}\n"
        "3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
        "4. If you need to use this API multiple times, please set \"Parameters\" to a list.\n"
        "5. You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. An examples output looks like:\n"
        "'''\n"
        "Example 1: ```json{\"Parameters\":{\"keyword\": \"Artificial Intelligence\", \"language\": \"English\"}}\n```"
        "'''\n"
        f"This is API tool documentation: {API_instruction}\n"
        f"This is user's question: {question}\n"
        "Output:\n"
    )
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    
    for attempt in range(1):
        try:
            print(f"[PROMPT] {prompt}")
            result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
            
            # Extract content from the response
            if result and hasattr(result, 'choices') and result.choices:
                content = result.choices[0].message.content
            else:
                print(f"⚠️  Attempt {attempt + 1}: No valid response from LLM")
                if attempt < 2:
                    continue
                else:
                    return {}
            
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
            except json.JSONDecodeError as e:
                print(f"⚠️  Attempt {attempt + 1}: JSON decode error: {e}")
                print(f"Raw content: {content}")
                if attempt < 2:
                    continue
                else:
                    return {}
            
            parameters = parsed_result.get("Parameters", {})
            
            # Handle case where LLM returns a list with single dict instead of just the dict
            if isinstance(parameters, list) and len(parameters) == 1 and isinstance(parameters[0], dict):
                parameters = parameters[0]
            
            # Check if parameters are empty and retry if needed
            if not parameters or parameters == {}:
                print(f"⚠️  Attempt {attempt + 1}: Got empty parameters, retrying...")
                if attempt < 2:  # Don't retry on the last attempt
                    continue
                else:
                    print(f"❌ All 3 attempts failed to get non-empty parameters")
                    return {}
            else:
                print(f"✅ Got parameters on attempt {attempt + 1}: {parameters}")
                return parameters
                
        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1} failed with error: {e}")
            if attempt < 2:  # Don't retry on the last attempt
                continue
            else:
                print(f"❌ All 3 attempts failed due to errors")
                return {}
    
    return {}


def choose_parameter_depend(API_instruction, question, previous_log, model_name, temperature=0.2, top_p=1.0, max_tokens=32768):
    """
    Choose parameters for API call based on the question and previous context.
    """
    template = "You are a helpful assistant."
    prompt = (
        "Given a user's question and a API tool documentation, you need to output parameters according to the API tool documentation to successfully call the API to solve the user's question.\n"
        "Please note that: \n"
        "1. The Example in the API tool documentation can help you better understand the use of the API.\n"
        "2. Ensure the parameters you output are correct. The output must contain the required parameters, and can contain the optional parameters based on the question. If no paremters in the required parameters and optional parameters, just leave it as {\"Parameters\":{}}\n"
        "3. If the user's question mentions other APIs, you should ONLY consider the API tool documentation I give and do not consider other APIs.\n"
        "4. IMPORTANT - Parameter Extraction from Previous Context: When the API requires path parameters (like person_id, movie_id, tv_id, company_id, etc.), you MUST extract them from the subtask_output of previous steps. Look for patterns like:\n"
        " - 'person ID is 190' or 'person ID for [name] is 190' → use 190 as person_id\n"
        " - 'movie ID is 12345' or 'movie ID for [title] is 12345' → use 12345 as movie_id\n"
        " - 'TV show ID is 67890' → use 67890 as tv_id\n"
        " - 'company ID is 54321' → use 54321 as company_id\n"
        " Extract the numeric ID values from these text descriptions and use them as the corresponding path parameters.\n"
        "5. If you need to use this API multiple times,, please set \"Parameters\" to a list.\n"
        "6. You must ONLY output in a parsable JSON format, with no extra explanations, notes, or comments. The output should strictly follow the JSON format. An examples output looks like:\n"
        "'''\n"
        "Example 1: ```json{\"Parameters\":{\"keyword\": \"Artificial Intelligence\", \"language\": \"English\"}}\n```"
        "Example 2: ```json{\"Parameters\":{\"person_id\": 190}}```\n"
        "'''\n"
        f"There are logs of previous questions and answers: \n {previous_log}\n"
        f"This is API tool documentation: {API_instruction}\n"
        f"This is the current user's question: {question}\n"
        "Output:\n"
    )
    messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": prompt}
        ]
    
    n_attempts = 3
    for attempt in range(n_attempts):  # Try up to 3 times
        try:
            print(f"[PROMPT] {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
            result = llm_call(messages, model=model_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            print(f"[RESULT] {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
            
            # Extract content from the response
            if result and hasattr(result, 'choices') and result.choices:
                content = result.choices[0].message.content
            else:
                print(f"⚠️  Attempt {attempt + 1}: No valid response from LLM")
                if attempt < n_attempts - 1:  # Continue if not the last attempt
                    continue
                else:
                    return {}
            
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
            except json.JSONDecodeError as e:
                print(f"⚠️  Attempt {attempt + 1}: JSON decode error: {e}")
                print(f"Raw content: {content}")
                if attempt < n_attempts - 1:  # Continue if not the last attempt
                    continue
                else:
                    return {}
            
            parameters = parsed_result.get("Parameters", {})
            
            # Handle case where LLM returns a list with single dict instead of just the dict
            if isinstance(parameters, list) and len(parameters) == 1 and isinstance(parameters[0], dict):
                parameters = parameters[0]
            
            # Check if parameters are empty and retry if needed
            if not parameters or parameters == {}:
                print(f"⚠️  Attempt {attempt + 1}: Got empty parameters, retrying...")
                if attempt < n_attempts - 1:  # Continue if not the last attempt
                    continue
                else:
                    print(f"❌ All 3 attempts failed to get non-empty parameters")
                    return {}
            else:
                print(f"✅ Got parameters on attempt {attempt + 1}: {parameters}")
                return parameters
                
        except Exception as e:
            print(f"⚠️  Attempt {attempt + 1} failed with error: {e}")
            if attempt < n_attempts - 1:  # Continue if not the last attempt
                continue
            else:
                print(f"❌ All {n_attempts} attempts failed due to errors")
                return {}
    
    return {}