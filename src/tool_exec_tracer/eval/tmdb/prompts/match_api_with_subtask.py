"""
Prompt for LLM-based semantic matching of ground truth APIs to decomposed subtasks.
"""

def get_semantic_matching_prompt(query_text: str, api_metadata_section: str, subtasks_section: str) -> str:
    """
    Generate prompt for semantic matching of subtasks to ground truth APIs.
    
    Args:
        query_text: Original user query
        api_metadata_section: Formatted string with API metadata (name, description, parameters)
        subtasks_section: Formatted string with decomposed subtasks
    
    Returns:
        Complete prompt string for LLM
    """
    prompt = f"""You are tasked with matching each ground truth API to a decomposed subtask based on semantic similarity.

Original Query: {query_text}

Ground Truth APIs:
{api_metadata_section}

Decomposed Subtasks:
{subtasks_section}

Task: You need to match each ground truth API to the most semantically similar subtask. A subtask may or may not need an API call. If it does not require an API call, match it to the most appropriate ground truth API.

- If a subtask requires an API call, match it to the appropriate ground truth API
- If a subtask is a processing step (no API needed), omit it from the output
- Consider the semantic meaning of both the subtask and API descriptions

Output your answer as a JSON object mapping ONLY the subtask IDs that need APIs:
{{
  "1": "api_name",
  "3": "api_name",
  ...
}}

Only include subtasks that require an API call. Omit processing steps (e.g., counting, selecting, comparing results).

JSON Output:"""
    
    return prompt

