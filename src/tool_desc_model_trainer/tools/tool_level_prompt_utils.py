#!/usr/bin/env python3
"""
Utility functions for creating tool-level prompts.

These functions format tool groups into prompts that show multiple
example queries for comprehensive API description generation.
"""

from typing import Dict, List, Any, Optional
import random


def extract_error_details(query: Dict[str, Any]) -> str:
    """
    Extract detailed error information from query run_details.

    Args:
        query: Query record containing run_details

    Returns:
        Formatted string with error details, or empty string if no errors found
    """
    run_details = query.get("run_details", [])
    if not run_details:
        return ""

    error_info = []

    # Collect parameter validation errors
    param_errors = []
    api_errors = []

    for run in run_details:
        if not run.get("api_success", True):  # Failed runs
            # Parameter validation errors
            param_quality = run.get("parameter_quality_evaluation", {})
            if param_quality:
                validation_errors = param_quality.get("parameter_validation_errors", [])
                if validation_errors:
                    param_errors.extend(validation_errors)

                # Type mismatches
                validation_details = param_quality.get("parameter_validation_details", {})
                type_mismatches = validation_details.get("type_mismatches", [])
                for mismatch in type_mismatches:
                    error_msg = f"'{mismatch.get('param')}' expected {mismatch.get('expected_type')}, got {mismatch.get('actual_value')} ({type(mismatch.get('actual_value')).__name__})"
                    param_errors.append(error_msg)

                # API-level errors
                api_error_reason = param_quality.get("api_error_reason", "")
                if api_error_reason and api_error_reason != "UNKNOWN":
                    api_errors.append(api_error_reason)

    # Format error details if we have any
    if param_errors or api_errors:
        error_info.append("\n  **Error Details:**")

        # Deduplicate and format parameter errors
        unique_param_errors = list(set(param_errors))
        for error in unique_param_errors[:3]:  # Show max 3 parameter errors
            error_info.append(f"    • Parameter: {error}")

        # Deduplicate and format API errors
        unique_api_errors = list(set(api_errors))
        for error in unique_api_errors[:2]:  # Show max 2 API errors
            error_info.append(f"    • API: {error}")

    return "\n".join(error_info) if error_info else ""


def format_query_example(query: Dict[str, Any], index: int) -> str:
    """
    Format a single query as an example use case.

    Args:
        query: Query record with query_text, subtask_text, etc.
        index: 1-based index for numbering

    Returns:
        Formatted string for this query example
    """
    query_text = query.get("query_text", "")
    subtask_text = query.get("subtask_text", "")
    
    # Get API selection accuracy and determine what was selected
    api_selection_accuracy = query.get("avg_api_selection_accuracy", None)
    expected_api = query.get("expected_golden_api", "")
    selection_status = ""
    
    if api_selection_accuracy is not None:
        if api_selection_accuracy >= 0.9:  # Consider >=0.9 as correct (handles floating point)
            selection_status = " [CORRECT]"
        else:
            # Check run_details to see what was actually selected in FAILED runs
            selected_api = None
            run_details = query.get("run_details", [])
            if run_details and len(run_details) > 0:
                # Get selected APIs from runs where api_selection_correct=False
                failed_selected_apis = [
                    run.get("selected_api", "") 
                    for run in run_details 
                    if not run.get("api_selection_correct", True) and run.get("selected_api", "")
                ]
                
                if failed_selected_apis:
                    # Use the most common wrong API, or first one
                    from collections import Counter
                    selected_api = Counter(failed_selected_apis).most_common(1)[0][0]
            
            # Show what was actually selected (if known and different from expected)
            # Note: We're in the context of training for 'expected_api', so all queries
            # shown should have expected_api as the correct answer.
            if selected_api and selected_api != expected_api:
                # Clear API confusion - wrong API name selected
                selection_status = f" [FAILED - selected '{selected_api}']"
            else:
                # Same API name selected but marked as incorrect - determine why
                avg_api_success_rate = query.get("avg_api_success_rate", 0)
                if avg_api_success_rate < 0.5:
                    # Low success rate suggests parameter/execution issues
                    selection_status = " [CORRECT API - bad parameters]"
                else:
                    # High success rate suggests context/timing issues
                    selection_status = " [CORRECT API - wrong context/timing]"
    
    # Get API execution success information
    api_success_info = ""
    run_details = query.get("run_details", [])
    successful_runs = 0  # Initialize to handle empty run_details
    if run_details:
        total_runs = len(run_details)
        successful_runs = sum(1 for run in run_details if run.get("api_success", False))

        if successful_runs == total_runs:
            api_success_info = f" | API execution: {successful_runs}/{total_runs} succeeded"
        elif successful_runs == 0:
            api_success_info = f" | API execution: {successful_runs}/{total_runs} succeeded (all failed)"
        else:
            api_success_info = f" | API execution: {successful_runs}/{total_runs} succeeded"
    
    # Format previous context if available
    previous_context = query.get("previous_context", [])
    context_str = ""
    if previous_context and len(previous_context) > 0:
        context_items = []
        for ctx in previous_context[:3]:  # Show max 3 previous steps
            if isinstance(ctx, dict):
                api = ctx.get("api_name", "")
                result = ctx.get("result_summary", ctx.get("result", ""))
                if api:
                    context_items.append(f"  - Used {api}: {result}")
            elif isinstance(ctx, str):
                context_items.append(f"  - {ctx}")
        if context_items:
            context_str = "\n  **Previous steps:**\n" + "\n".join(context_items)
    
    # Extract detailed error information for failed cases
    error_details = ""
    if successful_runs == 0 or (api_selection_accuracy is not None and api_selection_accuracy < 0.9):
        error_details = extract_error_details(query)

    # Format the example
    example = f"""**Example {index}:**{selection_status}{api_success_info}
- **User Query:** {query_text}
- **Subtask:** {subtask_text}{context_str}{error_details}"""

    return example


def _select_balanced_queries(queries: List[Dict[str, Any]], num_examples: int) -> List[Dict[str, Any]]:
    """
    Select a balanced mix of successful/failed and golden/non-golden queries.

    Args:
        queries: List of all available queries
        num_examples: Number of examples to select

    Returns:
        List of selected queries with balanced representation
    """
    if len(queries) <= num_examples:
        return queries

    # Categorize queries
    successful_queries = [q for q in queries if q.get("avg_api_success_rate", 0) >= 0.8]
    failed_queries = [q for q in queries if q.get("avg_api_success_rate", 0) < 0.8]

    golden_queries = [q for q in queries if q.get("is_golden", True)]
    non_golden_queries = [q for q in queries if not q.get("is_golden", True)]

    print(f"   📊 Query distribution: {len(successful_queries)} successful, {len(failed_queries)} failed, {len(golden_queries)} golden, {len(non_golden_queries)} non-golden")

    selected = []

    # Strategy: Try to get balanced representation
    target_successful = max(1, min(num_examples // 2, len(successful_queries)))
    target_failed = max(0, min(num_examples - target_successful, len(failed_queries)))
    target_non_golden = max(0, min(num_examples // 3, len(non_golden_queries)))

    print(f"   🎯 Target distribution: {target_successful} successful, {target_failed} failed, {target_non_golden} non-golden")

    # First, prioritize failed examples (more learning signal)
    if failed_queries and target_failed > 0:
        selected.extend(random.sample(failed_queries, target_failed))

    # Then, add non-golden examples (negative examples of when NOT to use this API)
    remaining_non_golden = [q for q in non_golden_queries if q not in selected]
    if remaining_non_golden and target_non_golden > 0:
        to_add = min(target_non_golden, len(remaining_non_golden))
        selected.extend(random.sample(remaining_non_golden, to_add))

    # Fill remaining slots with successful examples
    remaining_slots = num_examples - len(selected)
    if remaining_slots > 0:
        remaining_successful = [q for q in successful_queries if q not in selected]
        if remaining_successful:
            to_add = min(remaining_slots, len(remaining_successful))
            selected.extend(random.sample(remaining_successful, to_add))

    # If still need more, add any remaining queries
    remaining_slots = num_examples - len(selected)
    if remaining_slots > 0:
        remaining_queries = [q for q in queries if q not in selected]
        if remaining_queries:
            to_add = min(remaining_slots, len(remaining_queries))
            selected.extend(random.sample(remaining_queries, to_add))

    # Final distribution info
    final_successful = sum(1 for q in selected if q.get("avg_api_success_rate", 0) >= 0.8)
    final_failed = sum(1 for q in selected if q.get("avg_api_success_rate", 0) < 0.8)
    final_golden = sum(1 for q in selected if q.get("is_golden", True))
    final_non_golden = sum(1 for q in selected if not q.get("is_golden", True))

    print(f"   ✅ Selected distribution: {final_successful} successful, {final_failed} failed, {final_golden} golden, {final_non_golden} non-golden")

    return selected


def format_query_examples(
    queries: List[Dict[str, Any]],
    num_examples: int = 5,
    selection_strategy: str = "random",
    seed: int = 42
) -> str:
    """
    Format multiple queries as example use cases.
    
    Args:
        queries: List of query records
        num_examples: Maximum number of examples to show
        selection_strategy: How to select examples:
            - "balanced": Balanced mix of successful/failed and golden/non-golden
            - "random": Random selection
            - "first": First N queries
            - "diverse": Try to select diverse queries (placeholder for future)
        seed: Random seed for reproducibility
    
    Returns:
        Formatted string with all query examples
    """
    if not queries:
        return "No example queries available."
    
    # Select queries based on strategy
    num_to_select = min(num_examples, len(queries))
    
    if selection_strategy == "balanced":
        # Balanced selection: mix of successful/failed and golden/non-golden examples
        random.seed(seed)
        selected_queries = _select_balanced_queries(queries, num_to_select)
    elif selection_strategy == "random":
        random.seed(seed)
        selected_queries = random.sample(queries, num_to_select)
    elif selection_strategy == "first":
        selected_queries = queries[:num_to_select]
    elif selection_strategy == "diverse":
        # TODO: Implement diversity-based selection
        # For now, fall back to random
        random.seed(seed)
        selected_queries = random.sample(queries, num_to_select)
    else:
        # Default to random
        random.seed(seed)
        selected_queries = random.sample(queries, num_to_select)
    
    # Format each query
    formatted_examples = []
    for i, query in enumerate(selected_queries, 1):
        formatted_examples.append(format_query_example(query, i))
    
    # Add summary line if we're showing subset
    header = ""
    if num_to_select < len(queries):
        header = f"(Showing {num_to_select} of {len(queries)} total use cases)\n\n"
    
    return header + "\n\n".join(formatted_examples)


def create_tool_level_prompt(
    tool_name: str,
    queries: List[Dict[str, Any]],
    original_description: str,
    prompt_template_path: str,
    num_examples: int = 5,
    selection_strategy: str = "random",
    seed: int = 42,
    version_mode: str = "auto",
    parameter_json: str = "",
    server_name: str = "",
    start_token: str = "<|extra_0|>",
    end_token: str = "<|extra_1|>",
    version: Optional[str] = None,
) -> str:
    """
    Create a complete tool-level prompt from template.

    Args:
        tool_name: Name of the API/tool
        queries: List of queries using this tool
        original_description: Current/baseline API description
        prompt_template_path: Path to prompt template file
        num_examples: Number of example queries to show
        selection_strategy: How to select example queries
        seed: Random seed
        version: Template version to use ("v0", "v1", "v2", "v3", or "auto" to detect from filename)
        parameter_json: JSON string with parameter schema (for v3 templates)
        server_name: Name of the server/API provider
        start_token: Token that should wrap the model's final output
        end_token: Token that should close the model's final output

    Returns:
        Complete formatted prompt string
    """
    # Detect template version from filename if auto
    # detected_version = version
    if version is not None:
        version_mode = version

    if version_mode == "auto":
        import os
        import re
        filename = os.path.basename(prompt_template_path)
        # Match v followed by a number (e.g., v3, v4, v5, v10)
        version_match = re.search(r'v(\d+)', filename)
        if version_match:
            detected_version = f"v{version_match.group(1)}"
        else:
            detected_version = "v0"
    else:
        detected_version = version_mode  # kept for future version-specific logic

    # Load prompt template
    with open(prompt_template_path, "r") as f:
        template = f.read()

    # Format query examples
    query_examples = format_query_examples(
        queries,
        num_examples=num_examples,
        selection_strategy=selection_strategy,
        seed=seed
    )

    # Prepare template values
    template_values = {
        "tool_name": tool_name,
        "query_examples": query_examples,
        "original_description": original_description,
        "server_name": server_name,
        "start_token": start_token,
        "end_token": end_token,
        "version": detected_version,
    }

    # Add parameter JSON for v3+ templates (any template that might use it)
    # The template.format() below will only use fields that are actually in the template
    if parameter_json:
        template_values["parameter_json"] = parameter_json
    else:
        template_values["parameter_json"] = "{}"

    # Fill in template (only use values that are actually needed)
    try:
        # Extract required fields from template
        import string
        required_fields = []
        for literal_text, field_name, format_spec, conversion in string.Formatter().parse(template):
            if field_name:
                required_fields.append(field_name)

        # Only pass required fields to avoid KeyError
        scoped_values = {k: template_values.get(k, "") for k in required_fields}
        prompt = template.format(**scoped_values)
    except KeyError as e:
        print(f"⚠️  Warning: Template field missing: {e}")
        print(f"   Available fields: {list(template_values.keys())}")
        print(f"   Required fields: {required_fields}")
        # Fallback to basic formatting
        raise KeyError(f"Template field missing: {e}")
        # prompt = template.format(
        #     tool_name=tool_name,
        #     query_examples=query_examples,
        #     original_description=original_description,
        #     parameter_json=parameter_json if parameter_json else "{}"
        # )

    return prompt


def create_tool_level_prompt_simple(
    tool_group: Dict[str, Any],
    prompt_template_path: str,
    num_examples: int = 5,
    use_train_queries: bool = True,
    selection_strategy: str = "random",
    seed: int = 42,
    start_token: str = "",
    end_token: str = "",
) -> str:
    """
    Simplified interface: Create prompt from a tool group dict.
    
    Args:
        tool_group: Tool group dictionary from group_by_tool.py
        prompt_template_path: Path to prompt template file
        num_examples: Number of example queries to show
        use_train_queries: If True, use train_queries; else use all_queries
        selection_strategy: How to select example queries
        seed: Random seed
    
    Returns:
        Complete formatted prompt string
    """
    queries = tool_group["train_queries"] if use_train_queries else tool_group["all_queries"]
    
    return create_tool_level_prompt(
        tool_name=tool_group["tool_name"],
        queries=queries,
        original_description=tool_group["original_description"],
        prompt_template_path=prompt_template_path,
        num_examples=num_examples,
        selection_strategy=selection_strategy,
        seed=seed,
        server_name=tool_group["server_name"],
        start_token=start_token,
        end_token=end_token,
    )


# Example usage
if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tool_level_prompt_utils.py <tool_grouped.json>")
        sys.exit(1)
    
    # Load tool groups
    with open(sys.argv[1], "r") as f:
        tool_groups = json.load(f)
    
    # Show prompt for first tool
    if tool_groups:
        tool_group = tool_groups[0]
        prompt_template = "DRAFT/policy_learn/prompts/policy_prompt_tool_level_v2.txt"
        
        print("=" * 80)
        print(f"EXAMPLE PROMPT FOR: {tool_group['tool_name']}")
        print("=" * 80)
        print()
        
        prompt = create_tool_level_prompt_simple(
            tool_group,
            prompt_template,
            num_examples=3
        )
        
        print(prompt)
        print()
        print("=" * 80)
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Approximate tokens: ~{len(prompt.split())}")

