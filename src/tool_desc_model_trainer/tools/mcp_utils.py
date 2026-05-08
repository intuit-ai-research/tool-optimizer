import json
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
# from utils.llm import llm_call

# Add StableToolBench to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from StableToolBench.server.utils import standardize


def _load_single_mcp_yaml(yaml_path: str) -> Tuple[Dict[tuple, Dict[str, Any]], Optional[str]]:
    """
    Load a single MCP YAML file.

    Args:
        yaml_path: Path to MCP YAML file

    Returns:
        Tuple of (tools_dict, error_message). error_message is None on success.
    """
    tools_dict = {}

    if not os.path.exists(yaml_path):
        return {}, f"⚠️  Warning: MCP YAML file not found: {yaml_path}"

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)

        # Extract tools from MCP YAML structure
        mcp_servers = yaml_data.get('mcp_servers', {})
        for server_name, server_info in mcp_servers.items():
            server_name = standardize(server_name)
            tools = server_info.get('tools', [])
            for tool in tools:
                tool_name = tool.get('tool_name', '')
                if tool_name:
                    tools_dict[(server_name, tool_name)] = tool

        return tools_dict, None

    except Exception as e:
        return {}, f"⚠️  Warning: Failed to load MCP YAML {yaml_path}: {e}"


def load_mcp_yaml_files(yaml_paths: List[str], max_workers: Optional[int] = None) -> Dict[tuple, Dict[str, Any]]:
    """
    Load and merge MCP YAML files into a unified tools dictionary using parallel processing.

    Args:
        yaml_paths: List of paths to MCP YAML files
        max_workers: Maximum number of threads to use. Defaults to min(32, len(yaml_paths), os.cpu_count() + 4)

    Returns:
        Dictionary mapping tool_name to tool info including parameters
    """
    if not yaml_paths:
        return {}

    # For single file, skip threading overhead
    if len(yaml_paths) == 1:
        tools_dict, error = _load_single_mcp_yaml(yaml_paths[0])
        if error:
            print(error)
        return tools_dict

    # Determine optimal worker count
    if max_workers is None:
        max_workers = min(len(yaml_paths), (os.cpu_count() or 1) + 4)

    tools_dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_load_single_mcp_yaml, path): path for path in yaml_paths}

        for future in as_completed(future_to_path):
            result, error = future.result()
            if error:
                print(error)
            tools_dict.update(result)

    return tools_dict


def extract_parameter_json(server_name: str, tool_name: str, mcp_tools: Dict[tuple, Dict[str, Any]]) -> str:
    """
    Extract parameter information for a tool and format as JSON.

    Args:
        tool_name: Name of the tool to extract parameters for
        mcp_tools: Dictionary of MCP tool definitions

    Returns:
        JSON string containing parameter schema, or empty JSON if not found
    """
    if (server_name, tool_name) not in mcp_tools:
        return json.dumps({"parameters": {}, "metadata": {}}, indent=2)

    tool_info = mcp_tools[(server_name, tool_name)]
    parameters = tool_info.get('parameters', {})
    metadata = tool_info.get('_metadata', {})

    # Convert parameters to our JSON schema format
    parameter_schema = {}
    for param_name, param_info in parameters.items():
        parameter_schema[param_name] = {
            "type": param_info.get('type', ''),
            "required": param_info.get('required', ''),
            "description": param_info.get('description', ''),
        }

    schema = {
        "parameters": parameter_schema,
        "metadata": {
            "endpoint": metadata.get('endpoint', ''),
            "method": metadata.get('method', ''),
            "category": metadata.get('category', '')
        }
    }

    return json.dumps(schema, indent=2)



def generate_compatible_ground_truth(
    improved_description: str,
    prompt_template_path: str,
    start_token: str = "",
    end_token: str = "",
    output_format: str = "auto",
    reasoning: Optional[str] = None,
) -> str:
    """
    Generate ground truth format compatible with different prompt versions.

    Args:
        improved_description: The improved API description to use as target
        prompt_template_path: Path to prompt template to determine version
        start_token: Start delimiter token (ignored for JSON format)
        end_token: End delimiter token (ignored for JSON format)
        output_format: "auto" (detect from template), "json", or "token"

    Returns:
        Formatted ground truth string
    """
    # Detect prompt version from filename
    prompt_filename = os.path.basename(prompt_template_path)
    
    # Determine output format
    if output_format == "auto":
        # v7+ uses JSON format
        if "v7" in prompt_filename or "v8" in prompt_filename or "v9" in prompt_filename or "v10" in prompt_filename:
            output_format = "json"
        else:
            output_format = "token"
    
    if output_format == "json":
        # JSON format: easy to parse with json.loads()
        if "v9" in prompt_filename or "v10" in prompt_filename:
            output_obj = {
                "reasoning": reasoning or "",
                "description": improved_description,
            }
        else:
            output_obj = {"description": improved_description}
        ground_truth = json.dumps(output_obj, ensure_ascii=False)
    elif "v2" in prompt_filename:
        # v2 format: reasoning in <think> tags + tokens
        ground_truth = f"<think></think>\n{start_token}\n{improved_description}\n{end_token}"
    else:
        # default token-based format
        ground_truth = f"{start_token}\n{improved_description}\n{end_token}"

    return ground_truth