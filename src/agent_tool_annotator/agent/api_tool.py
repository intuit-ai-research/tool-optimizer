from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import json

from smolagents import Tool
from smolagents.models import get_tool_json_schema

# FunctionWrapper is added to sys.path in agent_tool_annotator/main.py before imports
from eval.tmdb.utils.api_caller import call_api_mcp  # type: ignore

# """
# Shortest, no-file-edit workaround: monkey‑patch the imported symbol that tools.py uses.
# import smolagents.tools as stst.is_valid_name = lambda name: True  # do this before creating any Tool instances
# This overrides the is_valid_name function that tools.py calls at L167, effectively skipping the check.
# """
# import smolagents.tools as st
# st.is_valid_name = lambda name: True  # do this before creating any Tool instances

JSON_TO_SMOL: Dict[str, str] = {
    # JSON Schema
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    # Python-ish shorthands observed in specs
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def inputs_from_tool(tool: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert tool into smolagents inputs schema.
    """
    """
    tool = tool_manager.tools[0] = {'tool_name': 'Get Word by Length and Start', 'description': 'Returns a random word of specified length and that starts with specified string.\nFor example, 7 and "fru" will return any word that is 7 letters long and starts with "fru", such as "fruiter".', 'parameters': {'length': {'type': 'float', 'required': True, 'description': '', 'default': '7'}, 'start': {'type': 'str', 'required': True, 'description': '', 'default': 'fru'}}, '_metadata': {'endpoint': '/LS/{length}/{start}', 'method': 'GET', 'platform': 'stabletoolbench'}}
    """
    params = tool.get("parameters") or {}

    inputs = {}
    for key, spec in params.items():
        raw_type = str(spec.get("type", "any")).lower()
        smol_type = JSON_TO_SMOL.get(raw_type, "any")
        required_flag = bool(spec.get("required", False))
        inputs[key] = {
            "type": smol_type,
            "description": spec.get("description", ""),
            "nullable": not required_flag,
            # "default": spec.get("default", ""),
            # "required": required_flag,
            # "tool_from_tool_manager": tool,
        }
        # add items to the inputs if the type is list
        if spec.get("type") == "list":
            inputs[key]["items"] = {"type": "string"}
    return inputs

def _sanitize_tool_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name

class APISpecTool(Tool):
    skip_forward_signature_validation = True
    
    def __init__(
        self,
        tool,
        env,
        tool_call_by_prompt: bool = False,
    ):
        """
        attributes to change:
            name: str
            description: str
            inputs: dict[str, dict[str, str | type | bool]]
            output_type: str
            output_schema: dict[str, Any] | None = None
        """
        self.raw_name = tool["tool_name"]
        self.name = _sanitize_tool_name(self.raw_name)
        self.description = tool["description"]
        self.inputs = inputs_from_tool(tool)
        self.output_type = "object"
        
        self.tool = tool
        self.env = env
        self.is_initialized = True
        self.tool_call_by_prompt = tool_call_by_prompt

    def forward(self, **kwargs):
        if "parameters" in kwargs and len(self.inputs) == 1:
            params = kwargs["parameters"]
        else:
            params = kwargs
        """
        params = {'length': '7', 'start': 'fru'}

        what call_api_mcp needs:
        api_name
        'Get Word by Length and Start'
        parameters
        {'length': 7, 'start': 'fru'}
        metadata
        {'endpoint': '/LS/{length}/{start}', 'method': 'GET', 'platform': 'stabletoolbench'}
        platform
        'StableToolBench'
        rapidapi_wrapper
        <toolbench.inference.Downstream_tasks.rapidapi_multithread.rapidapi_wrapper object at 0x7fd42a54f140>
        """
        return call_api_mcp(
            api_name=self.raw_name,
            parameters=params,
            metadata=self.tool["_metadata"],
            platform=self.tool["_metadata"]["platform"],
            rapidapi_wrapper=self.env,
        )
    

    def __repr__(self):
        return f"APISpecTool(name={self.name}, description={self.description}, inputs={self.inputs}, output_type={self.output_type})"

    def to_tool_calling_prompt(self) -> str:
        if self.tool_call_by_prompt:
            # Render OpenAI-style function tool schema as JSON text for insertion in the system prompt
            return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)
        return super().to_tool_calling_prompt()


def build_tools_from_yaml_tools(
    tool_manager, #: ToolManager,
    env, #: rapidapi_wrapper,
    tool_call_by_prompt: bool = False,
) -> List[Tool]:
    tools: List[Tool] = []
    for tool in tool_manager.tools:
        tools.append(APISpecTool(tool, env, tool_call_by_prompt=tool_call_by_prompt))


    used = {}
    # Deduplicate tool names
    for tool in tools:
        base = tool.name
        i = 1
        while tool.name in used:
            i += 1
            tool.name = f"{base}_{i}"
        used[tool.name] = True
    return tools


