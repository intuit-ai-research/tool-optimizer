from pathlib import Path
from smolagents import Tool
from typing import Any
import yaml
from smolagents.default_tools import FinalAnswerTool as _BaseFinalAnswerTool
from smolagents.models import get_tool_json_schema
import json
import jsonschema

class PromptFinalAnswerTool(_BaseFinalAnswerTool):
    def to_tool_calling_prompt(self) -> str:
        return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)

class SchemaRewriteFinalAnswer(PromptFinalAnswerTool):

    def __init__(self, schema: dict, schema_file_path:Path):
        self.schema = schema # mutable. can be modified in place by other tools.
        self.schema_file_path = schema_file_path
        self.setup()

    def forward(self, answer: Any) -> Any:
        jsonschema.validate(instance=self.schema, schema=MCP_YAML_JSON_SCHEMA)
        self.schema_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.schema_file_path, "w") as f:
            yaml.dump(self.schema, f)
        return f"{answer}\nSchema saved to {self.schema_file_path}."

class UpdateProviderDescription(Tool):
    name = "utility_update_provider_description"
    description = "Update the description of the API provider."
    inputs = {"description": {"type": "string", "description": "The new description for the API provider."}}
    output_type = "string"

    def __init__(self, schema: dict):
        self.schema = schema
        self.setup()

    def forward(self, description: str) -> str:
        provider_name = extract_name(self.schema)
        self.schema["mcp_servers"][provider_name]["description"] = description
        return "Provider description updated."

    def to_tool_calling_prompt(self) -> str:
        return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)

class UpdateAPIDescription(Tool):
    name = "utility_update_api_description"
    description = "Update the description of a specific API."
    inputs = {
        "api_name": {"type": "string", "description": "The name of the API to update."},
        "description": {"type": "string", "description": "The new description for the API."}
    }
    output_type = "string"

    def __init__(self, schema: dict):
        self.schema = schema
        self.setup()

    def forward(self, api_name: str, description: str) -> str:
        provider_name = extract_name(self.schema)
        tools = self.schema["mcp_servers"][provider_name]["tools"]
        for tool in tools:
            if tool["tool_name"] == api_name:
                tool["description"] = description
                return f"Description for API '{api_name}' updated."
        else:
            raise ValueError(f"API '{api_name}' not found. The API names are {extract_tools(self.schema)}")

    def to_tool_calling_prompt(self) -> str:
        return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)

class UpdateAPIParameters(Tool):
    name = "utility_update_api_parameters"
    description = "Update the parameters for a specific API using a JSON string."
    inputs = {
        "api_name": {"type": "string", "description": "The name of the API to update."},
        "parameters_json": {"type": "string", "description": "The parameters as a JSON string. Must validate against the parameter schema."}
    }
    output_type = "string"

    def __init__(self, schema: dict):
        self.schema = schema
        self.setup()

    def forward(self, api_name: str, parameters_json: str) -> str:
        parameters = json.loads(parameters_json)
        
        # Validate parameters against schema
        # Use the structure from MCP_YAML_JSON_SCHEMA for validation
        jsonschema.validate(instance=parameters, schema=PARAMETER_SCHEMA)

        provider_name = extract_name(self.schema)
        tools = self.schema["mcp_servers"][provider_name]["tools"]
        for tool in tools:
            if tool["tool_name"] == api_name:
                tool["parameters"] = parameters
                return f"Parameters for API '{api_name}' updated."
        else:
            raise ValueError(f"API '{api_name}' not found. The API names are {extract_tools(self.schema)}")

    def to_tool_calling_prompt(self) -> str:
        return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)

class RemoveAPIParameters(Tool):
    name = "utility_remove_api_parameters"
    description = "Remove the 'parameters' field from an API, making it accept no parameters."
    inputs = {
        "api_name": {"type": "string", "description": "The name of the API."}
    }
    output_type = "string"

    def __init__(self, schema: dict):
        self.schema = schema
        self.setup()

    def forward(self, api_name: str) -> str:
        provider_name = extract_name(self.schema)
        tools = self.schema["mcp_servers"][provider_name]["tools"]
        for tool in tools:
            if tool["tool_name"] == api_name:
                if "parameters" in tool:
                    del tool["parameters"]
                    return f"Parameters field removed from API '{api_name}'."
                return f"Parameters field not found in API '{api_name}'."
        else:
            raise ValueError(f"API '{api_name}' not found. The API names are {extract_tools(self.schema)}")

    def to_tool_calling_prompt(self) -> str:
        return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)

class PrintSchema(Tool):
    name = "utility_print_schema"
    description = "Print the current schema."
    inputs = {}
    output_type = "string"

    def __init__(self, schema: dict):
        self.schema = schema
        self.setup()

    def forward(self) -> str:
        return json.dumps(self.schema, indent=2, ensure_ascii=False)

    def to_tool_calling_prompt(self) -> str:
        return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)

class TakeNotes(Tool):
    name = "utility_take_notes"
    description = "Take notes of what should be corrected for the schema."
    inputs = {"notes": {"type": "string", "description": "The notes of what should be corrected for the schema."}}
    output_type = "string"

    def forward(self, notes: str) -> str:
        return "Notes taken."

    def to_tool_calling_prompt(self) -> str:
        # Render OpenAI-style function tool schema as JSON text for insertion in the system prompt
        return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)


def extract_tools(mcp_yaml):
    tools = mcp_yaml["mcp_servers"][list(mcp_yaml["mcp_servers"].keys())[0]]["tools"]
    return {tool["tool_name"] for tool in tools}

def extract_name(mcp_yaml):
    return list(mcp_yaml["mcp_servers"].keys())[0]

def extract_category(mcp_yaml):
    return mcp_yaml["mcp_servers"][list(mcp_yaml["mcp_servers"].keys())[0]]["category"]

def validate_two_schemas(source_schema: dict, target_schema: dict):
    # check if the category and name are the same
    source_name = extract_name(source_schema)
    target_name = extract_name(target_schema)
    if source_name != target_name:
        raise ValueError(f"Name are different. The original name is {source_name} and the new name is {target_name}")
    source_category = extract_category(source_schema)
    target_category = extract_category(target_schema)
    if source_category != target_category:
        raise ValueError(f"Category are different. The original category is {source_category} and the new category is {target_category}")
    # check tools are the same
    source_tools = extract_tools(source_schema)
    target_tools = extract_tools(target_schema)
    if len(source_tools) != len(target_tools):
        error_message = f"Tools are different:"
        if source_tools - target_tools:
            error_message += f" The original tools used to have {source_tools - target_tools}, but now they are removed.\n"
        if target_tools - source_tools:
            error_message += f" You should not invent new tools:{target_tools - source_tools}."
        raise ValueError(error_message)
    

# Define the JSON schema for validation
MCP_YAML_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "mcp_servers": {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 1,
            "patternProperties": {
                ".*": {  # Matches any API provider name
                    "type": "object",
                    "required": ["command", "enabled", "description", "category", "tools"],
                    "properties": {
                        "tools": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["tool_name", "description", "_metadata"],
                                "optional": ["parameters"],
                                "properties": {
                                    "parameters": {
                                        "type": "object",
                                        "patternProperties": {
                                            ".*": {  # Matches any parameter name
                                                "type": "object",
                                                "required": ["type", "required", "description"]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "required": ["mcp_servers"]
}

PARAMETER_SCHEMA = {
    "type": "object",
    "patternProperties": {
        ".*": {
            "type": "object",
            "required": ["type", "required", "description"],
            "properties": {
                "type": {"type": "string"},
                "required": {"type": "boolean"},
                "description": {"type": "string"}
            }
        }
    }
}