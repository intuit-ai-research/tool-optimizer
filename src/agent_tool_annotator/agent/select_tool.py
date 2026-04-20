
from smolagents import Tool
from smolagents.default_tools import FinalAnswerTool as _BaseFinalAnswerTool
from smolagents.models import get_tool_json_schema
import json
import jsonschema
import yaml
from pathlib import Path
from typing import Any
from agent_tool_annotator.agent.rewrite_tool import extract_name, extract_tools
from typing import Literal


class PromptFinalAnswerTool(_BaseFinalAnswerTool):
    def to_tool_calling_prompt(self) -> str:
        return json.dumps(get_tool_json_schema(self), indent=2, ensure_ascii=False)

"""
Tools:

1. annotate_health (tool_name: str, health: [good, bad, unknown], reason: str)
2. annotate_example (tool_name: str, example: str)
"""

def deduplicate_tools(schema: dict) -> dict:
    provider_name = extract_name(schema)
    tools = schema["mcp_servers"][provider_name]["tools"]
    new_tools = []
    seen_tools = set()
    for tool in tools:
        if tool["tool_name"] not in seen_tools:
            new_tools.append(tool)
            seen_tools.add(tool["tool_name"])
        else:
            pass
    schema["mcp_servers"][provider_name]["tools"] = new_tools
    return schema

class AnnotationFinalAnswer(PromptFinalAnswerTool):

    def __init__(self, schema: dict, schema_file_path:Path):
        self.schema = schema # mutable. can be modified in place by other tools.
        self.schema_file_path = schema_file_path
        self.setup()

    def forward(self, answer: Any) -> Any:
        self.schema = deduplicate_tools(self.schema)
        jsonschema.validate(instance=self.schema, schema=MCP_YAML_JSON_SCHEMA_ANNOTATION)
        self.schema_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.schema_file_path, "w") as f:
            yaml.dump(self.schema, f)
        return f"{answer}\nSchema saved to {self.schema_file_path}."


class AnnotateHealth(Tool):
    name = "utility_annotate_health"
    description = "Annotate the health of a API."
    inputs = {
        "api_name": {"type": "string", "description": "The name of the API."},
        "health": {"type": "string", "description": """The health of the API. Must be one of [good, bad, unknown].
    good: the API is working as expected, returning meaningful results.
    bad: the API is not working as expected. For example, the API call always fails, returns Authorization error, or the API cannot be found.
    unknown: the health of the API is unknown. For example, the right parameters are not provided and cannot be inferred."""},
        "reason": {"type": "string", "description": "The reason for the health annotation."}
    }
    output_type = "string"

    def __init__(self, schema: dict):
        self.schema = schema
        self.setup()

    def forward(self, api_name: str, health: Literal["good", "bad", "unknown"], reason: str) -> str:
        provider_name = extract_name(self.schema)
        tools = self.schema["mcp_servers"][provider_name]["tools"]
        for tool in tools:
            if tool["tool_name"] == api_name:
                tool["health"] = {"health": health, "reason": reason}
                return f"Health for API '{api_name}' annotated."
        else:
            raise ValueError(f"API '{api_name}' not found. The API names are {extract_tools(self.schema)}")

class AnnotateExample(Tool):
    name = "utility_annotate_example"
    description = "Annotate the examples of calling a API."
    inputs = {
        "api_name": {"type": "string", "description": "The name of the API."},
        "example": {"type": "string", "description": 'The examples of calling the API. Should be a valid JSON string, like `[{"parameter1": "value1"}, {"parameter1": "value2"}]`'}
    }
    output_type = "string"

    def __init__(self, schema: dict):
        self.schema = schema
        self.setup()

    def forward(self, api_name: str, example: str) -> str:
        provider_name = extract_name(self.schema)
        tools = self.schema["mcp_servers"][provider_name]["tools"]
        for tool in tools:
            if tool["tool_name"] == api_name:
                tool["example"] = example
                return f"Example for API '{api_name}' annotated."
        else:
            raise ValueError(f"API '{api_name}' not found. The API names are {extract_tools(self.schema)}")

# Define the JSON schema for validation
MCP_YAML_JSON_SCHEMA_ANNOTATION = {
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
                                "required": ["tool_name", "description", "_metadata", "health"], # TODO: health and example schema
                                "optional": ["parameters", "example"],
                                "properties": {
                                    "parameters": {
                                        "type": "object",
                                        "patternProperties": {
                                            ".*": {  # Matches any parameter name
                                                "type": "object",
                                                "required": ["type", "required", "description"]
                                            }
                                        }
                                    },
                                    "health": {
                                        "type": "object",
                                        "required": ["health", "reason"],
                                        "properties": {
                                            "health": {
                                                "type": "string",
                                                "enum": ["good", "bad", "unknown"]
                                            },
                                            "reason": {
                                                "type": "string"
                                            }
                                        }
                                    },
                                    "example": {
                                        "type": "string"
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