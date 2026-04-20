import json
import yaml
from pathlib import Path
import sys
import os
import jsonschema

# Add current directory to path to find agent_tool_annotator module if running from inside
sys.path.append(os.getcwd())

try:
    from agent_tool_annotator.rewrite_tool import SaveSchema
except ImportError:
    # Fallback if running from inside agent_tool_annotator dir
    sys.path.append("..")
    from agent_tool_annotator.rewrite_tool import SaveSchema

# Use a dummy path for testing
test_output = Path("test_output.yaml")
tool = SaveSchema(test_output)

# 1. Valid schema test
valid_schema = """
{
    "mcp_servers": {
        "test_provider": {
            "command": [],
            "enabled": true,
            "description": "test desc",
            "category": "test cat",
            "tools": [
                {
                    "tool_name": "test_tool",
                    "description": "test tool desc",
                    "parameters": {
                        "arg1": {
                            "type": "string",
                            "required": true,
                            "description": "arg desc"
                        }
                    },
                    "_metadata": {}
                }
            ]
        }
    }
}
"""
print("Testing valid schema...")
result = tool.forward(valid_schema)
print(f"Result: {result}")
assert "Schema saved" in result
assert test_output.exists()

# 2. Invalid schema: Missing root
invalid_root = '{"foo": "bar"}'
print("\nTesting invalid root...")
try:
    tool.forward(invalid_root)
    print("FAIL: Should have raised ValidationError")
except jsonschema.ValidationError as e:
    print(f"Caught expected error: {e.message}")
    assert "'mcp_servers' is a required property" in e.message

# 3. Invalid schema: Missing provider field
invalid_provider_dict = {
    "mcp_servers": {
        "test_provider": {
            "command": [],
            "enabled": True,
            "description": "test desc",
            "category": "test cat"
            # Missing tools
        }
    }
}
print("\nTesting missing provider field...")
try:
    tool.forward(json.dumps(invalid_provider_dict))
    print("FAIL: Should have raised ValidationError")
except jsonschema.ValidationError as e:
    print(f"Caught expected error: {e.message}")
    assert "'tools' is a required property" in e.message

# 4. Invalid schema: Missing tool field
invalid_tool_dict = {
    "mcp_servers": {
        "test_provider": {
            "command": [],
            "enabled": True,
            "description": "test desc",
            "category": "test cat",
            "tools": [
                {
                    "tool_name": "test_tool",
                    # Missing description
                    "parameters": {},
                    "_metadata": {}
                }
            ]
        }
    }
}
print("\nTesting missing tool field...")
try:
    tool.forward(json.dumps(invalid_tool_dict))
    print("FAIL: Should have raised ValidationError")
except jsonschema.ValidationError as e:
    print(f"Caught expected error: {e.message}")
    assert "'description' is a required property" in e.message

# 5. Invalid schema: Missing parameter field
invalid_param_dict = {
    "mcp_servers": {
        "test_provider": {
            "command": [],
            "enabled": True,
            "description": "test desc",
            "category": "test cat",
            "tools": [
                {
                    "tool_name": "test_tool",
                    "description": "desc",
                    "parameters": {
                        "arg1": {
                            "type": "string"
                            # Missing required and description
                        }
                    },
                    "_metadata": {}
                }
            ]
        }
    }
}
print("\nTesting missing parameter field...")
try:
    tool.forward(json.dumps(invalid_param_dict))
    print("FAIL: Should have raised ValidationError")
except jsonschema.ValidationError as e:
    print(f"Caught expected error: {e.message}")
    # The error message format depends on jsonschema version, checking for field name
    assert "'required' is a required property" in e.message or "'description' is a required property" in e.message

print("\nAll tests passed!")

# Cleanup
if test_output.exists():
    test_output.unlink()

