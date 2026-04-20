import argparse
import json
import yaml
from pathlib import Path
import sys
import os
import jsonschema

from agent_tool_annotator.agent.rewrite_tool import MCP_YAML_JSON_SCHEMA

def main(mcp_yaml_path: Path):
    # read all mcp yaml files in the folder
    # first is the category name, then is the mcp yaml file
    mcp_yaml_dict = {}
    for category_folder in mcp_yaml_path.iterdir():
        for mcp_yaml_file in category_folder.iterdir():
            with open(mcp_yaml_file, "r") as f:
                mcp_yaml = yaml.safe_load(f)
                mcp_yaml_dict[mcp_yaml_file] = mcp_yaml

    # validate
    for mcp_yaml_file in mcp_yaml_dict.keys():
        with open(mcp_yaml_file, "r") as f:
            mcp_yaml = yaml.safe_load(f)
        try:
            jsonschema.validate(instance=mcp_yaml, schema=MCP_YAML_JSON_SCHEMA)
        except jsonschema.ValidationError as e:
            print(mcp_yaml_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp_yaml_path", type=str, required=True)
    args = parser.parse_args()
    main(Path(args.mcp_yaml_path))