from pathlib import Path
import argparse
import yaml
import sys
from agent_tool_annotator.agent.rewrite_tool import extract_tools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp_source_folder", type=str, help="the source folder of mcp yaml files")
    parser.add_argument("--mcp_target_folder", type=str, help="the target folder of mcp yaml files")
    args = parser.parse_args()

    # both has the same structure. category folder --> mcp yaml file

    mcp_source_folder = Path(args.mcp_source_folder)
    mcp_target_folder = Path(args.mcp_target_folder)
    for category_folder in mcp_source_folder.iterdir():
        for mcp_yaml_file in category_folder.iterdir():
            if not mcp_yaml_file.suffix == ".yaml":
                continue
            with open(mcp_yaml_file, "r") as f:
                mcp_yaml_source = yaml.safe_load(f)
            folder_name = mcp_yaml_file.parent.name
            file_name = mcp_yaml_file.name
            mcp_yaml_target = mcp_target_folder / folder_name / file_name
            with open(mcp_yaml_target, "r") as f:
                mcp_yaml_target = yaml.safe_load(f)
            
            # check tools are the same 
            tools_source = extract_tools(mcp_yaml_source)
            tools_target = extract_tools(mcp_yaml_target)
            if set(tools_source) != set(tools_target):
                # print(f"tools are different for {mcp_yaml_file}. \nsource only: {tools_source - tools_target}\ntarget only: {tools_target - tools_source}")
                # print('--------------------------------')
                print(mcp_yaml_file)
                continue

if __name__ == "__main__":
    main()