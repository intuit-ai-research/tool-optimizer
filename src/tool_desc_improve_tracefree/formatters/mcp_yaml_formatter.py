"""
Universal MCP YAML formatter that converts improved descriptions from any dataset
to MCP YAML format compatible with Biomni tutorials/examples/add_mcp_server/mcp_config.yaml
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from copy import deepcopy
from ..interfaces.dataset_interface import StandardizedTool


class MCPYamlFormatter:
    """Universal formatter that converts improved tools to MCP YAML format."""
    
    def __init__(self, server_name: str = "biomni_builtin", 
                 command: Optional[List[str]] = None,
                 preserve_comments: bool = True,
                 dataset_name: str = "biomni"):
        """
        Initialize MCP YAML formatter.
        
        Args:
            server_name: Name of the MCP server to use in output
            command: Command to use for the MCP server
            preserve_comments: Whether to preserve comments in output
            dataset_name: Name of the dataset (determines tool name key)
        """
        self.server_name = server_name
        self.command = command or [
            "python", 
            "/home/sagemaker-user/user-default-efs/FunctionWrapper/Biomni/tutorials/examples/add_mcp_server/builtin_tools_mcp.py"
        ]
        self.preserve_comments = preserve_comments
        self.dataset_name = dataset_name
    
    def format_tools(self, server_config: Dict[str, Any], tools: List[StandardizedTool], 
                    output_path: str,
                    header_comment: Optional[str] = None) -> str:
        """
        Format improved tools as MCP YAML configuration.
        
        Args:
            tools: List of improved tools to format
            output_path: Path where to save the formatted YAML
            header_comment: Optional header comment for the file
            
        Returns:
            Path to the saved YAML file
        """
        output_path = os.path.expanduser(output_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build MCP YAML structure
        mcp_config = self._build_mcp_config(server_config, tools, header_comment)
        
        # Save to file with proper formatting
        self._save_yaml_with_formatting(mcp_config, output_path, header_comment)
        
        return output_path
    
    def _build_mcp_config(self, server_config: Dict[str, Any], tools: List[StandardizedTool], 
                         header_comment: Optional[str] = None) -> Dict[str, Any]:
        """Build the MCP configuration structure."""
        server_config = deepcopy(server_config)
        server_config['tools'] = []
        mcp_config = {
            'mcp_servers': {
                server_config['name']: server_config
            }
        }
        
        # Convert tools to MCP format
        for tool in tools:
            tool_config = self._tool_to_mcp_format(tool)
            mcp_config['mcp_servers'][server_config['name']]['tools'].append(tool_config)
        
        return mcp_config
    
    def _tool_to_mcp_format(self, tool: StandardizedTool) -> Dict[str, Any]:
        """Convert a StandardizedTool to MCP YAML tool configuration."""
        # Use 'biomni_name' for biomni datasets, 'tool_name' for others
        name_key = 'biomni_name' if 'biomni' in self.dataset_name.lower() or 'dbqa' in self.dataset_name.lower() else 'tool_name'
        
        config = {
            name_key: tool.name,
            'description': tool.description,
            'parameters': self._format_parameters(tool.parameters)
        }
        
        # Include metadata if present (preserves _metadata from D0 for evaluation)
        if tool.metadata:
            config['_metadata'] = tool.metadata
        
        return config
    
    def _format_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Format parameters to match MCP YAML structure."""
        formatted_params = {}
        
        for param_name, param_def in parameters.items():
            if isinstance(param_def, dict):
                # Already in correct format
                formatted_params[param_name] = param_def
            else:
                # Convert simple format to MCP format
                formatted_params[param_name] = {
                    'type': 'str',
                    'required': True,
                    'description': str(param_def) if param_def else f"Parameter {param_name}"
                }
        
        return formatted_params
    
    def _save_yaml_with_formatting(self, config: Dict[str, Any], 
                                  output_path: str,
                                  header_comment: Optional[str] = None) -> None:
        """Save YAML with proper formatting and comments."""
        with open(output_path, 'w') as f:
            # Write header comment if provided
            if header_comment:
                f.write(f"# {header_comment}\n\n")
            elif self.preserve_comments:
                # Write default header
                f.write("# MCP Server Config for Biomni - Improved Descriptions\n")
                f.write("# Generated by description improvement pipeline\n\n")
            
            # Write YAML content with proper formatting
            yaml.dump(config, f, 
                     default_flow_style=False, 
                     indent=2, 
                     sort_keys=False,
                     allow_unicode=True)
    
    @classmethod
    def format_from_any_dataset(cls, server_config: Dict[str, Any], tools: List[StandardizedTool], 
                               output_path: str,
                               dataset_name: str = "unknown",
                               iteration: str = "improved") -> str:
        """
        Class method to format tools from any dataset type.
        
        Args:
            tools: Improved tools from any dataset
            output_path: Where to save the MCP YAML
            dataset_name: Name of the source dataset
            iteration: Iteration identifier (e.g., "D1", "D2")
            
        Returns:
            Path to saved file
        """
        formatter = cls(dataset_name=dataset_name)
        
        header_comment = f"Improved tool descriptions for {dataset_name} dataset ({iteration})"
        
        return formatter.format_tools(server_config, tools, output_path, header_comment)


def save_as_mcp_yaml(server_config: Dict[str, Any], tools: List[StandardizedTool], 
                     output_path: str,
                     dataset_name: str = "unknown",
                     iteration: str = "improved") -> str:
    """
    Convenience function to save any improved tools as MCP YAML.
    
    Args:
        tools: List of improved tools
        output_path: Path to save the MCP YAML file
        dataset_name: Name of the source dataset 
        iteration: Iteration identifier
        
    Returns:
        Path to saved file
    """
    return MCPYamlFormatter.format_from_any_dataset(
        server_config, tools, output_path, dataset_name, iteration
    )
