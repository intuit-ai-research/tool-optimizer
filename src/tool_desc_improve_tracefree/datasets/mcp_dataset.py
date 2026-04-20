"""
MCP YAML dataset implementation for description improvement pipeline.
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..interfaces.dataset_interface import DatasetInterface, StandardizedTool, DatasetQuery, ToolFormat
from ..core.registries import register_dataset


@register_dataset("mcp_yaml", "MCP YAML configuration files for tool descriptions")
class MCPYamlDataset(DatasetInterface):
    """Dataset implementation for MCP YAML configuration files."""
    
    def __init__(self, server_name: str = "biomni_builtin", **kwargs):
        """
        Initialize MCP YAML dataset.
        
        Args:
            server_name: Name of the MCP server section to process
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.server_name = server_name
        self._name = "mcp_yaml"
        self._original_server_config = None  # Store original server config
    
    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return self._name
    
    def get_tool_format(self) -> ToolFormat:
        """Return the tool format used by this dataset."""
        return ToolFormat.MCP_YAML
    
    def load_tools(self, tools_path: str) -> List[StandardizedTool]:
        """Load tools from MCP YAML configuration file."""
        tools_path = os.path.expanduser(tools_path)
        if not os.path.exists(tools_path):
            raise FileNotFoundError(f"Tools file not found: {tools_path}")
        
        with open(tools_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'mcp_servers' not in data:
            raise ValueError(f"Invalid MCP YAML: missing 'mcp_servers' section in {tools_path}")
        
        if self.server_name not in data['mcp_servers']:
            available_servers = list(data['mcp_servers'].keys())
            raise ValueError(f"Server '{self.server_name}' not found. Available servers: {available_servers}")
        
        server_config = data['mcp_servers'][self.server_name]
        
        # Store original server config (excluding tools) for later use in save_tools
        self._original_server_config = {k: v for k, v in server_config.items() if k != 'tools'}
        
        if 'tools' not in server_config:
            return []  # No tools defined for this server
        
        tools = []
        for tool_config in server_config['tools']:
            tool = self._parse_tool_config(tool_config)
            tools.append(tool)
        
        return tools
    
    def save_tools(self, tools: List[StandardizedTool], output_path: str) -> str:
        """Save tools to MCP YAML configuration file format."""
        output_path = os.path.expanduser(output_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use original server config if available, otherwise use defaults
        if self._original_server_config:
            server_config = dict(self._original_server_config)
        else:
            server_config = {
                'command': ["python", "/path/to/server.py"],
                'enabled': True,
            }
        
        # Add tools list
        server_config['tools'] = []
        
        # Build MCP YAML structure
        mcp_config = {
            'mcp_servers': {
                self.server_name: server_config
            }
        }
        
        # Convert tools to MCP format
        for tool in tools:
            tool_config = self._tool_to_mcp_config(tool)
            mcp_config['mcp_servers'][self.server_name]['tools'].append(tool_config)
        
        # Save to file
        with open(output_path, 'w') as f:
            yaml.dump(mcp_config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        return output_path
    
    def load_queries(self, queries_path: str) -> List[DatasetQuery]:
        """Load queries from JSON file (standard format)."""
        import json
        
        queries_path = os.path.expanduser(queries_path)
        if not os.path.exists(queries_path):
            raise FileNotFoundError(f"Queries file not found: {queries_path}")
        
        with open(queries_path, 'r') as f:
            data = json.load(f)
        
        # Handle different query file formats
        if isinstance(data, dict):
            # Try common keys for query lists
            if 'generated_questions' in data:
                data = data['generated_questions']
            elif 'queries' in data:
                data = data['queries']
            elif 'questions' in data:
                data = data['questions']
        
        queries = []
        for item in data:
            # Handle string items (JSON-encoded strings)
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except json.JSONDecodeError:
                    # If it's just a plain string question, use it directly
                    item = {'question': item}
            
            # Ensure item is a dict
            if not isinstance(item, dict):
                continue
                
            query = DatasetQuery(
                query_id=str(item.get('id', len(queries))),
                query_text=item.get('question', item.get('query', '')),
                expected_tools=item.get('expected_tools'),
                expected_solution=item.get('expected_solution'),
                metadata=item
            )
            queries.append(query)
        
        return queries
    
    def validate_tool(self, tool: StandardizedTool) -> bool:
        """Validate tool conforms to MCP YAML requirements."""
        # Check required fields
        if not tool.name or not tool.description:
            return False
        
        # Check parameters structure
        if not isinstance(tool.parameters, dict):
            return False
        
        # Validate parameter definitions
        for param_name, param_def in tool.parameters.items():
            if not isinstance(param_def, dict):
                return False
            if 'type' not in param_def:
                return False
        
        return True
    
    def _parse_tool_config(self, tool_config: Dict[str, Any]) -> StandardizedTool:
        """Parse a tool configuration from MCP YAML format."""
        # Support both 'tool_name' and 'biomni_name' keys
        tool_name = tool_config.get('tool_name') or tool_config.get('biomni_name')
        if not tool_name:
            raise ValueError(f"Tool configuration missing 'tool_name' or 'biomni_name': {tool_config}")
        
        tool = StandardizedTool(
            name=tool_name,
            description=tool_config['description'],
            parameters=tool_config.get('parameters', {}),
            format_specific={
                'tool_name': tool_name,
                'biomni_name': tool_name  # Keep for backward compatibility
            },
            metadata=tool_config.get('_metadata', {})
        )
        
        return tool
    
    def _tool_to_mcp_config(self, tool: StandardizedTool) -> Dict[str, Any]:
        """Convert a StandardizedTool to MCP YAML configuration format."""
        # Use 'tool_name' by default (more standard), but can be overridden
        config = {
            'tool_name': tool.name,
            'description': tool.description,
            'parameters': tool.parameters
        }
        
        # Include metadata if present
        if tool.metadata:
            config['_metadata'] = tool.metadata
        
        return config


# Auto-register when module is imported
MCPYamlDataset.__module__ = __name__
