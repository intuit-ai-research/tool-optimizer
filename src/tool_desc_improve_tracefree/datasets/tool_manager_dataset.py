"""
ToolManager-based dataset implementation for description improvement pipeline.
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..interfaces.dataset_interface import DatasetInterface, StandardizedTool, DatasetQuery, ToolFormat
from ..core.registries import register_dataset
from utils.tool_manager import ToolManager


@register_dataset("tool_manager", "ToolManager-based MCP YAML dataset with API endpoint mapping")
class ToolManagerDataset(DatasetInterface):
    """Dataset implementation using ToolManager for MCP YAML configuration files."""

    def __init__(self, server_name: str = "biomni_builtin", platform: str = "tmdb", verbose: bool = False, **kwargs):
        """
        Initialize ToolManager dataset.

        Args:
            server_name: Name of the MCP server section to process
            platform: Platform identifier ('tmdb', 'spotify', etc.)
            verbose: Enable verbose output
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.server_name = server_name
        self.platform = platform
        self.verbose = verbose
        self._name = "tool_manager"
        self._original_server_config = None  # Store original server config
        self._tool_manager = None  # Will be initialized when loading tools

    @property
    def name(self) -> str:
        """Dataset name identifier."""
        return self._name

    def get_tool_format(self) -> ToolFormat:
        """Return the tool format used by this dataset."""
        return ToolFormat.MCP_YAML

    def load_tools(self, tools_path: str) -> tuple[List[StandardizedTool], Dict[str, Any]]:
        """Load tools from MCP YAML configuration file using ToolManager."""
        tools_path = os.path.expanduser(tools_path)
        if not os.path.exists(tools_path):
            raise FileNotFoundError(f"Tools file not found: {tools_path}")

        # Load raw tools from MCP YAML
        def load_raw_tools(mcp_config_path):
            """Load raw tools list from MCP YAML (same as main_tmdb.py)"""
            with open(mcp_config_path, 'r') as f:
                mcp_config = yaml.safe_load(f)
                servers = mcp_config['mcp_servers']
                server_name = list(servers.keys())[0]  # only one server supported
                server_config = servers[server_name]
                tools = server_config['tools']
            return tools, server_config, server_name

        raw_tools, server_config, actual_server_name = load_raw_tools(tools_path)
        server_config["name"] = actual_server_name

        # Store actual server name and original server config
        self.server_name = actual_server_name
        self._original_server_config = {k: v for k, v in server_config.items() if k != 'tools'}

        # Initialize ToolManager with the raw tools
        self._tool_manager = ToolManager(raw_tools, platform=self.platform, verbose=self.verbose)

        # Build endpoint mapping from YAML metadata instead of relying on external JSON files
        self._build_endpoint_mapping_from_metadata(raw_tools)

        if self.verbose:
            print(f"🎯 ToolManagerDataset: Loaded {len(raw_tools)} tools for platform {self.platform}")

        # Convert raw tools to StandardizedTool objects
        tools = []
        for tool_config in raw_tools:
            tool = self._parse_tool_config(tool_config)
            tools.append(tool)

        return tools, server_config

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

    def get_tool_manager(self) -> Optional[ToolManager]:
        """Get the underlying ToolManager instance for direct access."""
        return self._tool_manager

    def _parse_tool_config(self, tool_config: Dict[str, Any]) -> StandardizedTool:
        """Parse a tool configuration from MCP YAML format."""
        # Support both 'tool_name' and 'biomni_name' keys
        tool_name = tool_config.get('tool_name') or tool_config.get('biomni_name')
        if not tool_name:
            raise ValueError(f"Tool configuration missing 'tool_name' or 'biomni_name': {tool_config}")

        # Add platform information to metadata if available
        metadata = tool_config.get('_metadata', {})
        if hasattr(self, '_tool_manager') and self._tool_manager:
            metadata['platform'] = self.platform

        tool = StandardizedTool(
            name=tool_name,
            description=tool_config['description'],
            parameters=tool_config.get('parameters', {}),
            format_specific={
                'tool_name': tool_name,
                'biomni_name': tool_name  # Keep for backward compatibility
            },
            metadata=metadata
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

    def _build_endpoint_mapping_from_metadata(self, raw_tools: List[Dict[str, Any]]):
        """
        Build endpoint to function name mapping from YAML metadata instead of external JSON files.
        This replaces the ToolManager's reliance on external JSON files.
        """
        endpoint_mapping = {}

        for tool_config in raw_tools:
            tool_name = tool_config.get('tool_name') or tool_config.get('biomni_name')
            metadata = tool_config.get('_metadata', {})

            if tool_name and metadata:
                endpoint = metadata.get('endpoint')
                method = metadata.get('method', 'GET')

                if endpoint:
                    # Add both formats for flexibility (with and without HTTP method)
                    endpoint_with_method = f"{method} {endpoint}"
                    endpoint_mapping[endpoint_with_method] = tool_name
                    endpoint_mapping[endpoint] = tool_name  # Fallback for endpoints without method prefix

        # Inject the mapping into the ToolManager to replace its broken JSON loading
        if self._tool_manager:
            self._tool_manager.endpoint_mapping = endpoint_mapping

            if self.verbose and endpoint_mapping:
                print(f"📋 Built endpoint mapping from YAML metadata: {len(endpoint_mapping)} endpoints")
                # Show a few examples
                examples = list(endpoint_mapping.items())[:3]
                for endpoint, function_name in examples:
                    print(f"   {endpoint} -> {function_name}")
                if len(endpoint_mapping) > 3:
                    print(f"   ... and {len(endpoint_mapping) - 3} more")


# Auto-register when module is imported
ToolManagerDataset.__module__ = __name__