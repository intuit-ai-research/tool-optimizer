import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class ToolManager:
    """Manages tool information from MCP YAML configuration files"""

    def __init__(self, tools: List[Dict], platform: str = 'tmdb', verbose: bool = False):
        """Initialize ToolManager with MCP YAML tools

        Args:
            tools: List of tool dictionaries from MCP YAML configuration
            platform: Platform identifier ('tmdb' or 'spotify'). Defaults to 'tmdb'.
            verbose: If True, print initialization details
        """
        self.tools = tools
        self.platform = platform.lower()
        self.verbose = verbose
        self.endpoint_mapping = self._load_endpoint_mapping()

        # Add platform to each tool's metadata for API caller to use
        for tool in self.tools:
            if isinstance(tool, dict) and '_metadata' in tool:
                tool['_metadata']['platform'] = self.platform

        if self.verbose:
            print(f"🎯 ToolManager initialized for platform: {self.platform.upper()}")
            print(f"📊 Loaded {len(self.tools)} tools")
    
    def _load_endpoint_mapping(self) -> Dict[str, str]:
        """
        Load endpoint to API name mapping from platform-specific JSON file.
        This replaces the hardcoded mapping to ensure consistency and prevent mismatches.
        """
        # Determine which JSON file to load based on platform
        json_filename = "spotify_apis_complete.json" if self.platform == "spotify" else "tmdb_57_apis_complete.json"
        
        # Try to find the JSON file in the current directory or parent directories
        current_dir = Path(__file__).parent
        possible_paths = [
            current_dir / "tmdb_apis" / json_filename,
            current_dir / json_filename,
            current_dir / ".." / "data" / json_filename,
            current_dir / ".." / ".." / json_filename,
        ]
        
        json_file_path = None
        for path in possible_paths:
            if path.exists():
                json_file_path = path
                break
        
        if json_file_path is None:
            print(f"⚠️  Warning: {json_filename} not found, using fallback empty mapping")
            return {}
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                api_data = json.load(f)
            
            endpoint_mapping = {}
            
            # Extract endpoint -> function_name mappings from all categories
            if "categories" in api_data:
                for category_name, category_data in api_data["categories"].items():
                    if "apis" in category_data:
                        for api in category_data["apis"]:
                            endpoint = api.get("endpoint", "")
                            function_name = api.get("function_name", "")
                            
                            # For Spotify, use endpoint_with_method if available to handle same endpoint with different methods
                            endpoint_with_method = api.get("endpoint_with_method", "")
                            
                            if endpoint and function_name:
                                # Add both formats for flexibility
                                if endpoint_with_method:
                                    endpoint_mapping[endpoint_with_method] = function_name  # e.g., "POST /users/{user_id}/playlists"
                                endpoint_mapping[endpoint] = function_name  # Fallback for endpoints without duplicates
            
            if self.verbose:
                print(f"✅ Loaded {len(endpoint_mapping)} endpoint mappings from {json_file_path}")
            
            # Print a few examples for verification
            if endpoint_mapping and self.verbose:
                print("📋 Sample mappings:")
                for i, (endpoint, function_name) in enumerate(list(endpoint_mapping.items())[:5]):
                    print(f"   {endpoint} -> {function_name}")
                if len(endpoint_mapping) > 5:
                    print(f"   ... and {len(endpoint_mapping) - 5} more")
            
            return endpoint_mapping
            
        except Exception as e:
            raise ValueError(f"Error loading tmdb_57_apis_complete.json: {e}")

    def extract_api_name_from_endpoint(self, endpoint: str) -> Optional[str]:
        """Extract API name from endpoint string using loaded mapping"""
        if not endpoint:
            return None
        
        # Strip any leading/trailing whitespace first
        endpoint = endpoint.strip()
        original_endpoint = endpoint
        
        # Try lookup with HTTP method first (for Spotify APIs with same endpoint but different methods)
        if endpoint in self.endpoint_mapping:
            return self.endpoint_mapping[endpoint]
        
        # Remove HTTP method prefix and try again
        http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', ' GET', ' POST', ' PUT', ' DELETE', ' PATCH']
        for method in http_methods:
            if endpoint.startswith(method):
                endpoint = endpoint[len(method)+1:].strip()
                break
        
        # Try lookup without HTTP method
        if endpoint in self.endpoint_mapping:
            return self.endpoint_mapping[endpoint]
        
        # Raise error instead of using fallback logic
        available_endpoints = list(self.endpoint_mapping.keys())[:10]  # Show first 10 for debugging
        raise ValueError(
            f"No mapping found for endpoint '{endpoint}'. "
            f"Available endpoints include: {available_endpoints}... "
            f"(Total: {len(self.endpoint_mapping)} endpoints loaded). "
            f"Please ensure tmdb_57_apis_complete.json is properly loaded and contains this endpoint."
        )
    
