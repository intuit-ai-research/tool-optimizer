#!/usr/bin/env python3
"""
TMDB MCP Server following Biomni patterns

Uses FastMCP and dynamic tool creation to serve TMDB API tools.
Follows the same patterns as Biomni's pubmed_mcp.py and other MCP tools.
"""

import asyncio
import json
import os
import sys
import yaml
import time
import random
from typing import Any, Dict, List, Optional
import httpx
# from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Initialize FastMCP server
# mcp = FastMCP("tmdb-tools")

# TMDB API configuration
TMDB_API_KEY = os.getenv("TMDB_ACCESS_TOKEN", "41339c143c0bc92f94712b78c551701f")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# TMDB MCP Server - Always uses real TMDB API data

# Global tools configuration loaded from YAML
TOOLS_CONFIG = {}


async def retry_with_backoff_async(func, max_retries=3, base_delay=1.0):
    """
    Async retry function with exponential backoff for handling rate limits and transient errors.
    
    Args:
        func: Async callable to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
    
    Returns:
        Result from func() or error dict if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return await func()
        except httpx.HTTPStatusError as e:
            error_msg = str(e).lower()
            status_code = e.response.status_code
            
            # Check for rate limiting errors (429)
            if status_code == 429 or 'rate limit' in error_msg or 'too many requests' in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter for rate limits
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️  Rate limit hit (HTTP {status_code}). Retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"❌ Max retries reached. Rate limit error: {e}")
                    return {"error": f"Rate limit exceeded after {max_retries} retries: HTTP {status_code}: {e.response.text}"}
            
            # Check for server errors (5xx)
            elif 500 <= status_code < 600:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️  Server error (HTTP {status_code}). Retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"❌ Max retries reached. Server error: {e}")
                    return {"error": f"Server error after {max_retries} retries: HTTP {status_code}: {e.response.text}"}
            
            # Check for temporary errors (503 Service Unavailable, 502 Bad Gateway)
            elif status_code in [502, 503, 504]:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"⚠️  Temporary error (HTTP {status_code}). Retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"❌ Max retries reached. Temporary error: {e}")
                    return {"error": f"Temporary error after {max_retries} retries: HTTP {status_code}: {e.response.text}"}
            
            # Non-retryable client errors (4xx except 429)
            else:
                print(f"❌ Non-retryable HTTP error: {status_code} - {e.response.text}")
                return {"error": f"HTTP {status_code}: {e.response.text}"}
                
        except (httpx.TimeoutException, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            # Timeout errors are retryable
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️  Timeout error. Retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            else:
                print(f"❌ Max retries reached. Timeout error: {e}")
                return {"error": f"Timeout error after {max_retries} retries: {str(e)}"}
                
        except (httpx.ConnectError, httpx.NetworkError) as e:
            # Network errors are retryable
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️  Network error. Retrying in {delay:.2f} seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            else:
                print(f"❌ Max retries reached. Network error: {e}")
                return {"error": f"Network error after {max_retries} retries: {str(e)}"}
                
        except Exception as e:
            # Unexpected errors - not retryable
            print(f"❌ Unexpected error (non-retryable): {e}")
            return {"error": f"API error: {str(e)}"}
    
    return {"error": "Unexpected retry loop exit"}


class TMDBAPIClient:
    """Helper class for making TMDB API calls."""
    
    @staticmethod
    async def make_request(endpoint: str, params: Dict[str, Any], method: str = 'GET', max_retries: int = 3) -> Dict[str, Any]:
        """Make authenticated request to TMDB API with retry logic.
        
        Args:
            endpoint: API endpoint to call
            params: Parameters to pass to the API
            method: HTTP method (GET, POST)
            max_retries: Maximum number of retry attempts for transient failures
            
        Returns:
            API response as dictionary or error dict
        """
        # Use the default API key from the original system
        api_key = TMDB_API_KEY or "41339c143c0bc92f94712b78c551701f"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Clean up endpoint
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]

        # Handle path parameter substitution (e.g., /tv/{tv_id} -> /tv/12345)
        import re
        path_params = re.findall(r'\{(\w+)\}', endpoint)
        for param_name in path_params:
            if param_name in params:
                endpoint = endpoint.replace(f'{{{param_name}}}', str(params[param_name]))
                # Remove path parameters from query parameters
                params = {k: v for k, v in params.items() if k != param_name}

        url = f"{TMDB_BASE_URL}/{endpoint}"

        # Filter out empty parameters and add API key
        clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
        clean_params['api_key'] = api_key  # Add API key to query parameters (TMDB v3 method)
        
        # Add default language if not specified
        # if 'language' not in clean_params:
        #     clean_params['language'] = 'en-US'
        
        # Define the async function to retry
        async def _make_single_request():
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method.upper() == 'GET':
                    response = await client.get(url, headers=headers, params=clean_params)
                elif method.upper() == 'POST':
                    response = await client.post(url, headers=headers, json=clean_params)
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}
                
                response.raise_for_status()
                return response.json()
        
        # Execute with retry logic
        return await retry_with_backoff_async(_make_single_request, max_retries=max_retries)
    


def load_tools_config(config_path: str):
    """Load tools configuration from MCP YAML file."""
    tools_config = {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract tools from first MCP server
        for server_name, server_config in config.get('mcp_servers', {}).items():
            if 'tools' in server_config:
                for tool in server_config['tools']:
                    tool_name = tool.get('biomni_name', '')
                    if tool_name:
                        tools_config[tool_name] = tool
                break
        
        print(f"✅ Loaded {len(tools_config)} tools from {config_path}")
        
        # Dynamically create MCP tools
        create_dynamic_tools(tools_config)
        
    except Exception as e:
        print(f"Warning: Failed to load tools config from {config_path}: {e}")
        tools_config = {}


def create_dynamic_tools(tools_config: Dict[str, Any]):
    """Dynamically create MCP tools based on loaded configuration."""
    for tool_name, tool_config in tools_config.items():
        metadata = tool_config.get('_metadata', {})
        endpoint = metadata.get('endpoint', '')
        method = metadata.get('method', 'GET')
        description = tool_config.get('description', '')
        
        if endpoint:
            # Create a dynamic tool function
            create_tool_function(tool_name, description, endpoint, method, tool_config.get('parameters', {}))


def create_tool_function(tool_name: str, description: str, endpoint: str, method: str, params_config: Dict[str, Any]):
    """Create a dynamic tool function and register it with MCP."""
    
    async def tool_function(**kwargs) -> str:
        """Dynamic tool function that makes TMDB API calls."""
        try:
            result = await TMDBAPIClient.make_request(endpoint, kwargs, method)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    # Set function metadata
    tool_function.__name__ = tool_name
    tool_function.__doc__ = description
    
    # Create parameter annotations for better type hinting
    annotations = {}
    for param_name, param_config in params_config.items():
        param_type = param_config.get('type', 'str')
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool
        }
        annotations[param_name] = type_mapping.get(param_type, str)
    
    tool_function.__annotations__ = annotations
    
    # Register with MCP
    mcp.tool()(tool_function)
    print(f"📋 Registered tool: {tool_name}")


# All tools are loaded dynamically from YAML configuration
# No hardcoded tools needed!


def main():
    """Main entry point."""
    # Check if config path is provided as argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if os.path.exists(config_path):
            print(f"🔄 Loading tools from {config_path}")
            load_tools_config(config_path)
        else:
            print(f"❌ Config file not found: {config_path}")
            print("No tools will be available!")
            return
    else:
        print("❌ No config provided! Usage: python tmdb_mcp_server.py <config.yaml>")
        print("No tools will be available!")
        return
    
    if not TMDB_API_KEY:
        print("⚠️  Warning: TMDB_ACCESS_TOKEN not set. API calls will fail.")
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()