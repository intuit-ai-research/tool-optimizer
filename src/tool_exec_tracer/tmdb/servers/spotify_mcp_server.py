#!/usr/bin/env python3
"""
Spotify MCP Server

Uses FastMCP and dynamic tool creation to serve Spotify API tools.
Follows the same patterns as TMDB MCP server.
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
import re
from typing import Set, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..utils.spotify_auth import get_spotify_access_token

# Initialize FastMCP server
# mcp = FastMCP("spotify-tools")

# Spotify API configuration
SPOTIFY_BASE_URL = "https://api.spotify.com/v1"

# Global tools configuration loaded from YAML
TOOLS_CONFIG = {}

# Load Spotify OpenAPI specification
# Use path relative to this file's location
_current_dir = os.path.dirname(os.path.abspath(__file__))
_oas_path = os.path.join(_current_dir, 'specs', 'spotify_oas.json')
with open(_oas_path, 'r') as f:
    SPOTIFY_OAS = json.load(f)


async def retry_with_backoff_async(func, max_retries=3, base_delay=10.0):
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
                    # Fixed 30-second delay for rate limits
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"⚠️  Rate limit hit (HTTP {status_code}). Waiting {delay:.0f} seconds before retry... (attempt {attempt + 1}/{max_retries})")
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

def _match_oas_path(endpoint: str) -> Tuple[str, dict]:
    """Return the OAS path template and path item that match a concrete endpoint."""
    ep = endpoint.strip("/")
    for oas_path, item in SPOTIFY_OAS.get("paths", {}).items():
        pat = "^" + re.sub(r"\{[^}]+\}", r"[^/]+", oas_path.strip("/")) + "$"
        if re.match(pat, ep):
            return oas_path, item
    return "", {}

def _get_operation(oas_path_item: dict, method: str) -> dict:
    return oas_path_item.get(method.lower(), {}) if oas_path_item else {}

def _extract_param_names(op: dict) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (path_names, query_names, header_names) from OAS operation.parameters."""
    path_names, query_names, header_names = set(), set(), set()
    for p in op.get("parameters", []):
        where = p.get("in")
        name = p.get("name")
        if where == "path": path_names.add(name)
        elif where == "query": query_names.add(name)
        elif where == "header": header_names.add(name)
    return path_names, query_names, header_names

def _has_json_body(op: dict) -> bool:
    rb = op.get("requestBody", {})
    content = rb.get("content", {}) if isinstance(rb, dict) else {}
    return "application/json" in content

class SpotifyAPIClient:
    """Helper class for making Spotify API calls."""
    
    _cached_user_id = None  # Cache user ID to avoid repeated calls
    
    @staticmethod
    async def get_current_user_id() -> Optional[str]:
        """Fetch current user's Spotify ID. Returns None if auth fails."""
        if SpotifyAPIClient._cached_user_id:
            return SpotifyAPIClient._cached_user_id
            
        access_token = get_spotify_access_token()
        if not access_token:
            return None
            
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{SPOTIFY_BASE_URL}/me", headers=headers)
                if response.status_code == 200:
                    user_data = response.json()
                    user_id = user_data.get('id')
                    SpotifyAPIClient._cached_user_id = user_id
                    print(f"✅ Fetched user ID: {user_id}")
                    return user_id
        except Exception as e:
            print(f"⚠️  Could not fetch user ID: {e}")
        return None
    
    @staticmethod
    async def make_request(endpoint: str, params: Dict[str, Any], method: str = 'GET', max_retries: int = 3) -> Dict[str, Any]:
        """Make authenticated request to Spotify API with retry logic.
        
        Args:
            endpoint: API endpoint to call
            params: Parameters to pass to the API
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            max_retries: Maximum number of retry attempts for transient failures
            
        Returns:
            API response as dictionary or error dict
        """
        # Get access token using proper OAuth flow
        access_token = get_spotify_access_token()
        
        if not access_token:
            return {"error": "Failed to obtain Spotify access token. Please configure authentication."}
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Clean up endpoint
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        
        # Discover OAS operation for smart routing of params
        oas_path, path_item = _match_oas_path(endpoint)
        op = _get_operation(path_item, method)
        path_names, query_names, header_names = _extract_param_names(op)
        needs_json_body = _has_json_body(op)

        # Handle path parameter substitution (e.g., /albums/{id} -> /albums/12345)
        import re
        # path_params = re.findall(r'\{(\w+)\}', endpoint)
        path_params = re.findall(r'\{(\w+)\}', oas_path or endpoint)

        # for param_name in path_params:
        #     if param_name in params:
        #         endpoint = endpoint.replace(f'{{{param_name}}}', str(params[param_name]))
        #         # Remove path parameters from query/body parameters
        #         params = {k: v for k, v in params.items() if k != param_name}
        #     elif param_name == 'user_id':
        #         # Auto-fetch user_id if missing
        #         user_id = await SpotifyAPIClient.get_current_user_id()
        #         if user_id:
        #             endpoint = endpoint.replace(f'{{{param_name}}}', user_id)
        #             print(f"✅ Auto-filled user_id: {user_id}")
        #         else:
        #             return {"error": f"Missing required path parameter '{param_name}' and could not auto-fetch it. User authentication required."}

        path_params = re.findall(r'\{(\w+)\}', oas_path or endpoint)
        for pname in path_params:
            if pname in params:
                endpoint = endpoint.replace(f'{{{pname}}}', str(params.pop(pname)))
            elif pname == 'user_id':  # special convenience: auto fetch
                user_id = await SpotifyAPIClient.get_current_user_id()
                if user_id:
                    endpoint = endpoint.replace("{user_id}", user_id)
                else:
                    return {"error": "Missing required path parameter 'user_id' and could not auto-fetch it. User authentication required."}
            else:
                return {"error": f"Missing required path parameter '{pname}'."}
        
        url = f"{SPOTIFY_BASE_URL}/{endpoint}"

        # Filter out empty parameters (including empty lists)
        clean_params = {k: v for k, v in (params or {}).items() 
                       if v is not None and v != "" and (not isinstance(v, list) or len(v) > 0)}
        
        # Convert list parameters to comma-separated strings for GET requests
        # Spotify API expects lists as comma-separated values in query parameters
        # if method.upper() == 'GET':
        #     for key, value in list(clean_params.items()):
        #         if isinstance(value, list):
        #             clean_params[key] = ','.join(str(v) for v in value)
        
        # Define the async function to retry
        # async def _make_single_request():
        #     async with httpx.AsyncClient(timeout=30.0) as client:
        #         if method.upper() == 'GET':
        #             response = await client.get(url, headers=headers, params=clean_params)
        #         elif method.upper() == 'POST':
        #             # For POST requests, check if we should use query params or JSON body
        #             # Many Spotify endpoints (e.g., player endpoints) use POST with query params
        #             # If all params are simple types (str, int, float, bool), use query params
        #             # Otherwise, use JSON body for complex data structures
        #             has_complex_params = any(isinstance(v, (dict, list)) for v in clean_params.values())
        #             if has_complex_params:
        #                 response = await client.post(url, headers=headers, json=clean_params)
        #             else:
        #                 # Use query parameters for simple types (common in Spotify API)
        #                 response = await client.post(url, headers=headers, params=clean_params)
        #         elif method.upper() == 'PUT':
        #             # For PUT requests, check if we should use query params or JSON body
        #             # Similar to POST - simple types use query params, complex types use JSON body
        #             has_complex_params = any(isinstance(v, (dict, list)) for v in clean_params.values())
        #             if has_complex_params:
        #                 response = await client.put(url, headers=headers, json=clean_params)
        #             else:
        #                 response = await client.put(url, headers=headers, params=clean_params)
        #         elif method.upper() == 'DELETE':
        #             # For DELETE requests, check if we should use query params or JSON body
        #             # Many endpoints (e.g., remove_albums_user) use DELETE with query params
        #             if clean_params:
        #                 has_complex_params = any(isinstance(v, (dict, list)) for v in clean_params.values())
        #                 if has_complex_params:
        #                     response = await client.delete(url, headers=headers, json=clean_params)
        #                 else:
        #                     response = await client.delete(url, headers=headers, params=clean_params)
        #             else:
        #                 response = await client.delete(url, headers=headers)
        #         elif method.upper() == 'PATCH':
        #             # For PATCH requests, typically used for partial updates with structured data
        #             # But apply same logic for consistency
        #             has_complex_params = any(isinstance(v, (dict, list)) for v in clean_params.values())
        #             if has_complex_params:
        #                 response = await client.patch(url, headers=headers, json=clean_params)
        #             else:
        #                 response = await client.patch(url, headers=headers, params=clean_params)
        #         else:
        #             return {"error": f"Unsupported HTTP method: {method}"}
                
        #         # Raise for HTTP errors first
        #         response.raise_for_status()
                
        #         # Handle 204 No Content responses
        #         if response.status_code == 204:
        #             return {"status": "success", "message": "Operation completed successfully"}
                
        #         # Some endpoints return empty responses or whitespace-only responses
        #         # Check if content is empty or just whitespace before trying to parse JSON
        #         if not response.content or not response.content.strip():
        #             return {"status": "success", "message": "Operation completed successfully"}
                
        #         # Try to parse JSON response
        #         try:
        #             return response.json()
        #         except ValueError as e:
        #             # If JSON parsing fails, return the raw text
        #             return {"status": "success", "raw_response": response.text}

                # Convert list query params to comma-separated strings (any method)

        # Split params by OAS location (query vs body vs headers). Anything not declared as query/path/header goes to body when a body exists.
        query_dict, header_dict, body_dict = {}, {}, {}
        for k, v in clean_params.items():
            if k in query_names:
                query_dict[k] = v
            elif k in header_names:
                header_dict[k] = v
            else:
                body_dict[k] = v

        for k, v in list(query_dict.items()):
            if isinstance(v, list):
                query_dict[k] = ",".join(str(x) for x in v)

        # Merge discovered header params into headers (if any)
        if header_dict:
            headers.update({str(k): str(v) for k, v in header_dict.items()})

        async def _make_single_request():
            async with httpx.AsyncClient(timeout=30.0) as client:
                m = method.upper()

                # Decide how to send body according to OAS
                send_json = body_dict if (needs_json_body and m in {'POST','PUT','PATCH'}) else None

                if m == 'GET':
                    response = await client.get(url, headers=headers, params=query_dict)
                elif m == 'POST':
                    response = await client.post(url, headers=headers, params=query_dict, json=send_json)
                elif m == 'PUT':
                    response = await client.put(url, headers=headers, params=query_dict, json=send_json)
                elif m == 'DELETE':
                    # FIXME: Need to further check when body is used for DELETE.
                    response = await client.delete(url, headers=headers, params=body_dict)
                elif m == 'PATCH':
                    response = await client.patch(url, headers=headers, params=query_dict, json=send_json)
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}

                response.raise_for_status()
                if response.status_code == 204:
                    return {"status": "success", "message": "Operation completed successfully"}
                if not response.content or not response.content.strip():
                    return {"status": "success", "message": "Operation completed successfully"}
                try:
                    return response.json()
                except ValueError:
                    return {"status": "success", "raw_response": response.text}
        
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
                    tool_name = tool.get('tool_name', '')
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
        """Dynamic tool function that makes Spotify API calls."""
        try:
            result = await SpotifyAPIClient.make_request(endpoint, kwargs, method)
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
        print("❌ No config provided! Usage: python spotify_mcp_server.py <config.yaml>")
        print("No tools will be available!")
        return
    
    # Check authentication
    from utils.spotify_auth import get_spotify_auth_manager
    auth_manager = get_spotify_auth_manager()
    
    if not auth_manager.is_authenticated():
        print("⚠️  Warning: Spotify authentication not configured properly!")
        print("⚠️  Please set one of the following:")
        print("    1. SPOTIFY_ACCESS_TOKEN (manual token)")
        print("    2. SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET (app-only access)")
        print("    3. SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET + SPOTIFY_REFRESH_TOKEN (full access)")
    else:
        print("✅ Spotify authentication configured successfully")
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
