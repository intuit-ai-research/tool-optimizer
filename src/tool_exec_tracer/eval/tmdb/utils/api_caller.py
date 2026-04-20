"""
Generic API caller that supports both TMDB and Spotify platforms.
Automatically detects platform from metadata and routes to appropriate client.
"""

from tool_exec_tracer.eval.tmdb.servers.tmdb_mcp_server import TMDBAPIClient
from tool_exec_tracer.eval.tmdb.servers.spotify_mcp_server import SpotifyAPIClient
from tool_exec_tracer.eval.tmdb.utils.mcp_call_logger import log_mcp_call
import asyncio
import json
import os
from typing import Dict, Any, Optional


def call_api_mcp(api_name: str, 
                 parameters: Dict[str, Any], 
                 metadata: Dict[str, Any],
                 platform: Optional[str] = None,
                 query_id: Optional[str] = None,
                 subtask_id: Optional[int] = None, rapidapi_wrapper=None,
                 tool_provider:str = "") -> Dict[str, Any]:
    """
    Generic API caller that works for both TMDB and Spotify.
    
    Args:
        api_name: Name of the API function
        parameters: Parameters to pass to the API
        metadata: Metadata containing endpoint, method, etc.
        platform: Platform identifier ('tmdb' or 'spotify'). If None, auto-detects from endpoint.
        query_id: Optional query identifier for logging
        subtask_id: Optional subtask identifier for logging
    
    Returns:
        API response as dictionary
    """
    
    async def mcp_client_call(api_name: str, parameters: Dict[str, Any], metadata: Dict[str, Any], platform: str, rapidapi_wrapper=None, tool_provider="") -> Dict[str, Any]:
        endpoint = metadata.get("endpoint", "")
        method = metadata.get("method", "GET")
        
        # Use platform directly from parameter (should come from args.dataset)
        if not platform:
            # Check metadata for platform hint as fallback
            platform = metadata.get("platform", "")
            
            if not platform:
                # Default to tmdb for backwards compatibility
                platform = "tmdb"
        
        # Route to appropriate client
        if rapidapi_wrapper is not None:
            api_name_in_rapidapi_wrapper = rapidapi_wrapper.rapidapi_wrapper_name_mapping[f"{tool_provider}::{api_name}"]
            response, status_code = rapidapi_wrapper.step(action_name=api_name_in_rapidapi_wrapper, action_input=json.dumps(parameters))
            response = json.loads(response)
        elif platform.lower() == "spotify":
            response = await SpotifyAPIClient.make_request(endpoint, parameters, method)
        else:  # default to tmdb
            response = await TMDBAPIClient.make_request(endpoint, parameters, method)
        
        # Log the call (will no-op if logger not initialized)
        log_mcp_call(
            api_name=api_name,
            parameters=parameters,
            metadata=metadata,
            response=response,
            query_id=query_id,
            subtask_id=subtask_id,
            platform=platform
        )
        
        return response
    
    return asyncio.run(mcp_client_call(api_name, parameters, metadata, platform or "", rapidapi_wrapper=rapidapi_wrapper, tool_provider=tool_provider))


# Backwards compatibility functions
def call_tmdb_api_mcp(api_name: str, 
                      parameters: Dict[str, Any], 
                      metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    TMDB-specific API caller (backwards compatible).
    """
    return call_api_mcp(api_name, parameters, metadata, platform="tmdb")


def call_spotify_api_mcp(api_name: str, 
                         parameters: Dict[str, Any], 
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Spotify-specific API caller.
    """
    return call_api_mcp(api_name, parameters, metadata, platform="spotify")
