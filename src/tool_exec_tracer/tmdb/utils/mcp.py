from tool_exec_tracer.tmdb.servers.tmdb_mcp_server import TMDBAPIClient
import asyncio
import json
import os
from typing import Dict, Any

TMDB_API_KEY = os.getenv("TMDB_ACCESS_TOKEN", "41339c143c0bc92f94712b78c551701f")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def call_tmdb_api_mcp(api_name: str, 
                    parameters: Dict[str, Any], 
                    metadata: Dict[str, Any]) -> Dict[str, Any]:

    async def mcp_client_call(api_name: str, parameters: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        return await TMDBAPIClient.make_request(metadata["endpoint"], parameters, metadata["method"])

    return asyncio.run(mcp_client_call(api_name, parameters, metadata))