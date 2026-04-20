"""
MCP Call Logger - Logs all MCP API call signatures and responses
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class MCPCallLogger:
    """Logger for MCP API calls including signatures (name, parameters) and responses"""
    
    def __init__(self, output_dir: Optional[str] = None, enabled: bool = True):
        """
        Initialize MCP call logger
        
        Args:
            output_dir: Directory to save logs. If None, uses current working directory
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.output_dir = output_dir
        self.call_history = []
        
        if self.enabled and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.log_file = os.path.join(self.output_dir, "mcp_call_log.jsonl")
            print(f"📝 MCP Call Logger initialized - logging to: {self.log_file}")
        else:
            self.log_file = None
    
    def log_call(self, 
                 api_name: str,
                 parameters: Dict[str, Any],
                 metadata: Dict[str, Any],
                 response: Dict[str, Any],
                 query_id: Optional[str] = None,
                 subtask_id: Optional[int] = None,
                 platform: Optional[str] = None) -> Dict[str, Any]:
        """
        Log an MCP API call with its signature and response
        
        Args:
            api_name: Name of the API being called
            parameters: Parameters passed to the API
            metadata: API metadata (endpoint, method, etc.)
            response: Response from the API
            query_id: Optional query identifier
            subtask_id: Optional subtask identifier
            platform: Platform (tmdb/spotify)
            
        Returns:
            Log entry dictionary
        """
        if not self.enabled:
            return {}
        
        timestamp = datetime.now().isoformat()
        
        # Determine if the call was successful
        success = not bool(response.get("error", ""))
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "query_id": query_id,
            "subtask_id": subtask_id,
            "call_signature": {
                "api_name": api_name,
                "parameters": parameters,
                "endpoint": metadata.get("endpoint", ""),
                "method": metadata.get("method", "GET"),
                "platform": platform or metadata.get("platform", "unknown")
            },
            "response": {
                "success": success,
                **response,
            },
            "metadata": {
                "response_size": len(str(response.get("response", {}))),
                "has_error": bool(response.get("error", ""))
            }
        }
        
        # Add to history
        self.call_history.append(log_entry)
        
        # Write to file immediately (append mode)
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"⚠️  Warning: Failed to write MCP call log: {e}")
        
        # Print summary
        status = "✅" if success else "❌"
        print(f"   {status} MCP Call: {api_name} | Platform: {platform or metadata.get('platform', 'unknown')} | Params: {len(parameters)} | Response Size: {log_entry['metadata']['response_size']}")
        
        return log_entry
    
    def save_summary(self, summary_file: Optional[str] = None) -> None:
        """
        Save a summary of all logged calls
        
        Args:
            summary_file: Path to summary file. If None, saves to output_dir
        """
        if not self.enabled or not self.call_history:
            return
        
        # Calculate statistics
        total_calls = len(self.call_history)
        successful_calls = sum(1 for entry in self.call_history if entry["response"]["success"])
        failed_calls = total_calls - successful_calls
        
        # Count by API
        api_counts = {}
        for entry in self.call_history:
            api_name = entry["call_signature"]["api_name"]
            api_counts[api_name] = api_counts.get(api_name, 0) + 1
        
        # Count by platform
        platform_counts = {}
        for entry in self.call_history:
            platform = entry["call_signature"]["platform"]
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        summary = {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "api_call_counts": api_counts,
            "platform_counts": platform_counts,
            "call_history": self.call_history
        }
        
        # Determine output file
        if summary_file is None and self.output_dir:
            summary_file = os.path.join(self.output_dir, "mcp_call_summary.json")
        
        if summary_file:
            try:
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"📊 MCP Call Summary saved to: {summary_file}")
                print(f"   Total Calls: {total_calls} | Success: {successful_calls} | Failed: {failed_calls}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to save MCP call summary: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged calls"""
        if not self.call_history:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "success_rate": 0.0,
                "error_analysis": {}
            }
        
        total_calls = len(self.call_history)
        successful_calls = sum(1 for entry in self.call_history if entry["response"]["success"])
        
        # Analyze error types
        error_analysis = self.analyze_errors()
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
            "error_analysis": error_analysis
        }
    
    def analyze_errors(self) -> Dict[str, Any]:
        """
        Analyze and categorize errors from the call history
        
        Returns:
            Dictionary containing error statistics by type, HTTP status, and API
        """
        if not self.call_history:
            return {
                "total_errors": 0,
                "error_types": {},
                "http_status_codes": {},
                "errors_by_api": {},
                "errors_by_platform": {},
                "common_error_messages": []
            }
        
        error_entries = [entry for entry in self.call_history if not entry["response"]["success"]]
        
        error_types = {}
        http_status_codes = {}
        errors_by_api = {}
        errors_by_platform = {}
        error_messages = {}
        
        for entry in error_entries:
            error_msg = entry["response"]["error"]
            api_name = entry["call_signature"]["api_name"]
            platform = entry["call_signature"]["platform"]
            
            # Categorize error type
            error_type = self._categorize_error(error_msg)
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Extract HTTP status code if present
            http_status = self._extract_http_status(error_msg)
            if http_status:
                http_status_codes[http_status] = http_status_codes.get(http_status, 0) + 1
            
            # Count by API
            errors_by_api[api_name] = errors_by_api.get(api_name, 0) + 1
            
            # Count by platform
            errors_by_platform[platform] = errors_by_platform.get(platform, 0) + 1
            
            # Track error messages
            error_messages[error_msg] = error_messages.get(error_msg, 0) + 1
        
        # Get top 10 most common error messages
        common_errors = sorted(error_messages.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_errors": len(error_entries),
            "error_types": error_types,
            "http_status_codes": http_status_codes,
            "errors_by_api": errors_by_api,
            "errors_by_platform": errors_by_platform,
            "common_error_messages": [
                {"message": msg, "count": count, "percentage": count / len(error_entries) * 100}
                for msg, count in common_errors
            ]
        }
    
    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error based on error message"""
        error_msg_lower = error_msg.lower()
        
        if "http 400" in error_msg_lower or "bad request" in error_msg_lower:
            return "HTTP_400_BAD_REQUEST"
        elif "http 401" in error_msg_lower or "unauthorized" in error_msg_lower:
            return "HTTP_401_UNAUTHORIZED"
        elif "http 403" in error_msg_lower or "forbidden" in error_msg_lower:
            return "HTTP_403_FORBIDDEN"
        elif "http 404" in error_msg_lower or "not found" in error_msg_lower:
            return "HTTP_404_NOT_FOUND"
        elif "http 429" in error_msg_lower or "rate limit" in error_msg_lower:
            return "HTTP_429_RATE_LIMIT"
        elif "http 500" in error_msg_lower or "internal server error" in error_msg_lower:
            return "HTTP_500_SERVER_ERROR"
        elif "http 503" in error_msg_lower or "service unavailable" in error_msg_lower:
            return "HTTP_503_SERVICE_UNAVAILABLE"
        elif "timeout" in error_msg_lower:
            return "TIMEOUT"
        elif "missing" in error_msg_lower or "required parameter" in error_msg_lower:
            return "MISSING_PARAMETER"
        elif "invalid" in error_msg_lower:
            return "INVALID_PARAMETER"
        elif "connection" in error_msg_lower:
            return "CONNECTION_ERROR"
        elif "empty" in error_msg_lower or "no results" in error_msg_lower:
            return "EMPTY_RESPONSE"
        else:
            return "OTHER"
    
    def _extract_http_status(self, error_msg: str) -> Optional[str]:
        """Extract HTTP status code from error message"""
        import re
        match = re.search(r'HTTP (\d{3})', error_msg, re.IGNORECASE)
        if match:
            return f"HTTP_{match.group(1)}"
        return None


# Global logger instance - can be initialized once and reused
_global_logger: Optional[MCPCallLogger] = None


def initialize_global_logger(output_dir: Optional[str] = None, enabled: bool = True) -> MCPCallLogger:
    """Initialize the global MCP call logger"""
    global _global_logger
    _global_logger = MCPCallLogger(output_dir=output_dir, enabled=enabled)
    return _global_logger


def get_global_logger() -> Optional[MCPCallLogger]:
    """Get the global MCP call logger instance"""
    return _global_logger


def log_mcp_call(api_name: str,
                 parameters: Dict[str, Any],
                 metadata: Dict[str, Any],
                 response: Dict[str, Any],
                 query_id: Optional[str] = None,
                 subtask_id: Optional[int] = None,
                 platform: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to log MCP call using global logger
    
    Returns empty dict if logger not initialized
    """
    if _global_logger:
        return _global_logger.log_call(api_name, parameters, metadata, response, 
                                       query_id, subtask_id, platform)
    return {}
