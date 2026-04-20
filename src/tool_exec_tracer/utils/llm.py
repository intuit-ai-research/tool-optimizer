import os
import sys
import json
from openai import OpenAI

import time
import uuid
from datetime import datetime

import random

import threading
_LLM_LOG_WRITE_LOCK = threading.Lock()

# Default max chars for truncation (can be overridden via environment variable)
DEFAULT_MAX_CHARS = 2000

def get_max_chars():
    """Get the maximum character limit for log truncation from environment or default."""
    return int(os.environ.get("LLM_LOG_TRUNCATE_CHARS", DEFAULT_MAX_CHARS))

# Resolve OpenAI API key from environment
_OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")

def set_openai_api_key(key: str):
    """Set the OpenAI API key at runtime (called from CLI argument parsing)."""
    global _OPENAI_API_KEY
    _OPENAI_API_KEY = key

def _get_openai_api_key() -> str:
    """Return the resolved API key, preferring the runtime override over the env var."""
    key = _OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required. Pass --openai_api_key or set OPENAI_API_KEY env var.")
    return key

def llm_call(messages, model='gpt-4.1-2025-04-14',
    temperature=0.,
    max_tokens=32768,
    tools=None,
    tool_choice=None,
    logprobs=False,
    max_retries=4,
    base_delay=6.0, seed: int | None = None, step_name="", **kwargs):

    api_key = _get_openai_api_key()

    use_vllm = model.startswith("vllm:")
    if use_vllm:
        client = OpenAI(
            api_key="placeholder",
            base_url="http://localhost:8000/v1"
        )
    else:
        client = OpenAI(api_key=api_key)

    # Prepare API call parameters
    api_params = {
        "model": model.replace("vllm:", "") if use_vllm else model,
        "messages": messages,
        **kwargs
    }

    if not use_vllm:
        api_params["max_completion_tokens"] = max_tokens

    api_params["temperature"] = temperature

    # If backend supports deterministic seeding, pass it through in the params payload
    if seed is not None:
        try:
            api_params["seed"] = int(seed)
        except Exception:
            pass

    # Add tools and tool_choice if provided
    if tools:
        api_params["tools"] = tools
        if tool_choice:
            api_params["tool_choice"] = tool_choice
        else:
            api_params["tool_choice"] = "required"  # Force exactly one function call
    
    # Resolve logging directory from environment
    llm_log_dir = os.environ.get("LLM_LOG_DIR")
    llm_logs_subdir = None
    log_file_path = None
    if llm_log_dir:
        try:
            llm_logs_subdir = os.path.join(llm_log_dir, "llm_logs")
            os.makedirs(llm_logs_subdir, exist_ok=True)
            log_file_path = os.path.join(llm_logs_subdir, f"llm_calls_{step_name}.json")
        except Exception as _e:
            print(f"Warning: Unable to create llm_logs directory: {_e}")
            llm_logs_subdir = None
            log_file_path = None

    def _truncate_content(content, max_chars=None):
        """Truncate content to specified number of characters."""
        if max_chars is None:
            max_chars = get_max_chars()
        if content is None:
            return None
        if isinstance(content, str):
            return content[:max_chars] + "..." if len(content) > max_chars else content
        elif isinstance(content, list):
            # Handle list of content items (e.g., for multi-modal messages)
            truncated_list = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    truncated_item = item.copy()
                    text = item['text']
                    truncated_item['text'] = text[:max_chars] + "..." if len(text) > max_chars else text
                    truncated_list.append(truncated_item)
                else:
                    truncated_list.append(item)
            return truncated_list
        else:
            # For other types, convert to string and truncate
            content_str = str(content)
            return content_str[:max_chars] + "..." if len(content_str) > max_chars else content_str

    def _append_llm_io(request_messages, request_parameters: dict, response_obj=None, error: str | None = None):
        
        if not log_file_path:
            return
        call_id = uuid.uuid4().hex
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Truncate request messages content
        truncated_request_messages = []
        for msg in request_messages:
            truncated_msg = {}
            if isinstance(msg, dict):
                truncated_msg = msg.copy()
                if 'content' in msg:
                    truncated_msg['content'] = _truncate_content(msg['content'])
            else:
                # Handle message objects
                truncated_msg = {
                    'role': getattr(msg, 'role', 'unknown'),
                    'content': _truncate_content(getattr(msg, 'content', None))
                }
            truncated_request_messages.append(truncated_msg)
        
        entry = {
            "id": call_id,
            "timestamp": timestamp,
            "model": model,
            "request": {
                "messages": truncated_request_messages,
                "parameters": request_parameters
            }
        }
        if error is None and response_obj is not None:
            try:
                entry["response"] = {
                    "choices": [
                        {
                            "message": {
                                "role": getattr(choice.message, 'role', 'assistant'),
                                "content": _truncate_content(getattr(choice.message, 'content', None))
                            }
                        } for choice in getattr(response_obj, 'choices', [])
                    ]
                }
            except Exception:
                entry["response"] = {"repr": str(response_obj)}
        else:
            entry["error"] = error if error is not None else "Unknown error"

        # Read existing log file
        with _LLM_LOG_WRITE_LOCK:
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        data = {"calls": []}
            else:
                data = {"calls": []}

            data["calls"].append(entry)

            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    # Retry logic for handling rate limiting and other transient errors
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**api_params)

            # print("About to make API call...", flush=True)
            # print(response, flush=True)
            
            # Check if we got a valid response
            if response and hasattr(response, 'choices') and response.choices:
                # Build sanitized parameters for logging
                sanitized_params = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    # "logprobs": logprobs
                }
                if seed is not None:
                    sanitized_params["seed"] = int(seed)
                sanitized_params.update(kwargs)
                _append_llm_io(messages, sanitized_params, response_obj=response, error=None)
                return response
            else:
                # Invalid response structure - this is retryable
                if attempt < max_retries - 1:
                    delay = base_delay * (1.2 ** attempt)  # Light backoff for invalid responses
                    print(f"⚠️  Invalid response structure (attempt {attempt + 1}/{max_retries}). Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Invalid response structure persisted after {max_retries} attempts")
                    sanitized_params = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "tools": tools,
                        "tool_choice": tool_choice,
                        # "logprobs": logprobs
                    }
                    if seed is not None:
                        sanitized_params["seed"] = int(seed)
                    sanitized_params.update(kwargs)
                    _append_llm_io(messages, sanitized_params, response_obj=None, error="Invalid response structure")
                    return None
                
        except Exception as e:
            # print(response, flush=True)
            error_msg = str(e)
            print(f"Error in llm_call (attempt {attempt + 1}/{max_retries}): {error_msg}")
            sanitized_params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "tools": tools,
                "tool_choice": tool_choice,
                # "logprobs": logprobs
            }
            if seed is not None:
                sanitized_params["seed"] = int(seed)
            sanitized_params.update(kwargs)
            _append_llm_io(messages, sanitized_params, response_obj=None, error=error_msg)
            
            # Check for rate limiting (429 Too Many Requests)
            if "429" in error_msg or "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    # Exponential backoff with jitter for rate limiting
                    delay = base_delay * (2 ** attempt) + (0.1 * attempt)  # Add small jitter
                    print(f"⚠️  Rate limit hit. Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} attempts")
                    return None
            
            # Check for other retryable errors (timeouts, connection issues)
            elif any(error_type in error_msg.lower() for error_type in [
                "timeout", "connection", "network", "temporary", "service unavailable", "502", "503", "504"
            ]):
                if attempt < max_retries - 1:
                    delay = base_delay * (1.5 ** attempt)  # Lighter backoff for network issues
                    print(f"⚠️  Network/timeout error. Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Network error persisted after {max_retries} attempts")
                    return None
            
            # Check for server errors (500 series)
            elif any(error_code in error_msg for error_code in ["500", "501", "502", "503", "504"]):
                if attempt < max_retries - 1:
                    delay = base_delay * (1.3 ** attempt)  # Backoff for server errors
                    print(f"⚠️  Server error. Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Server error persisted after {max_retries} attempts")
                    return None
            
            # Check for model-specific errors that might be retryable
            elif any(error_type in error_msg.lower() for error_type in [
                "model overloaded", "model unavailable", "service overloaded", "try again"
            ]):
                if attempt < max_retries - 1:
                    delay = base_delay * (1.8 ** attempt)  # Backoff for model issues
                    print(f"⚠️  Model/service issue. Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Model/service issue persisted after {max_retries} attempts")
                    return None
            
            # Check for 401 Unauthorized specifically - terminate process if persists
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                if attempt < max_retries - 1:
                    delay = base_delay * (1.1 ** attempt)  # Light backoff for auth issues
                    print(f"⚠️  401 Unauthorized error. Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ 401 Unauthorized error persisted after {max_retries} attempts. Terminating process.")
                    print(f"   Error: {error_msg}")
                    sys.exit(1)  # Kill the process with exit code 1
            
            # Check for other authentication issues (might be temporary) - don't terminate
            elif any(auth_error in error_msg.lower() for auth_error in [
                "authentication", "forbidden", "invalid token"
            ]):
                if attempt < max_retries - 1:
                    delay = base_delay * (1.1 ** attempt)  # Light backoff for auth issues
                    print(f"⚠️  Authentication issue. Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Authentication issue persisted after {max_retries} attempts")
                    return None
            # For truly non-retryable errors (client errors like invalid input)
            elif any(error_code in error_msg for error_code in ["400", "404", "422"]) and not any(retryable in error_msg.lower() for retryable in ["timeout", "temporary"]):
                print(f"❌ Non-retryable client error: {error_msg}")
                return None
            
            # For any other unexpected errors, try to retry (defensive approach)
            else:
                if attempt < max_retries - 1:
                    delay = base_delay * (1.4 ** attempt)  # Moderate backoff for unknown errors
                    print(f"⚠️  Unexpected error, retrying. Waiting {delay:.2f} seconds before retry...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"❌ Unexpected error persisted after {max_retries} attempts: {error_msg}")
                    return None
    
    # All retries exhausted
    print(f"❌ All {max_retries} retry attempts exhausted")
    return None


def llm_embedding(messages, model='text-embedding-3-small', **kwargs):
    try:
        api_key = _get_openai_api_key()
        client = OpenAI(api_key=api_key)

        api_params = {
            "model": model,
            "input": messages,
            "encoding_format": "float",
        }

        response = client.embeddings.create(**api_params)

        if response and hasattr(response, 'data') and response.data:
            return response

    except Exception as e:
        print(f"Error in llm_embedding: {e}")
        print(f"Messages: {messages}")
        return None

    return None