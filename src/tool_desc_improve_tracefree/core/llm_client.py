"""
Thin wrapper around the OpenAI client for LLM calls used by improvement strategies.
"""

import os
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


def create_openai_client(api_key: str | None = None, max_retries: int = 4) -> OpenAI:
    """Create an OpenAI client, resolving the API key from the parameter or environment."""
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError("OpenAI API key required. Pass --openai_api_key or set OPENAI_API_KEY env var.")
    return OpenAI(api_key=resolved_key, max_retries=max_retries)


def llm_call(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """
    Make a chat completion call and return the response content as a string.

    Args:
        client: An initialised OpenAI client.
        messages: Chat messages in OpenAI format.
        model: Model name (``openai:`` or ``vllm:`` prefixes are stripped automatically).
        temperature: Sampling temperature.
        max_tokens: Maximum completion tokens.

    Returns:
        The assistant message content (stripped).

    Raises:
        RuntimeError: When the API returns an empty or invalid response.
    """
    # Strip known prefixes (e.g. "openai:gpt-4.1-..." -> "gpt-4.1-...")
    clean_model = model.split(":", 1)[1] if ":" in model else model

    response = client.chat.completions.create(
        model=clean_model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )

    if not response or not response.choices:
        raise RuntimeError("LLM call returned an empty response")

    content = response.choices[0].message.content
    return content.strip() if content else ""
