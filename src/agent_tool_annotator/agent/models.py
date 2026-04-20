import os
from typing import Any

from smolagents.models import OpenAIServerModel
from smolagents import VLLMModel


def get_model(model_id: str, tool_call_by_prompt: bool = False, api_key: str | None = None) -> OpenAIServerModel:
    if model_id.startswith("openai:"):
        model_name = model_id.split(":", 1)[1]
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OpenAI API key required. Pass --openai_api_key or set OPENAI_API_KEY env var.")
        return OpenAIModel(model_id=model_name, api_key=resolved_key, tool_call_by_prompt=tool_call_by_prompt)
    elif model_id.startswith("vllm:"):
        return VLLMModel(model_id=model_id.split(":", 1)[1])
    else:
        raise ValueError(f"Unknown model id prefix: {model_id}. Use 'openai:<model>' or 'vllm:<model>'.")


class OpenAIModel(OpenAIServerModel):
    def __init__(
        self,
        model_id: str,
        api_key: str,
        tool_call_by_prompt: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_id=model_id, api_key=api_key, **kwargs)
        self.tool_call_by_prompt = tool_call_by_prompt

    def _prepare_completion_kwargs(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        completion_kwargs = super()._prepare_completion_kwargs(**kwargs)
        if self.tool_call_by_prompt:
            completion_kwargs.pop("tools", None)
            completion_kwargs.pop("tool_choice", None)
        return completion_kwargs
