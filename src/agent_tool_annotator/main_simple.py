from smolagents import ToolCallingAgent, DuckDuckGoSearchTool
from agent_tool_annotator.agent.models import get_model

model = get_model(model_id="vllm:Qwen/Qwen2.5-7B-Instruct", tool_call_by_prompt=False)
agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=model)
agent.run("What is the weather in Tokyo?")