from dotenv import load_dotenv
from phoenix.client import Client
import pandas as pd
import os
import json
import time
load_dotenv()
from typing import Any, Dict, List, Optional, Set, Tuple
from smolagents import Tool
from phoenix.client.types.spans import SpanQuery

def json_dump_columns(df: pd.DataFrame):
    for col in df.columns:
        if isinstance(df[col].iloc[0], dict) or isinstance(df[col].iloc[0], list):
            df[col] = df[col].apply(lambda x: json.dumps(x))
    return df

class PhoenixUtils:
    def __init__(self):
        self.client = Client()

    def get_client(self):
        return self.client

    def list_projects(self):
        return self.client.projects.list()

    # def get_spans_dataframe(self, project_name: str, timeout: int = 1000):
    #     return self.client.spans.get_spans_dataframe(project_name=project_name, timeout=timeout)

    def get_annotations_dataframe(self, project_name: str, timeout: int = 1000, df: pd.DataFrame = None):
        raise NotImplementedError("get_annotations_dataframe is not implemented")
        # if df is None:
        #     df = self.get_spans_dataframe(project_name=project_name, timeout=timeout)
        # return self.client.spans.get_span_annotations_dataframe(spans_dataframe=df, timeout=timeout, 
        # project_identifier=project_name)

    def save_project(self, save_path: str, project_name: str, timeout: int = 1000):
        df = self.get_spans_dataframe(project_name=project_name, timeout=timeout)
        df_annotation = self.get_annotations_dataframe(project_name=project_name, timeout=timeout, df=df[df["span_kind"]=="AGENT"])
        os.makedirs(save_path, exist_ok=True)
        df_annotation = json_dump_columns(df_annotation)
        df = json_dump_columns(df)
        df_annotation.to_csv(f"{save_path}/annotations.csv")
        df.to_csv(f"{save_path}/spans.csv")
        print(f"Project {project_name} saved to {save_path}")
        return df, df_annotation

    def _safe_json_loads(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def _extract_user_query_from_input(self, raw_input: Any) -> Optional[str]:
        data = self._safe_json_loads(raw_input)
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            # common keys
            for key in ("query", "prompt", "task", "question"):
                if isinstance(data.get(key), str) and data.get(key):
                    return data[key]
            # messages style
            messages = data.get("messages")
            if isinstance(messages, list):
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("role") in ("user", "system"):
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            return content
                        if isinstance(content, list):
                            # OpenAI style content = [{type: 'text', text: '...'}]
                            for part in content:
                                if isinstance(part, dict) and isinstance(part.get("text"), str):
                                    return part["text"]
        return None

    def _extract_text_from_output(self, raw_output: Any) -> Optional[str]:
        out = self._safe_json_loads(raw_output)
        if out is None:
            return None
        if isinstance(out, str):
            return out if out.strip() else None
        if isinstance(out, dict):
            # OpenAI-like
            choices = out.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0] or {}
                # chat
                message = choice.get("message") or {}
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                # completions
                text = choice.get("text")
                if isinstance(text, str) and text.strip():
                    return text
            # direct content
            content = out.get("content")
            if isinstance(content, str) and content.strip():
                return content
        if isinstance(out, list):
            parts = [p for p in out if isinstance(p, str) and p.strip()]
            if parts:
                return "\n".join(parts)
        return None

    def get_trace_id_from_root_span_id(self, root_span_id: str, project_name: str, timeout: int = 1000) -> Optional[str]:
        # first get the root span
        # retry 10 seconds until fetch the root span
        for _ in range(10):
            spans_df = self.client.spans.get_spans_dataframe(project_name=project_name, timeout=timeout)
            if spans_df is None or spans_df.empty:
                raise RuntimeError("No spans available for project to build ToolEval payload.")
            if root_span_id in spans_df["context.span_id"].values:
                break
            time.sleep(1)
        else:
            raise RuntimeError("No root span found for project to build ToolEval payload.")

        # get the trace id
        trace_id = spans_df[spans_df["context.span_id"] == root_span_id].iloc[0].get("context.trace_id")
        return trace_id

    def collect_trace_df(self, trace_id: str, project_name: str, timeout: int = 1000) -> pd.DataFrame:
        # get the trace df
        trace_df = self.client.spans.get_spans_dataframe(project_name=project_name, timeout=timeout, query=SpanQuery().where(f"context.trace_id == '{trace_id}'"))
        return trace_df
        

    def build_tooleval_payload(
        self,
        project_name: str,
        root_span_id: str,
        tools: List[Tool],
        timeout: int = 1000,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build ToolEval (task_description, answer) payload from Phoenix spans given a root span id.
        """
        trace_id = self.get_trace_id_from_root_span_id(root_span_id, project_name, timeout)
        trace_df = self.collect_trace_df(trace_id, project_name, timeout)

        # Sort by start time if available to maintain order
        time_cols = [c for c in ("start_time", "start_timestamp", "start") if c in trace_df.columns]
        if time_cols:
            trace_df = trace_df.sort_values(by=time_cols[0], ascending=True)

        # Identify span kind column, if present
        span_kind_col = "span_kind" if "span_kind" in trace_df.columns else None

        nodes: List[Dict[str, Any]] = []
        last_assistant_text: Optional[str] = None
        finish_candidate: Optional[str] = None

        # Helper to get a column safely
        def col(row: pd.Series, name: str) -> Any:
            return row[name] if name in trace_df.columns else None

        for _, row in trace_df.iterrows():
            kind = col(row, span_kind_col) if span_kind_col else None
            raw_output = col(row, "attributes.output.value")

            # Tool spans
            if kind == "TOOL":
                """
                {'name': 'APISpecTool', 'span_kind': 'TOOL', 'parent_id': 'ccd5c16237a1baba', 'start_time': Timestamp('2025-11-18 23:33:09.780627+0000', tz='UTC'), 'end_time': Timestamp('2025-11-18 23:33:11.788006+0000', tz='UTC'), 'status_code': 'OK', 'status_message': '', 'events': [], 'context.span_id': '75e2268e72c464f2', 'context.trace_id': '6606142c1157749c4d6af247cb7f0584', 'attributes.openinference.span.kind': 'TOOL', 'attributes.input.value': '{"args": [], "sanitize_inputs_outputs": true, "kwargs": {"colisId": "CA107308006SI"}}', 'attributes.tool.description': "L'état courant (ie. le dernier état du colis).", 'attributes.tool.name': 'latest', 'attributes.tool.parameters': {'colisId': {...}}, 'attributes.output.value': '{"error": "Function executing from toolenv.tools.Logistics.suivi_colis.api import lat...yword argument \'colisId\'", "response": ""}', 'attributes.llm.token_count.completion': nan, 'attributes.llm.invocation_parameters': None, 'attributes.llm.token_count.prompt': nan, 'attributes.llm.tools': None, 'attributes.llm.output_messages': None, 'attributes.llm.model_name': None, 'attributes.input.mime_type': None, 'attributes.output.mime_type': 'application/json', 'attributes.llm.token_count.total': nan}
                """
                tool_name = col(row, "attributes.tool.name")
                args_obj = self._safe_json_loads(col(row, "attributes.input.value"))
                resp_obj = self._safe_json_loads(raw_output)
                msg_obj = {
                    "name": tool_name,
                    "arguments": args_obj,
                    "response": resp_obj,
                }
                # Track Finish final answer if present
                if tool_name == "final_answer":
                    # Prefer arguments as final; fallback to response text
                    if isinstance(args_obj, dict):
                        # pick first string value
                        for v in args_obj.values():
                            if isinstance(v, dict) and 'answer' in v:
                                finish_candidate = v['answer']
                                break
                    if not finish_candidate:
                        text = self._extract_text_from_output(resp_obj)
                        if text:
                            finish_candidate = text
                nodes.append({
                    "role": "tool",
                    "message": str(msg_obj),
                })
                continue

            # LLM assistant spans
            if kind == "LLM":
                text = self._extract_text_from_output(raw_output)
                if text and text.strip():
                    last_assistant_text = text.strip()
                    nodes.append({
                        "role": "assistant",
                        "message": last_assistant_text,
                    })
                continue

        # Derive query from root span input or first LLM input
        # Try root span row
        root_rows = trace_df[trace_df["context.span_id"] == root_span_id]
        query: Optional[str] = None
        if root_rows.empty:
            query = ""
        else:
            query = self._extract_user_query_from_input(root_rows.iloc[0].get("attributes.input.value"))


        final_answer = last_assistant_text or finish_candidate or ""
        total_steps = sum(1 for n in nodes if n.get("role") == "tool")

        available_tools = [format_tool_for_tooleval(tool) for tool in tools]
        task_description = {
            "query": query,
            "available_tools": available_tools,
        }
        answer = {
            "final_answer": final_answer,
            "total_steps": total_steps,
            "answer_details": nodes,  # list form is accepted by the judge
        }
        return task_description, answer

    def extract_tool_calls(
        self,
        project_name: str,
        root_span_id: str,
        timeout: int = 1000,
    ) -> Tuple[Set[str], int, int]:
        """
        Extract tool invocation names and success stats from Phoenix spans.
        Returns: (used_tool_names, success_count, total_count)
        Excludes the 'final_answer' tool from counts.
        """
        trace_id = self.get_trace_id_from_root_span_id(root_span_id, project_name, timeout)
        trace_df = self.collect_trace_df(trace_id, project_name, timeout)
        span_kind_col = "span_kind" if "span_kind" in trace_df.columns else None

        used_tool_names: Set[str] = set()
        success_count = 0
        total_count = 0

        def col(row: pd.Series, name: str) -> Any:
            return row[name] if name in trace_df.columns else None

        for _, row in trace_df.iterrows():
            kind = col(row, span_kind_col) if span_kind_col else None
            if kind != "TOOL":
                continue
            tool_name = col(row, "attributes.tool.name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            # exclude final answer tool from metrics
            if tool_name == "final_answer":
                continue
            used_tool_names.add(tool_name)

            output_obj = self._safe_json_loads(col(row, "attributes.output.value"))
            # success criterion: not bool(error)
            error_val = ""
            if isinstance(output_obj, dict):
                error_val = output_obj.get("error", "")
            total_count += 1
            if not bool(error_val):
                success_count += 1

        return used_tool_names, success_count, total_count

    def extract_query_from_project(self, project_name: str, timeout: int = 1000) -> Dict[str, str]:
        df = self.client.spans.get_spans_dataframe(project_name=project_name, timeout=timeout, root_spans_only=True, limit=2000)
        if df.shape[0] == 0:
            return {}
        df_agent = df[df["span_kind"]=="AGENT"][["attributes.input.value", "context.trace_id"]].to_dict(orient="records")
        # return a dict of query --> span_id
        return {self._extract_user_query_from_input(each_dict["attributes.input.value"]): each_dict["context.trace_id"] for each_dict in df_agent}


def format_tool_for_tooleval(tool: Tool) -> Dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputs,
    }