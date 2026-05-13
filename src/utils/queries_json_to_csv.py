"""Convert a StableToolBench ``combined_queries.json`` file into the CSV format
consumed by ``agent_tool_optimizer/inference_main.py``.

Input shape (list of query objects):
    [
        {
            "api_list": [
                {
                    "category_name": "...",
                    "tool_name": "...",
                    "api_name": "...",
                    "api_description": "...",
                    "required_parameters": [{"name": "...", "type": "...", "description": "...", "default": "..."}, ...],
                    "optional_parameters": [...],
                    "method": "GET"
                },
                ...
            ],
            "query": "...",
            "relevant APIs": [...],
            "query_id": 147
        },
        ...
    ]

Output columns: ``tool_name``, ``parameters``, ``original_description`` — matching
``data/inference/tool_descs.example.csv`` and the schema read by
``PromptsBuilder.build_dataset_from_csv``.

Usage:
    python src/utils/queries_json_to_csv.py \\
        --input data/StableToolBench/tools_synthetic_queries/ToolUse_smithery_198_3tool_1775806399/combined_queries.json \\
        --output data/inference/smithery_198_3tool.csv
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _build_param_entry(raw: dict[str, Any], required: bool) -> tuple[str, dict[str, Any]]:
    """Translate one StableToolBench parameter into the inference schema entry."""
    name = raw.get("name") or raw.get("parameter_name") or ""
    entry: dict[str, Any] = {
        "type": raw.get("type", "str"),
        "required": required,
        "description": raw.get("description", ""),
    }
    if "default" in raw and raw["default"] not in (None, ""):
        entry["default"] = raw["default"]
    return name, entry


def build_parameter_schema(api: dict[str, Any]) -> dict[str, Any]:
    """Construct the ``{"parameters": {...}, "metadata": {...}}`` dict expected by the prompt template."""
    params: dict[str, Any] = {}
    for raw in api.get("required_parameters") or []:
        name, entry = _build_param_entry(raw, required=True)
        if name:
            params[name] = entry
    for raw in api.get("optional_parameters") or []:
        name, entry = _build_param_entry(raw, required=False)
        if name:
            params[name] = entry

    return {
        "parameters": params,
        "metadata": {
            "endpoint": api.get("api_name", ""),
            "method": api.get("method", ""),
            "category": api.get("category_name", ""),
        },
    }


def collect_unique_apis(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten every ``api_list`` entry and deduplicate on (tool_name, api_name)."""
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for entry in queries:
        for api in entry.get("api_list") or []:
            key = (api.get("tool_name", ""), api.get("api_name", ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(api)
    return unique


def convert(input_path: Path, output_path: Path) -> int:
    """Read JSON at ``input_path`` and write the CSV at ``output_path``. Returns row count."""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {input_path}, got {type(data).__name__}")

    apis = collect_unique_apis(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tool_name", "parameters", "original_description"])
        writer.writeheader()
        for api in apis:
            schema = build_parameter_schema(api)
            writer.writerow(
                {
                    "tool_name": api.get("tool_name", ""),
                    "parameters": json.dumps(schema, indent=4),
                    "original_description": api.get("api_description", ""),
                }
            )

    return len(apis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Path to combined_queries.json")
    parser.add_argument("--output", required=True, type=Path, help="Path to write the CSV file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    n = convert(args.input, args.output)
    log.info("Wrote %d unique APIs to %s", n, args.output)


if __name__ == "__main__":
    main()
