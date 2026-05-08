#!/usr/bin/env python3
"""
Convert new TMDB evaluation format to legacy flattened format.

New format (from eval/tmdb main_tmdb.py):
  - step_wise_eval_results.json with nested structure
  - Each entry has main_results.step_wise_results[] with runs[]

Legacy format (expected by prepare_pl_dataset.py):
  - Flattened structure with records[] array
  - Each record represents one (API, subtask) pair with aggregated metrics

Usage:
    python convert_tmdb_eval_format.py \
        --input /path/to/step_wise_eval_results.json \
        --output /path/to/desc_eval_res_step_wise_eval_flattened.json \
        --tool-desc-file /path/to/tool_descriptions.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def convert_new_to_legacy_format(
    new_data: List[Dict[str, Any]],
    tool_desc_file: str = None
) -> Dict[str, Any]:
    """
    Convert new TMDB evaluation format to legacy flattened format.
    
    Args:
        new_data: List of query evaluation results from step_wise_eval_results.json
        tool_desc_file: Path to tool description file (for metadata)
        
    Returns:
        Dictionary in legacy flattened format with metadata and records
    """
    
    records = []
    
    for query_idx, query_data in enumerate(new_data):
        try:
            # Extract main results
            main_results = query_data.get('main_results', {})
            step_wise_results = main_results.get('step_wise_results', [])
            
            # Get query-level info
            # Note: query_text is not always available in new format, so we'll use
            # the first subtask's input as a proxy
            query_text = query_data.get('query', '')
            query_id = query_data.get('query_id', f'query_{query_idx}')
            
            # If no query_text, use first subtask's input as proxy
            if not query_text and step_wise_results:
                first_subtask = step_wise_results[0]
                first_scenario = first_subtask.get('scenario', {})
                query_text = first_scenario.get('subtask_input', '')
            
            # Build mapping from api_name to server_name (tool_name in api_list)
            # In StableToolBench format, api_list[].tool_name is actually the server name
            # and api_list[].api_name is the actual tool/API name
            api_list = query_data['query_data']['api_list']

            api_name_to_server = {}
            for api in api_list:
                api_name = api['api_name']
                server_name = api['tool_name']
                if api_name and server_name:
                    api_name_to_server[api_name] = server_name
            
            # Process each subtask
            for subtask_result in step_wise_results:
                try:
                    subtask_id = subtask_result.get('subtask_id', 0)
                    scenario = subtask_result['scenario']
                    expected_golden_api = subtask_result['expected_golden_api']
                    runs = subtask_result['runs']

                    if expected_golden_api == "" or not runs:
                        continue
                    
                    # Extract subtask info from scenario
                    subtask_text = scenario['subtask_input']
                    dependencies = scenario['dependencies']
                    selected_api_name = scenario['selected_api_name']
                    selected_description = scenario['selected_description']
                    selected_metadata = scenario['selected_metadata']
                    
                    # Look up server names for golden API and selected API
                    server_name_golden_api = api_name_to_server[expected_golden_api]
                    server_name_selected_api = api_name_to_server[selected_api_name]
                    
                    # Build previous context from dependencies (simplified)
                    previous_context = []
                    for dep_id in dependencies:
                        previous_context.append({
                            'subtask_id': dep_id,
                            'subtask_input': f"Subtask {dep_id}",
                            'subtask_output': f"Output of subtask {dep_id}",
                            'api_used': ''
                        })
                    
                    # Aggregate metrics across runs
                    api_selection_accuracies = []
                    api_success_rates = []
                    subtask_success_rates = []
                    run_details = []
                    
                    for run_idx, run_data in enumerate(runs):
                        # Extract run metrics
                        # Derive api_selection_correct from selected_api vs expected_golden_api
                        selected_api = run_data['selected_api']
                        expected_api = run_data['expected_golden_api']
                        api_selection_correct = (selected_api == expected_api) if selected_api and expected_api else False

                        api_success = run_data['api_success']

                        # For subtask_success, use api_success as proxy if subtask_success doesn't exist
                        # This makes sense because if the API call succeeded, the subtask likely succeeded
                        # subtask_success = run_data['subtask_success']
                        
                        # Convert boolean to float accuracy
                        api_selection_accuracy = 1.0 if api_selection_correct else 0.0
                        api_success_rate = 1.0 if api_success else 0.0
                        # subtask_success_rate = 1.0 if subtask_success else 0.0
                        
                        api_selection_accuracies.append(api_selection_accuracy)
                        api_success_rates.append(api_success_rate)
                        # subtask_success_rates.append(subtask_success_rate)
                        
                        # Build run detail
                        run_detail = {
                            'run_index': run_idx + 1,
                            'selected_api': selected_api,
                            'api_selection_correct': api_selection_correct,
                            'api_selection_accuracy': api_selection_accuracy,
                            'api_success': api_success,
                            # 'subtask_success': subtask_success,
                            'api_selection_reasoning': run_data.get('api_selection_reasoning', run_data.get('reasoning', ''))
                        }

                        # Add parameter quality evaluation if present
                        parameter_quality = run_data.get('parameter_quality_evaluation', {})
                        if parameter_quality:
                            run_detail['parameter_quality_evaluation'] = parameter_quality
                        run_details.append(run_detail)
                    
                    # Calculate averages
                    avg_api_selection_accuracy = sum(api_selection_accuracies) / len(api_selection_accuracies) if api_selection_accuracies else 0.0
                    avg_api_success_rate = sum(api_success_rates) / len(api_success_rates) if api_success_rates else 0.0
                    avg_subtask_success_rate = sum(subtask_success_rates) / len(subtask_success_rates) if subtask_success_rates else 0.0

                    # print(f"DEBUG: scenario keys = {scenario.keys()}")
                    # print(f"DEBUG: selected_metadata = {selected_metadata}")
                    # print(f"DEBUG: selected_metadata type = {type(selected_metadata)}")
                    
                    # Build legacy record
                    record = {
                        # Query and subtask info
                        'query_text': query_text,
                        'query_id': query_id,
                        'subtask_id': subtask_id,
                        'subtask_text': subtask_text,
                        'subtask_dependencies': dependencies,
                        
                        # API info
                        'selected_api_name': selected_api_name,
                        'expected_golden_api': expected_golden_api,

                        'server_name_selected_api': server_name_selected_api,
                        'server_name_golden_api': server_name_golden_api,

                        'api_description': selected_description,
                        
                        # Metadata
                        'description_index': 0, # TODO: get the real description index
                        'task_decomposition_source': selected_metadata.get('improvement_source', 'tmdb_eval'),
                        'task_decomposition_file_path': tool_desc_file or '',
                        'task_decomposition_prompt_version': 'v3',
                        # 'task_decomposition_timestamp': datetime.fromtimestamp(selected_metadata.get('improvement_timestamp', datetime.now().timestamp())).isoformat(),
                        'parameter_generation_prompt_version': 'v2',
                        
                        # Context
                        'previous_context': previous_context,
                        'context_length': len(previous_context),
                        'pmpt_without_api_res': True,
                        
                        # Evaluation settings
                        'model_name': 'gpt-41-2025-04-14',
                        'temperature': 1.0,
                        'seed': 42,
                        'runs_per_scenario': len(runs),
                        'description_index_config': 0,
                        
                        # Aggregated metrics
                        'avg_api_selection_accuracy': avg_api_selection_accuracy,
                        'avg_api_success_rate': avg_api_success_rate,
                        'avg_subtask_success_rate': avg_subtask_success_rate,
                        
                        # Per-run metrics
                        'api_selection_accuracies': api_selection_accuracies,
                        'api_success_rates': api_success_rates,
                        'subtask_success_rates': subtask_success_rates,
                        
                        # Run details
                        'run_details': run_details
                    }
                    
                    records.append(record)
                    
                except Exception as e:
                    raise ValueError(f"Error processing subtask {subtask_id} in query {query_idx}: {e}")
                    
        except Exception as e:
            raise ValueError(f"Error processing query {query_idx}: {e}")
    
    # Build output with metadata
    output = {
        'metadata': {
            'description': 'Flattened (API, subtask) pair data from step-wise evaluation',
            'total_records': len(records),
            'evaluation_type': 'step_wise_evaluation',
            'generated_at': datetime.now().isoformat(),
            'source_file': 'step_wise_eval_results.json',
            'conversion_note': 'Converted from new TMDB evaluation format to legacy format',
            'tool_desc_file': tool_desc_file or ''
        },
        'records': records
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Convert new TMDB evaluation format to legacy flattened format'
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input file: step_wise_eval_results.json (new format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file: desc_eval_res_step_wise_eval_flattened.json (legacy format)'
    )
    
    parser.add_argument(
        '--tool-desc-file',
        default=None,
        help='Path to tool description file (for metadata)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Load new format data
    print(f"📂 Loading new format data from: {args.input}")
    with open(input_path, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    
    print(f"   Found {len(new_data)} queries")
    
    # Convert to legacy format
    print(f"🔄 Converting to legacy flattened format...")
    legacy_data = convert_new_to_legacy_format(new_data, args.tool_desc_file)
    
    print(f"   ✅ Converted to {legacy_data['metadata']['total_records']} records")
    
    # Save legacy format data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving legacy format data to: {args.output}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(legacy_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Conversion complete!")
    print(f"\n📊 Summary:")
    print(f"   Input queries: {len(new_data)}")
    print(f"   Output records: {legacy_data['metadata']['total_records']}")
    print(f"   Output file: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())

