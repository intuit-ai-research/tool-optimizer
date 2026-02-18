#!/bin/bash

echo "Starting the application..."

export PYTHONPATH=${WORKSPACE_DIR}/src
python src/agent_tool_optimizer/inference_main.py --model_name /opt/ml/model