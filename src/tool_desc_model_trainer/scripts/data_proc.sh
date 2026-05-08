#!/bin/bash

# Script translated from .vscode/launch.json debug configurations

# Get the workspace root directory (parent of DRAFT/policy_learn/scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_FOLDER="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Set PYTHONPATH
export PYTHONPATH="${WORKSPACE_FOLDER}:${WORKSPACE_FOLDER}/DRAFT/policy_learn"

DATE=$(date +%Y%m%d_%H%M%S)

# Usage: ./data_proc.sh [case]
# case: 1 = prepare_training_data (split2b_no_eval)
#       2 = prepare_inference_data (split2a_no_eval_inference)
#       3 = original STB case (legacy)

CASE="${1:-1}"
SCALING_EXPERIMENT="${2:-False}"

# Common defaults
# BASELINE_DESC_FILES="/home/sagemaker-user/user-default-efs/FunctionWrapper_STBd1/eval/StableToolBench/"

if [ "${SCALING_EXPERIMENT}" = True ]; then
    # scaling experiment test data: all tools' D0 except split2b
    BASELINE_DESC_FILES="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/StableToolBench_minus_split2b_D0"
else
    # original STB case: all tools in test queries
    BASELINE_DESC_FILES="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/StableToolBench/"
fi

NUM_QUERY_EXAMPLES="3"
MIN_QUERIES_PER_TOOL="1"
TRAIN_QUERY_RATIO="0.8"
MAX_SAMPLES_PER_TOOL="3"
MODEL_NAME="gpt-4-turbo-2024-04-09"
PARAM_GEN_PROMPT_VERSION="v3"
W1="0.5"
W2="0.5"
GROUND_TRUTH_STRATEGY="select_best"
QUERY_SELECTION_STRATEGY="random"
START_TOKEN=""
END_TOKEN=""
REFINE_DESC=False

case "$CASE" in
    1)
        echo "Running: prepare_training_data (split2b_no_eval)"
        EVAL_DIR=""
        GROUND_TRUTH_DESC_FILES="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2b_StableToolBench_D2_2"
        FIXED_PARAM_PATH="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2b_StableToolBench_D1_fix_only2"
        # v9: with reasoning
        # v7: without reasoning, was the one working on 0109
        POLICY_PROMPT_VERSION="v7" # v7, v5, v6
        POLICY_PROMPT_PATH="${WORKSPACE_FOLDER}/DRAFT/policy_learn/prompts/policy_prompt_tool_level_${POLICY_PROMPT_VERSION}.txt"
        DATASET_NAME="split2b_no_eval_prompt_${POLICY_PROMPT_VERSION}"
        VAL_RATIO="0.1"
        ;;
    2)
        echo "Running: prepare_training_data (split2b_w_eval)"
        EVAL_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper/experiments_track/20260102_190139_split2b_D0"
        GROUND_TRUTH_DESC_FILES="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2b_StableToolBench_D2_2"
        FIXED_PARAM_PATH="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2b_StableToolBench_D1_fix_only2"
        # v8: without reasoning
        # v10: with reasoning
        POLICY_PROMPT_VERSION="v8" # v8, v10
        POLICY_PROMPT_PATH="${WORKSPACE_FOLDER}/DRAFT/policy_learn/prompts/policy_prompt_tool_level_${POLICY_PROMPT_VERSION}.txt"
        if [ "${REFINE_DESC}" = True ]; then
            DATASET_NAME="split2b_w_eval_refined_prompt_${POLICY_PROMPT_VERSION}"
        else
            DATASET_NAME="split2b_w_eval_prompt_${POLICY_PROMPT_VERSION}"
        fi
        VAL_RATIO="0.1"
        ;;
    3)
        echo "Running: prepare_inference_data (split2a_no_eval_inference)"
        EVAL_DIR="" 
        FIXED_PARAM_PATH="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2a_StableToolBench_D1_fix_only2"
        POLICY_PROMPT_VERSION="v9" # v9, v7
        if [ "${SCALING_EXPERIMENT}" = True ]; then
            DATASET_NAME="split2a_no_eval_inference_prompt_${POLICY_PROMPT_VERSION}_scaling_experiment"
            GROUND_TRUTH_DESC_FILES="${BASELINE_DESC_FILES}"
        else
            DATASET_NAME="split2a_no_eval_inference_prompt_${POLICY_PROMPT_VERSION}"
            GROUND_TRUTH_DESC_FILES="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2a_StableToolBench_D1_fix_only2"
        fi
        POLICY_PROMPT_PATH="${WORKSPACE_FOLDER}/DRAFT/policy_learn/prompts/policy_prompt_tool_level_${POLICY_PROMPT_VERSION}.txt"
        VAL_RATIO="0.0"
        ;;
    4)
        echo "Running: prepare_inference_data (split2a_w_eval_inference)"
        EVAL_DIR="PATH to the traces of D0 of split2a"
        GROUND_TRUTH_DESC_FILES="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2a_StableToolBench_D1_fix_only2"
        FIXED_PARAM_PATH="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2a_StableToolBench_D1_fix_only2"
        POLICY_PROMPT_VERSION="v8"
        POLICY_PROMPT_PATH="${WORKSPACE_FOLDER}/DRAFT/policy_learn/prompts/policy_prompt_tool_level_${POLICY_PROMPT_VERSION}.txt"
        DATASET_NAME="split2a_w_eval_inference_prompt_${POLICY_PROMPT_VERSION}"
        VAL_RATIO="0.0"
        ;;
    *)
        echo "Usage: $0 [case]"
        echo "  1 = prepare_training_data (split2b_no_eval)"
        echo "  2 = prepare_inference_data (split2a_no_eval_inference)"
        echo "  3 = original STB case (legacy)"
        exit 1
        ;;
esac

OUTPUT_DIR="${WORKSPACE_FOLDER}/data_stb/${DATASET_NAME}"

# Build command args

CMD_ARGS=(
    --eval-dirs "${EVAL_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --baseline-desc-files "${BASELINE_DESC_FILES}"
    --ground-truth-desc-files "${GROUND_TRUTH_DESC_FILES}"
    --prompt-path "${POLICY_PROMPT_PATH}"
    --dataset-name "${DATASET_NAME}"
    --num-query-examples "${NUM_QUERY_EXAMPLES}"
    --min-queries-per-tool "${MIN_QUERIES_PER_TOOL}"
    --train-query-ratio "${TRAIN_QUERY_RATIO}"
    --max-samples-per-tool "${MAX_SAMPLES_PER_TOOL}"
    --model-name "${MODEL_NAME}"
    --parameter-generation-prompt-version "${PARAM_GEN_PROMPT_VERSION}"
    --w1 "${W1}"
    --w2 "${W2}"
    --ground-truth-strategy "${GROUND_TRUTH_STRATEGY}"
    --query-selection-strategy "${QUERY_SELECTION_STRATEGY}"
    --val-ratio "${VAL_RATIO}"
    --start-token "${START_TOKEN}"
    --end-token "${END_TOKEN}"
)

if [ "${REFINE_DESC}" = True ]; then
    CMD_ARGS+=(--refine-desc)
fi

# Add fixed-param-path only if set
if [ -n "${FIXED_PARAM_PATH}" ]; then
    CMD_ARGS+=(--fixed-param-path "${FIXED_PARAM_PATH}")
fi

python "${WORKSPACE_FOLDER}/DRAFT/policy_learn/tools/prepare_training_data.py" "${CMD_ARGS[@]}"
