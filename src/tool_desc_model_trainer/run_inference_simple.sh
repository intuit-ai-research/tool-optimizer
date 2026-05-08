SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/inference_utils.sh"
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-0}

PROMPT_NO_EVAL_VERSION="v7"

NUM_SAMPLES=1
INFER_ON_SPLIT2B=False
SCALING_EXPERIMENT=False


SEED=42
MAX_NEW_TOKENS=1024



if [ "${SCALING_EXPERIMENT}" = True ]; then
    TEST_DATA_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2x_no_eval_inference_prompt_${PROMPT_NO_EVAL_VERSION}_scaling_experiment/dataset"
    MERGE_TARGET_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/StableToolBench/"
elif [ "${INFER_ON_SPLIT2B}" = True ]; then
    TEST_DATA_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval_prompt_${PROMPT_NO_EVAL_VERSION}/dataset"
    MERGE_TARGET_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2b_StableToolBench_D1_fix_only2"
    MAX_MODEL_LEN=16384
else
    TEST_DATA_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2a_no_eval_inference_prompt_${PROMPT_NO_EVAL_VERSION}/dataset"
    MERGE_TARGET_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2a_StableToolBench_D1_fix_only2"
    MAX_MODEL_LEN=8192
fi

echo "TEST_DATA_DIR: $TEST_DATA_DIR"
echo "MERGE_TARGET_DIR: $MERGE_TARGET_DIR"

DATE=$(date +%Y%m%d_%H%M%S)


# CKPT_DIR shows the training dataset directory
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval/dataset/checkpoints_20260103_215911/tool-level-sft-simple-20260103_215911/global_step_105"
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval/dataset/checkpoints/lora_rank_128_lr_0.001_ebs_8_max_prompt_length_2048_20260109_000645/global_step_105"
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval/dataset/checkpoints/lora_rank_128_lr_0.001_ebs_8_max_prompt_length_2048_20260109_004026/global_step_105"
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval_prompt_v5/dataset/checkpoints/lora_rank_128_lr_0.001_ebs_8_max_prompt_length_2048_20260109_005414/global_step_105"
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval_prompt_v5/dataset/checkpoints/lora_rank_128_lr_0.001_ebs_8_max_prompt_length_2048_20260109_010039/global_step_105"

# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/split2b_mix_0.7_epoch_1/dataset/checkpoints/lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260109_021002/global_step_210/"
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/split2b_mix_0.3_epoch_2/dataset/checkpoints_20260108_075314/tool-level-sft-simple-20260108_075314/global_step_27"
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/split2b_mix_0.7_epoch_1/dataset/checkpoints/lora_rank_64_lr_0.0001_ebs_64_max_prompt_length_2048_20260108_225123/global_step_54"
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/split2b_mix_0.7_epoch_1/dataset/checkpoints/lora_rank_128_lr_0.001_ebs_64_max_prompt_length_2048_20260108_233140/global_step_54"

# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/2/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.7_epoch_1/dataset/checkpoints/lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260109_040047/global_step_210/"

# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/2/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_1200/split2b_mix_0.7_epoch_1/dataset/checkpoints/_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260114_071356/global_step_300"

# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/2/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.9_epoch_1/dataset/checkpoints/_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260114_095553/global_step_210/"

# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/2/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.5_epoch_1/dataset/checkpoints/_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260114_103450/global_step_210"

# Trace-free 1 epoch
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval_prompt_v7/dataset/checkpoints/lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260109_084129/global_step_105/"

# CKPT_DIR=""
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.9_epoch_1/dataset/checkpoints/_lora_rank_0_lr_0.00005_ebs_8_max_prompt_length_2048_20260112_093335/global_step_210"

# (0.9 0.1)
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/2/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.9_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260114_095553/global_step_210"

# SFT 3
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_w_eval_prompt_v8/dataset/checkpoints/SFT3/global_step_628"

# Qwen3-4B Concat 1 epoch CoT
# CKPT_DIR="""

# Qwen3-4B Mixed 0.9 0.1 2 epochs CoT
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/2/split2b_w_eval_prompt_v10_split2b_no_eval_prompt_v9/split2b_mix_0.9_epoch_1/dataset/checkpoints/Qwen3-4B_lora_rank_64_mixed_91_CoT/global_step_210"

# Qwen3-4B Instruct 2507 Mixed 0.9 0.1 2 epochs CoT
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/2/split2b_w_eval_prompt_v10_split2b_no_eval_prompt_v9/split2b_mix_0.9_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_mixed_91_CoT/global_step_210"

# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.7_epoch_1/dataset/checkpoints/Llama-3.2-3B-Instruct_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260112_024518/global_step_210/"

# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.7_epoch_1/dataset/checkpoints/Llama-3.2-3B-Instruct_lora_rank_0_lr_0.00005_ebs_8_max_prompt_length_2048_20260112_041808/global_step_210"

# 3 Epochs 
# 210 steps

# Mixed 0.9 0.5 0.1

## lora rank 0 is bad
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_3_epochs_951/split2b_mix_0.9_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_0_lr_0.00005_ebs_8_max_prompt_length_2048_20260112_093335/global_step_210"

## lora rank 64 is also not that good
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_3_epochs_951/split2b_mix_0.9_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048/global_step_210"

#llama 3.1 8B (pretty bad, probably worse than llama 3.2 3B)
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_3_epochs_951/split2b_mix_0.9_epoch_1/dataset/checkpoints/Llama-3.1-8B-Instruct_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048/global_step_210"

# Mixed 0.5 0.5 0.5
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v10_split2b_no_eval_prompt_v9/split2b_mix_0.5_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_mixed_555_CoT/global_step_210"



## 315 steps

# 3 Epochs Mixed 0.9 0.5 0.1
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_846/split2b_mix_0.9_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048/global_step_315"

# 3 Epochs Mixed 0.7 0.5 0.3
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_846/split2b_mix_0.7_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048/global_step_315"

# 3 Epochs Mixed 0.5 0.5 0.5
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_846/split2b_mix_0.5_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048/global_step_315"


# CKPT_DIR=""
# Setup MODEL_DIR and OUTPUT_DIR (uses base model if CKPT_DIR is empty)
# Extracts split and prompt_version from TEST_DATA_DIR automatically
setup_model_paths "stb" "Qwen/Qwen3-4B-Instruct-2507"
# setup_model_paths "stb" "Qwen/Qwen3-4B"

# setup_model_paths "stb" "meta-llama/Llama-3.2-3B-Instruct"
# setup_model_paths "stb" "google/gemma-3-4b-pt"
# setup_model_paths "stb" "meta-llama/Llama-3.2-3B"
# setup_model_paths "stb" "google/gemma-3-4b-it"
# ALL_API_JSON="/home/sagemaker-user/user-default-efs/FunctionWrapper/StableToolBench/solvable_queries/test_apis/all_apis_combined.json"

USE_FORCE_REPLACE=false
if [ "${USE_FORCE_REPLACE:-false}" = "true" ]; then
    USE_FORCE_REPLACE="--force-replace"
else
    USE_FORCE_REPLACE=""
fi

# Set LLM_EXTRACTION_FLAG=true to use LLM for extraction (slower but more robust)
if [ "${LLM_EXTRACTION_FLAG:-false}" = "true" ]; then
    LLM_EXTRACTION_FLAG="--use-llm-extraction"
else
    LLM_EXTRACTION_FLAG=""
fi

echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"

if [ "$NUM_SAMPLES" -gt 1 ]; then
    temperature=0.8
    top_p=0.95
    repetition_penalty=1.05
else
    temperature=0.3
    top_p=0.9
    repetition_penalty=1.1
fi

echo "temperature: $temperature"
echo "top_p: $top_p"
echo "repetition_penalty: $repetition_penalty"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "SEED: $SEED"
echo "TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"


python ./tools/generate_tool_descriptions_vllm.py \
    --model-checkpoint "$MODEL_DIR" \
    --prompts-parquet ${TEST_DATA_DIR}/train.parquet \
    --merge-target-yaml ${MERGE_TARGET_DIR} \
    --output-mcp-yaml ${OUTPUT_DIR} \
    --start-token "" \
    --end-token "" \
    $LLM_EXTRACTION_FLAG \
    $USE_FORCE_REPLACE \
    --temperature $temperature \
    --top-p $top_p \
    --top-k 0 \
    --repetition-penalty $repetition_penalty \
    --max-new-tokens $MAX_NEW_TOKENS \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --num-samples $NUM_SAMPLES \
    --seed $SEED