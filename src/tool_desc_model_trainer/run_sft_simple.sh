#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# VERL source configuration (set to true to use local verl)
export USE_LOCAL_VERL=true
# export LOCAL_VERL_PATH="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/verl"
export LOCAL_VERL_PATH="/home/sagemaker-user/user-default-efs/verl"

MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
# MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_PATH="Qwen/Qwen3-4B"
# MODEL_PATH="google/gemma-3-4b-it"
# MODEL_PATH="meta-llama/Llama-3.2-3B"
# MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"

# Derive a readable model name for checkpoint suffixes
MODEL_NAME="$(basename "$MODEL_PATH")"


# DATA_DIR1="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval/dataset"
# DATA_DIR1="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval_prompt_v5/dataset"
NUM_EPOCHS=3


MUTUALLY_EXCLUSIVE=False
SAMPLES_PER_EPOCH_MANUAL=846

PROMPT_NO_EVAL_VERSION="v7"
PROMPT_W_EVAL_VERSION="v8"

SUFFIX="3_epochs_951"

MIXED_DATA_ROOT="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/${NUM_EPOCHS}/split2b_w_eval_prompt_${PROMPT_W_EVAL_VERSION}_split2b_no_eval_prompt_${PROMPT_NO_EVAL_VERSION}_${SUFFIX}"

DATA_DIR1="${MIXED_DATA_ROOT}/split2b_mix_0.9_epoch_1/dataset"
DATA_DIR2="${MIXED_DATA_ROOT}/split2b_mix_0.5_epoch_2/dataset"
DATA_DIR3="${MIXED_DATA_ROOT}/split2b_mix_0.1_epoch_3/dataset"

# DATA_DIR3="${MIXED_DATA_ROOT}/split2b_mix_0.1_epoch_3/dataset"

# DATA_DIR1=/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval_prompt_v7/dataset
# DATA_DIR1=/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_w_eval_prompt_v10_split2b_no_eval_prompt_v9/dataset


# Reuse DATA_DIR1 for both epochs
# if DATA_DIR2 is not set and NUM_EPOCHS is 2, use DATA_DIR1
if [ -z "${DATA_DIR2}" ] && [ "${NUM_EPOCHS}" -eq 2 ]; then
    DATA_DIR2="${DATA_DIR1}"
fi
# TRAIN_PARQUET=${DATA_DIR1}/train.parquet

# Build EPOCH_TRAIN_FILES array conditionally
EPOCH_TRAIN_FILES=()
EPOCH_VAL_FILES=()
if [ -d "${DATA_DIR1}" ]; then
    EPOCH_TRAIN_FILES+=("${DATA_DIR1}/train.parquet")
    EPOCH_VAL_FILES+=("${DATA_DIR1}/val.parquet")
fi

if [ -d "${DATA_DIR2}" ]; then
    EPOCH_TRAIN_FILES+=("${DATA_DIR2}/train.parquet")
    EPOCH_VAL_FILES+=("${DATA_DIR2}/val.parquet")
fi

if [ -d "${DATA_DIR3}" ]; then
    EPOCH_TRAIN_FILES+=("${DATA_DIR3}/train.parquet")
    EPOCH_VAL_FILES+=("${DATA_DIR3}/val.parquet")
fi

# Convert array to JSON-style string for verl trainer: ["file1", "file2"]
EPOCH_TRAIN_FILES_STR=$(printf '"%s",' "${EPOCH_TRAIN_FILES[@]}")
EPOCH_TRAIN_FILES_STR="[${EPOCH_TRAIN_FILES_STR%,}]"
export EPOCH_TRAIN_FILES_STR

# Use first val file for validation (validation set is same across epochs)
export VAL_PARQUET="${EPOCH_VAL_FILES[0]}"

# EPOCHS=length of EPOCH_TRAIN_FILES
EPOCHS=$NUM_EPOCHS
# assert $NUM_EPOCHS == length of EPOCH_TRAIN_FILES
if [ $NUM_EPOCHS -ne ${#EPOCH_TRAIN_FILES[@]} ]; then
    echo "Error: NUM_EPOCHS ($NUM_EPOCHS) does not match length of EPOCH_TRAIN_FILES (${#EPOCH_TRAIN_FILES[@]})"
    exit 1
fi

echo "EPOCH_TRAIN_FILES_STR: $EPOCH_TRAIN_FILES_STR"
echo "VAL_PARQUET: $VAL_PARQUET"
echo "EPOCHS: $EPOCHS"


export LEARNING_RATE=0.00005
export LORA_RANK=64
export LR_SCHEDULER="cosine"
export WEIGHT_DECAY=0.01
export WARMUP_RATIO=0.05
export MAX_PROMPT_LENGTH=2048
export EFFECTIVE_BATCH_SIZE=8

# export START_TOKEN="<output>"
# export END_TOKEN="</output>"

DATE=$(date +%Y%m%d_%H%M%S)

# Source the training library
source "$SCRIPT_DIR/lib/training_sft.sh"

# Data configuration (fallback values, not used when EPOCH_*_FILES_STR are set)
# export TRAIN_PARQUET="${DATA_DIR1}/train.parquet"
# export VAL_PARQUET="${DATA_DIR1}/val.parquet"
export PROMPT_KEY="prompt"
export RESPONSE_KEY="ground_truth"
export TRUNCATION="right"

# Batch configuration
export MICRO_BATCH_PER_GPU=1
export WORLD_SIZE=8

# Model configuration
export TRUST_REMOTE_CODE=true
export MODEL_DTYPE="bf16"
export ATTN_IMPLEMENTATION="flash_attention_2"

# Trainer configuration
# export CHECKPOINT_DIR="${DATA_DIR}/checkpoints"
export CHECKPOINT_DIR="${DATA_DIR1}/checkpoints"
export WANDB_PROJECT="tool-level-sft-simple"
export RESUME_MODE="disable"
export SAVE_STRATEGY="epoch"
export SAVE_ONLY_MODEL=true
export TRAINER_LOGGER='["console","wandb"]'

# Call the function

# not need unless you want to train after SFT (e.g., RS-SFT)
# check if "checkpoints" is in the model path string
# if [[ "$MODEL_PATH" == *"checkpoints"* ]]; then
# # this is the 2nd round of training, we need to convert the checkpoint to huggingface format
#     if ! ls "$MODEL_PATH/huggingface"/*.safetensors 1>/dev/null 2>&1; then
#         python -m verl.model_merger merge \
#             --backend fsdp \
#             --local_dir "$MODEL_PATH" \
#             --target_dir "$MODEL_PATH/huggingface" \
#             --trust-remote-code
#     fi
#     if [ -d "${MODEL_PATH}/huggingface/lora_adapter" ]; then
#         export LORA_ADAPTER_PATH="${MODEL_PATH}/huggingface/lora_adapter"
#         export LORA_RANK=0 # with lora adapter, we do not need to specify the lora rank
#     else
#         export LORA_ADAPTER_PATH=""
#     fi
#     MODEL_PATH="${MODEL_PATH}/huggingface"

#     # export LEARNING_RATE=$(python3 -c "print($LEARNING_RATE / 2.0)")
# fi

SUFFIX="${MODEL_NAME}_lora_rank_${LORA_RANK}_lr_${LEARNING_RATE}_ebs_${EFFECTIVE_BATCH_SIZE}_max_prompt_length_${MAX_PROMPT_LENGTH}"

# SUFFIX="3_epochs_951"


run_sft_training "${MODEL_PATH}" "${EPOCHS}" "${SUFFIX}"