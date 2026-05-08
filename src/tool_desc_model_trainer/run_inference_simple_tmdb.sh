SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/inference_utils.sh"

# TEST_DATA_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2a_no_eval_inference_prompt_v7/dataset"
TEST_DATA_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_tmdb/split2a_no_eval_inference_prompt_v7/dataset"

# MERGE_TARGET_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_STBd1/eval/StableToolBench/"
# MERGE_TARGET_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper/descriptions/split2a_StableToolBench_D1_fix_only2"

# for our models, the merging would not take any description from D1 since it will generate it for all tools
MERGE_TARGET_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_DESC/eval/tmdb/desc_mcp_yaml/tmdb/tmdb_d1_1007_mcp.yaml"

DATE=$(date +%Y%m%d_%H%M%S)


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


# 3 Epochs Mixed 0.9 0.5 0.1
CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.9_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_0_lr_0.00005_ebs_8_max_prompt_length_2048_20260112_093335/global_step_210"


# 3 Epochs more samples

# 3 Epochs Mixed 0.9 0.5 0.1
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_846/split2b_mix_0.9_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048/global_step_315"

# 3 Epochs Mixed 0.7 0.5 0.3
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_846/split2b_mix_0.7_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048/global_step_315"

# 3 Epochs Mixed 0.5 0.5 0.5
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/3/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7_846/split2b_mix_0.5_epoch_1/dataset/checkpoints/Qwen3-4B-Instruct-2507_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048/global_step_315"

# 2 Epochs Mixed 0.5 0.5
# CKPT_DIR="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/2/split2b_w_eval_prompt_v8_split2b_no_eval_prompt_v7/split2b_mix_0.5_epoch_1/dataset/checkpoints/_lora_rank_64_lr_0.00005_ebs_8_max_prompt_length_2048_20260114_103450/global_step_210"


# Setup MODEL_DIR and OUTPUT_DIR (uses base model if CKPT_DIR is empty)
# Extracts split and prompt_version from TEST_DATA_DIR automatically
setup_model_paths "tmdb" "Qwen/Qwen3-4B-Instruct-2507"

# Set LLM_EXTRACTION_FLAG=true to use LLM for extraction (slower but more robust)
if [ "${LLM_EXTRACTION_FLAG:-false}" = "true" ]; then
    LLM_EXTRACTION_FLAG="--use-llm-extraction"
else
    LLM_EXTRACTION_FLAG=""
fi

echo "MODEL_DIR: $MODEL_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"

python ./tools/generate_tool_descriptions_vllm.py \
    --model-checkpoint "$MODEL_DIR" \
    --prompts-parquet ${TEST_DATA_DIR}/train.parquet \
    --merge-target-yaml ${MERGE_TARGET_DIR} \
    --output-mcp-yaml ${OUTPUT_DIR} \
    --dataset-name TMDB \
    --start-token "" \
    --end-token "" \
    $LLM_EXTRACTION_FLAG \
    --temperature 0.3 \
    --top-p 0.9 \
    --top-k 0 \
    --repetition-penalty 1.1 \
    --max-new-tokens 1024