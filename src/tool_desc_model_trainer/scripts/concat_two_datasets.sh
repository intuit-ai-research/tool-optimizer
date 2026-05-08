#!/usr/bin/env bash
set -euo pipefail

# Example: concat split2b_w_eval_prompt_v8 and split2b_no_eval_prompt_v7
DATA_NAME1=split2b_w_eval_prompt_v10
DATA_NAME2=split2b_no_eval_prompt_v9

# Optional custom output (otherwise defaults to .../data_stb/${DATA_NAME1}_${DATA_NAME2}/dataset)
# OUTPUT_DATASET="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/${DATA_NAME1}_${DATA_NAME2}/dataset"

python /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/DRAFT/policy_learn/tools/concat_two_datasets.py \
  --dataset1 /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/${DATA_NAME1}/dataset \
  --dataset2 /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/${DATA_NAME2}/dataset
# If you want a custom output path, add:
#   --output "${OUTPUT_DATASET}"
