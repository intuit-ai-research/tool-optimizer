# Example: mix split2b_w_eval1 and split2b_no_eval1 with per-epoch ratios

DATA_NAME1=split2b_w_eval_prompt_v8
DATA_NAME2=split2b_no_eval_prompt_v7

DATETIME=$(date +%Y%m%d_%H%M%S)

# python /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/DRAFT/policy_learn/tools/mix_two_datasets.py \
#   --dataset_folder1 /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/$DATA_NAME1 \
#   --dataset_folder2 /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/$DATA_NAME2 \
#   --mix_ratios 0.5 0.5 \
#   --num_epochs 2 \
#   --output_root /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/${DATA_NAME1}_${DATA_NAME2} \
#   --seed 42 \
#   --mutually_exclusive \
#   --total_train_size 846

MIX_RATIOS=(0.5 0.5 0.5)
NUM_EPOCHS=${#MIX_RATIOS[@]}
MUTUALLY_EXCLUSIVE=True

SAMPLES_PER_EPOCH_MANUAL=846 # total train size for each epoch

if [ "${MUTUALLY_EXCLUSIVE}" = True ]; then
    echo "Mutually exclusive mode"
    SAMPLES_PER_EPOCH=$((846*2/${NUM_EPOCHS}))
    mutually_exclusive="--mutually_exclusive"
else
    echo "Non-mutually exclusive mode"
    SAMPLES_PER_EPOCH=$SAMPLES_PER_EPOCH_MANUAL
    mutually_exclusive=""
fi
#SAMPLES_PER_EPOCH=$((846*2/${NUM_EPOCHS}))

if [ "${MUTUALLY_EXCLUSIVE}" = True ]; then
    # standard total train size
    echo "Standard total train size"
    OUTPUT_ROOT="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/${NUM_EPOCHS}/${DATA_NAME1}_${DATA_NAME2}_{$DATETIME}"
else
    echo "Custom total train size"
    OUTPUT_ROOT="/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb_mixed/${NUM_EPOCHS}/${DATA_NAME1}_${DATA_NAME2}_${SAMPLES_PER_EPOCH}_{$DATETIME}"
fi

echo "MIX_RATIOS: ${MIX_RATIOS[@]}"
echo "NUM_EPOCHS: $NUM_EPOCHS"
echo "SAMPLES_PER_EPOCH: $SAMPLES_PER_EPOCH" # train size for each epoch

python /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/DRAFT/policy_learn/tools/mix_two_datasets.py \
  --dataset_folder1 /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/$DATA_NAME1 \
  --dataset_folder2 /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/$DATA_NAME2 \
  --mix_ratios ${MIX_RATIOS[@]} \
  --num_epochs $NUM_EPOCHS \
  --output_root $OUTPUT_ROOT \
  --seed 42 \
  ${mutually_exclusive} \
  --total_train_size $SAMPLES_PER_EPOCH
