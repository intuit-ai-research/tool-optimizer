## Environment

start with a new virtual env with python 3.10
```
conda create -n verl python=3.10
cd xxx/FunctionWrapper/DRAFT/policy_learn
poetry install --sync
```

there might be a small issue with flash attention, you can check if it is available by
```
python - <<'PY'
from transformers.utils import is_flash_attn_2_available
print("FA2 available:", is_flash_attn_2_available())
PY
```

If not, you can reinstall it
```
# 1) Remove any mixed/old installs
python -m pip uninstall -y flash-attn flash_attn

# 2) Reinstall a FA2 build that matches your Torch/CUDA
# (try your current pin first; if it fails, try a nearby 2.7.x or 2.6.x)
python -m pip install --no-cache-dir --no-build-isolation "flash-attn==2.7.4.post1"
# If that still fails to load, try:
#   python -m pip install --no-cache-dir --no-build-isolation "flash-attn==2.7.3"
# or: python -m pip install --no-cache-dir --no-build-isolation "flash-attn==2.6.3"

```

and check again

```
python - <<'PY'
import importlib
for m in ["flash_attn", "flash_attn_2_cuda"]:
    try:
        importlib.import_module(m)
        print(m, "OK")
    except Exception as e:
        print(m, "FAILED:", e)
PY
```

```
python - <<'PY'
from transformers.utils import is_flash_attn_2_available
print("FA2 available:", is_flash_attn_2_available())
PY
```


## Summary of SFT

1. Data preprocessing
2. SFT
3. Inference

## Data Preprocessing

```
# normal SFT data
# MODE can be 1/2/3 depends on what you need
bash DRAFT/policy_learn/scripts/data_proc.sh $MODE

# It actually uses the code
DRAFT/policy_learn/tools/generate_tool_descriptions_vllm.py

# this produces datasets into
data_stb/spilt2x_yyy

# w_eval means the data is ((D0, D0_traces), D2)
# no_eval means the data is (D0, D2)
# _inference means the data is for inference, the prompt here will decide whether it is trace-free or trace-based eval

# Then if needed
# mixed (D0, D2) and ((D0, D0_traces), D2) data
bash /home/sagemaker-user/user-default-efs/FunctionWrapper_PL/DRAFT/policy_learn/scripts/run_mix_data.sh
```


### Tool-level Prompt

There are many versions
```
# Trace-free
DRAFT/policy_learn/prompts/policy_prompt_tool_level_v5.txt
DRAFT/policy_learn/prompts/policy_prompt_tool_level_v6.txt

# Trace-based
DRAFT/policy_learn/prompts/policy_prompt_tool_level_v3.txt
DRAFT/policy_learn/prompts/policy_prompt_tool_level_v3.5.txt
```

Each prompt repr. a tool.
The trace-based prompt will contain few example tool use traces.
Each example shows sth like:
1. API Selection: whether the AI agent used this API correctly
   - [CORRECT] - Correct API selected with valid parameters
   - [FAILED - selected 'X'] - Wrong API selected: 'X' was chosen instead
   - [CORRECT API - bad parameters] - Right API but wrong/invalid parameters generated
2. API Execution: whether the API call succeeded when executed
   - "API execution: N/M succeeded" - N out of M runs executed successfully
   - "(all failed)" - All attempts failed (usually due to bad parameters)


## SFT

SFT produces ckpt for (1) next steps (e.g., Rejection Sampling) and (2) direct evaluation.

It will use the training/validation data you just generated under data_stb. It will not perform evaluation on the test, you need to run inference before that.

The current implementation adds special tokens, which needs to modify the code of verl 0.5.0 a little bit.

I have a local version at (a submodule of this repo)

```
https://github.intuit.com/AIResearch/verl
```

You can specify the path in run_sft_simple.sh using the variable
```
LOCAL_VERL_PATH
```
Now the local verl is from the submodule 


```
# first cd to the policy_learn folder

cd DRAFT/policy_learn

bash run_sft_simple.sh
```

It uses the function in the script
```
DRAFT/policy_learn/lib/training_sft.sh
```

The ckpt will be saved into the training data folder

For example
```
/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval/dataset/checkpoints_20260103_215911/tool-level-sft-simple-20260103_215911/global_step_105
```
which means it was trained with the data
```
/home/sagemaker-user/user-default-efs/FunctionWrapper_PL/data_stb/split2b_no_eval/dataset/
```



## Inference
We need to generate a tool description for each tool for evaluation.

The inference data is also from the previous data_proc.sh

For trace-free
```
data_stb/split2a_no_eval_inference
```

For trace-based, you need to prepare data with
```
cd DRAFT/policy_learn/scripts
bash data_proc.sh 4
```
As it has been done by me.
You need to specify the path
```
EVAL_DIR="PATH to the traces of D0 of split2a"
```

After inference, you will get a folder similar to
```
/home/sagemaker-user/user-default-efs/FunctionWrapper/eval/split2a_StableToolBench_D1_fix_only2
```

Then you can run evaluation on it
