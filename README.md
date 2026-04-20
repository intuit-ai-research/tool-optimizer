# Agent Tool Interface Optimizer

Python package to enhance software tool descriptions in an effort to improve agentic performance. Supports direct LLM based inference on already trained model (via Hugging Face or vLLM), or training of a new model.

---

## Inference

Inference runs a language model over a prompt dataset and logs model responses. Two backends are supported: **vLLM** (default, recommended for throughput) and **Hugging Face** (Transformers pipeline).

### Requirements

- Python ≥ 3.12
- CUDA for GPU inference

### Environment setup
- Install dependencies with `uv`:
  - If using **vLLM** for inference: `uv sync --active --no-install-project --extra vllm`, or use the Docker image 
  - If using **Hugging Face** for inference: `uv sync --active --no-install-project`
- Start virtual environment: `source .venv/bin/activate` 

### CLI

Entry point: `src/agent_tool_optimizer/inference_main.py`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_name` | Yes | `intuit/agent-tool-optimizer` | Hugging Face model id or local path (e.g. `/opt/ml/model`) |
| `--input_data_path` | Yes | — | Path to a comma-delimited data file for inference |
| `--inference_engine` | No | `vllm` | Engine: `vllm` or `hf` |
| `--hf_access_token` | No | `None` | Hugging Face access token for authentication (required for gated models) |

**Examples**

```bash
# vLLM (default) with a local model and a CSV data file
python src/agent_tool_optimizer/inference_main.py --model_name /opt/ml/model --input_data_path data/inference/tool_descs.example.csv

# Hugging Face engine with a Hub model
python src/agent_tool_optimizer/inference_main.py --model_name Qwen/Qwen3-8B --inference_engine hf --input_data_path data/inference/tool_descs.example.csv

# Using a gated model with Hugging Face access token
python src/agent_tool_optimizer/inference_main.py --model_name meta-llama/Llama-3.1-8B --input_data_path data/inference/tool_descs.example.csv --hf_access_token hf_your_token_here
```

Set `PYTHONPATH` to include `src` (e.g. `export PYTHONPATH=/path/to/project/src`).

### Docker

- **Build** (from project root):

  ```bash
  ./scripts/docker_build.sh
  ```

  Or: `docker build --network=host --progress plain -f Dockerfile -t agent-tool-interface-optimizer:0.1 .`

- **Run**: Container expects the model at `/opt/ml/model`. Mount your model and run:

  ```bash
  ./scripts/docker_run.sh
  ```

  Or: `docker run -e CUDA_VISIBLE_DEVICES=0 --gpus all --network=host --ipc=host -v /path/to/model:/opt/ml/model -d -t agent-tool-interface-optimizer:0.1`

The image uses the Dockerfile’s conda env, installs PyTorch (CUDA 12.6) and the `vllm` extra, and runs `entrypoint.sh`.

### Dataset

- **CSV file**: Pass `--input_data_path path/to/file.csv`. The file must be comma-delimited with columns: `tool_name`, `parameters`, `original_description`.
- **Local / demo**: If no `input_data_path` is provided, `PromptsBuilder` uses built-in demo prompts (e.g. tool descriptions and parameters) from `inference/application/prompts_builder.py`.

### Components

- **`src/agent_tool_optimizer/inference_main.py`** — Parses CLI, selects engine (vLLM or HF), runs inference.
- **`src/agent_tool_optimizer/inference/application/vllm_inference.py`** — vLLM backend; configurable sampling (max_tokens, temperature, top_p, top_k).
- **`src/agent_tool_optimizer/inference/application/hf_inference.py`** — Hugging Face `text-generation` pipeline with bfloat16 and Flash Attention 2.
- **`src/agent_tool_optimizer/inference/application/prompts_builder.py`** — Builds `DatasetDict` from a CSV file or local demo data; prompts use a template from `data/inference/`.

---

## Training

### Training Data (Option 1) - Download existing example training data

Sample training data is available for this library through [intuit/tool-optimizer-dataset](https://huggingface.co/datasets/intuit/tool-optimizer-dataset) stored in HuggingFace. 
To retrieve that data, execute `python src/utils/pull_hf_data.py` using the following arguments:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--hf_access_token` | No | `None` | Hugging Face access token for authentication. Pass as arg or set as env variable (HF_TOKEN) |
| `--repo_id` | Yes | `intuit/agent-tool-optimizer` | Hugging Face Dataset repo ID |
| `--data_files` | No | `None` | Optional. Pulls only specific files from HF repo |
| `--data_dir` | No | `None` | Optional. Pulls only specific directory from HF repo |
| `--output_path` | No | `"../../data"` | Local path where to store files pulled from HF repo |

#### Training Data (Option 2) - Generate training data using pipeline

The pipeline has **three main stages** that progressively improves tool descriptions:

![Pipeline Diagram](misc/data_syn_better.png)

Across these stages, there are six main steps:
1. **Step 1 — Agentic Seed Tool Annotation**: Curate raw tools by checking health and collecting running examples (optional).
2. **Step 2 — D0 to D1 (Data-Independent Improvement)**: Improve sparse, vague descriptions (D0) into structured, clear descriptions (D1) using LLM guidelines — no execution data needed.
3. **Step 3 — User Query Synthesis**: Generate realistic multi-step user queries for the tools.
4. **Step 4 — Execution Traces**: Run synthesized queries against tools (using D1 descriptions) to produce success/failure execution traces.
5. **Step 5 — D1 to D2 (Trace-Aware Refinement)**: Refine D1 descriptions into rule-enriched, robust D2 descriptions using the execution traces.
6. **Step 6 — SFT (Supervised Fine-Tuning)**: Train a model to generate D2-quality descriptions from D0 input.

#### Environment setup

```bash
# Go into root directory
cd tool-optimizer

# Install dependencies
uv sync --active --no-install-project
uv pip install -e . --no-deps

# Start virtual environment
source .venv/bin/activate

# TODO: provide more detailed instructions on this
# Configure environment variables (first time only)
cp .env.example .env
# Edit .env — key fields:
#   PYTHONPATH=<path_to_StableToolBench>:${PYTHONPATH}
#   TOOLBENCH_KEY=<your_toolbench_key>
#   ROOT_DIR=<path_to_FunctionWrapper>
#   SKIP_REAL_REQUEST=False
#   SKIP_SIMULATION=True
```

#### Step 1. Annotate tool usage and health signals

> **Diagram mapping (Stage 1: Agentic Seed Tool Query Synthesis)**
> - **Input**: `Raw ToolBench Tools` + `Tool Usage` 
> - **Process**: `Tool Call Collection` + `Health Check`
> - **Output**: `Synthesized Seed Tools` + `Tool Call Queries`

**Purpose**: For each tool, an LLM agent calls the tool's APIs to check health and collect running examples. This produces annotated YAML files with health status and example call/response pairs.

> **Skip this step?** Pre-computed annotations are available using `src/utils/pull_hf_data.py` (pulled as part of `tools_mcp_yaml_annotated/`). You can proceed directly to Step 3 using these, or use the raw D0 YAMLs from (pulled as part of `tools_mcp_yaml_raw/`).


##### Step 1.1. Install the CUDA
Check this [link](https://verl.readthedocs.io/en/latest/start/install.html) for more details.

Quick setup if you restart the SageMaker instance:
```bash
# At where the file is downloaded
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo update-alternatives --set cuda /usr/local/cuda-12.8
```

```bash
# At where the file is downloaded
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

Check if CUDA is 12.8:
```bash
nvcc --version
```

If not, run the following command:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && export PATH=/usr/local/cuda/bin:$PATH && nvcc --version
```

##### Step 1.2. Install the python dependencies

Use [environment.yml](environment.yml)
```bash
conda env create -f environment.yml
conda activate agent_train
```

Install based on the docs
```bash
conda create -n verl python==3.12
conda activate verl
```

Then, execute the install.sh script that we provided in verl:
```bash
# Make sure you have activated verl conda env
# If you need to run with megatron
bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

##### Step 1.3. Install the verl
```bash
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
```

Currently, Verl is at commit b9bd00efba253ea90072555c45692054cf703de2.

**TODO: Confirm that this can be removed.**

~~##### Step 1.4. Install the smolagents lib~~

~~pip install "smolagents[toolkit,openai,telemetry,vllm]>=1.23.0" "tqdm>=4.67.1" "termcolor>=3.2.0" "rank-bm25>=0.2.2" "tenacity>=9.1.2" "joblib>=1.5.2"~~

##### Step 1.4. Execute script to generate output data

TODO: Change arg "tool_root_dir" with "tools_root_dir"
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--eval_results_folders` | Yes | `None` | Directory path to tools usage logs |
| `--mcp_yaml_path` | Yes | `None` | Directory path to raw tools mcp yamls |
| `--tool_root_dir` | Yes | `None` | Directory path to tools API definitions |
| `--openai_api_key` | No | `None` | API key for LLM calls using OpenAI models. Must either be provided as CLI arg or already exist as OPENAI_API_KEY env variable |
| `--model_name` | No | `gpt-4.1-2025-04-14` | OpenAI model name to be used for LLM calls |
| `--output_path` | No | `"../../data/tools_mcp_yaml_annotated_<timestamp>"` | Local path where to store output of annotated tools mcp yamls |

```bash
cd src/agent_tool_annotator

python main_select.py \
  --eval_results_folders <path_to_tool_usage_logs_dir> \
  --mcp_yaml_path <path_to_tools_mcp_yaml_raws> \
  --tool_root_dir <path_to_tools_apis> \
  --openai_api_key <openai_key_str> \
  --model_name <openai_model_name> \
  --output_path <path_to_output_dir>

cd ../..
```

**Example command used** (from history):

```bash
cd src/agent_tool_annotator

python main_select.py \
  --eval_results_folders data/StableToolBench/tools_usage/ \
  --mcp_yaml_path data/StableToolBench/tools_mcp_yaml_raw/ \
  --tool_root_dir data/StableToolBench/tools_api/ \
  --model_name  "gpt-4.1-2025-04-14" \
  --output_path ../../data/tools_mcp_yaml_annotated

cd ../..
```

**Output**: Annotated YAML files in `<path_to_output_dir>/<Category>/`, one per tool. These contain the original tool descriptions plus `_metadata` annotations for health status and example calls.

---

#### Step 2. Data-Independent Description Improvement

> **Diagram mapping (First half of Stage 3: Two-Stage Description Improvement)**
> - **Input**: `Initial Description (D0)` (Sparse, Vague) → `eval/StableToolBench/`
> - **Process**: `Data-Independent Improvement`
> - **Output**: `Improved Description (D1)` (Structured, Clear) → `description_improvement/results/StableToolBench_D1/`

**Purpose**: Transform sparse, vague D0 descriptions into structured, clear D1 descriptions using LLM-based guidelines. No execution data is needed — this is purely prompt-driven improvement.

> **Skip this step?** Pre-computed trace-free and data independent descriptions are available using `src/utils/pull_hf_data.py` (pulled as part of `tools_mcp_yaml_tracefree_desc_improve/`).

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--mcp_yaml_path` | Yes | `None` | Directory path to raw tools mcp yamls |
| `--openai_api_key` | No | `None` | API key for LLM calls using OpenAI models. Must either be provided as CLI arg or already exist as OPENAI_API_KEY env variable |
| `--model_name` | No | `gpt-4.1-2025-04-14` | OpenAI model name to be used for LLM calls |
| `--output_path` | No | `"../../data/StableToolBench/tools_mcp_yaml_desc_improve_tracefree_<timestamp>"` | Local path where to store output of tools mcp yamls with trace-free improved descriptions |
| `--config` | Yes | -- | Path to config file |

```bash
cd src/description_improvement

python main_StableToolBench.py \
  --mcp_yaml_path <path_to_tools_mcp_yaml_raws> \
  --openai_api_key <openai_key_str> \
  --model_name <openai_model_name> \
  --output_path <path_to_output_dir> \
  --config <path_to_config_file>

cd ../..
```

**Actual command used** (from history):

```bash
cd src/description_improvement

python main_StableToolBench.py \
  --mcp_yaml_path ../../data/StableToolBench/tools_mcp_yaml_raw \
  --model_name "gpt-4.1-2025-04-14" \
  --output_path ../../data/StableToolBench/tools_mcp_yaml_desc_improve_tracefree
  --config config/StableToolBench.yaml

cd ../..
```

> The script was run with multiple parallel processes (4+) to speed up processing all ~700 tools.

This automatically loops through all YAML files under `<path_to_tools_mcp_yaml_raws>/<Category>/`. It skips files that already exist in the output folder, so re-running is safe.

**Output**: YAML files with improved descriptions stored in `<path_to_output_dir>/<Category>/<tool>.yaml`.


**Config details** (`StableToolBench.yaml`):
- Strategy: `generic_llm_guidelines` (data-independent)
- Model: `gpt-41-2025-04-14-oai` (temperature 0.3)
- Prompt: `data_indep_v1.txt`

---

#### Step 3: User Query Synthesis (TOUCAN Pipeline)

> **Diagram mapping (Stage 2: User Query Synthesis — Annotation & Filtering)**
> - **Input**: `Synthesized Seed Tools` → annotated YAMLs from Step 1 (or D1 from Step 2)
> - **Process**: `Agentic Annotator` → `Generate Natural, Multi-Step Queries` → `LLM` → `Annotate Filter &`
# TODO: change output to allow for user defined path
> - **Output**: `Tool Call Queries` + `History` → `TOUCAN/data/ToolUse_<name>/combined_queries.json`

**Purpose**: Generate realistic, multi-step user queries that exercise the tools. The TOUCAN pipeline converts tool YAMLs to MCP JSON, synthesizes questions via LLM, validates quality, generates agent trajectories, and produces a combined query file.

> **Skip this step?** Synthetic queries using MCP servers with at least 3 tools and generating 24 questions per server are available using `src/utils/pull_hf_data.py` (pulled as part of `tools_synthetic_queries/`).

**New Repo**: `TOUCAN` (branch `main`)
**Script**: `datagen/run_pipeline.sh`


##### Environment setup
```bash
# Exit tool-optimizer directory
cd ..
# Pull separate TOUCAN package
git clone https://github.com/intuit-ai-research/TOUCAN
cd TOUCAN
```

**OPTION 1: USING UV**
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch
uv pip install -r requirements.txt
cd Qwen-Agent && pip install -e . && cd ..
```

**OPTION 2: USING CONDA**
```bash
# Create Env
conda create -n toucan python=3.12 -y
conda activate toucan
# Install Required Packages
pip install torch
pip install -r requirements.txt
# Install Qwen Agent from Source
cd Qwen-Agent && pip install -e . && cd ..
```


**IF RUNNING ON MAC OS**
There is a known issue with PyTorch + tokenizers on Apple Silicon, leading to segfault errors when SentenceTransformer (using PyTorch) tries to encode embeddings on macOS ARM (aarch64). This is often caused by:
  1. OMP/MKL threading conflicts — PyTorch's OpenMP and the tokenizers' parallelism clash
  2. A buggy PyTorch or tokenizers version on macOS ARM

To mitigate, set the following before running the scripts below:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```


| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input_dir` | Yes | -- | Input directory for YAML tool specs. Can use annotated files from --output_path of Step 1 or filtered files from --output_path of Step 2 |
| `--tools_root_dir` | Yes | -- | Directory path to tools API definitions |
| `--mcp_servers_dir` | No | `../mcp_servers` | Output directory for generated MCP server JSONs (Stage 0) |                                                        
| `--cache_dir` | No | `./tool_response_cache` | Cache directory for `convert_yaml_to_mcp_json.py` (Stage 0) |                                                     
| `--num_tools` | No | 2 | Controls complexity of generated questions by how many tools are included in each generated prompt/question |
| `--sampling_strategy` | No | `uniform` | How MCP servers are selected when generating question prompts. Options: `random`, `uniform`, `power_law`, or `featured` |
| `--samples_per_server` | No | 10 | How many questions are generated per MCP server |
| `--mode` | No | `single_server` | Question generation mode: `single_server` or `multi_server` |
| `--output_folder` | No | `../data` | Output root directory for Stage 1-1 |
| `--timestamp` | No | Current epoch time | Timestamp/seed used in output file naming for Stage 1-1 |
| `--model_name` | No | `gpt-4.1-2025-04-14` | OpenAI model name to be used for LLM calls |
| `--openai_api_key` | No | `None` | API key for LLM calls using OpenAI models. Must either be provided as CLI arg or already exist as OPENAI_API_KEY env variable |
| `--engine` | No | `openai` | Inference backend: `vllm_api`, `vllm`, `hf`, `together_api`, `openai`, or `openrouter_api` |
| `--start_vllm_service` | No | `false` | Whether to auto-start a vLLM server (`true` or `false`) |


##### Script execution
```bash
cd datagen

./run_pipeline.sh \
  --input_dir ../../tool-optimizer/data/StableToolBench/tools_mcp_yaml_desc_improve_tracefree \
  --samples_per_server 6 \
  --num_tools 3
  --model_name "gpt-4.1-2025-04-14" \
  --tools_root_dir ../../tool-optimizer/data/StableToolBench/tools_api/ \
  --output_folder ../../tool-optimizer/data/StableToolBench/tools_synthetic_queries
# Output: ../../tool-optimizer/data/StableToolBench/tools_synthetic_queries/ToolUse_smithery_2508_3tool_1766028185

cd ../../tool-optimizer
```

**Sub-stages** (run automatically by `run_pipeline.sh`):
1. **Stage 0**: Convert YAML to MCP JSON (`convert_yaml_to_mcp_json.py`)
2. **Stage 1.1**: Generate question prompts (`step1.1_gen_questions.py`)
3. **Stage 1.2**: LLM completion (`step1.2_completion.sh`)
4. **Stage 1.3**: Process completions (`step1.3_process_completion.py`)
5. **Convert**: Convert to query format (`convert_preview_to_g1.py`)
6. **Combine**: Merge all queries into `combined_queries.json`

For manual step-by-step execution (including optional quality checks in Steps 2-4), see `TOUCAN/datagen/README.MD`.

**Output**: `<output_folder>/ToolUse_<name>_<timestamp>/combined_queries.json`

---

## Step 4: Synthesized Queries to Execution Traces

> **Diagram mapping (Stage 3: Two-Stage Description Improvement — trace generation)**
> - **Input**: `Tool Call Queries` (from Step 3) + `Improved Description (D1)` (from Step 2)
> - **Process**: Execute queries against actual tool APIs
> - **Output**: `(Success & Failure) Execution Traces` → `experiments/<timestamp>/`

**Purpose**: Execute the synthesized queries against the actual tools to produce success/failure execution traces. These traces are used in Step 5 to refine descriptions.

**Directory**: `tool_exec_tracer`
**Script**: `eval/tmdb/examples/main_tmdb.py`

```bash
cd src/tool_exec_tracer/StableToolBench/server 

# Start the StableToolBench server (and leave it running)
python main.py --openai_api_key "<key>" \
  --model_name "gpt-4.1-2025-04-14"
  --tool_root_dir ../../../../data/StableToolBench/tools_api 
```

**In a separate terminal window**

```bash
cd src/tool_exec_tracer/StableToolBench/server 

cd FunctionWrapper

python eval/tmdb/examples/main_tmdb.py \
  --config eval/tmdb/configs/tmdb_base.yaml \
  --dataset <path_to_combined_queries.json> \
  --mcp_yaml_path description_improvement/results/StableToolBench_D1 \
  --decompo_mcp_yaml_path description_improvement/results/StableToolBench_D1 \
  --tool_root_dir StableToolBench/data/toolenv/tools/ \
  --output_dir experiments/<experiment_name>
```

**Actual commands used** (from history, chained from Step 3 output):


python eval/tmdb/examples/main_tmdb.py \
  --config eval/tmdb/configs/tmdb_base.yaml \
  --dataset /Users/csoares1/dev/git-repos/tool-optimizer/data/StableToolBench/tools_synthetic_queries/ToolUse_smithery_198_3tool_1775806399/combined_queries.json \
  --mcp_yaml_path /Users/csoares1/dev/git-repos/tool-optimizer/data/StableToolBench/tools_mcp_yaml_desc_improve_tracefree \
  --tool_root_dir /Users/csoares1/dev/git-repos/tool-optimizer/data/StableToolBench/tools_api \
  --output_dir /Users/csoares1/dev/git-repos/tool-optimizer/data/StableToolBench/tools_exec_traces/20260415_111700 \
  --openai_api_key "<key>" \
  --model_name "gpt-4.1-2025-04-14"



```bash
cd FunctionWrapper

# Running synthesized queries from TOUCAN against D1 descriptions (2025-11-12)
# Input: combined_queries.json from Step 3
python eval/tmdb/examples/main_tmdb.py \
  --config eval/tmdb/configs/tmdb_base.yaml \
  --dataset /path/to/TOUCAN/data/ToolUse_smithery_49350_2tool_1762908421/queries/advertising_as.json \
  --mcp_yaml_path description_improvement/results/StableToolBench_D1/ \
  --tool_root_dir StableToolBench/data/toolenv/tools/ \
  --output_dir experiments/20251112_212230/advertising_as

# Running with standard StableToolBench eval queries using D1 descriptions (2025-11-13)
python eval/tmdb/examples/main_tmdb.py \
  --config eval/tmdb/configs/tmdb_base.yaml \
  --dataset StableToolBench/solvable_queries/test_instruction/G2_instruction.json \
  --mcp_yaml_path description_improvement/results/StableToolBench_D1/ \
  --tool_root_dir StableToolBench/data/toolenv/tools/ \
  --output_dir experiments/20251113_141608_D1_EBL/BM25/baseline/G2_instruction
```

- `--dataset`: The `combined_queries.json` from Step 3, or individual query JSONs from `queries/` subfolder
- `--mcp_yaml_path`: Use D1 descriptions (from Step 2)
- `--output_dir`: Where to write execution traces

**Output**: Each tool gets a subfolder under the output directory containing:
- `step_wise_eval_results.json` — detailed per-step evaluation
- `evaluation_statistics.json` — summary statistics
- `run_parameters.json` — reproducibility metadata
- `mcp_call_log.jsonl` — raw MCP API call logs




- Dataset format and preparation
  - Set your HuggingFace token: `export HF_TOKEN=your_token_here`
- Training scripts and configs
- Hardware and environment setup
- Checkpointing and evaluation

When training is added, usage and examples will be documented here.
