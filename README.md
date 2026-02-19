# Agent Tool Interface Optimizer

Python package for running LLM inference (and future training) for agent tool interface optimization. Uses Hugging Face Transformers and optional vLLM.

---

## Inference

Inference runs a language model over a prompt dataset and logs model responses. Two backends are supported: **vLLM** (default, recommended for throughput) and **Hugging Face** (Transformers pipeline).

### Requirements

- Python ≥ 3.12
- CUDA for GPU inference
- For vLLM: install with `uv sync --active --no-install-project` and the `vllm` extra, or use the Docker image

### CLI

Entry point: `src/agent_tool_optimizer/inference_main.py`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_name` | Yes | `intuit/agent-tool-optimizer` | Hugging Face model id or local path (e.g. `/opt/ml/model`) |
| `--dataset_id` | No | `intuit/tool-optimizer-dataset` | Hugging Face dataset id; if empty, uses built-in local demo dataset |
| `--inference_engine` | No | `vllm` | Engine: `vllm` or `hf` |
| `--hf_access_token` | No | `None` | Hugging Face access token for authentication (required for gated models or private datasets) |

**Examples**

```bash
# vLLM (default) with a local model, local/demo dataset
python src/agent_tool_optimizer/inference_main.py --model_name /opt/ml/model

# Hugging Face engine with a Hub model and Hub dataset
python src/agent_tool_optimizer/inference_main.py --model_name Qwen/Qwen3-8B --inference_engine hf --dataset_id your-org/your-dataset

# Using a gated model with Hugging Face access token
python src/agent_tool_optimizer/inference_main.py --model_name meta-llama/Llama-3.1-8B --hf_access_token hf_your_token_here

# vLLM with private dataset requiring authentication
python src/agent_tool_optimizer/inference_main.py --model_name /opt/ml/model --dataset_id your-org/private-dataset --hf_access_token hf_your_token_here
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

The image uses the Dockerfile’s conda env, installs PyTorch (CUDA 12.6) and the `vllm` extra, and runs `entrypoint.sh`, which starts inference with `--model_name /opt/ml/model` (no `--dataset_id`, so the built-in local dataset is used).

### Dataset

- **Hugging Face**: Pass `--dataset_id your-org/dataset-name`. The code expects a split named `test` and records with a `"prompt"` field.
- **Local / demo**: Omit `--dataset_id` or pass `""`. `PromptsBuilder` uses built-in demo prompts (e.g. tool descriptions and parameters) from `inference/application/prompts_builder.py`.

### Components

- **`inference_main.py`** — Parses CLI, selects engine (vLLM or HF), runs inference.
- **`inference/application/vllm_inference.py`** — vLLM backend; configurable sampling (max_tokens, temperature, top_p, top_k).
- **`inference/application/hf_inference.py`** — Hugging Face `text-generation` pipeline with bfloat16 and Flash Attention 2.
- **`inference/application/prompts_builder.py`** — Builds `DatasetDict` from Hugging Face or local demo data; prompts use a template from `inference/data/`.

---

## Training

Training pipelines for the agent tool interface optimizer are not implemented in this repository yet. This section is a placeholder for future documentation on:

- Dataset format and preparation
- Training scripts and configs
- Hardware and environment setup
- Checkpointing and evaluation

When training is added, usage and examples will be documented here.
