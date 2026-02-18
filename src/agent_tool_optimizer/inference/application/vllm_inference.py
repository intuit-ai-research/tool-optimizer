import datetime
import logging
import traceback
from typing import Any

from datasets import DatasetDict
from vllm import LLM
from transformers import AutoTokenizer
from agent_tool_optimizer.inference.application.prompts_builder import PromptsBuilder
from agent_tool_optimizer.inference.utils.console_output import (
    SPACE_BETWEEN_INFERENCES,
    print_inference_output,
)

log = logging.getLogger(__name__)

SAMPLE_PARAMS_MAX_COMPLETION_TOKENS = 1024
SAMPLE_PARAMS_TEMPERATURE = 0.4
SAMPLE_PARAMS_TOP_K = 40
SAMPLE_PARAMS_TOP_P = 0.95

MAX_NUM_SEQS = 1
GPU_MEMORY_UTILIZATION = 0.95
MAX_MODEL_LEN = 32768
REASONING_PARSER = "qwen3"


class VLLMInference:
    def __init__(self, model_name: str, **llm_kwargs: Any):
        self.model_name = model_name
        self.llm: LLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.prompts_builder = PromptsBuilder()        
        self.llm_kwargs = llm_kwargs

        log.info("Initializing vLLM with model name: %s", self.model_name)

        self.load_model()

    def load_model(self) -> None:
        log.info("Start loading model from %s", self.model_name)
        
        load_start_time = datetime.datetime.now()
        self.llm = LLM(model=self.model_name, 
                    tokenizer = self.model_name,
                    trust_remote_code=True,    
                    max_model_len=MAX_MODEL_LEN,
                    enforce_eager=False,
                    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                    max_num_seqs=MAX_NUM_SEQS                    
        )

        end_time = datetime.datetime.now()
        log.info("Model loaded in %s seconds", (end_time - load_start_time).total_seconds())

    def run_inference(self, dataset_id: str) -> None:
        if self.llm is None:
            raise RuntimeError("Model not loaded")

        try:
            dataset: DatasetDict = self.prompts_builder.build_dataset(dataset_id)

            sampling_params = self.llm.get_default_sampling_params()
            sampling_params.max_tokens = SAMPLE_PARAMS_MAX_COMPLETION_TOKENS
            sampling_params.temperature = SAMPLE_PARAMS_TEMPERATURE
            sampling_params.top_p = SAMPLE_PARAMS_TOP_P
            sampling_params.top_k = SAMPLE_PARAMS_TOP_K

            log.info("Starting inference...")
            if dataset is None:
                log.error("No dataset found")
                return

            for record in dataset["test"]:
                prompt = record["prompt"]
                conversation = [{"role": "user", "content": prompt}]

                outputs = self.llm.chat([conversation], sampling_params, use_tqdm=False)
                if outputs:
                    content = outputs[0].outputs[0].text                    
                    print_inference_output(prompt, content)
                    print("\n" * SPACE_BETWEEN_INFERENCES)
                else:
                    log.error("No model responses found")

        except Exception as e:
            tb = traceback.format_exc()
            log.error("Error in model generation: %s, tb - %s", e, tb)
            raise
