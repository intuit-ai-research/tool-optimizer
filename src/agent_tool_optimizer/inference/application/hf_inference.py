import datetime
import logging
import traceback

import torch
from datasets import DatasetDict
from huggingface_hub import login
from transformers import AutoTokenizer, pipeline

from agent_tool_optimizer.inference.application.prompts_builder import PromptsBuilder
from utils.console_output import (
    SPACE_BETWEEN_INFERENCES,
    print_inference_output,
)

log = logging.getLogger(__name__)

SAMPLE_PARAMS_MAX_COMPLETION_TOKENS = 1024
SAMPLE_PARAMS_TEMPERATURE = 0.6
SAMPLE_PARAMS_TOP_K = 40
SAMPLE_PARAMS_TOP_P = 0.95


class HFInference:
    def __init__(self, model_name: str, hf_access_token: str | None = None):
        self.model_name = model_name
        self.model = None
        self.prompts_builder = PromptsBuilder()

        log.info("Initializing LLM with model name: %s", self.model_name)

        if hf_access_token:
            self.login_to_huggingface(hf_access_token)

        self.load_model()

    def login_to_huggingface(self, access_token: str) -> None:
        try:            
            log.info("Logging in to Hugging Face using access token")
            login(token=access_token)
            log.info("Successfully logged in to Hugging Face")
        except Exception as e:
            log.error("Failed to login to Hugging Face: %s", e)
            raise

    def load_model(self):
        log.info("Start loading model from  %s", self.model_name)

        load_start_time = datetime.datetime.now()
        self.model = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        end_time = datetime.datetime.now()
        log.info("Model loaded in %s seconds", (end_time - load_start_time).total_seconds())

    def run_inference(self, input_data_path: str):
        try:
            # get the dataset
            dataset: DatasetDict = self.prompts_builder.build_dataset(input_data_path)

            # sample parameters for the model
            generation_args = {
                "max_new_tokens": SAMPLE_PARAMS_MAX_COMPLETION_TOKENS,
                "return_full_text": False,
                "do_sample": True,
                "num_beams": 1,
                "temperature": SAMPLE_PARAMS_TEMPERATURE,
                "top_p": SAMPLE_PARAMS_TOP_P,
                "top_k": SAMPLE_PARAMS_TOP_K,
            }

            log.info("Starting inference...")
            if dataset is not None:
                messages = []
                for record in dataset["test"]:
                    prompt = record["prompt"]
                    messages.clear()
                    messages.append({"role": "user", "content": prompt})

                    model_responses = self.model(messages, **generation_args)
                    if model_responses:
                        content = model_responses[0]["generated_text"]
                        print_inference_output(prompt, content)
                        print("\n" * SPACE_BETWEEN_INFERENCES)
                    else:
                        log.error("No model responses found")
            else:
                log.error("No dataset found")
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Error in model generation: {e}, tb - {tb}")
            raise e
