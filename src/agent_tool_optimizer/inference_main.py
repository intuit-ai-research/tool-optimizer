import argparse
import logging
import logging.config
import os

import yaml

from agent_tool_optimizer.inference.application.hf_inference import HFInference
from agent_tool_optimizer.inference.application.vllm_inference import VLLMInference

LOG_CONFIG_YAML = '''
---
version: 1
disable_existing_loggers: False
formatters:
  simple:
    format: '%(asctime)s [%(levelname)s] - [ProcessId %(process)d] %(name)s:%(lineno)d - %(message)s'  

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: simple
    stream: ext://sys.stdout
root:
  level: INFO
  handlers: [console]
'''

def init_logging():
    try:
        # set up logging to console
        log_config = yaml.safe_load(LOG_CONFIG_YAML)
        logging.config.dictConfig(log_config)
    except Exception as e:
        print(f"Error initializing logging: {e}")

init_logging()
log = logging.getLogger(__name__)


if __name__ == '__main__':
    try:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--model_name", type=str, required=True, help="Model name on Huggingface or local path to model")
        arg_parser.add_argument("--dataset_id", type=str, required=False, default="", help="The Huggingface dataset id to use for inference or empty to use a local dataset")
        arg_parser.add_argument("--inference_engine", type=str, required=False, default="vllm", help="Whether to use VLLM for inference")

        args = arg_parser.parse_args()
        log.info("Arguments parsed")

        for key, value in vars(args).items():
            log.info(f"{key}: {value}")

        if args.inference_engine == "vllm":
            log.info("Using VLLM for inference")
            inference_engine = VLLMInference(model_name=args.model_name)
            inference_engine.run_inference(dataset_id=args.dataset_id)
        else:
            log.info("Using HF for inference")
            inference_engine = HFInference(model_name=args.model_name)
            inference_engine.run_inference(dataset_id=args.dataset_id)
        
        log.info("Inference completed successfully")
    except Exception as e:
        log.error(f"Error running inference: {e}")
        raise