import json
import logging
import os
from typing import Optional

from datasets import Dataset, DatasetDict

from agent_tool_optimizer.inference.utils.string_utils import (
    str_is_empty,
    str_is_not_empty,
)

log = logging.getLogger(__name__)

class PromptsBuilder:
    def __init__(self,):        
        log.info("Initializing PromptsBuilder")

    def build_dataset(self, dataset_id: str = "") -> DatasetDict:
        """
        Get the dataset.
        Returns:
            DatasetDict: The dataset
        """
        log.info("Getting dataset")

        if str_is_empty(dataset_id):
            return self.build_dataset_from_local_data()
        else:
            return self.build_dataset_from_huggingface(dataset_id)

    def build_dataset_from_huggingface(self, dataset_id: str) -> DatasetDict:
        """
        Build a dataset from Huggingface.
        Returns:
            DatasetDict: The dataset
        """
        log.info("Building dataset from Huggingface")
        return DatasetDict.from_hf_hub(dataset_id)

    def build_dataset_from_local_data(self) -> DatasetDict:
        """
        Build a dataset from local data.
        Returns:
            DatasetDict: The dataset
        """
        log.info("Building dataset from local data")

        dataset_dicts = []
        prompt_template = self.read_prompt_template()

        # record 1
        tool_name = "Reels Downloader"
        parameters = {
            "parameters": {
                "link": {
                "type": "str",
                "required": True,
                "description": "The full URL of a public Instagram reel or post to download. Only public reels and posts are supported."
                }
            },
            # metadata is the metadata of the API, if the metadata is missing, leave it as default. but may have distribution shitft problem since we train the model on such API data.
            "metadata": {
                "endpoint": "/n/",
                "method": "GET",
                "category": ""
            }
        }
        original_description = "You only need to provide the link to your Instagram media, and our API will give you the results in Download links of reels and posts"
        dataset_dicts.append({    
            "prompt_template": prompt_template,
            "prompt": prompt_template.format(tool_name=tool_name, parameter_json=json.dumps(parameters), original_description=original_description),
            "tool_name" : tool_name,
            "parameters" : json.dumps(parameters, indent=4),
            "original_description" : original_description        
        })

        # record 2
        tool_name = "Weather Forecast API"
        parameters = {
            "parameters": {
                "location": {
                    "type": "str",
                    "required": True,
                    "description": "City name or coordinates"
                },
                "days": {
                    "type": "int",
                    "required": False,
                    "description": "Number of days for forecast (1-14)"
                },
                "units": {
                    "type": "str",
                    "required": False,
                    "description": "Temperature units: metric or imperial"
                }
            },
            "metadata": {
                "endpoint": "/forecast",
                "method": "GET",
                "category": "weather"
            }
        }
        original_description = "Get weather forecast for any location"
        dataset_dicts.append({    
            "prompt_template": prompt_template,
            "prompt": prompt_template.format(tool_name=tool_name, parameter_json=json.dumps(parameters), original_description=original_description),
            "tool_name" : tool_name,
            "parameters" : json.dumps(parameters, indent=4),
            "original_description" : original_description        
        })

        # record 3
        tool_name = "Email Sender"
        parameters = {
            "parameters": {
                "to": {
                    "type": "list",
                    "required": True,
                    "description": "List of recipient email addresses"
                },
                "subject": {
                    "type": "str",
                    "required": True,
                    "description": "Email subject line"
                },
                "body": {
                    "type": "str",
                    "required": True,
                    "description": "Email body content"
                },
                "attachments": {
                    "type": "list",
                    "required": False,
                    "description": "List of file URLs to attach"
                },
                "cc": {
                    "type": "list",
                    "required": False,
                    "description": "CC recipients"
                }
            },
            "metadata": {
                "endpoint": "/send",
                "method": "POST",
                "category": "email"
            }
        }
        original_description = "Send emails with optional attachments and CC"
        dataset_dicts.append({    
            "prompt_template": prompt_template,
            "prompt": prompt_template.format(tool_name=tool_name, parameter_json=json.dumps(parameters), original_description=original_description),
            "tool_name" : tool_name,
            "parameters" : json.dumps(parameters, indent=4),
            "original_description" : original_description        
        })

        # record 4
        tool_name = "Image Resizer"
        parameters = {
            "parameters": {
                "image_url": {
                    "type": "str",
                    "required": True,
                    "description": "URL of the image to resize. Supports JPEG, PNG, GIF, WebP formats"
                },
                "width": {
                    "type": "int",
                    "required": True,
                    "description": "Target width in pixels (1-4000)"
                },
                "height": {
                    "type": "int",
                    "required": True,
                    "description": "Target height in pixels (1-4000)"
                },
                "maintain_aspect_ratio": {
                    "type": "bool",
                    "required": False,
                    "description": "Whether to maintain aspect ratio when resizing"
                }
            },
            "metadata": {
                "endpoint": "/resize",
                "method": "POST",
                "category": "image"
            }
        }
        original_description = "Resize images to your desired dimensions"
        dataset_dicts.append({    
            "prompt_template": prompt_template,
            "prompt": prompt_template.format(tool_name=tool_name, parameter_json=json.dumps(parameters), original_description=original_description),
            "tool_name" : tool_name,
            "parameters" : json.dumps(parameters, indent=4),
            "original_description" : original_description        
        })

        # record 5
        tool_name = "Database Query API"
        parameters = {
            "parameters": {
                "query": {
                    "type": "str",
                    "required": True,
                    "description": "Search query text"
                },
                "filters": {
                    "type": "dict",
                    "required": False,
                    "description": "Filter criteria including date_range, status, category"
                },
                "sort_by": {
                    "type": "str",
                    "required": False,
                    "description": "Sort field: relevance, created_at, updated_at, or title"
                },
                "limit": {
                    "type": "int",
                    "required": False,
                    "description": "Maximum number of results (1-100)"
                },
                "offset": {
                    "type": "int",
                    "required": False,
                    "description": "Number of results to skip for pagination"
                }
            },
            "metadata": {
                "endpoint": "/query",
                "method": "POST",
                "category": "database"
            }
        }
        original_description = "Query database with filters and sorting"
        dataset_dicts.append({    
            "prompt_template": prompt_template,
            "prompt": prompt_template.format(tool_name=tool_name, parameter_json=json.dumps(parameters), original_description=original_description),
            "tool_name" : tool_name,
            "parameters" : json.dumps(parameters, indent=4),
            "original_description" : original_description        
        })

        ds = DatasetDict({        
            "test": Dataset.from_list(dataset_dicts)      # ← here
        })
    
        log.info(ds["test"])
        log.info(len(ds["test"]))          # → 3 (or however many rows you have)
        log.info(ds["test"].features)    
        return ds


    def read_prompt_template(self) -> str:
        """
        Read the prompt template from tool_prompt.txt file.        
        Returns:
            str: The content of the prompt template file            
        Raises:
            FileNotFoundError: If tool_prompt.txt is not found
            IOError: If there's an error reading the file
        """
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_file_path = os.path.join(script_dir, "../data/tool_prompt.txt")
            
            log.info(f"Reading prompt template from {prompt_file_path}")
            
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            
            log.info("Successfully loaded prompt template")
            return prompt_template
            
        except FileNotFoundError as e:
            log.error(f"Prompt template file not found: {e}")
            raise
        except IOError as e:
            log.error(f"Error reading prompt template file: {e}")
            raise
        

