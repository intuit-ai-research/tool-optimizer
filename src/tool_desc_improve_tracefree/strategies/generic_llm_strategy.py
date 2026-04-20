"""
Generic LLM-based improvement strategy that can work with any dataset.
"""

from typing import List, Dict, Any, Optional
import time
import json
import re
import logging
import os
from pathlib import Path
from datetime import datetime

from ..interfaces.improvement_interface import (
    ImprovementStrategy, ImprovementMethod, ImprovementStage, 
    ImprovementContext, ImprovementResult
)
from ..interfaces.dataset_interface import StandardizedTool
from ..core.registries import ImprovementStrategyRegistry
from ..core.llm_client import create_openai_client, llm_call

logger = logging.getLogger(__name__)


class GenericLLMStrategy(ImprovementStrategy):
    """Generic LLM strategy that can work with any dataset by detecting the dataset from context."""
    
    def __init__(self, prompt_template_path: str, model_name: str = "gpt-4",
                 temperature: float = 0.3, max_tokens: int = 500,
                 dataset_name: str = None, save_reasoning: bool = True,
                 output_dir: str = None, improve_parameters: bool = False,
                 run_timestamp: str = None, guidelines_file_path: str = None,
                 openai_api_key: str = None, **kwargs):
        """
        Initialize generic LLM strategy.

        Args:
            prompt_template_path: Path to prompt template file (required)
            model_name: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            dataset_name: Override dataset name (otherwise auto-detect)
            save_reasoning: Whether to save full JSON responses with reasoning (default: True)
            output_dir: Directory to save reasoning JSON files (default: auto-detect from context)
            improve_parameters: Whether to improve parameter descriptions in addition to tool descriptions (default: False)
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dataset_name = dataset_name  # Can be overridden
        self.prompt_template_path = prompt_template_path  # Path to template file
        self.save_reasoning = save_reasoning  # Save JSON responses
        self.output_dir = output_dir  # Output directory for JSON files
        self.improve_parameters = improve_parameters  # Whether to improve parameter descriptions
        self.run_timestamp = run_timestamp  # Consistent timestamp from pipeline (e.g., 20251106_075323)
        self.guidelines_file_path = guidelines_file_path  # Path to guidelines file for metadata
        self.openai_api_key = openai_api_key  # OpenAI API key for standard API calls
        self.client = create_openai_client(api_key=openai_api_key)
    
    @property
    def name(self) -> str:
        return "generic_llm_guidelines"
    
    def get_method_type(self) -> ImprovementMethod:
        return ImprovementMethod.LLM_BASED
    
    def get_supported_stages(self) -> List[ImprovementStage]:
        return [ImprovementStage.DATA_INDEPENDENT, ImprovementStage.DATA_DEPENDENT]
    
    def can_handle_tools(self, tools: List[StandardizedTool]) -> bool:
        """Check if this strategy can handle the given tools."""
        return len(tools) > 0  # Can handle any tools
    
    def improve_descriptions(self, context: ImprovementContext) -> ImprovementResult:
        """
        Improve tool descriptions using LLM and auto-detected or specified dataset.
        
        Args:
            context: Improvement context with tools, guidelines, etc.
            
        Returns:
            ImprovementResult with improved tools
        """
        try:
            # Auto-detect dataset if not specified
            dataset_name = self.dataset_name or self._detect_dataset(context)
            
            improved_tools = []
            
            # For data-independent: use only guidelines
            # For data-dependent: use guidelines + usage patterns
            if context.stage == ImprovementStage.DATA_INDEPENDENT:
                improvement_source = f"{dataset_name}_guidelines_only"
                guidelines_text = "\n".join(context.guidelines) if context.guidelines else "Use best practices for API documentation"
            else:
                improvement_source = f"{dataset_name}_guidelines_and_patterns"
                guidelines_text = "\n".join(context.guidelines) if context.guidelines else "Use best practices"
                # Could incorporate usage patterns here for data-dependent stages
            
            for tool in context.tools:
                # Make dataset-specific LLM call
                if self.improve_parameters:
                    improved_description, improved_parameters, reasoning_data = self._llm_improve_with_parameters(tool, guidelines_text, context, dataset_name)
                    final_parameters = improved_parameters if improved_parameters else tool.parameters
                else:
                    # Original behavior - only improve description
                    improved_description, reasoning_data = self._llm_improve(tool, guidelines_text, context, dataset_name)
                    final_parameters = tool.parameters

                # Create improved tool
                improved_tool = StandardizedTool(
                    name=tool.name,
                    description=improved_description,
                    parameters=final_parameters,
                    format_specific=tool.format_specific,
                    metadata={
                        **(tool.metadata or {}),
                        "improvement_method": "generic_llm_guidelines",
                        "improvement_stage": context.stage.value,
                        "improvement_source": improvement_source,
                        "dataset": dataset_name,
                        "improvement_timestamp": self.run_timestamp or time.strftime("%Y%m%d_%H%M%S"),
                        "guidelines_applied": self.guidelines_file_path or f"guidelines/datasets/{dataset_name}/generic_v1.txt",
                        "reasoning": reasoning_data.get('reasoning') if reasoning_data else None,
                        "original_description": tool.description  # Keep original for comparison
                    }
                )
                improved_tools.append(improved_tool)
            
            return ImprovementResult(
                improved_tools=improved_tools,
                improvement_metadata={
                    "strategy": "generic_llm_guidelines",
                    "model": self.model_name,
                    "dataset": dataset_name,
                    "stage": context.stage.value,
                    "guidelines_count": len(context.guidelines) if context.guidelines else 0,
                    "tools_processed": len(improved_tools)
                },
                success=True
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error in improve_descriptions: {e}")
            logger.debug(f"Full traceback:\n{error_details}")
            return ImprovementResult(
                improved_tools=context.tools,  # Return original tools on error
                improvement_metadata={},
                success=False,
                error_message=f"{str(e)}\n{error_details}"
            )
    
    def _detect_dataset(self, context: ImprovementContext) -> str:
        """
        Auto-detect dataset from context.
        
        This is a simple implementation - could be enhanced with:
        - Tool name patterns
        - Guidelines content analysis
        - Context metadata
        """
        # Simple heuristics for dataset detection
        if context.guidelines:
            guidelines_text = " ".join(context.guidelines).lower()
            if "biomedical" in guidelines_text or "clinvar" in guidelines_text or "ensembl" in guidelines_text:
                return "dbqa"
            elif "movie" in guidelines_text or "tmdb" in guidelines_text:
                return "tmdb"
        
        # Check tool names for patterns
        if context.tools:
            tool_names = " ".join([tool.name for tool in context.tools]).lower()
            if "query_" in tool_names and ("clinvar" in tool_names or "ensembl" in tool_names):
                return "dbqa"
            elif "movie" in tool_names or "actor" in tool_names:
                return "tmdb"
        
        # Default fallback
        return "generic"
    
    def _llm_improve(self, tool: StandardizedTool, guidelines: str, context: ImprovementContext, dataset_name: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Make LLM call to improve tool description using dataset-specific prompt.
        
        Returns:
            tuple: (improved_description, reasoning_data)
                - improved_description: The improved tool description
                - reasoning_data: Dict with full JSON response and file path (if saved), or None
        """
        try:
            # Load dataset-specific prompt template
            prompt_template = self._load_prompt_template(context.stage, dataset_name)
            
            # Format the prompt with tool information
            formatted_prompt = self._format_prompt(prompt_template, tool, guidelines, context.server_config)
            
            # Make LLM call
            raw_response = self._call_llm(formatted_prompt)
            logger.info(f"Received LLM response for tool '{tool.name}' (length: {len(raw_response)} chars)")
            logger.debug(f"Raw response preview: {raw_response[:300]}...")
            
            # Try to parse JSON response (if prompt expects structured output)
            improved_description, parsed_json = self._parse_llm_response(raw_response, return_full_json=True)
            
            if not improved_description:
                logger.error(f"No improved description extracted for tool '{tool.name}'")
                logger.error(f"Full response: {raw_response}")
                raise ValueError(f"Failed to extract improved description from LLM response for tool '{tool.name}'")
            
            # Save reasoning if enabled and JSON was parsed
            reasoning_data = None
            if self.save_reasoning and parsed_json:
                try:
                    reasoning_data = self._save_reasoning_json(tool, parsed_json, context, dataset_name)
                except Exception as e:
                    logger.warning(f"Failed to save reasoning JSON: {e}")
            
            return improved_description, reasoning_data

        except Exception as e:
            logger.error(f"Error in _llm_improve for tool '{tool.name}': {e}")
            raise

    def _llm_improve_with_parameters(self, tool: StandardizedTool, guidelines: str, context: ImprovementContext, dataset_name: str) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Make LLM call to improve both tool description and parameter descriptions.

        Returns:
            tuple: (improved_description, improved_parameters, reasoning_data)
                - improved_description: The improved tool description
                - improved_parameters: Dict of improved parameter specs or None
                - reasoning_data: Dict with full JSON response and file path (if saved), or None
        """
        try:
            # Load dataset-specific prompt template
            prompt_template = self._load_prompt_template(context.stage, dataset_name)

            # Format the prompt with tool information
            formatted_prompt = self._format_prompt(prompt_template, tool, guidelines, context.server_config)

            # Make LLM call
            raw_response = self._call_llm(formatted_prompt)
            logger.info(f"Received LLM response for tool '{tool.name}' (with parameters, length: {len(raw_response)} chars)")
            logger.debug(f"Raw response preview: {raw_response[:300]}...")

            # Try to parse JSON response for both description and parameters
            improved_description, improved_parameters, parsed_json = self._parse_llm_response_with_parameters(raw_response, tool, return_full_json=True)

            if not improved_description:
                logger.error(f"No improved description extracted for tool '{tool.name}'")
                logger.error(f"Full response: {raw_response}")
                raise ValueError(f"Failed to extract improved description from LLM response for tool '{tool.name}'")

            # Save reasoning if enabled and JSON was parsed
            reasoning_data = None
            if self.save_reasoning and parsed_json:
                try:
                    reasoning_data = self._save_reasoning_json(tool, parsed_json, context, dataset_name)
                except Exception as e:
                    logger.warning(f"Failed to save reasoning JSON: {e}")

            return improved_description, improved_parameters, reasoning_data

        except Exception as e:
            logger.error(f"Error in _llm_improve_with_parameters for tool '{tool.name}': {e}")
            raise

    def _parse_llm_response(self, response: str, return_full_json: bool = False) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Parse LLM response, extracting the improved description.
        Handles both plain text and JSON responses.
        
        Args:
            response: Raw LLM response
            return_full_json: If True, also return the full parsed JSON
            
        Returns:
            tuple: (improved_description, parsed_json or None)
        """
        if not response:
            logger.warning("Empty response from LLM")
            return "", None
        
        # Clean up the response - remove any leading/trailing whitespace and decode escape sequences
        cleaned_response = response.strip()
        
        # Try to parse as JSON first (handles clean JSON responses)
        try:
            parsed = json.loads(cleaned_response)
            if isinstance(parsed, dict) and 'improved_description' in parsed:
                logger.debug("Successfully parsed JSON response (direct parse)")
                return parsed['improved_description'], parsed if return_full_json else None
            elif isinstance(parsed, dict):
                logger.warning(f"JSON parsed but missing 'improved_description'. Keys: {list(parsed.keys())}")
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
            logger.debug(f"Response preview (first 500 chars): {cleaned_response[:500]}")
        
        # Try to find JSON in markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            try:
                parsed = json.loads(matches[0])
                if isinstance(parsed, dict) and 'improved_description' in parsed:
                    logger.debug("Successfully parsed JSON from markdown code block")
                    return parsed['improved_description'], parsed if return_full_json else None
            except json.JSONDecodeError as e:
                logger.debug(f"Markdown JSON parse failed: {e}")
        
        # Try to extract JSON object more carefully (handle nested braces)
        # Find the first { and try to parse from there
        start_idx = response.find('{')
        if start_idx != -1:
            # Try to find matching closing brace by counting braces
            brace_count = 0
            for i in range(start_idx, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching closing brace
                        json_str = response[start_idx:i+1]
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and 'improved_description' in parsed:
                                logger.debug("Successfully parsed JSON with brace matching")
                                return parsed['improved_description'], parsed if return_full_json else None
                        except json.JSONDecodeError as e:
                            logger.debug(f"Brace-matched JSON parse failed: {e}")
                        break
        
        # If no JSON found or parsing failed, check if response looks like incomplete JSON
        if cleaned_response.lstrip().startswith('{') and 'improved_description' in cleaned_response:
            # Response looks like JSON but couldn't be parsed - likely truncated
            logger.warning(f"Response appears to be incomplete JSON (starts with '{{' but unparseable)")
            logger.warning("This usually means max_tokens was too low and response was cut off")
            # Try to extract improved_description with regex as last resort
            desc_pattern = r'"improved_description"\s*:\s*"((?:[^"\\]|\\.)*)"'
            match = re.search(desc_pattern, cleaned_response, re.DOTALL)
            if match:
                desc = match.group(1)
                # Unescape the description
                desc = desc.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                logger.info("Successfully extracted improved_description from malformed JSON using regex")
                return desc, None
            else:
                logger.error("Could not extract improved_description from malformed JSON")
                return cleaned_response, None
        
        # Treat as plain text response
        logger.info("No valid JSON found in response, treating as plain text description")
        logger.debug(f"Response preview: {cleaned_response[:200]}...")
        return cleaned_response, None

    def _parse_llm_response_with_parameters(self, response: str, tool: StandardizedTool, return_full_json: bool = False) -> tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Parse LLM response, extracting both improved description and parameter descriptions.
        Handles both plain text and JSON responses.

        Args:
            response: Raw LLM response
            tool: Tool being improved (for parameter structure)
            return_full_json: If True, also return the full parsed JSON

        Returns:
            tuple: (improved_description, improved_parameters, parsed_json or None)
        """
        if not response:
            logger.warning("Empty response from LLM")
            return "", None, None

        # Clean up the response
        cleaned_response = response.strip()

        # Try to parse as JSON first (handles clean JSON responses)
        try:
            parsed = json.loads(cleaned_response)
            if isinstance(parsed, dict) and 'improved_description' in parsed:
                logger.debug("Successfully parsed JSON response with parameters (direct parse)")
                improved_params = self._process_improved_parameters(parsed.get('improved_parameters'), tool)
                return parsed['improved_description'], improved_params, parsed if return_full_json else None
            elif isinstance(parsed, dict):
                logger.warning(f"JSON parsed but missing 'improved_description'. Keys: {list(parsed.keys())}")
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")

        # Try to find JSON in markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            try:
                parsed = json.loads(matches[0])
                if isinstance(parsed, dict) and 'improved_description' in parsed:
                    logger.debug("Successfully parsed JSON from markdown code block with parameters")
                    improved_params = self._process_improved_parameters(parsed.get('improved_parameters'), tool)
                    return parsed['improved_description'], improved_params, parsed if return_full_json else None
            except json.JSONDecodeError as e:
                logger.debug(f"Markdown JSON parse failed: {e}")

        # Try to extract JSON object more carefully (handle nested braces)
        start_idx = response.find('{')
        if start_idx != -1:
            brace_count = 0
            for i in range(start_idx, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = response[start_idx:i+1]
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and 'improved_description' in parsed:
                                logger.debug("Successfully parsed JSON with brace matching and parameters")
                                improved_params = self._process_improved_parameters(parsed.get('improved_parameters'), tool)
                                return parsed['improved_description'], improved_params, parsed if return_full_json else None
                        except json.JSONDecodeError:
                            continue

        # Fallback: treat as plain text response (no parameter improvements)
        logger.info("No valid JSON found in response, treating as plain text description (no parameter improvements)")
        return cleaned_response, None, None

    def _process_improved_parameters(self, improved_params_dict: Optional[Dict[str, str]], tool: StandardizedTool) -> Optional[Dict[str, Any]]:
        """
        Process improved parameter descriptions and merge them with original parameter structure.

        Args:
            improved_params_dict: Dict mapping parameter names to improved descriptions
            tool: Original tool with parameter structure

        Returns:
            Updated parameters dict with improved descriptions, or None if no improvements
        """
        if not improved_params_dict or not tool.parameters:
            return None

        # Start with original parameters structure
        updated_params = dict(tool.parameters)

        # Update descriptions for parameters that have improvements
        for param_name, improved_desc in improved_params_dict.items():
            if param_name in updated_params and improved_desc:
                # Keep original structure but update description
                updated_params[param_name] = {
                    **updated_params[param_name],
                    'description': improved_desc
                }
                logger.debug(f"Updated parameter '{param_name}' description")

        return updated_params

    def _save_reasoning_json(self, tool: StandardizedTool, parsed_json: Dict[str, Any], 
                            context: ImprovementContext, dataset_name: str) -> Dict[str, Any]:
        """
        Save the full JSON response (with reasoning) to a file.
        
        Args:
            tool: The tool being improved
            parsed_json: The full parsed JSON response
            context: Improvement context
            dataset_name: Dataset name
            
        Returns:
            Dict with file path and metadata
        """
        try:
            # Use output_dir passed from pipeline (which is the timestamped run directory)
            if self.output_dir:
                output_dir = Path(self.output_dir) / "reasoning"
            else:
                # Fallback to default if not provided
                output_dir = Path("results") / dataset_name / "reasoning"
            
            # Create reasoning subdirectory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize tool name for filename
            safe_tool_name = re.sub(r'[^\w\-_]', '_', tool.name)
            filename = f"{safe_tool_name}_{timestamp}.json"
            file_path = output_dir / filename
            
            # Prepare data to save
            save_data = {
                "tool_name": tool.name,
                "timestamp": timestamp,
                "stage": context.stage.value,
                "dataset": dataset_name,
                "model": self.model_name,
                "prompt_file": self.prompt_template_path or "default",
                "original_description": tool.description,
                "improved_description": parsed_json.get('improved_description', ''),
                "reasoning": parsed_json.get('reasoning', '')
            }
            
            # Save to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved reasoning JSON for tool '{tool.name}' to: {file_path}")
            
            # Return only the content needed for metadata embedding (no file paths)
            return {
                "reasoning": parsed_json.get('reasoning', ''),
                "original_description": tool.description,
                "improved_description": parsed_json.get('improved_description', ''),
                "saved_at": timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to save reasoning JSON for tool '{tool.name}': {e}")
            return {
                "error": str(e)
            }
    
    def _load_prompt_template(self, stage: ImprovementStage, dataset_name: str) -> str:
        """Load prompt template from provided path."""

        # Use the provided template path
        prompt_path = Path(self.prompt_template_path)
        logger.debug(f"Using template path: {self.prompt_template_path}")
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt template not found: {prompt_path}. "
                f"Please ensure the prompt file exists for dataset '{dataset_name}' and stage '{stage.value}'"
            )
    
    def _format_prompt(self, template: str, tool: StandardizedTool, guidelines: str, server_config: Dict[str, Any]) -> str:
        """Format prompt template with tool information."""
        
        # Extract required and optional parameters
        required_params = []
        optional_params = []
        
        for param_name, param_def in (tool.parameters or {}).items():
            if isinstance(param_def, dict):
                param_type = param_def.get('type', 'unknown')
                param_desc = param_def.get('description', 'No description')
                param_required = param_def.get('required', False)
                param_default = param_def.get('default', None)

                # Enhanced parameter formatting with complete type information
                type_info = param_type
                if param_default is not None:
                    # Show all defaults, including empty strings (common in APIs)
                    type_info += f", default: {repr(param_default)}"

                # Build comprehensive parameter line
                param_line = f"- {param_name} ({type_info}): {param_desc}"

                # Parse required field more accurately (handle string 'true'/'false')
                is_required = self._is_parameter_required(param_required)

                if is_required:
                    required_params.append(param_line)
                else:
                    optional_params.append(param_line)
            else:
                # Simple parameter format
                required_params.append(f"- {param_name}: {param_def}")
        
        # Use string replacement instead of .format() to avoid issues with JSON braces in template
        formatted_prompt = template.replace("{tool_name}", tool.name)
        formatted_prompt = formatted_prompt.replace("{current_description}", tool.description)
        formatted_prompt = formatted_prompt.replace("{required_parameters}", 
                                                   "\n".join(required_params) if required_params else "None")
        formatted_prompt = formatted_prompt.replace("{optional_parameters}", 
                                                   "\n".join(optional_params) if optional_params else "None")
        formatted_prompt = formatted_prompt.replace("{guidelines}", guidelines)
        formatted_prompt = formatted_prompt.replace("{api_name}", server_config['name'])
        formatted_prompt = formatted_prompt.replace("{api_provider_description}", server_config['description'])
        return formatted_prompt

    def _is_parameter_required(self, required_value) -> bool:
        """
        Accurately determine if a parameter is required, handling various formats.

        Args:
            required_value: The 'required' field value (could be bool, str, etc.)

        Returns:
            bool: True if parameter is required
        """
        if isinstance(required_value, bool):
            return required_value
        elif isinstance(required_value, str):
            return required_value.lower() in ('true', '1', 'yes', 'required')
        else:
            return bool(required_value)

    def _call_llm(self, prompt: str) -> str:
        """Make an LLM API call via the OpenAI client."""
        try:
            messages = [{"role": "user", "content": prompt}]
            return llm_call(
                client=self.client,
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"Generic LLM call failed with model {self.model_name}: {str(e)}")


# Register the generic strategy
ImprovementStrategyRegistry.register("generic_llm_guidelines", GenericLLMStrategy, 
                                   "Generic LLM-based improvement with auto-dataset detection")
