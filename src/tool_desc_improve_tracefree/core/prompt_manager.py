"""
Prompt management for contrastive learning strategies.

This component handles loading and formatting prompts for different improvement approaches.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

from ..interfaces.dataset_interface import StandardizedTool

logger = logging.getLogger(__name__)


class ContrastivePromptManager:
    """
    Manages prompt loading and formatting for contrastive learning approaches.
    """
    
    def __init__(self, base_prompt_dir: str = "prompts"):
        """
        Initialize prompt manager.
        
        Args:
            base_prompt_dir: Base directory containing prompt templates
        """
        self.base_prompt_dir = Path(base_prompt_dir)
    
    def load_template(self, dataset_name: str, template_type: str = "data_dep") -> str:
        """
        Load contrastive prompt template for specific dataset.
        
        Args:
            dataset_name: Dataset name (e.g., "dbqa", "tmdb") - must have explicit prompt file
            template_type: Type of template ("data_dep", "data_indep")
            
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If the specific prompt file doesn't exist
        """
        # Require explicit dataset-specific prompt - no fallbacks
        prompt_path = self.base_prompt_dir / "datasets" / dataset_name / f"{template_type}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template required for dataset '{dataset_name}' not found. "
                f"Expected: {prompt_path}. "
                f"Each dataset must have its own explicit prompt file - no fallbacks allowed."
            )
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template_content = f.read().strip()
                if not template_content:
                    raise RuntimeError(f"Prompt template is empty: {prompt_path}")
                return template_content
        except Exception as e:
            raise RuntimeError(f"Error loading prompt template from {prompt_path}: {e}")
    
    def load_template_from_path(self, prompt_path: str) -> str:
        """
        Load prompt template from explicit file path.
        
        Args:
            prompt_path: Explicit path to prompt template file
            
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        path = Path(prompt_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found at: {path}")
        
        if not path.is_file():
            raise ValueError(f"Prompt path is not a file: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                template_content = f.read().strip()
                if not template_content:
                    raise RuntimeError(f"Prompt template is empty: {path}")
                return template_content
        except Exception as e:
            raise RuntimeError(f"Error loading prompt template from {path}: {e}")
    
    def format_prompt_with_patterns(self, template: str, tool: StandardizedTool, 
                                   guidelines: List[str], positive_patterns: str, 
                                   negative_patterns: str, error_patterns: List[str], 
                                   success_patterns: List[str]) -> str:
        """
        Format contrastive prompt template with tool and pattern information.
        
        Args:
            template: Prompt template string
            tool: Tool being improved
            guidelines: Context guidelines
            positive_patterns: Formatted positive usage patterns
            negative_patterns: Formatted negative usage patterns
            error_patterns: List of error patterns
            success_patterns: List of success patterns
            
        Returns:
            Formatted prompt string
        """
        # Format parameters
        required_params, optional_params = self._format_tool_parameters(tool)
        
        # Format pattern lists
        error_patterns_text = "\n".join([f"- {pattern}" for pattern in error_patterns]) if error_patterns else "No error patterns identified."
        success_patterns_text = "\n".join([f"- {pattern}" for pattern in success_patterns]) if success_patterns else "No success patterns identified."
        guidelines_text = "\n".join([f"- {guideline}" for guideline in guidelines]) if guidelines else "No specific guidelines provided."
        
        # Format the template
        formatted_prompt = template.format(
            tool_name=tool.name,
            current_description=tool.description,
            required_parameters=required_params,
            optional_parameters=optional_params,
            positive_patterns=positive_patterns,
            negative_patterns=negative_patterns,
            error_patterns=error_patterns_text,
            success_patterns=success_patterns_text,
            guidelines=guidelines_text
        )
        
        return formatted_prompt
    
    def format_prompt_with_guidelines(self, template: str, tool: StandardizedTool,
                                     guidelines: List[str], selection_guidelines: List[str],
                                     usage_guidelines: List[str]) -> str:
        """
        Format prompt template with extracted guidelines instead of raw patterns.

        Uses semantic variable names for clarity:
        - {selection_guidelines}: When to select this API (extracted from positive usage patterns)
        - {usage_guidelines}: How to use this API correctly (extracted from error patterns)

        Args:
            template: Prompt template string (should use semantic variable names)
            tool: Tool being improved
            guidelines: Context guidelines
            selection_guidelines: Extracted guidelines about when to select this API
            usage_guidelines: Extracted guidelines about how to use this API correctly

        Returns:
            Formatted prompt string
        """
        # Format parameters
        required_params, optional_params = self._format_tool_parameters(tool)
        
        # Format guidelines for prompt
        selection_text = "\n".join([f"- {g}" for g in selection_guidelines]) if selection_guidelines else "No selection guidelines extracted"
        usage_text = "\n".join([f"- {g}" for g in usage_guidelines]) if usage_guidelines else "No usage guidelines extracted"
        
        # Format the template with extracted guidelines using clear, semantic naming
        format_dict = {
            # Tool information
            'tool_name': tool.name,
            'api_name': tool.name,  # DBQA template compatibility
            'current_description': tool.description,

            # Parameters
            'required_parameters': required_params,
            'optional_parameters': optional_params,
            'required_parameters_text': required_params,  # DBQA template compatibility
            'optional_parameters_text': optional_params,  # DBQA template compatibility

            # Primary semantic naming (preferred for new templates)
            'selection_guidelines': selection_text,
            'usage_guidelines': usage_text,

            # Alternative semantic naming for compatibility
            'selection_guidelines_text': selection_text,
            'usage_guidelines_text': usage_text,

            # Legacy naming for backwards compatibility with old templates
            # TODO: Remove after all templates are updated to use semantic names
            'positive_patterns': selection_text,  # DEPRECATED: Use 'selection_guidelines' instead
            'negative_patterns': usage_text,      # DEPRECATED: Use 'usage_guidelines' instead

            # Context guidelines
            'guidelines': "\n".join(guidelines or []),

            # Pattern placeholders for templates that expect separate error/success patterns
            # (These would be populated by format_prompt_with_patterns method instead)
            'error_patterns': "No error patterns available in guidelines mode",
            'success_patterns': "No success patterns available in guidelines mode"
        }
        
        # Use safe formatting to handle template variables
        try:
            formatted_prompt = template.format(**format_dict)
        except KeyError as e:
            missing_key = str(e).strip("'")
            # Suggest better variable names for common deprecated ones
            suggestion = ""
            if missing_key == "positive_patterns":
                suggestion = " (Consider using 'selection_guidelines' instead for better clarity)"
            elif missing_key == "negative_patterns":
                suggestion = " (Consider using 'usage_guidelines' instead for better clarity)"

            raise ValueError(
                f"Template variable '{missing_key}' not found{suggestion}. "
                f"Available variables: {', '.join(format_dict.keys())}"
            )

        return formatted_prompt
    
    def load_pattern_analysis_prompt(self, prompt_type: str) -> str:
        """
        Load pattern analysis prompt template.
        
        Args:
            prompt_type: Type of pattern analysis prompt ("selection_guidelines", "usage_guidelines")
            
        Returns:
            Prompt template string
            
        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        prompt_path = self.base_prompt_dir / "pattern_analysis" / f"{prompt_type}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Pattern analysis prompt template not found: {prompt_path}. "
                f"Available types: selection_guidelines, usage_guidelines"
            )
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template_content = f.read().strip()
                if not template_content:
                    raise RuntimeError(f"Pattern analysis prompt template is empty: {prompt_path}")
                return template_content
        except Exception as e:
            raise RuntimeError(f"Error loading pattern analysis prompt from {prompt_path}: {e}")
    
    def load_system_prompt(self, prompt_type: str) -> str:
        """
        Load system prompt for pattern analysis.
        
        Args:
            prompt_type: Type of system prompt ("system_selection", "system_usage")
            
        Returns:
            System prompt string
            
        Raises:
            FileNotFoundError: If the system prompt file doesn't exist
        """
        prompt_path = self.base_prompt_dir / "pattern_analysis" / f"{prompt_type}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"System prompt template not found: {prompt_path}. "
                f"Available types: system_selection, system_usage"
            )
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template_content = f.read().strip()
                if not template_content:
                    raise RuntimeError(f"System prompt template is empty: {prompt_path}")
                return template_content
        except Exception as e:
            raise RuntimeError(f"Error loading system prompt from {prompt_path}: {e}")

    def format_pattern_analysis_prompt(self, template: str, tool_name: str,
                                     positive_patterns: str, negative_patterns: str,
                                     tool=None, existing_guidelines: List[str] = None, max_guidelines: int = None) -> str:
        """
        Format pattern analysis prompt with tool and pattern data.

        Args:
            template: Prompt template string
            tool_name: Name of the tool being analyzed
            positive_patterns: Formatted positive usage patterns
            negative_patterns: Formatted negative usage patterns
            tool: Optional StandardizedTool object for description and parameters
            existing_guidelines: Optional list of existing guidelines for iterative refinement
            max_guidelines: Maximum number of guidelines to generate (for constrained prompts, None for unconstrained)

        Returns:
            Formatted prompt string
        """
        # Build format dict with basic fields
        format_dict = {
            'tool_name': tool_name,
            'api_name': tool_name,  # Alias for compatibility
            'positive_patterns': positive_patterns,
            'negative_patterns': negative_patterns,
            'list of all positive_sample_api_selection': positive_patterns,  # Alias
            'list of all negative_sample_api_selection': negative_patterns,  # Alias
            'list of all positive_sample_api_usage': positive_patterns,  # Alias
            'list of all negative_sample_api_usage': negative_patterns,  # Alias
        }

        # Only add max_guidelines if provided (for constrained prompts)
        if max_guidelines is not None:
            format_dict['max_guidelines'] = max_guidelines
        
        # Add tool description and parameters if tool object provided
        if tool:
            format_dict['api_description'] = tool.description or "No description available"
            format_dict['current_description'] = tool.description or "No description available"
            
            # Format parameters
            required_params, optional_params = self._format_tool_parameters(tool)
            format_dict['api_parameters'] = f"Required: {required_params}\nOptional: {optional_params}"
            format_dict['required_parameters_text'] = required_params
            format_dict['optional_parameters_text'] = optional_params
        else:
            format_dict['api_description'] = "No description available"
            format_dict['current_description'] = "No description available"
            format_dict['api_parameters'] = "No parameters available"
            format_dict['required_parameters_text'] = "None"
            format_dict['optional_parameters_text'] = "None"
        
        # Add existing guidelines
        if existing_guidelines:
            guidelines_text = "\n".join([f"- {g}" for g in existing_guidelines])
        else:
            guidelines_text = "No existing guidelines (creating from scratch)"
        
        format_dict['existing_api_selection_guidelines'] = guidelines_text
        format_dict['existing_api_usage_guidelines'] = guidelines_text
        
        return template.format(**format_dict)

    def format_samples(self, samples: List[Dict[str, Any]], sample_type: str) -> str:
        """
        Format positive or negative samples for prompt.
        
        Args:
            samples: List of sample dictionaries
            sample_type: Type of samples ("positive" or "negative")
            
        Returns:
            Formatted samples string
        """
        if not samples:
            return f"No {sample_type} samples available."
        
        formatted_samples = []
        for i, sample in enumerate(samples[:5], 1):  # Limit to top 5 samples
            kwargs = sample.get("kwargs", {})
            result_preview = sample.get("result_preview", "")
            status = sample.get("status", "unknown")
            
            # Format kwargs
            kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            
            sample_text = f"Sample {i}: Called with parameters ({kwargs_str}), "
            sample_text += f"Status: {status}, Result: {result_preview[:100]}..."
            
            if sample_type == "negative" and sample.get("error"):
                sample_text += f", Error: {sample['error']}"
            
            formatted_samples.append(sample_text)
        
        return "\n".join(formatted_samples)
    
    def _format_tool_parameters(self, tool: StandardizedTool) -> tuple:
        """Format tool parameters for prompt inclusion."""
        required_params = []
        optional_params = []
        
        for param_name, param_def in (tool.parameters or {}).items():
            if isinstance(param_def, dict):
                param_type = param_def.get('type', 'unknown')
                param_desc = param_def.get('description', 'No description')
                param_required = param_def.get('required', False)
                
                param_line = f"- {param_name} ({param_type}): {param_desc}"
                
                if param_required:
                    required_params.append(param_line)
                else:
                    optional_params.append(param_line)
            else:
                # Simple parameter format
                required_params.append(f"- {param_name}: {param_def}")
        
        required_params_text = "\n".join(required_params) if required_params else "None"
        optional_params_text = "\n".join(optional_params) if optional_params else "None"
        
        return required_params_text, optional_params_text
