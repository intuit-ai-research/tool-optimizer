#!/usr/bin/env python3
"""
Prompt Loader Utility

This module provides a clean interface for loading and formatting prompt templates
used in the description quality evaluation system.
"""

import os
from typing import Dict, Optional


class PromptLoader:
    """Utility class for loading and formatting prompt templates"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the prompt loader
        
        Args:
            prompts_dir: Directory containing prompt template files
        """
        self.prompts_dir = prompts_dir
        self.prompts_cache = {}  # Cache loaded prompts
        
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template from file
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            
        Returns:
            The prompt template as a string
        """
        if prompt_name in self.prompts_cache:
            return self.prompts_cache[prompt_name]
            
        prompt_file = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read().strip()
            
        self.prompts_cache[prompt_name] = prompt_template
        return prompt_template
    
    def format_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Load and format a prompt template with the given parameters
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            **kwargs: Parameters to substitute in the template
            
        Returns:
            The formatted prompt string
        """
        template = self.load_prompt(prompt_name)
        return template.format(**kwargs)
    
    def get_api_evaluation_judge_prompt(self, user_query: str, sub_task: str, 
                                      apis_with_descriptions: str, 
                                      previous_context: Optional[str] = None) -> str:
        """
        Get the API evaluation judge prompt with parameters
        
        Args:
            user_query: The original user query
            sub_task: The specific subtask being evaluated
            apis_with_descriptions: JSON string of APIs and their descriptions
            previous_context: Optional previous context information
            
        Returns:
            Formatted prompt for API evaluation judging
        """
        context_section = ""
        if previous_context:
            context_section = f"\nPrevious subtask results for context:\n{previous_context}"
            
        return self.format_prompt(
            "api_evaluation_judge",
            user_query=user_query,
            sub_task=sub_task,
            apis_with_descriptions=apis_with_descriptions,
            previous_context=context_section
        )
    
    def get_api_selection_prompt(self, query: str, tools_info: str, 
                               previous_context: Optional[str] = None) -> str:
        """
        Get the API selection prompt with parameters
        
        Args:
            query: The user query
            tools_info: JSON string of available tools and APIs
            previous_context: Optional previous context information
            
        Returns:
            Formatted prompt for API selection
        """
        if previous_context:
            return self.format_prompt(
                "api_selection_with_context",
                query=query,
                tools_info=tools_info,
                previous_context=previous_context
            )
        else:
            return self.format_prompt(
                "api_selection_simple",
                query=query,
                tools_info=tools_info
            )
    
    def get_single_api_selection_prompt(self, query: str, tools_info: str, 
                                       previous_context: Optional[str] = None,
                                       consider_provider_in_exact_match: bool = False) -> str:
        """
        Get the single API selection prompt with parameters for step-wise evaluation
        
        Args:
            query: The subtask query
            tools_info: JSON string of available tools and APIs
            previous_context: Optional previous context information
            
        Returns:
            Formatted prompt for selecting exactly one API
        """
        if previous_context:
            context_section = f"Previous Context (logs of previous subtasks):\n{previous_context}\n\n"
        else:
            context_section = ""

        if consider_provider_in_exact_match:
            return self.format_prompt(
                "api_selection_single_with_provider",
                query=query,
                tools_info=tools_info,
                context_section=context_section
            )
        else:
            return self.format_prompt(
                "api_selection_single",
                query=query,
                tools_info=tools_info,
                context_section=context_section
            )
    
    def get_task_decomposition_prompt(self, question: str, tools: str) -> str:
        """Get the task decomposition prompt"""
        return self.format_prompt("task_decomposition", question=question, tools=tools)
    
    def get_task_topology_prompt(self, question: str, task_list: str) -> str:
        """Get the task topology analysis prompt"""
        return self.format_prompt("task_topology", question=question, task_list=task_list)
    
    def get_parameter_generation_prompt(self, question: str, api_instruction: str) -> str:
        """Get the parameter generation prompt"""
        return self.format_prompt("parameter_generation", question=question, api_instruction=api_instruction)
    
    def get_parameter_generation_depend_prompt(self, question: str, api_instruction: str, previous_log: str) -> str:
        """Get the parameter generation with dependencies prompt"""
        return self.format_prompt("parameter_generation_depend", 
                                question=question, 
                                api_instruction=api_instruction, 
                                previous_log=previous_log)
    
    def get_answer_generation_prompt(self, question: str, call_result: str) -> str:
        """Get the answer generation prompt"""
        return self.format_prompt("answer_generation", question=question, call_result=call_result)
    
    def get_answer_generation_depend_prompt(self, question: str, call_result: str, previous_log: str) -> str:
        """Get the answer generation with dependencies prompt"""
        return self.format_prompt("answer_generation_depend", 
                                question=question, 
                                call_result=call_result, 
                                previous_log=previous_log)
    
    def get_answer_check_prompt(self, question: str, answer: str) -> str:
        """Get the answer check prompt"""
        return self.format_prompt("answer_check", question=question, answer=answer)
    
    def get_answer_summarize_prompt(self, question: str, answer_task: str) -> str:
        """Get the answer summarization prompt"""
        return self.format_prompt("answer_summarize", question=question, answer_task=answer_task)
    
    def get_tool_selection_prompt(self, question: str, tool_list: str) -> str:
        """Get the tool selection prompt"""
        return self.format_prompt("tool_selection", question=question, tool_list=tool_list)

    def list_available_prompts(self) -> list:
        """
        List all available prompt templates
        
        Returns:
            List of available prompt names (without .txt extension)
        """
        if not os.path.exists(self.prompts_dir):
            return []
            
        prompt_files = [f for f in os.listdir(self.prompts_dir) if f.endswith('.txt')]
        return [f[:-4] for f in prompt_files]  # Remove .txt extension


# Convenience function for creating a default prompt loader
def get_prompt_loader(prompts_dir: str = "prompts") -> PromptLoader:
    """
    Create a prompt loader instance
    
    Args:
        prompts_dir: Directory containing prompt templates
        
    Returns:
        PromptLoader instance
    """
    return PromptLoader(prompts_dir) 