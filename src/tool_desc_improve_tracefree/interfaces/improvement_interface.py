"""
Abstract interfaces for description improvement strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .dataset_interface import StandardizedTool, DatasetQuery


class ImprovementMethod(Enum):
    """Types of improvement methods."""
    LLM_BASED = "llm_based"
    RULE_BASED = "rule_based"
    RETRIEVAL_BASED = "retrieval_based"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class ImprovementStage(Enum):
    """Stages of improvement process."""
    DATA_INDEPENDENT = "data_independent"  # Using guidelines only
    DATA_DEPENDENT = "data_dependent"      # Using usage patterns
    ITERATIVE = "iterative"                # Multiple rounds
    FEEDBACK_BASED = "feedback_based"      # Using evaluation feedback
    CUSTOM = "custom"


@dataclass
class ImprovementContext:
    """Context information for improvement strategies."""
    tools: List[StandardizedTool]
    queries: List[DatasetQuery] = None
    guidelines: List[str] = None
    usage_patterns: Dict[str, Any] = None
    evaluation_results: Dict[str, Any] = None
    iteration_number: int = 0
    stage: ImprovementStage = ImprovementStage.DATA_INDEPENDENT
    custom_context: Dict[str, Any] = None
    server_config: Dict[str, Any] = None


@dataclass
class ImprovementResult:
    """Result of an improvement operation."""
    improved_tools: List[StandardizedTool]
    improvement_metadata: Dict[str, Any]
    success: bool = True
    error_message: str = None


class ImprovementStrategy(ABC):
    """Abstract base class for improvement strategies."""
    
    @abstractmethod
    def get_method_type(self) -> ImprovementMethod:
        """Return the improvement method type."""
        pass
    
    @abstractmethod
    def get_supported_stages(self) -> List[ImprovementStage]:
        """Return supported improvement stages."""
        pass
    
    @abstractmethod
    def improve_descriptions(self, context: ImprovementContext) -> ImprovementResult:
        """
        Improve tool descriptions based on context.
        
        Args:
            context: ImprovementContext with tools, guidelines, patterns, etc.
            
        Returns:
            ImprovementResult with improved tools and metadata.
        """
        pass
    
    @abstractmethod
    def can_handle_tools(self, tools: List[StandardizedTool]) -> bool:
        """Check if this strategy can handle the given tools."""
        pass
    
    def validate_context(self, context: ImprovementContext) -> bool:
        """Validate that context has required information for this strategy."""
        return True  # Default implementation
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name identifier."""
        pass

    # Common LLM response extraction methods
    def _try_all_extraction_methods(self, content: str, tool_name: str, llm_attempt: int = 1) -> str:
        """
        Try all extraction methods on the LLM response content.

        Args:
            content: Raw LLM response content
            tool_name: Name of tool being improved (for logging)
            llm_attempt: Which LLM attempt this is (for logging)

        Returns:
            Extracted description or empty string if all methods fail
        """
        import logging
        logger = logging.getLogger(__name__)

        extraction_methods = self._get_extraction_methods()

        for method_name, method_func in extraction_methods:
            try:
                logger.debug(f"Trying {method_name} for tool: {tool_name}")
                result = method_func(content)
                if result:
                    logger.debug(f"{method_name} succeeded for tool: {tool_name}")
                    return result
            except Exception as e:
                logger.debug(f"{method_name} failed for tool {tool_name}: {e}")
                continue

        logger.warning(f"All extraction methods failed for tool: {tool_name} (LLM attempt {llm_attempt})")
        return ""

    def _get_extraction_methods(self):
        """
        Get list of extraction methods to try. Can be overridden by subclasses.

        Returns:
            List of tuples: (method_name, method_function)
        """
        return [
            ("JSON parsing", self._extract_with_json_parsing),
            ("Regex extraction", self._extract_with_regex),
        ]

    def _extract_with_json_parsing(self, content: str) -> str:
        """Extract using standard JSON parsing with markdown cleanup."""
        import json

        cleaned_content = content.strip()

        # Handle markdown code blocks
        if cleaned_content.startswith('```json'):
            cleaned_content = cleaned_content[7:]
        elif cleaned_content.startswith('```'):
            cleaned_content = cleaned_content[3:]

        if cleaned_content.endswith('```'):
            cleaned_content = cleaned_content[:-3]

        cleaned_content = cleaned_content.strip()

        # Handle common JSON formatting issues
        if not cleaned_content.startswith('{'):
            # Look for JSON object in the response
            start_idx = cleaned_content.find('{')
            if start_idx != -1:
                cleaned_content = cleaned_content[start_idx:]

        result = json.loads(cleaned_content)
        return result.get("improved_description", "")

    def _extract_with_regex(self, content: str) -> str:
        """Extract using regex for malformed JSON."""
        import re

        desc_pattern = r'"improved_description"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        match = re.search(desc_pattern, content, re.DOTALL)
        if match:
            extracted_desc = match.group(1)
            # Unescape JSON characters
            extracted_desc = extracted_desc.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            return extracted_desc
        return ""


class GuidelinesProvider(ABC):
    """Abstract interface for providing improvement guidelines."""
    
    @abstractmethod
    def load_guidelines(self, guidelines_path: str, context: Dict[str, Any] = None) -> List[str]:
        """Load guidelines from source."""
        pass
    
    @abstractmethod
    def get_contextual_guidelines(self, tool: StandardizedTool, 
                                 context: ImprovementContext) -> List[str]:
        """Get guidelines specific to a tool and context."""
        pass


class PatternAnalyzer(ABC):
    """Abstract interface for analyzing usage patterns."""
    
    @abstractmethod
    def analyze_usage_patterns(self, evaluation_results: Dict[str, Any], 
                              tools: List[StandardizedTool]) -> Dict[str, Any]:
        """Analyze usage patterns from evaluation results."""
        pass
    
    @abstractmethod
    def extract_tool_insights(self, tool_name: str, 
                             patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights for specific tool."""
        pass
