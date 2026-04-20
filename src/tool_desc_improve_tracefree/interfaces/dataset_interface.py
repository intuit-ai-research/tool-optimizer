"""
Abstract interfaces for dataset handling in description improvement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Iterator
from dataclasses import dataclass
from enum import Enum


class ToolFormat(Enum):
    """Supported tool configuration formats."""
    MCP_YAML = "mcp_yaml"
    OPENAPI = "openapi"  
    FUNCTION_CALLING = "function_calling"
    CUSTOM = "custom"


@dataclass
class StandardizedTool:
    """Universal tool representation across different formats."""
    name: str
    description: str
    parameters: Dict[str, Any]
    format_specific: Dict[str, Any] = None  # Format-specific fields
    metadata: Dict[str, Any] = None         # Improvement tracking
    
    def get_field(self, field_name: str, default=None):
        """Get field from any part of the tool definition."""
        if hasattr(self, field_name):
            return getattr(self, field_name)
        if self.format_specific and field_name in self.format_specific:
            return self.format_specific[field_name]
        if self.metadata and field_name in self.metadata:
            return self.metadata[field_name]
        return default


@dataclass
class DatasetQuery:
    """Universal query representation across different datasets."""
    query_id: str
    query_text: str
    expected_tools: List[str] = None        # Expected tool names
    expected_solution: List[str] = None     # Expected API sequence
    metadata: Dict[str, Any] = None         # Dataset-specific fields


class DatasetInterface(ABC):
    """Abstract interface for different dataset types."""
    
    @abstractmethod
    def load_tools(self, tools_path: str) -> List[StandardizedTool]:
        """Load tools from dataset-specific format."""
        pass
    
    @abstractmethod
    def save_tools(self, tools: List[StandardizedTool], output_path: str) -> str:
        """Save tools to dataset-specific format."""
        pass
    
    @abstractmethod
    def load_queries(self, queries_path: str) -> List[DatasetQuery]:
        """Load queries/questions from dataset."""
        pass
    
    @abstractmethod
    def get_tool_format(self) -> ToolFormat:
        """Return the tool format used by this dataset."""
        pass
    
    @abstractmethod
    def validate_tool(self, tool: StandardizedTool) -> bool:
        """Validate tool conforms to dataset requirements."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass


class EvaluationInterface(ABC):
    """Abstract interface for evaluation systems."""
    
    @abstractmethod
    def evaluate_tools(self, tools_config_path: str, queries: List[DatasetQuery]) -> Dict[str, float]:
        """Evaluate tools using dataset queries."""
        pass
    
    @abstractmethod
    def extract_usage_patterns(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Extract tool usage patterns from evaluation results."""
        pass
