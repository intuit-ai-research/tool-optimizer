"""
Standardized pattern analysis interface with consistent data contracts.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .dataset_interface import StandardizedTool


class PatternSourceType(Enum):
    """Types of pattern data sources."""
    MCP_LOGS = "mcp_logs"           # MCP call history + question outcomes
    EVALUATION_METRICS = "metrics"  # Basic evaluation metrics only
    API_LOGS = "api_logs"          # API call logs
    USER_FEEDBACK = "feedback"     # User feedback data
    HYBRID = "hybrid"              # Multiple sources combined


@dataclass
class PatternAnalysisInput:
    """Standardized input for pattern analysis."""
    # Core data
    tools: List[StandardizedTool]
    evaluation_results: Dict[str, Any]
    
    # Optional context
    source_type: PatternSourceType = PatternSourceType.EVALUATION_METRICS
    dataset_name: Optional[str] = None
    analysis_params: Optional[Dict[str, Any]] = None
    
    def get_output_directory(self) -> Optional[str]:
        """Extract output directory from evaluation results if available."""
        return self.evaluation_results.get('output_directory')
    
    def get_metrics(self) -> Dict[str, Any]:
        """Extract metrics from evaluation results."""
        return {
            'accuracy': self.evaluation_results.get('accuracy', 0.0),
            'total_questions': self.evaluation_results.get('total_questions', 0),
            'execution_time': self.evaluation_results.get('execution_time', 0.0)
        }


@dataclass
class ToolPattern:
    """Standardized tool pattern representation with separate selection and usage patterns."""
    tool_name: str

    # Overall usage data
    total_calls: int = 0

    # Selection-related patterns (API choice accuracy)
    selection_positive_samples: List[Dict[str, Any]] = None
    selection_negative_samples: List[Dict[str, Any]] = None
    selection_success_rate: float = 0.0

    # Usage-related patterns (parameter correctness)
    usage_positive_samples: List[Dict[str, Any]] = None
    usage_negative_samples: List[Dict[str, Any]] = None
    usage_success_rate: float = 0.0

    # Legacy patterns (for backward compatibility)
    positive_samples: List[Dict[str, Any]] = None
    negative_samples: List[Dict[str, Any]] = None
    success_rate: float = 0.0
    success_indicators: List[str] = None
    common_errors: List[str] = None

    # Parameter patterns
    parameter_patterns: Dict[str, Any] = None

    # Extracted guidelines (if guideline extraction is enabled)
    selection_guidelines: List[str] = None
    usage_guidelines: List[str] = None

    # Analysis metadata
    data_source: PatternSourceType = PatternSourceType.EVALUATION_METRICS
    insights: str = ""
    
    def __post_init__(self):
        """Initialize empty lists/dicts if None."""
        # Initialize new selection/usage pattern fields
        if self.selection_positive_samples is None:
            self.selection_positive_samples = []
        if self.selection_negative_samples is None:
            self.selection_negative_samples = []
        if self.usage_positive_samples is None:
            self.usage_positive_samples = []
        if self.usage_negative_samples is None:
            self.usage_negative_samples = []

        # Initialize legacy fields for backward compatibility
        if self.positive_samples is None:
            self.positive_samples = []
        if self.success_indicators is None:
            self.success_indicators = []
        if self.negative_samples is None:
            self.negative_samples = []
        if self.common_errors is None:
            self.common_errors = []
        if self.parameter_patterns is None:
            self.parameter_patterns = {}
        if self.selection_guidelines is None:
            self.selection_guidelines = []
        if self.usage_guidelines is None:
            self.usage_guidelines = []

        # Populate legacy fields from new fields for compatibility
        if not self.positive_samples and self.usage_positive_samples:
            self.positive_samples = self.usage_positive_samples
        if not self.negative_samples and self.usage_negative_samples:
            self.negative_samples = self.usage_negative_samples
        if self.success_rate == 0.0 and self.usage_success_rate > 0.0:
            self.success_rate = self.usage_success_rate
    
    def has_positive_patterns(self) -> bool:
        """Check if tool has positive usage patterns."""
        return len(self.positive_samples) > 0 or len(self.success_indicators) > 0
    
    def has_negative_patterns(self) -> bool:
        """Check if tool has negative usage patterns."""
        return len(self.negative_samples) > 0 or len(self.common_errors) > 0
    
    def has_selection_patterns(self) -> bool:
        """Check if tool has API selection pattern data."""
        return len(self.selection_positive_samples) > 0 or len(self.selection_negative_samples) > 0

    def has_usage_patterns(self) -> bool:
        """Check if tool has API usage/parameter pattern data."""
        return len(self.usage_positive_samples) > 0 or len(self.usage_negative_samples) > 0

    def has_sufficient_data(self, min_samples: int = 2) -> bool:
        """Check if tool has sufficient data for reliable patterns."""
        return (self.has_positive_patterns() or self.has_negative_patterns()) and self.total_calls >= min_samples


@dataclass
class PatternAnalysisResult:
    """Standardized result from pattern analysis."""
    # Tool-specific patterns
    tool_patterns: Dict[str, ToolPattern]
    
    # Global analysis metadata
    total_tools_analyzed: int
    total_data_points: int
    data_source: PatternSourceType
    
    # Summary statistics
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize summary if None."""
        if self.summary is None:
            self.summary = self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary."""
        tools_with_positive = sum(1 for p in self.tool_patterns.values() if p.has_positive_patterns())
        tools_with_negative = sum(1 for p in self.tool_patterns.values() if p.has_negative_patterns())
        tools_with_sufficient_data = sum(1 for p in self.tool_patterns.values() if p.has_sufficient_data())
        
        return {
            'tools_with_positive_patterns': tools_with_positive,
            'tools_with_negative_patterns': tools_with_negative,
            'tools_with_sufficient_data': tools_with_sufficient_data,
            'avg_success_rate': sum(p.success_rate for p in self.tool_patterns.values()) / len(self.tool_patterns) if self.tool_patterns else 0.0,
            'data_coverage': tools_with_sufficient_data / self.total_tools_analyzed if self.total_tools_analyzed > 0 else 0.0
        }
    
    def get_tool_pattern(self, tool_name: str) -> Optional[ToolPattern]:
        """Get pattern for specific tool."""
        return self.tool_patterns.get(tool_name)
    
    def get_tools_with_patterns(self, min_calls: int = 2) -> List[ToolPattern]:
        """Get tools with reliable patterns based on actual data sufficiency."""
        return [
            pattern for pattern in self.tool_patterns.values()
            if pattern.has_sufficient_data(min_calls)
        ]


class PatternAnalyzer(ABC):
    """
    Abstract interface for analyzing usage patterns with standardized contracts.
    """
    
    @property
    @abstractmethod
    def supported_source_types(self) -> List[PatternSourceType]:
        """Return list of supported pattern source types."""
        pass
    
    @abstractmethod
    def analyze_usage_patterns(self, analysis_input: PatternAnalysisInput) -> PatternAnalysisResult:
        """
        Analyze usage patterns from standardized input.
        
        Args:
            analysis_input: Standardized pattern analysis input
            
        Returns:
            PatternAnalysisResult with standardized tool patterns
        """
        pass
    
    def extract_tool_insights(self, tool_name: str, 
                             analysis_result: PatternAnalysisResult) -> Dict[str, Any]:
        """
        Extract insights for specific tool from analysis result.
        
        Args:
            tool_name: Name of the tool
            analysis_result: Result from pattern analysis
            
        Returns:
            Tool-specific insights dictionary
        """
        tool_pattern = analysis_result.get_tool_pattern(tool_name)
        if not tool_pattern:
            return {
                "tool_name": tool_name,
                "has_patterns": False,
                "insights": "No usage data available for this tool"
            }
        
        return {
            "tool_name": tool_name,
            "has_patterns": True,
            "positive_samples": tool_pattern.positive_samples,
            "negative_samples": tool_pattern.negative_samples,
            "usage_stats": {
                "total_calls": tool_pattern.total_calls,
                "success_rate": tool_pattern.success_rate
            },
            "common_errors": tool_pattern.common_errors,
            "success_indicators": tool_pattern.success_indicators,
            "parameter_patterns": tool_pattern.parameter_patterns,
            "insights": tool_pattern.insights
        }
    
    def supports_source_type(self, source_type: PatternSourceType) -> bool:
        """Check if analyzer supports given source type."""
        return source_type in self.supported_source_types
