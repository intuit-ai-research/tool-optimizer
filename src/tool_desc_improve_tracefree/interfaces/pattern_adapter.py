"""
Adapter to bridge old PatternAnalyzer interface with new standardized interface.
This allows gradual migration while maintaining backward compatibility.
"""

from typing import Dict, Any, List
import logging

from .improvement_interface import PatternAnalyzer as LegacyPatternAnalyzer
from .pattern_interface import (
    PatternAnalyzer, PatternAnalysisInput, PatternAnalysisResult, 
    ToolPattern, PatternSourceType
)
from .dataset_interface import StandardizedTool

logger = logging.getLogger(__name__)


class LegacyPatternAnalyzerAdapter(PatternAnalyzer):
    """
    Adapter that wraps legacy PatternAnalyzer implementations to use new interface.
    """
    
    def __init__(self, legacy_analyzer: LegacyPatternAnalyzer):
        """
        Initialize adapter with legacy analyzer.
        
        Args:
            legacy_analyzer: Instance of old PatternAnalyzer interface
        """
        self.legacy_analyzer = legacy_analyzer
        self._supported_types = [PatternSourceType.EVALUATION_METRICS]  # Default assumption
        
        # Try to detect supported types from analyzer class name
        analyzer_name = legacy_analyzer.__class__.__name__.lower()
        if 'mcp' in analyzer_name:
            self._supported_types = [PatternSourceType.MCP_LOGS]
        elif 'api' in analyzer_name:
            self._supported_types = [PatternSourceType.API_LOGS]
    
    @property
    def supported_source_types(self) -> List[PatternSourceType]:
        """Return supported source types."""
        return self._supported_types
    
    def analyze_usage_patterns(self, analysis_input: PatternAnalysisInput) -> PatternAnalysisResult:
        """
        Analyze patterns using legacy analyzer and convert to new format.
        
        Args:
            analysis_input: Standardized input
            
        Returns:
            Standardized PatternAnalysisResult
        """
        try:
            # Call legacy analyzer with old interface
            legacy_result = self.legacy_analyzer.analyze_usage_patterns(
                analysis_input.evaluation_results, 
                analysis_input.tools
            )
            
            # Convert legacy result to new format
            return self._convert_legacy_result(legacy_result, analysis_input)
            
        except Exception as e:
            logger.error(f"Error in legacy analyzer adapter: {e}")
            # Return empty result on error
            return PatternAnalysisResult(
                tool_patterns={},
                total_tools_analyzed=len(analysis_input.tools),
                total_data_points=0,
                data_source=analysis_input.source_type
            )
    
    def _convert_legacy_result(self, legacy_result: Dict[str, Any], 
                              analysis_input: PatternAnalysisInput) -> PatternAnalysisResult:
        """Convert legacy result format to new standardized format."""
        
        tool_patterns = {}
        total_data_points = 0
        
        # Handle MCP-style results
        if 'tool_patterns' in legacy_result:
            mcp_tool_patterns = legacy_result['tool_patterns']
            total_data_points = legacy_result.get('total_mcp_calls', 0)
            
            for tool_name, tool_data in mcp_tool_patterns.items():
                tool_patterns[tool_name] = self._convert_mcp_tool_pattern(tool_name, tool_data)
        
        # Handle basic analyzer results  
        elif 'usage_frequency' in legacy_result:
            total_data_points = sum(legacy_result.get('usage_frequency', {}).values())
            
            for tool in analysis_input.tools:
                tool_patterns[tool.name] = self._convert_basic_tool_pattern(tool.name, legacy_result)
        
        return PatternAnalysisResult(
            tool_patterns=tool_patterns,
            total_tools_analyzed=len(analysis_input.tools),
            total_data_points=total_data_points,
            data_source=analysis_input.source_type
        )
    
    def _convert_mcp_tool_pattern(self, tool_name: str, mcp_data: Dict[str, Any]) -> ToolPattern:
        """Convert MCP tool data to ToolPattern."""
        usage_stats = mcp_data.get('usage_stats', {})
        

        
        return ToolPattern(
            tool_name=tool_name,
            total_calls=usage_stats.get('total_calls', 0),
            success_rate=usage_stats.get('success_rate', 0.0),
            positive_samples=mcp_data.get('positive_samples', []),
            success_indicators=mcp_data.get('success_indicators', []),
            negative_samples=mcp_data.get('negative_samples', []),
            common_errors=mcp_data.get('common_errors', []),
            parameter_patterns=mcp_data.get('parameter_patterns', {}),
            data_source=PatternSourceType.MCP_LOGS,
            insights=mcp_data.get('insights', '')
        )
    
    def _convert_basic_tool_pattern(self, tool_name: str, basic_data: Dict[str, Any]) -> ToolPattern:
        """Convert basic analyzer data to ToolPattern."""
        usage_freq = basic_data.get('usage_frequency', {}).get(tool_name, 0)
        
        return ToolPattern(
            tool_name=tool_name,
            total_calls=usage_freq,
            success_rate=0.5,  # Unknown, assume neutral
            positive_samples=[],
            success_indicators=basic_data.get('success_indicators', {}).get(tool_name, []),
            negative_samples=[],
            common_errors=basic_data.get('common_errors', []),
            parameter_patterns=basic_data.get('parameter_patterns', {}).get(tool_name, {}),
            data_source=PatternSourceType.EVALUATION_METRICS,
            insights=f"Basic usage frequency: {usage_freq} calls"
        )


class PatternAnalyzerFactory:
    """
    Factory for creating pattern analyzers with automatic adaptation.
    """
    
    @staticmethod
    def create_analyzer(analyzer_instance: LegacyPatternAnalyzer) -> PatternAnalyzer:
        """
        Create standardized analyzer from legacy instance.
        
        Args:
            analyzer_instance: Legacy analyzer instance
            
        Returns:
            Standardized PatternAnalyzer (possibly adapted)
        """
        # If it's already a new-style analyzer, return as-is
        if isinstance(analyzer_instance, PatternAnalyzer):
            return analyzer_instance
        
        # Otherwise, wrap with adapter
        return LegacyPatternAnalyzerAdapter(analyzer_instance)
    
    @staticmethod
    def create_analysis_input(evaluation_results: Dict[str, Any], 
                             tools: List[StandardizedTool],
                             source_type: PatternSourceType = None,
                             dataset_name: str = None) -> PatternAnalysisInput:
        """
        Create standardized analysis input from legacy parameters.
        
        Args:
            evaluation_results: Legacy evaluation results
            tools: Tools to analyze
            source_type: Type of data source (auto-detected if None)
            dataset_name: Name of dataset
            
        Returns:
            StandardizedPatternAnalysisInput
        """
        # Auto-detect source type if not provided
        if source_type is None:
            if evaluation_results.get('output_directory'):
                source_type = PatternSourceType.MCP_LOGS
            else:
                source_type = PatternSourceType.EVALUATION_METRICS
        
        return PatternAnalysisInput(
            tools=tools,
            evaluation_results=evaluation_results,
            source_type=source_type,
            dataset_name=dataset_name
        )
