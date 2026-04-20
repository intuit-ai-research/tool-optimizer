"""
File-based guidelines provider for loading improvement guidelines from text files.
"""

import os
from typing import List, Dict, Any
from pathlib import Path

from ..interfaces.improvement_interface import GuidelinesProvider, ImprovementContext
from ..interfaces.dataset_interface import StandardizedTool
from ..core.registries import GuidelinesRegistry


class FileBasedGuidelinesProvider(GuidelinesProvider):
    """Guidelines provider that loads guidelines from text files."""
    
    def load_guidelines(self, guidelines_path: str, context: Dict[str, Any] = None) -> List[str]:
        """
        Load guidelines from text files in the specified directory.
        
        Args:
            guidelines_path: Path to directory containing .txt files with guidelines
            context: Optional context that may contain 'guidelines_file' to load specific file
            
        Returns:
            List of guideline strings
        """
        guidelines = []
        guidelines_dir = Path(guidelines_path)
        
        if not guidelines_dir.exists():
            print(f"Warning: Guidelines directory not found: {guidelines_dir}")
            return []
        
        # Check if a specific file is requested in context
        specific_file = context.get('guidelines_file') if context else None
        
        if specific_file:
            # Load only the specific file
            specific_path = guidelines_dir / specific_file
            if not specific_path.exists():
                print(f"Warning: Specific guidelines file not found: {specific_path}")
                return []
            
            try:
                with open(specific_path, 'r', encoding='utf-8') as f:
                    file_guidelines = [line.strip() for line in f if line.strip()]
                    guidelines.extend(file_guidelines)
            except Exception as e:
                print(f"Warning: Error loading {specific_path.name}: {e}")
        else:
            # Load all .txt files in the directory (original behavior)
            txt_files = list(guidelines_dir.glob("*.txt"))
            
            if not txt_files:
                print(f"Warning: No .txt files found in {guidelines_dir}")
                return []
            
            for txt_file in sorted(txt_files):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        file_guidelines = [line.strip() for line in f if line.strip()]
                        guidelines.extend(file_guidelines)
                except Exception as e:
                    print(f"Warning: Error loading {txt_file.name}: {e}")
        
        return guidelines
    
    def get_contextual_guidelines(self, tool: StandardizedTool, 
                                 context: ImprovementContext) -> List[str]:
        """
        Get guidelines specific to a tool and context.
        
        Args:
            tool: Tool to get guidelines for
            context: Improvement context
            
        Returns:
            List of contextual guidelines
        """
        # For file-based provider, return all loaded guidelines
        # Could be enhanced to filter based on tool type, domain, etc.
        return context.guidelines if context.guidelines else []


# Register the provider
GuidelinesRegistry.register("file_based", FileBasedGuidelinesProvider)
