"""
Terminal Fixer

Provides fixes for terminal issues.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .diagnostician import Diagnosis


@dataclass
class Fix:
    """Represents a fix for a terminal issue."""
    command: str
    description: str
    confidence: float
    side_effects: List[str]


class TerminalFixer:
    """Provides fixes for terminal issues."""
    
    def __init__(self):
        self.fix_templates = self._create_fix_templates()
    
    def generate_fixes(self, diagnosis: Diagnosis, context: Dict[str, Any]) -> List[Fix]:
        """Generate fixes based on diagnosis."""
        fixes = []
        
        for suggested_fix in diagnosis.suggested_fixes:
            fix = self._create_fix_from_suggestion(suggested_fix, diagnosis, context)
            if fix:
                fixes.append(fix)
        
        # Sort by confidence
        fixes.sort(key=lambda f: f.confidence, reverse=True)
        
        return fixes
    
    def _create_fix_from_suggestion(self, suggestion: str, diagnosis: Diagnosis, context: Dict[str, Any]) -> Optional[Fix]:
        """Create a fix from a suggestion string."""
        templates = self.fix_templates.get(diagnosis.issue_type, [])
        
        for template in templates:
            if suggestion in template["description"]:
                return Fix(
                    command=template["command"],
                    description=template["description"],
                    confidence=template["confidence"],
                    side_effects=template.get("side_effects", [])
                )
        
        # Generic fix
        return Fix(
            command=suggestion,
            description=suggestion,
            confidence=0.5,
            side_effects=[]
        )
    
    def _create_fix_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create fix templates for common issues."""
        return {
            "file_permission": [
                {
                    "command": "sudo {original_command}",
                    "description": "Use sudo to run with elevated privileges",
                    "confidence": 0.8,
                    "side_effects": ["Runs with root privileges", "May require password"]
                },
                {
                    "command": "chmod +x {file_path}",
                    "description": "Make file executable",
                    "confidence": 0.7,
                    "side_effects": ["Modifies file permissions"]
                }
            ],
            "missing_command": [
                {
                    "command": "which {command}",
                    "description": "Check if command exists",
                    "confidence": 0.9,
                    "side_effects": []
                },
                {
                    "command": "apt install {command}",
                    "description": "Install missing package",
                    "confidence": 0.6,
                    "side_effects": ["Requires internet", "Installs software"]
                }
            ]
        }
