"""
Tradeoff Analyzer

Analyzes tradeoffs between decision options.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re


@dataclass
class Tradeoff:
    """Represents a tradeoff between options."""
    dimension: str
    option1: str
    option2: str
    comparison: str  # "better", "worse", "equal"
    magnitude: float  # 0-1, how significant difference is


class TradeoffAnalyzer:
    """Analyzes tradeoffs between decision options."""
    
    def __init__(self):
        self.error_patterns = self._create_error_patterns()
        self.command_patterns = self._create_command_patterns()
    
    def analyze_tradeoffs(self, options: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tradeoffs between options."""
        tradeoffs = []
        
        # Compare each pair of options
        for i, option1 in enumerate(options):
            for j, option2 in enumerate(options[i+1:], i+1):
                pair_tradeoffs = self._compare_options(option1, option2)
                tradeoffs.extend(pair_tradeoffs)
        
        return {
            "tradeoffs": tradeoffs,
            "summary": self._summarize_tradeoffs(tradeoffs)
        }
    
    def _get_option_name(self, option: Any, default: str) -> str:
        """Safely extract option name from dataclass or dict."""
        if hasattr(option, 'name'):
            return option.name
        elif isinstance(option, dict):
            return option.get("name", default)
        else:
            return default
    
    def _get_option_score(self, option: Any, dimension: str) -> float:
        """Safely extract option score from dataclass or dict."""
        if hasattr(option, 'properties'):
            # Dataclass style - has properties attribute
            return option.properties.get(f"{dimension}_score", 0.5)
        elif isinstance(option, dict):
            # Dict style - check if it has the score directly
            if 'properties' in option:
                return option['properties'].get(f"{dimension}_score", 0.5)
            else:
                return option.get(f"{dimension}_score", 0.5)
        else:
            return 0.5
    
    def _compare_options(self, option1: Dict[str, Any], option2: Dict[str, Any]) -> List[Tradeoff]:
        """Compare two options across dimensions."""
        tradeoffs = []
        
        dimensions = ["cost", "performance", "security", "maintainability"]
        
        for dimension in dimensions:
            # Use helper functions for safe extraction
            score1 = self._get_option_score(option1, dimension)
            score2 = self._get_option_score(option2, dimension)
            option1_name = self._get_option_name(option1, "Option 1")
            option2_name = self._get_option_name(option2, "Option 2")
            
            if abs(score1 - score2) > 0.1:  # Significant difference
                if score1 > score2:
                    comparison = "better"
                    magnitude = score1 - score2
                else:
                    comparison = "worse"
                    magnitude = score2 - score1
                
                tradeoff = Tradeoff(
                    dimension=dimension,
                    option1=option1_name,
                    option2=option2_name,
                    comparison=comparison,
                    magnitude=magnitude
                )
                tradeoffs.append(tradeoff)
        
        return tradeoffs
    
    def _summarize_tradeoffs(self, tradeoffs: List[Tradeoff]) -> str:
        """Summarize tradeoff analysis."""
        if not tradeoffs:
            return "No significant tradeoffs identified."
        
        summary = f"Found {len(tradeoffs)} significant tradeoffs:\n"
        
        for tradeoff in tradeoffs[:5]:  # Top 5 tradeoffs
            summary += f"- {tradeoff.option1} is {tradeoff.comparison} than {tradeoff.option2} in {tradeoff.dimension}\n"
        
        return summary
    
    def _create_error_patterns(self) -> Dict[str, str]:
        """Create regex patterns for common errors."""
        return {
            "permission_denied": r"(permission denied|access denied).*?(?:<context>[\w/]+)?",
            "command_not_found": r"(command not found|not recognized).*?(?:<message>[\w\s]+)?",
            "network_error": r"(connection refused|timeout|unreachable).*?(?:<context>[\w\.:]+)?",
            "syntax_error": r"syntax error.*?(?:<context>[\w\s]+)?",
            "git_error": r"fatal:.*git|git.*fatal|error:.*git",
            "docker_error": r"(error|failed).*?(?:<message>.+)?",
            "npm_error": r"(ERR!|error).*?(?:<message>.+)?",
            "python_error": r"(?:<code>\w+Error):.*?(?:<message>.+)?"
        }
    
    def _create_command_patterns(self) -> Dict[str, str]:
        """Create regex patterns for command recognition."""
        return {
            "git": r"^git\s+",
            "docker": r"^docker\s+",
            "npm": r"^npm\s+",
            "pip": r"^pip\s+",
            "python": r"^(python|python3)\s+",
            "ssh": r"^ssh\s+",
            "curl": r"^curl\s+"
        }
    
    def _determine_severity(self, error_type: str) -> str:
        """Determine error severity."""
        high_severity = ["permission_denied", "command_not_found", "syntax_error"]
        medium_severity = ["network_error", "git_error", "docker_error"]
        
        if error_type in high_severity:
            return "error"
        elif error_type in medium_severity:
            return "warning"
        else:
            return "info"
    
    def extract_command_from_error(self, error_text: str) -> Optional[str]:
        """Extract the command that caused an error."""
        # First check if it's a git error
        parsed_error = self.parse_error(error_text)
        if parsed_error.error_type == "git_error":
            return "git"
        
        # Look for command patterns in error text
        for command, pattern in self.command_patterns.items():
            match = re.search(pattern, error_text)
            if match:
                return command
        
        return None
