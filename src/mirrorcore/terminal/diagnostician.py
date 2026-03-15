"""
Terminal Diagnostician

Diagnoses terminal issues based on parsed errors and context.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .parser import ParsedError


@dataclass
class Diagnosis:
    """Represents a terminal issue diagnosis."""
    issue_type: str
    root_cause: str
    confidence: float
    suggested_fixes: List[str]
    related_commands: List[str]


class TerminalDiagnostician:
    """Diagnoses terminal issues."""
    
    def __init__(self):
        self.diagnostic_rules = self._create_diagnostic_rules()
    
    def diagnose(self, parsed_error: ParsedError, context: Dict[str, Any]) -> Diagnosis:
        """Diagnose a terminal error."""
        rules = self.diagnostic_rules.get(parsed_error.error_type, [])
        
        if not rules:
            # Generic diagnosis
            return Diagnosis(
                issue_type=parsed_error.error_type,
                root_cause="Unknown cause",
                confidence=0.3,
                suggested_fixes=["Check command syntax", "Verify permissions"],
                related_commands=[]
            )
        
        # Find best matching rule
        best_rule = max(rules, key=lambda r: self._rule_match_score(r, parsed_error, context))
        
        return Diagnosis(
            issue_type=best_rule["issue_type"],
            root_cause=best_rule["root_cause"],
            confidence=best_rule["confidence"],
            suggested_fixes=best_rule["suggested_fixes"],
            related_commands=best_rule.get("related_commands", [])
        )
    
    def _create_diagnostic_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create diagnostic rules for common issues."""
        return {
            "permission_denied": [
                {
                    "issue_type": "file_permission",
                    "root_cause": "Insufficient permissions to access file or directory",
                    "confidence": 0.9,
                    "suggested_fixes": [
                        "Use sudo to run with elevated privileges",
                        "Check file permissions with ls -la",
                        "Change ownership with chown",
                        "Modify permissions with chmod"
                    ],
                    "related_commands": ["sudo", "ls", "chmod", "chown"]
                }
            ],
            "command_not_found": [
                {
                    "issue_type": "missing_command",
                    "root_cause": "Command is not installed or not in PATH",
                    "confidence": 0.8,
                    "suggested_fixes": [
                        "Install the missing package",
                        "Check if command exists with which",
                        "Add command directory to PATH",
                        "Use full path to command"
                    ],
                    "related_commands": ["which", "whereis", "apt", "brew", "export"]
                }
            ],
            "network_error": [
                {
                    "issue_type": "connectivity",
                    "root_cause": "Network connectivity issue or service unavailable",
                    "confidence": 0.7,
                    "suggested_fixes": [
                        "Check network connection with ping",
                        "Verify service is running",
                        "Check firewall settings",
                        "Try different port or host"
                    ],
                    "related_commands": ["ping", "netstat", "telnet", "curl"]
                }
            ]
        }
    
    def _rule_match_score(self, rule: Dict[str, Any], error: ParsedError, context: Dict[str, Any]) -> float:
        """Calculate how well a rule matches the error."""
        score = rule["confidence"]
        
        # Adjust based on context matching
        if context.get("working_directory"):
            score += 0.1
        
        return score
