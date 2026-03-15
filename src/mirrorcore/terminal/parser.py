"""
Terminal Parser

Parses terminal commands and error messages.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
from dataclasses import dataclass


@dataclass
class ParsedCommand:
    """Represents a parsed terminal command."""
    command: str
    arguments: List[str]
    options: Dict[str, str]
    working_directory: str
    environment: Dict[str, str]


@dataclass
class ParsedError:
    """Represents a parsed error message."""
    error_type: str
    error_code: Optional[str]
    message: str
    context: str
    severity: str  # error, warning, info


class TerminalParser:
    """Parses terminal commands and error messages."""
    
    def __init__(self):
        self.error_patterns = self._create_error_patterns()
        self.command_patterns = self._create_command_patterns()
    
    def parse_command(self, command_line: str) -> ParsedCommand:
        """Parse a command line into components."""
        if not command_line.strip():
            raise ValueError("Empty command line")
        
        # Handle quoted values properly
        import shlex
        try:
            parts = shlex.split(command_line)
        except ValueError:
            # Fallback if shlex fails
            parts = command_line.split()
        
        if not parts:
            return ParsedCommand(
                command="",
                arguments=[],
                options={},
                working_directory="",
                environment={}
            )
        
        # Extract working directory
        working_dir = ""
        if len(parts) > 1 and parts[-1].startswith("cd "):
            working_dir = parts[-1][3:]  # Remove "cd "
            parts = parts[:-1]
        
        command = parts[0] if parts else ""
        arguments = []
        options = {}
        
        # Parse options and arguments
        i = 1
        while i < len(parts):
            part = parts[i]
            if part.startswith('--'):
                # Long option - handle quoted values
                if '=' in part:
                    key, value = part[2:].split('=', 1)
                    # Remove quotes if present
                    if value.startswith(('"', "'")) and len(value) > 1:
                        value = value[1:-1]
                    options[key] = value
                else:
                    key = part[2:]
                    if i + 1 < len(parts) and not parts[i + 1].startswith('-'):
                        next_part = parts[i + 1]
                        # Remove quotes from next part if present
                        if next_part.startswith(('"', "'")) and len(next_part) > 1:
                            next_part = next_part[1:-1]
                        options[key] = next_part
                        i += 1
                    else:
                        options[key] = True
            elif part.startswith('-') and len(part) > 1:
                # Combined flags (like -la = -l -a)
                # Split into individual flags
                for flag in part[1:]:
                    options[flag] = True
                # Don't increment i here - let the main loop handle it
            else:
                arguments.append(part)
            
            i += 1
        
        return ParsedCommand(
            command=command,
            arguments=arguments,
            options=options,
            working_directory=working_dir,
            environment={}  # Would be populated from actual environment
        )
    
    def parse_error(self, error_text: str) -> Optional[ParsedError]:
        """Parse an error message."""
        for error_type, pattern in self.error_patterns.items():
            match = re.search(pattern, error_text, re.IGNORECASE)
            if match:
                return ParsedError(
                    error_type=error_type,
                    error_code=match.groupdict().get('code'),
                    message=match.groupdict().get('message', error_text),
                    context=match.groupdict().get('context', ''),
                    severity=self._determine_severity(error_type)
                )
        
        # Fallback parsing
        return ParsedError(
            error_type="unknown",
            error_code=None,
            message=error_text,
            context="",
            severity="error"
        )
    
    def _create_error_patterns(self) -> Dict[str, str]:
        """Create regex patterns for common errors."""
        return {
            "permission_denied": r"(permission denied|access denied).*?(?:<context>[\w/]+)?",
            "command_not_found": r"(command not found|not recognized).*?(?:<message>[\w\s]+)?",
            "network_error": r"(connection refused|timeout|unreachable).*?(?:<context>[\w\.:]+)?",
            "syntax_error": r"syntax error.*?(?:<context>[\w\s]+)?",
            "git_error": r"fatal:.*git|git.*fatal|error:.*git",
            "docker_error": r"(error|failed).*?(?:<message>.+)",
            "npm_error": r"(ERR!|error).*?(?:<message>.+)",
            "python_error": r"(?:<code>\w+Error):.*?(?:<message>.+)"
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
        """Extract the command that caused the error."""
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
