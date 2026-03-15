"""
Log Parser

Parses raw terminal output, logs, and stack traces to extract meaningful error signals.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParsedLog:
    """Represents parsed log output with extracted signals."""
    original_text: str
    extracted_lines: List[str]
    likely_keywords: List[str]
    likely_error_category: str
    compact_summary_text: str
    context_lines: List[str]
    detected_command: Optional[str]  # Command that was detected
    detected_subsystem: Optional[str]  # Primary tool/subsystem
    command_context: Optional[str]  # Description of where command was detected
    output_lines: List[str]  # Lines identified as command output
    error_lines: List[str]  # Lines identified as error messages
    signal_families: List[str]  # New: detected signal families for conflict detection


class LogParser:
    """Parses raw terminal output to extract diagnostic signals."""
    
    def __init__(self):
        # Define error signal patterns
        self.error_patterns = {
            'traceback': [
                r'Traceback \(most recent call last\):',
                r'File ".*", line \d+',
                r'^\s+\d+\.',
                r'^\s+.*Error$',
                r'raise .*Error',
                r'raise .*Exception'
            ],
            'exception': [
                r'.*Exception.*',
                r'except .*:',
                r'raise .*Exception',
                r'Exception in thread',
                r'Unhandled exception'
            ],
            'error': [
                r'.*Error.*',
                r'ERROR.*',
                r'error:.*',
                r'Error:.*',
                r'FATAL.*',
                r'fatal:.*'
            ],
            'failed': [
                r'.*failed.*',
                r'FAILED.*',
                r'Failure:.*',
                r'failure:.*',
                r'Build failed',
                r'Deployment failed',
                r'Test failed'
            ],
            'permission_denied': [
                r'.*Permission denied.*',
                r'.*permission denied.*',
                r'Operation not permitted',
                r'Access denied',
                r'Unauthorized'
            ],
            'command_not_found': [
                r'.*command not found.*',
                r'.*Command not found.*',
                r'.*not found.*command',
                r'No such file or directory',
                r'is not recognized.*command'
            ],
            'connection_refused': [
                r'.*Connection refused.*',
                r'.*connection refused.*',
                r'Connection refused',
                r'connect: Connection refused'
            ],
            'timeout': [
                r'.*timeout.*',
                r'.*timed out.*',
                r'Connection timed out',
                r'Read timed out',
                r'Operation timed out'
            ],
            'dns': [
                r'.*DNS.*',
                r'.*Name or service not known.*',
                r'Could not resolve host',
                r'NXDOMAIN',
                r'DNS lookup failed'
            ],
            'module_not_found': [
                r'.*No module named.*',
                r'.*ModuleNotFoundError.*',
                r'ImportError: No module named',
                r'cannot import.*',
                r'Module not found'
            ],
            'import_error': [
                r'.*ImportError.*',
                r'import.*failed',
                r'Cannot import.*',
                r'Import error'
            ],
            'service_failed': [
                r'.*service.*failed.*',
                r'.*Service failed.*',
                r'Job for .* failed',
                r'Start: Job failed',
                r'Systemd.*failed',
                r'systemctl.*failed'
            ],
            'exited_with_code': [
                r'.*exited with code.*',
                r'.*Exited with status.*',
                r'exit code \d+',
                r'non-zero exit code',
                r'return code \d+'
            ],
            'segmentation_fault': [
                r'.*segmentation fault.*',
                r'.*Segmentation fault.*',
                r'Segmentation fault',
                r'core dumped'
            ],
            'syntax_error': [
                r'.*syntax error.*',
                r'.*SyntaxError.*',
                r'Syntax error near',
                r'Invalid syntax',
                r'Parse error'
            ],
            'parse_error': [
                r'.*parse error.*',
                r'.*ParseError.*',
                r'Could not parse',
                r'Invalid format',
                r'Malformed.*'
            ],
            # Evidence patterns for follow-up analysis
            'connection_confirmed': [
                r'.*connected.*',
                r'.*Connection successful.*',
                r'.*HTTP/1\.[01] 200.*',
                r'.*HTTP/2 200.*',
                r'.*200 OK.*',
                r'.*Transfer complete.*'
            ],
            'connection_refused': [
                r'.*Connection refused.*',
                r'.*connect.*refused.*',
                r'.*curl: \(7\) Failed to connect.*',
                r'.*nc: connect to.*refused.*',
                r'.*telnet: Unable to connect.*',
                r'.*No route to host.*'
            ],
            'timeout': [
                r'.*timeout.*',
                r'.*timed out.*',
                r'.*Connection timed out.*',
                r'.*Read timed out.*',
                r'.*Request timeout.*'
            ],
            'service_active': [
                r'.*Active: active.*',
                r'.*running.*',
                r'.*is running.*',
                r'.*started.*',
                r'.*UP.*',
                r'.*LISTEN.*'
            ],
            'service_inactive': [
                r'.*Active: inactive.*',
                r'.*stopped.*',
                r'.*not running.*',
                r'.*failed.*',
                r'.*DOWN.*',
                r'.*dead.*'
            ],
            'port_listening': [
                r'.*LISTEN.*',
                r'.*listening.*',
                r'.*Port.*open.*',
                r'.*bind.*success.*',
                r'.*Server listening.*'
            ],
            'port_not_listening': [
                r'.*No such file or directory.*',
                r'.*Address already in use.*',
                r'.*Port.*closed.*',
                r'.*Connection refused.*',
                r'.*bind.*failed.*'
            ],
            'config_valid': [
                r'.*config.*valid.*',
                r'.*syntax.*OK.*',
                r'.*Configuration.*valid.*',
                r'.*parsed.*successfully.*',
                r'.*Validation.*passed.*'
            ],
            'config_invalid': [
                r'.*config.*invalid.*',
                r'.*syntax.*error.*',
                r'.*Configuration.*invalid.*',
                r'.*failed.*parse.*',
                r'.*Validation.*failed.*'
            ],
            'permission_granted': [
                r'.*permission.*granted.*',
                r'.*Access.*granted.*',
                r'.*authorized.*',
                r'.*success.*permission.*',
                r'.*rw-.*',
                r'.*rwx.*'
            ],
            'permission_denied': [
                r'.*Permission denied.*',
                r'.*Access denied.*',
                r'.*unauthorized.*',
                r'.*Operation not permitted.*',
                r'.*read-only.*',
                r'.*permission.*denied.*'
            ],
            'module_present': [
                r'.*module.*found.*',
                r'.*installed.*',
                r'.*available.*',
                r'.*loaded.*',
                r'.*import.*success.*'
            ],
            'module_missing': [
                r'.*module.*not.*found.*',
                r'.*No module named.*',
                r'.*not installed.*',
                r'.*unavailable.*',
                r'.*cannot.*import.*'
            ]
        }
        
        # Define command patterns for detection
        self.command_patterns = {
            'pip': [
                r'^\s*pip\s+(install|uninstall|list|show|freeze|check|search)',
                r'^\s*pip3\s+(install|uninstall|list|show|freeze|check|search)',
                r'^\s*python\s+-m\s+pip\s+(install|uninstall|list|show|freeze|check|search)'
            ],
            'python': [
                r'^\s*python\s+[\w\-_\.\/\\]+\.(py|pyw)',
                r'^\s*python3\s+[\w\-_\.\/\\]+\.(py|pyw)',
                r'^\s*python\s+-c\s+',
                r'^\s*python3\s+-c\s+'
            ],
            'docker': [
                r'^\s*docker\s+(run|build|push|pull|logs|ps|stop|start|restart|exec|inspect)',
                r'^\s*docker\s+compose\s+(up|down|build|logs|ps|stop|start|restart)',
                r'^\s*docker\s+(container|image|volume|network)\s+'
            ],
            'systemctl': [
                r'^\s*systemctl\s+(start|stop|restart|status|enable|disable|reload)',
                r'^\s*sudo\s+systemctl\s+(start|stop|restart|status|enable|disable|reload)'
            ],
            'journalctl': [
                r'^\s*journalctl\s+',
                r'^\s*sudo\s+journalctl\s+'
            ],
            'git': [
                r'^\s*git\s+(clone|pull|push|commit|add|status|log|checkout|branch|merge|fetch)',
                r'^\s*git\s+remote\s+',
                r'^\s*git\s+config\s+'
            ],
            'npm': [
                r'^\s*npm\s+(install|uninstall|run|start|stop|test|build|publish)',
                r'^\s*npx\s+',
                r'^\s*node\s+[\w\-_\.\/\\]+\.(js|mjs)'
            ],
            'chmod': [
                r'^\s*chmod\s+',
                r'^\s*sudo\s+chmod\s+'
            ],
            'chown': [
                r'^\s*chown\s+',
                r'^\s*sudo\s+chown\s+'
            ],
            'sudo': [
                r'^\s*sudo\s+',
                r'^\s*doas\s+'
            ],
            'curl': [
                r'^\s*curl\s+',
                r'^\s*wget\s+'
            ],
            'ssh': [
                r'^\s*ssh\s+',
                r'^\s*scp\s+',
                r'^\s*sftp\s+'
            ]
        }
    
    def parse_log(self, raw_text: str) -> ParsedLog:
        """Parse raw log text to extract error signals and command context."""
        lines = raw_text.strip().split('\n')
        
        # Detect command context first
        command_info = self._detect_command_context(lines)
        
        # Separate command from output
        command_lines, output_lines = self._separate_command_from_output(lines, command_info)
        
        # Extract relevant lines from output
        extracted_lines = []
        likely_keywords = []
        error_lines = []
        
        for line in output_lines:
            line_stripped = line.strip()
            
            # Skip empty lines and obvious noise
            if not line_stripped or self._is_noise_line(line_stripped):
                continue
            
            # Check for error patterns
            matched_keywords = self._check_error_patterns(line_stripped)
            
            if matched_keywords:
                extracted_lines.append(line_stripped)
                likely_keywords.extend(matched_keywords)
                error_lines.append(line_stripped)
        
        # Determine likely error category
        likely_error_category = self._determine_error_category(likely_keywords)
        
        # Generate compact summary
        compact_summary = self._generate_summary(extracted_lines, likely_error_category)
        
        # Get context lines
        context_lines = self._get_context_lines(output_lines, extracted_lines)
        
        return ParsedLog(
            original_text=raw_text,
            extracted_lines=extracted_lines,
            likely_keywords=list(set(likely_keywords)),  # Remove duplicates
            likely_error_category=likely_error_category,
            compact_summary_text=compact_summary,
            context_lines=context_lines,
            detected_command=command_info.get('command'),
            detected_subsystem=command_info.get('subsystem'),
            command_context=command_info.get('context'),
            output_lines=output_lines,
            error_lines=error_lines,
            signal_families=self._detect_signal_families(raw_text)  # New: signal families for conflict detection
        )
    
    def _detect_command_context(self, lines: List[str]) -> Dict[str, Any]:
        """Detect command context from the input lines."""
        command_info = {
            'command': None,
            'subsystem': None,
            'context': None
        }
        
        # Look for command patterns in the first few lines
        for i, line in enumerate(lines[:5]):  # Check first 5 lines for commands
            line_stripped = line.strip()
            
            for subsystem, patterns in self.command_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        command_info['command'] = line_stripped
                        command_info['subsystem'] = subsystem
                        command_info['context'] = f"Line {i+1}: {subsystem} command detected"
                        return command_info
        
        return command_info
    
    def _separate_command_from_output(self, lines: List[str], command_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Separate command lines from output lines."""
        if not command_info.get('command'):
            # No command detected, treat all as output
            return [], lines
        
        command_found = False
        command_lines = []
        output_lines = []
        
        for line in lines:
            if line.strip() == command_info['command']:
                command_found = True
                command_lines.append(line)
            elif command_found:
                # After finding command, treat everything else as output
                output_lines.append(line)
            else:
                # Before finding command, treat as output (might be context)
                output_lines.append(line)
        
        return command_lines, output_lines
    
    def _is_noise_line(self, line: str) -> bool:
        """Check if line is obvious noise."""
        noise_patterns = [
            r'^$',  # Empty lines
            r'^\s*$',  # Whitespace only
            r'^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',  # Timestamps without content
            r'^\s*\[\d+\]\s*$',  # Just numbered list items
            r'^\s*[-=]{3,}\s*$',  # Just separators
            r'^\s*Process \d+ exited with code 0$',  # Successful exits
            r'^\s*INFO\s*:',  # Generic info logs
            r'^\s*DEBUG\s*:',  # Debug logs
            r'^\s*WARN\s*:',  # Warnings (not errors)
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _check_error_patterns(self, line: str) -> List[str]:
        """Check line against error patterns and return matching keywords."""
        matched_keywords = []
        
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    matched_keywords.append(category)
                    break  # Only count each category once per line
        
        return matched_keywords
    
    def parse_evidence(self, text: str) -> List[str]:
        """Parse follow-up evidence and return detected evidence signals."""
        evidence_signals = []
        lines = text.strip().split('\n')
        
        for line in lines:
            # Check against evidence patterns
            for category, patterns in self.error_patterns.items():
                # Only check evidence-related patterns
                if category in ['connection_confirmed', 'connection_refused', 'timeout', 
                              'service_active', 'service_inactive', 'port_listening', 
                              'port_not_listening', 'config_valid', 'config_invalid',
                              'permission_granted', 'permission_denied', 
                              'module_present', 'module_missing']:
                    
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            if category not in evidence_signals:
                                evidence_signals.append(category)
                            break
        
        return evidence_signals
    
    def _determine_error_category(self, keywords: List[str]) -> str:
        """Determine the most likely error category based on keywords."""
        if not keywords:
            return "unknown"
        
        # Priority mapping for more specific categories
        category_priority = {
            'traceback': 10,
            'segmentation_fault': 9,
            'syntax_error': 8,
            'permission_denied': 7,
            'command_not_found': 6,
            'module_not_found': 6,
            'import_error': 6,
            'service_failed': 5,
            'connection_refused': 5,
            'timeout': 4,
            'dns': 4,
            'exception': 3,
            'error': 2,
            'failed': 2,
            'parse_error': 1
        }
        
        # Sort by priority and frequency
        keyword_scores = {}
        for keyword in keywords:
            keyword_scores[keyword] = (
                category_priority.get(keyword, 0),
                keywords.count(keyword)  # Frequency as secondary sort key
            )
        
        # Return highest priority keyword
        if keyword_scores:
            return max(keyword_scores.items(), key=lambda x: (x[1][0], x[1][1]))[0]
        
        return "unknown"
    
    def _generate_summary(self, extracted_lines: List[str], error_category: str) -> str:
        """Generate compact summary of the parsed log."""
        if not extracted_lines:
            return "No clear error signals detected"
        
        # Take first few relevant lines for summary
        summary_lines = extracted_lines[:3]
        
        # Clean up lines for summary
        cleaned_lines = []
        for line in summary_lines:
            # Remove excessive whitespace
            cleaned = re.sub(r'\s+', ' ', line.strip())
            # Truncate very long lines
            if len(cleaned) > 100:
                cleaned = cleaned[:97] + "..."
            cleaned_lines.append(cleaned)
        
        if len(cleaned_lines) == 1:
            return f"Detected {error_category}: {cleaned_lines[0]}"
        elif len(cleaned_lines) == 2:
            return f"Detected {error_category}: {cleaned_lines[0]}; {cleaned_lines[1]}"
        else:
            return f"Detected {error_category}: {'; '.join(cleaned_lines)}"
    
    def _detect_signal_families(self, text: str) -> List[str]:
        """Detect signal families present in text."""
        text_lower = text.lower()
        detected_families = []
        
        for family, patterns in self.error_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                detected_families.append(family)
        
        return detected_families
    
    def _get_context_lines(self, all_lines: List[str], extracted_lines: List[str]) -> List[str]:
        """Get context lines around extracted error lines."""
        context_lines = []
        
        for error_line in extracted_lines[:5]:  # Limit to first 5 errors
            try:
                line_index = all_lines.index(error_line)
                
                # Get lines before and after (context window)
                start = max(0, line_index - 2)
                end = min(len(all_lines), line_index + 3)
                
                context = all_lines[start:end]
                context_lines.extend(context)
                
            except ValueError:
                # Line not found (shouldn't happen), just add the error line
                context_lines.append(error_line)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_context = []
        for line in context_lines:
            if line not in seen:
                seen.add(line)
                unique_context.append(line)
        
        return unique_context[:20]  # Limit total context
