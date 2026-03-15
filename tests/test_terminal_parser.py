"""
Tests for Terminal Parser

Tests terminal command and error parsing functionality.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.terminal.parser import TerminalParser, ParsedCommand, ParsedError


class TestTerminalParser(unittest.TestCase):
    """Test the terminal parser."""
    
    def setUp(self):
        self.parser = TerminalParser()
    
    def test_initialization(self):
        """Test parser initialization."""
        self.assertIsNotNone(self.parser.error_patterns)
        self.assertIsNotNone(self.parser.command_patterns)
        self.assertIn("permission_denied", self.parser.error_patterns)
        self.assertIn("git", self.parser.command_patterns)
    
    def test_parse_simple_command(self):
        """Test parsing a simple command."""
        command = "ls -la /home"
        parsed = self.parser.parse_command(command)
        
        self.assertIsInstance(parsed, ParsedCommand)
        self.assertEqual(parsed.command, "ls")
        self.assertTrue(parsed.options.get("l"))  # Check for 'l' flag
        self.assertTrue(parsed.options.get("a"))  # Check for 'a' flag  
        self.assertIn("/home", parsed.arguments)
    
    def test_parse_command_with_long_options(self):
        """Test parsing command with long options."""
        command = "git commit --message='Initial commit' --all"
        parsed = self.parser.parse_command(command)
        
        self.assertEqual(parsed.command, "git")
        self.assertEqual(parsed.options["message"], "Initial commit")
        self.assertTrue(parsed.options["all"])
    
    def test_parse_permission_error(self):
        """Test parsing permission error."""
        error_text = "bash: /usr/bin/docker: Permission denied"
        parsed = self.parser.parse_error(error_text)
        
        self.assertIsInstance(parsed, ParsedError)
        self.assertEqual(parsed.error_type, "permission_denied")
        self.assertEqual(parsed.severity, "error")
    
    def test_parse_command_not_found_error(self):
        """Test parsing command not found error."""
        error_text = "bash: kubectl: command not found"
        parsed = self.parser.parse_error(error_text)
        
        self.assertIsInstance(parsed, ParsedError)
        self.assertEqual(parsed.error_type, "command_not_found")
        self.assertEqual(parsed.severity, "error")
    
    def test_parse_network_error(self):
        """Test parsing network error."""
        error_text = "curl: (7) Failed to connect to github.com port 443: Connection refused"
        parsed = self.parser.parse_error(error_text)
        
        self.assertIsInstance(parsed, ParsedError)
        self.assertEqual(parsed.error_type, "network_error")
        self.assertEqual(parsed.severity, "warning")
    
    def test_extract_command_from_error(self):
        """Test extracting command from error."""
        error_text = "fatal: 'origin' does not appear to be a git repository"
        command = self.parser.extract_command_from_error(error_text)
        self.assertEqual(command, "git")
    
    def test_parse_empty_command(self):
        """Test parsing empty command."""
        with self.assertRaises(ValueError):
            self.parser.parse_command("")
    
    def test_parse_unknown_error(self):
        """Test parsing unknown error."""
        error_text = "Some unknown error message"
        parsed = self.parser.parse_error(error_text)
        
        self.assertIsInstance(parsed, ParsedError)
        self.assertEqual(parsed.error_type, "unknown")
        self.assertEqual(parsed.severity, "error")


if __name__ == "__main__":
    unittest.main()
