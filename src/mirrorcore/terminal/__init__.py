"""
Terminal Agent Package

The Terminal Agent parses, diagnoses, and resolves command-line issues.
"""

from .parser import TerminalParser
from .diagnostician import TerminalDiagnostician
from .fixer import TerminalFixer
from .signatures import IncidentSignatureDetector, IncidentSignature

__all__ = ["TerminalParser", "TerminalDiagnostician", "TerminalFixer", "IncidentSignatureDetector", "IncidentSignature"]
