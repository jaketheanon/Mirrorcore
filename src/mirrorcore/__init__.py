"""
Mirrorcore Package Initialization

This module initializes the Mirrorcore package and provides access to core functionality.
"""

__version__ = "0.1.0"
__author__ = "Mirrorcore Team"
__description__ = "A local-first reasoning assistant for personalized troubleshooting and decision support"

# Import core components for easy access
from .config import get_config, get_default_config
from .db.models import DatabaseManager
from .intake.questionnaire import get_assessment_definition
from .memory.retrieval import MemoryRetrieval
from .persona.voice import VoiceModel
from .terminal.parser import TerminalParser
from .decision.engine import DecisionEngine
from .llm.base import LLMBase

__all__ = [
    "get_config",
    "get_default_config", 
    "DatabaseManager",
    "get_assessment_definition",
    "MemoryRetrieval",
    "VoiceModel",
    "TerminalParser",
    "DecisionEngine",
    "LLMBase"
]


# Package execution entry point
def main():
    """Entry point for python -m mirrorcore execution."""
    from .main import main as cli_main
    cli_main()
