"""
LLM Agent Package

The LLM Agent abstracts language model interactions.
"""

from .base import LLMBase
from .local_adapter import LocalAdapter
from .api_adapter import APIAdapter

__all__ = ["LLMBase", "LocalAdapter", "APIAdapter"]
