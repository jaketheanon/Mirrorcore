"""
Decision Agent Package

The Decision Agent provides reasoning support and tradeoff analysis.
"""

from .engine import DecisionEngine
from .tradeoffs import TradeoffAnalyzer
from .confidence import ConfidenceCalculator

__all__ = ["DecisionEngine", "TradeoffAnalyzer", "ConfidenceCalculator"]
