"""
Decision Engine

Core decision-making engine for reasoning support.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .tradeoffs import TradeoffAnalyzer
from .confidence import ConfidenceCalculator


@dataclass
class DecisionOption:
    """Represents a decision option."""
    name: str
    description: str
    properties: Dict[str, Any]
    pros: List[str]
    cons: List[str]


@dataclass
class DecisionResult:
    """Result of a decision analysis."""
    recommended_option: str
    reasoning: str
    confidence: float
    alternatives: List[str]
    tradeoffs: Dict[str, Any]


class DecisionEngine:
    """Core decision-making engine."""
    
    def __init__(self):
        self.tradeoff_analyzer = TradeoffAnalyzer()
        self.confidence_calculator = ConfidenceCalculator()
    
    def analyze_decision(self, decision_context: Dict[str, Any], options: List[DecisionOption]) -> DecisionResult:
        """Analyze a decision with multiple options."""
        # Analyze tradeoffs
        tradeoffs = self.tradeoff_analyzer.analyze_tradeoffs(options, decision_context)
        
        # Calculate scores for each option
        scores = {}
        for option in options:
            score = self._calculate_option_score(option, decision_context, tradeoffs)
            scores[option.name] = score
        
        # Find best option
        best_option = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_option]
        
        # Calculate confidence
        confidence = self.confidence_calculator.calculate_confidence(scores, decision_context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_option, scores, tradeoffs)
        
        return DecisionResult(
            recommended_option=best_option,
            reasoning=reasoning,
            confidence=confidence,
            alternatives=[opt.name for opt in options if opt.name != best_option],
            tradeoffs=tradeoffs
        )
    
    def _calculate_option_score(self, option: DecisionOption, context: Dict[str, Any], tradeoffs: Dict[str, Any]) -> float:
        """Calculate score for an option."""
        score = 0.5  # Base score
        
        # Consider user values if available
        user_values = context.get("user_values", {})
        for value, weight in user_values.items():
            # Handle both dataclass and dict-style options
            if hasattr(option, 'properties'):
                property_score = option.properties.get(f"{value}_score", 0.5)
            else:
                property_score = option.get(f"{value}_score", 0.5)
            score += (property_score - 0.5) * weight
        
        # Consider pros and cons
        pros_weight = len(option.pros) * 0.1
        cons_weight = len(option.cons) * 0.1
        score += pros_weight - cons_weight
        
        return max(0.0, min(1.0, score))
    
    def _generate_reasoning(self, best_option: str, scores: Dict[str, float], tradeoffs: Dict[str, Any]) -> str:
        """Generate reasoning for the recommendation."""
        reasoning = f"Recommended option: {best_option}\n\n"
        reasoning += "This option scored highest based on:\n"
        
        # Add top scoring factors
        reasoning += "- Overall assessment score\n"
        reasoning += "- Balance of pros and cons\n"
        
        if tradeoffs:
            reasoning += "- Favorable tradeoff analysis\n"
        
        return reasoning
