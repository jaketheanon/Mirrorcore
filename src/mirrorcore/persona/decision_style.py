"""
Decision Style Model

Models user's decision-making patterns and preferences.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AnalysisApproach(Enum):
    DATA_DRIVEN = "data_driven"
    INTUITIVE = "intuitive"
    HYBRID = "hybrid"


@dataclass
class DecisionStyle:
    """User decision-making style profile."""
    risk_tolerance: RiskTolerance
    analysis_approach: AnalysisApproach
    time_horizon: str  # short_term, medium_term, long_term
    stakeholder_focus: str  # individual, team, system
    confidence_threshold: float


class DecisionStyleModel:
    """Models user decision-making patterns."""
    
    def __init__(self):
        self.current_style: Optional[DecisionStyle] = None
    
    def load_profile(self, profile_data: Dict[str, Any]) -> DecisionStyle:
        """Load decision style from profile data."""
        self.current_style = DecisionStyle(
            risk_tolerance=RiskTolerance(profile_data.get("risk_tolerance", "moderate")),
            analysis_approach=AnalysisApproach(profile_data.get("analysis_approach", "hybrid")),
            time_horizon=profile_data.get("time_horizon", "medium_term"),
            stakeholder_focus=profile_data.get("stakeholder_focus", "individual"),
            confidence_threshold=profile_data.get("confidence_threshold", 0.7)
        )
        return self.current_style
    
    def recommend_decision_process(self, decision_context: Dict[str, Any]) -> List[str]:
        """Recommend decision process based on user style."""
        if not self.current_style:
            return ["Analyze options", "Consider pros and cons", "Make decision"]
        
        process = []
        
        # Add analysis steps based on approach
        if self.current_style.analysis_approach == AnalysisApproach.DATA_DRIVEN:
            process.extend([
                "Gather quantitative data",
                "Analyze metrics and benchmarks",
                "Create data-driven comparison"
            ])
        elif self.current_style.analysis_approach == AnalysisApproach.INTUITIVE:
            process.extend([
                "Consider past experience",
                "Evaluate gut feelings",
                "Check intuition against known patterns"
            ])
        else:  # HYBRID
            process.extend([
                "Gather relevant data",
                "Consider intuitive insights",
                "Balance analysis with experience"
            ])
        
        # Add risk consideration based on tolerance
        if self.current_style.risk_tolerance == RiskTolerance.CONSERVATIVE:
            process.append("Evaluate risks and mitigation strategies")
            process.append("Choose safest viable option")
        elif self.current_style.risk_tolerance == RiskTolerance.AGGRESSIVE:
            process.append("Consider high-reward options")
            process.append("Plan for rapid iteration")
        else:  # MODERATE
            process.append("Balance risk and reward")
            process.append("Choose middle-ground approach")
        
        return process
    
    def should_recommend_option(self, option: Dict[str, Any]) -> bool:
        """Determine if option should be recommended based on risk tolerance."""
        if not self.current_style:
            return True
        
        risk_score = option.get("risk_score", 0.5)
        
        if self.current_style.risk_tolerance == RiskTolerance.CONSERVATIVE:
            return risk_score <= 0.4
        elif self.current_style.risk_tolerance == RiskTolerance.AGGRESSIVE:
            return risk_score >= 0.3
        else:  # MODERATE
            return 0.2 <= risk_score <= 0.8
