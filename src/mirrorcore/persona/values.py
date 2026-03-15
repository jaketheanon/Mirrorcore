"""
Values Model

Models user's values and ethical considerations.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ValueDimension:
    """Represents a value dimension."""
    name: str
    weight: float  # 0-1, importance to user
    description: str


@dataclass
class ValuesProfile:
    """User values profile."""
    dimensions: List[ValueDimension]
    ethical_constraints: List[str]
    decision_criteria: List[str]


class ValuesModel:
    """Models user values and ethical considerations."""
    
    def __init__(self):
        self.current_profile: Optional[ValuesProfile] = None
        self.default_dimensions = self._create_default_dimensions()
    
    def _create_default_dimensions(self) -> List[ValueDimension]:
        """Create default value dimensions."""
        return [
            ValueDimension("security", 0.8, "Security and privacy considerations"),
            ValueDimension("performance", 0.7, "System performance and efficiency"),
            ValueDimension("maintainability", 0.6, "Code maintainability and readability"),
            ValueDimension("cost", 0.5, "Financial cost considerations"),
            ValueDimension("time_to_market", 0.4, "Speed of implementation"),
            ValueDimension("innovation", 0.3, "Innovation and modern approaches")
        ]
    
    def load_profile(self, profile_data: Dict[str, Any]) -> ValuesProfile:
        """Load values profile from data."""
        dimensions = []
        
        for dim_data in profile_data.get("dimensions", []):
            dimension = ValueDimension(
                name=dim_data.get("name"),
                weight=dim_data.get("weight", 0.5),
                description=dim_data.get("description", "")
            )
            dimensions.append(dimension)
        
        self.current_profile = ValuesProfile(
            dimensions=dimensions or self.default_dimensions,
            ethical_constraints=profile_data.get("ethical_constraints", []),
            decision_criteria=profile_data.get("decision_criteria", [])
        )
        
        return self.current_profile
    
    def evaluate_option(self, option: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate an option against user values."""
        if not self.current_profile:
            return {"overall_score": 0.5}
        
        scores = {}
        total_score = 0.0
        total_weight = 0.0
        
        for dimension in self.current_profile.dimensions:
            # Score this option against the dimension
            dimension_score = self._score_against_dimension(option, dimension)
            scores[dimension.name] = dimension_score
            
            total_score += dimension_score * dimension.weight
            total_weight += dimension.weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.5
        scores["overall_score"] = overall_score
        
        return scores
    
    def _score_against_dimension(self, option: Dict[str, Any], dimension: ValueDimension) -> float:
        """Score option against a specific value dimension."""
        # Simple scoring based on option properties
        option_properties = option.get("properties", {})
        
        if dimension.name == "security":
            return option_properties.get("security_score", 0.5)
        elif dimension.name == "performance":
            return option_properties.get("performance_score", 0.5)
        elif dimension.name == "maintainability":
            return option_properties.get("maintainability_score", 0.5)
        elif dimension.name == "cost":
            cost_score = option_properties.get("cost_score", 0.5)
            # Lower cost is better, so invert if needed
            return 1.0 - cost_score if cost_score > 0.5 else cost_score
        else:
            return 0.5  # Default score
