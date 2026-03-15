"""
Confidence Calculator

Calculates confidence scores for decisions.
"""

from typing import Dict, Any, List
import statistics


class ConfidenceCalculator:
    """Calculates confidence scores for decisions."""
    
    def calculate_confidence(self, scores: Dict[str, float], context: Dict[str, Any]) -> float:
        """Calculate confidence in the decision."""
        if not scores:
            return 0.0
        
        # Calculate score distribution
        score_values = list(scores.values())
        max_score = max(score_values)
        second_max = sorted(score_values)[-2] if len(score_values) > 1 else 0.0
        
        # Confidence based on margin of victory
        margin = max_score - second_max
        base_confidence = min(margin * 2, 1.0)  # Scale to 0-1
        
        # Adjust based on data quality
        data_quality = context.get("data_quality", 0.7)
        
        # Adjust based on number of options
        option_count_factor = min(len(scores) / 5.0, 1.0)  # More options = higher confidence
        
        # Combine factors
        confidence = base_confidence * data_quality * option_count_factor
        
        return max(0.1, min(1.0, confidence))
    
    def calculate_option_confidence(self, option_score: float, all_scores: List[float]) -> float:
        """Calculate confidence for a specific option."""
        if not all_scores:
            return 0.0
        
        # Relative confidence based on percentile
        percentile = len([s for s in all_scores if s <= option_score]) / len(all_scores)
        
        # Adjust for score distribution
        if len(all_scores) > 1:
            std_dev = statistics.stdev(all_scores)
            if std_dev > 0:
                z_score = (option_score - statistics.mean(all_scores)) / std_dev
                confidence = 0.5 + (z_score * 0.1)  # Convert to 0-1 range
            else:
                confidence = 0.5  # All scores equal
        else:
            confidence = 0.8  # Only one option
        
        return max(0.0, min(1.0, confidence))
