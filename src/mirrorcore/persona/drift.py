"""
Persona Drift Controller

Manages persona trait drift evaluation and governance actions.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json


class DriftAction(Enum):
    """Actions for persona drift governance."""
    RETAIN = "retain"
    RETAIN_WITH_LOWER_CONFIDENCE = "retain_with_lower_confidence"
    SUGGEST_REVISION = "suggest_revision"
    APPLY_SMALL_SHIFT = "apply_small_shift"
    FLAG_FOR_REVIEW = "flag_for_review"


class EvidenceType(Enum):
    """Types of evidence for drift evaluation."""
    REINFORCING = "reinforcing"
    CONTRADICTORY = "contradictory"
    NEUTRAL = "neutral"


@dataclass
class DriftPressure:
    """Represents drift pressure on a persona trait."""
    trait_name: str
    baseline_value: float
    baseline_confidence: float
    
    # Recent evidence summary
    reinforcing_count: int
    contradictory_count: int
    neutral_count: int = 0
    recent_direction: str = "stable"  # "increasing", "decreasing", "stable"
    
    # Calculated metrics
    drift_pressure: float = 0.0  # -1 to 1, negative = contradictory pressure
    evidence_strength: float = 0.0  # 0 to 1, overall evidence strength
    recency_weight: float = 0.0  # 0 to 1, how recent is evidence
    
    def __post_init__(self):
        """Calculate drift metrics after initialization."""
        self._calculate_drift_pressure()
        self._calculate_evidence_strength()
        self._calculate_recency_weight()
    
    def _calculate_drift_pressure(self):
        """Calculate net drift pressure from evidence."""
        total_signals = self.reinforcing_count + self.contradictory_count + self.neutral_count
        if total_signals == 0:
            self.drift_pressure = 0.0
            return
        
        # Weight reinforcing vs contradictory
        reinforcing_weight = self.reinforcing_count / total_signals
        contradictory_weight = self.contradictory_count / total_signals
        
        # Calculate pressure (-1 to 1)
        self.drift_pressure = reinforcing_weight - contradictory_weight
        
        # Adjust for direction consistency
        if self.recent_direction == "stable":
            self.drift_pressure *= 0.5  # Reduce pressure if no clear direction
    
    def _calculate_evidence_strength(self):
        """Calculate overall evidence strength."""
        total_signals = self.reinforcing_count + self.contradictory_count + self.neutral_count
        if total_signals == 0:
            self.evidence_strength = 0.0
            return
        
        # More signals = stronger evidence
        signal_strength = min(total_signals / 10.0, 1.0)  # Cap at 10 signals
        
        # Adjust for consistency
        if self.contradictory_count > 0:
            consistency_penalty = self.contradictory_count / total_signals
            signal_strength *= (1.0 - consistency_penalty)
        
        self.evidence_strength = signal_strength
    
    def _calculate_recency_weight(self):
        """Calculate recency weight based on signal timing."""
        # This would be set from actual signal timestamps
        # For now, use a default moderate recency
        self.recency_weight = 0.7  # Moderate recency assumption


@dataclass
class DriftEvaluation:
    """Result of persona drift evaluation."""
    trait_name: str
    baseline_value: float
    baseline_confidence: float
    
    # Evidence summary
    drift_pressure: DriftPressure
    evaluation_timestamp: str
    
    # Governance decision
    recommended_action: DriftAction
    action_rationale: str
    
    # Outcomes
    confidence_adjustment: float  # Change to confidence (can be negative)
    value_adjustment: Optional[float] = None  # Small value shift if applied
    revision_suggestion: Optional[str] = None


class PersonaDriftController:
    """Controls persona trait drift based on observed evidence."""
    
    def __init__(self):
        self.drift_rules = self._initialize_drift_rules()
    
    def _initialize_drift_rules(self) -> Dict[str, Any]:
        """Initialize conservative drift governance rules."""
        return {
            "evidence_threshold": {
                "reinforcing_min": 3,  # Need 3+ reinforcing signals
                "contradiction_min": 2,  # 2+ contradictory signals trigger review
                "total_min": 5  # Need 5+ total signals for evaluation
            },
            "pressure_thresholds": {
                "high": 0.6,  # Strong pressure triggers action
                "medium": 0.3,  # Moderate pressure triggers consideration
                "low": 0.1  # Low pressure just monitoring
            },
            "adjustment_limits": {
                "max_confidence_change": 0.3,  # Max 30% confidence change
                "max_value_shift": 0.15,  # Max 15% value shift
                "min_confidence": 0.2  # Don't go below 20% confidence
            },
            "time_decay": {
                "half_life_days": 30,  # Evidence halves every 30 days
                "recent_days": 7  # Recent = within 7 days
            }
        }
    
    def evaluate_trait_drift(self, trait_name: str, baseline_value: float, 
                          baseline_confidence: float, recent_signals: List[Dict[str, Any]]) -> DriftEvaluation:
        """Evaluate drift for a single persona trait."""
        
        # Analyze recent signals
        signal_analysis = self._analyze_signals(recent_signals)
        
        # Create drift pressure object
        drift_pressure = DriftPressure(
            trait_name=trait_name,
            baseline_value=baseline_value,
            baseline_confidence=baseline_confidence,
            reinforcing_count=signal_analysis["reinforcing_count"],
            contradictory_count=signal_analysis["contradictory_count"],
            neutral_count=signal_analysis["neutral_count"],
            recent_direction=signal_analysis["recent_direction"]
        )
        
        # Determine governance action
        action, rationale = self._determine_action(drift_pressure, signal_analysis)
        
        # Calculate adjustments
        confidence_adj, value_adj = self._calculate_adjustments(
            action, baseline_confidence, drift_pressure
        )
        
        # Generate revision suggestion if needed
        revision_suggestion = self._generate_revision_suggestion(
            action, signal_analysis, drift_pressure
        )
        
        return DriftEvaluation(
            trait_name=trait_name,
            baseline_value=baseline_value,
            baseline_confidence=baseline_confidence,
            drift_pressure=drift_pressure,
            evaluation_timestamp=datetime.utcnow().isoformat(),
            recommended_action=action,
            action_rationale=rationale,
            confidence_adjustment=confidence_adj,
            value_adjustment=value_adj,
            revision_suggestion=revision_suggestion
        )
    
    def _analyze_signals(self, recent_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recent signals for a trait."""
        reinforcing_count = 0
        contradictory_count = 0
        neutral_count = 0
        
        # Track direction of value changes
        value_changes = []
        
        for signal in recent_signals:
            signal_type = signal.get("evidence_type", "neutral")
            
            if signal_type == "reinforcing":
                reinforcing_count += 1
                if "value_change" in signal:
                    value_changes.append(signal["value_change"])
            elif signal_type == "contradictory":
                contradictory_count += 1
                if "value_change" in signal:
                    value_changes.append(-signal["value_change"])  # Opposite direction
            else:
                neutral_count += 1
        
        # Determine recent direction
        if len(value_changes) > 0:
            avg_change = sum(value_changes) / len(value_changes)
            if avg_change > 0.05:  # 5% threshold
                recent_direction = "increasing"
            elif avg_change < -0.05:
                recent_direction = "decreasing"
            else:
                recent_direction = "stable"
        else:
            recent_direction = "stable"
        
        return {
            "reinforcing_count": reinforcing_count,
            "contradictory_count": contradictory_count,
            "neutral_count": neutral_count,
            "recent_direction": recent_direction,
            "value_changes": value_changes
        }
    
    def _determine_action(self, drift_pressure: DriftPressure, 
                       signal_analysis: Dict[str, Any]) -> Tuple[DriftAction, str]:
        """Determine governance action based on drift pressure."""
        rules = self.drift_rules
        
        # Check evidence thresholds
        total_signals = (signal_analysis["reinforcing_count"] + 
                        signal_analysis["contradictory_count"] + 
                        signal_analysis["neutral_count"])
        
        if total_signals < rules["evidence_threshold"]["total_min"]:
            return DriftAction.RETAIN, "Insufficient evidence for change"
        
        # High contradictory evidence
        if (signal_analysis["contradictory_count"] >= 
            rules["evidence_threshold"]["contradiction_min"]):
            if drift_pressure.drift_pressure < -0.3:
                return (DriftAction.FLAG_FOR_REVIEW, 
                       "Strong contradictory evidence detected")
            else:
                return (DriftAction.RETAIN_WITH_LOWER_CONFIDENCE,
                       "Mixed evidence, reducing confidence")
        
        # Strong reinforcing evidence
        if (signal_analysis["reinforcing_count"] >= 
            rules["evidence_threshold"]["reinforcing_min"]):
            
            if drift_pressure.drift_pressure > rules["pressure_thresholds"]["high"]:
                if drift_pressure.evidence_strength > 0.7:
                    return (DriftAction.APPLY_SMALL_SHIFT,
                           "Strong consistent evidence allows small adjustment")
                else:
                    return (DriftAction.SUGGEST_REVISION,
                           "Moderate evidence suggests review")
            
            elif drift_pressure.drift_pressure > rules["pressure_thresholds"]["medium"]:
                return (DriftAction.RETAIN_WITH_LOWER_CONFIDENCE,
                       "Some reinforcing evidence, strengthening confidence")
        
        # Default: retain current state
        return DriftAction.RETAIN, "No clear drift pressure detected"
    
    def _calculate_adjustments(self, action: DriftAction, 
                           baseline_confidence: float, 
                           drift_pressure: DriftPressure) -> Tuple[float, Optional[float]]:
        """Calculate confidence and value adjustments."""
        rules = self.drift_rules
        
        if action == DriftAction.RETAIN:
            return 0.0, None
        
        elif action == DriftAction.RETAIN_WITH_LOWER_CONFIDENCE:
            # Reduce confidence based on contradictions
            confidence_penalty = min(abs(drift_pressure.drift_pressure) * 0.2, 
                                  rules["adjustment_limits"]["max_confidence_change"])
            return -confidence_penalty, None
        
        elif action == DriftAction.APPLY_SMALL_SHIFT:
            # Small value adjustment proportional to drift pressure
            if abs(drift_pressure.drift_pressure) > 0.5:
                shift_magnitude = rules["adjustment_limits"]["max_value_shift"]
            else:
                shift_magnitude = rules["adjustment_limits"]["max_value_shift"] * 0.5
            
            # Small confidence increase for successful shift
            confidence_boost = 0.1
            return confidence_boost, drift_pressure.baseline_value * shift_magnitude
        
        elif action == DriftAction.SUGGEST_REVISION:
            # No automatic changes, just confidence adjustment
            return -0.1, None
        
        elif action == DriftAction.FLAG_FOR_REVIEW:
            # Significant confidence reduction
            return -0.3, None
        
        return 0.0, None
    
    def _generate_revision_suggestion(self, action: DriftAction, 
                                signal_analysis: Dict[str, Any],
                                drift_pressure: DriftPressure) -> Optional[str]:
        """Generate revision suggestion if needed."""
        if action not in [DriftAction.SUGGEST_REVISION, DriftAction.FLAG_FOR_REVIEW]:
            return None
        
        # Summarize evidence for suggestion
        total_evidence = (signal_analysis["reinforcing_count"] + 
                        signal_analysis["contradictory_count"] + 
                        signal_analysis["neutral_count"])
        
        suggestion_parts = [
            f"Trait '{drift_pressure.trait_name}' shows mixed evidence",
            f"Reinforcing signals: {signal_analysis['reinforcing_count']}",
            f"Contradictory signals: {signal_analysis['contradictory_count']}",
            f"Recent trend: {signal_analysis['recent_direction']}",
            f"Current value: {drift_pressure.baseline_value}"
        ]
        
        return " | ".join(suggestion_parts)
