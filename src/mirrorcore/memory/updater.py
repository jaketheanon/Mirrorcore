"""
Memory Updater

Updates memory entries based on new interactions and learning.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid


@dataclass
class MemoryUpdate:
    """Represents a memory update operation."""
    memory_id: str
    update_type: str
    new_data: Dict[str, Any]
    confidence_delta: float = 0.0


@dataclass
class PersonaRefinement:
    """Represents a suggested persona refinement."""
    trait_name: str
    current_value: float
    suggested_value: float
    confidence: float
    evidence_count: int
    reinforcing_signals: int
    conflicting_signals: int
    evidence_summary: str
    session_id: str
    timestamp: str


class ConservativePersonaUpdater:
    """Conservative persona profile updater based on conversation learning signals."""
    
    def __init__(self):
        self.min_signals_for_update = 3  # Require at least 3 signals
        self.majority_threshold = 0.6  # 60% majority for clear direction
        self.max_update_magnitude = 0.15  # Max 15% change per update
        self.signal_decay_days = 30  # Signals older than 30 days have less weight
        # Lazy initialize drift controller when needed
        self._drift_controller = None
    
    def evaluate_persona_updates(self, signals: List[Dict[str, Any]], 
                           current_persona: Dict[str, Any]) -> List[PersonaRefinement]:
        """Evaluate if persona refinements are warranted based on accumulated signals."""
        refinements = []
        
        # Group signals by trait
        signals_by_trait = self._group_signals_by_trait(signals)
        
        for trait_name, trait_signals in signals_by_trait.items():
            if len(trait_signals) < self.min_signals_for_update:
                continue  # Not enough evidence
            
            # Calculate signal direction balance
            positive_count = sum(1 for s in trait_signals if s.get("inferred_direction") == "positive")
            negative_count = sum(1 for s in trait_signals if s.get("inferred_direction") == "negative")
            total_signals = len(trait_signals)
            
            # Check if we have clear majority direction
            positive_ratio = positive_count / total_signals if total_signals > 0 else 0
            negative_ratio = negative_count / total_signals if total_signals > 0 else 0
            
            clear_direction = None
            if positive_ratio >= self.majority_threshold:
                clear_direction = "positive"
            elif negative_ratio >= self.majority_threshold:
                clear_direction = "negative"
            
            if not clear_direction:
                continue  # Mixed signals, prefer stability
            
            # Calculate suggested adjustment
            current_value = current_persona.get(trait_name, {}).get("value", 0.5)
            adjustment = self._calculate_adjustment(clear_direction, positive_ratio, negative_ratio)
            suggested_value = self._apply_conervative_bounds(current_value, adjustment)
            
            # Calculate confidence based on signal consistency and recency
            confidence = self._calculate_update_confidence(trait_signals)
            
            # Create refinement suggestion
            refinement = PersonaRefinement(
                trait_name=trait_name,
                current_value=current_value,
                suggested_value=suggested_value,
                confidence=confidence,
                evidence_count=total_signals,
                reinforcing_signals=max(positive_count, negative_count),
                conflicting_signals=min(positive_count, negative_count),
                evidence_summary=self._create_evidence_summary(trait_signals, clear_direction),
                session_id=trait_signals[0].get("session_id", "unknown"),
                timestamp=datetime.utcnow().isoformat()
            )
            
            refinements.append(refinement)
        
        return refinements
    
    def _group_signals_by_trait(self, signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group learning signals by trait type."""
        # Map signal types to persona traits
        signal_to_trait = {
            "directness_preference": "communication_preference",
            "evidence_preference": "evidence_threshold", 
            "ambiguity_tolerance": "ambiguity_tolerance",
            "action_bias": "action_bias",
            "trust_verification_style": "trust_verification_style",
            "self_reliance_level": "self_reliance_level",
            "communication_preference": "communication_preference",
            "uncertainty_response": "confidence_calibration"
        }
        
        grouped = {}
        
        for signal in signals:
            signal_type = signal.get("signal_type")
            trait_name = signal_to_trait.get(signal_type)
            
            if trait_name:
                if trait_name not in grouped:
                    grouped[trait_name] = []
                grouped[trait_name].append(signal)
        
        return grouped
    
    def _calculate_adjustment(self, direction: str, positive_ratio: float, negative_ratio: float) -> float:
        """Calculate adjustment magnitude based on signal strength."""
        if direction == "positive":
            # Move towards 1.0
            return min(positive_ratio * self.max_update_magnitude, self.max_update_magnitude)
        else:
            # Move towards 0.0
            return -min(negative_ratio * self.max_update_magnitude, self.max_update_magnitude)
    
    def _apply_conervative_bounds(self, current_value: float, adjustment: float) -> float:
        """Apply conservative bounds to ensure small, stable changes."""
        new_value = current_value + adjustment
        
        # Keep within 0-1 bounds
        new_value = max(0.0, min(1.0, new_value))
        
        # Ensure change is not too large
        max_change = self.max_update_magnitude
        if abs(new_value - current_value) > max_change:
            if adjustment > 0:
                new_value = current_value + max_change
            else:
                new_value = current_value - max_change
        
        return new_value
    
    def _calculate_update_confidence(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate confidence in suggested update based on signal quality."""
        if not signals:
            return 0.0
        
        # Base confidence on number of signals (more = better)
        signal_count_factor = min(len(signals) / 10.0, 1.0)  # Max at 10 signals
        
        # Factor in average signal confidence
        avg_signal_confidence = sum(s.get("confidence", 0.5) for s in signals) / len(signals)
        
        # Factor in recency (recent signals have more weight)
        recency_factor = self._calculate_recency_factor(signals)
        
        # Combined confidence
        confidence = (signal_count_factor * 0.4 + 
                     avg_signal_confidence * 0.4 + 
                     recency_factor * 0.2)
        
        return min(confidence, 0.8)  # Cap at 0.8 for learning signals
    
    def _calculate_recency_factor(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate recency factor for signals."""
        if not signals:
            return 0.0
        
        now = datetime.utcnow()
        recency_scores = []
        
        for signal in signals:
            try:
                signal_time = datetime.fromisoformat(signal.get("timestamp", ""))
                days_old = (now - signal_time).days
                
                # Recent signals (less than 7 days) get full weight
                if days_old <= 7:
                    recency_scores.append(1.0)
                # Signals up to 30 days get partial weight
                elif days_old <= self.signal_decay_days:
                    recency_scores.append(1.0 - (days_old / self.signal_decay_days))
                # Older signals get minimal weight
                else:
                    recency_scores.append(0.1)
            except:
                recency_scores.append(0.1)
        
        return sum(recency_scores) / len(recency_scores)
    
    def _create_evidence_summary(self, signals: List[Dict[str, Any]], direction: str) -> str:
        """Create a summary of evidence for the refinement."""
        reinforcing_count = sum(1 for s in signals if s.get("inferred_direction") == direction)
        conflicting_count = len(signals) - reinforcing_count
        
        if conflicting_count == 0:
            return f"All {reinforcing_count} signals indicate {direction} direction"
        else:
            return f"{reinforcing_count} signals support {direction} direction, {conflicting_count} conflict"
    
    def apply_refinement_safely(self, refinement: PersonaRefinement, 
                              db_store, confidence_threshold: float = 0.6) -> bool:
        """Apply a refinement if it meets safety criteria."""
        if refinement.confidence < confidence_threshold:
            return False
        
        # Additional safety: don't apply if change is too large
        change_magnitude = abs(refinement.suggested_value - refinement.current_value)
        if change_magnitude > self.max_update_magnitude:
            return False
        
        # Store refinement suggestion separately from main profile
        try:
            # Store as JSON for consistency
            refinement_data = {
                "type": "persona_refinement",
                "trait_name": refinement.trait_name,
                "current_value": refinement.current_value,
                "suggested_value": refinement.suggested_value,
                "confidence": refinement.confidence,
                "evidence_count": refinement.evidence_count,
                "evidence_summary": refinement.evidence_summary,
                "session_id": refinement.session_id,
                "applied": False
            }
            
            # Store as memory entry
            memory_id = db_store.add_memory_entry(
                memory_type="persona_refinement",
                content=json.dumps(refinement_data),  # JSON serialization
                source_module="learning_updater",
                confidence=refinement.confidence,
                tags=f"persona,refinement,{refinement.trait_name}"
            )
            
            print(f"✓ Stored persona refinement suggestion for {refinement.trait_name}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to store refinement: {e}")
            return False


class MemoryUpdater:
    """Updates memory entries based on new interactions and learning."""
    
    def __init__(self):
        self.persona_updater = ConservativePersonaUpdater()
        self.update_strategies = {
            "success_rate_update": self._create_success_rate_update,
            "confidence_update": self._update_confidence,
            "usage_increment": self._create_usage_update,
            "relationship_add": self._add_relationship
        }
    
    def evaluate_learning_signals(self, signals: List[Dict[str, Any]], 
                             current_persona: Dict[str, Any], db_store) -> List[PersonaRefinement]:
        """Evaluate persona refinements from learning signals."""
        return self.persona_updater.evaluate_persona_updates(signals, current_persona)
    
    def apply_conservative_updates(self, refinements: List[PersonaRefinement], 
                                db_store, auto_apply_threshold: float = 0.7) -> Tuple[int, int]:
        """Apply conservative persona updates."""
        applied_count = 0
        suggested_count = len(refinements)
        
        for refinement in refinements:
            # Only auto-apply high-confidence refinements
            if refinement.confidence >= auto_apply_threshold:
                if self.persona_updater.apply_refinement_safely(refinement, db_store, 0.5):
                    applied_count += 1
            else:
                # Store as suggestion for manual review
                self.persona_updater.apply_refinement_safely(refinement, db_store, 0.3)
        
        return applied_count, suggested_count
    
    def update_from_interaction(self, interaction: Dict[str, Any], memory_store) -> List[MemoryUpdate]:
        """Update memories based on a new interaction."""
        updates = []
        
        # Update related memories
        related_memories = interaction.get("related_memories", [])
        
        for memory_id in related_memories:
            memory = memory_store.search_memories("", limit=1000)  # Get all memories
            target_memory = next((m for m in memory if m.get("id") == memory_id), None)
            
            if target_memory:
                # Update usage count
                usage_update = self._create_usage_update(memory_id)
                updates.append(usage_update)
                
                # Update success rate based on interaction outcome
                if interaction.get("outcome") in ["success", "failure"]:
                    success_update = self._create_success_rate_update(memory_id, interaction)
                    updates.append(success_update)
        
        return updates
    
    def apply_updates(self, updates: List[MemoryUpdate], memory_store) -> bool:
        """Apply a batch of memory updates."""
        try:
            for update in updates:
                strategy = self.update_strategies.get(update.update_type)
                if strategy:
                    strategy(update, memory_store)
            
            return True
        except Exception as e:
            print(f"Failed to apply memory updates: {e}")
            return False
    
    def _create_usage_update(self, memory_id: str) -> MemoryUpdate:
        """Create a usage increment update."""
        return MemoryUpdate(
            memory_id=memory_id,
            update_type="usage_increment",
            new_data={"increment": 1}
        )
    
    def _create_success_rate_update(self, memory_id: str, interaction: Dict[str, Any]) -> MemoryUpdate:
        """Create a success rate update."""
        outcome = interaction.get("outcome", "unknown")
        
        return MemoryUpdate(
            memory_id=memory_id,
            update_type="success_rate_update",
            new_data={"outcome": outcome}
        )
    
    def _update_success_rate(self, update: MemoryUpdate, memory_store):
        """Update success rate for a memory."""
        # This would update the database
        # For now, it's a placeholder
        pass
    
    def _update_confidence(self, update: MemoryUpdate, memory_store):
        """Update confidence score for a memory."""
        # This would update the database
        # For now, it's a placeholder
        pass
    
    def _add_relationship(self, update: MemoryUpdate, memory_store):
        """Add relationship between memories."""
        # This would update the database
        # For now, it's a placeholder
        pass
    
    def update_pattern_from_usage(self, pattern_id: str, success: bool, memory_store) -> MemoryUpdate:
        """Update a pattern based on usage outcome."""
        update_data = {
            "usage_count": 1,  # Increment
            "last_used": datetime.utcnow().isoformat()
        }
        
        if success:
            update_data["success_increment"] = 1
        
        return MemoryUpdate(
            memory_id=pattern_id,
            update_type="pattern_usage_update",
            new_data=update_data
        )
    
    def evaluate_persona_drift(self, current_persona: Dict[str, Any], 
                           recent_signals: List[Dict[str, Any]], 
                           memory_store) -> List[Any]:
        """Evaluate persona drift using drift controller."""
        # Lazy import to avoid circular dependencies
        try:
            from ..persona.drift import PersonaDriftController, DriftEvaluation
        except ImportError:
            print("Warning: Drift controller not available")
            return []
        
        drift_evaluations = []
        drift_controller = PersonaDriftController()
        
        # Get baseline values from current persona
        for trait_name, trait_data in current_persona.items():
            if isinstance(trait_data, dict) and "value" in trait_data:
                baseline_value = float(trait_data["value"])
                baseline_confidence = float(trait_data.get("confidence", 0.5))
                
                # Filter signals for this trait
                trait_signals = [
                    signal for signal in recent_signals
                    if signal.get("trait_name") == trait_name
                ]
                
                if trait_signals:  # Only evaluate if we have signals
                    drift_eval = drift_controller.evaluate_trait_drift(
                        trait_name=trait_name,
                        baseline_value=baseline_value,
                        baseline_confidence=baseline_confidence,
                        recent_signals=trait_signals
                    )
                    drift_evaluations.append(drift_eval)
        
        return drift_evaluations
    
    def apply_drift_outcomes(self, drift_evaluations: List[Any], 
                           memory_store) -> Dict[str, Any]:
        """Apply drift evaluation outcomes to persona and memory."""
        # Lazy import to avoid circular dependencies
        try:
            from ..persona.drift import DriftEvaluation
        except ImportError:
            print("Warning: Drift evaluation not available")
            return {"traits_evaluated": 0, "actions_taken": {}}
        
        outcomes = {
            "traits_evaluated": len(drift_evaluations),
            "actions_taken": {},
            "confidence_changes": {},
            "value_adjustments": {},
            "revision_suggestions": []
        }
        
        for evaluation in drift_evaluations:
            trait_name = evaluation.trait_name
            
            # Store drift evaluation in memory
            self._store_drift_evaluation(evaluation, memory_store)
            
            # Apply outcomes based on recommended action
            if evaluation.recommended_action.value in ["retain", "retain_with_lower_confidence"]:
                outcomes["actions_taken"][trait_name] = evaluation.recommended_action.value
                outcomes["confidence_changes"][trait_name] = evaluation.confidence_adjustment
                
            elif evaluation.recommended_action.value == "apply_small_shift":
                if evaluation.value_adjustment is not None:
                    outcomes["actions_taken"][trait_name] = evaluation.recommended_action.value
                    outcomes["confidence_changes"][trait_name] = evaluation.confidence_adjustment
                    outcomes["value_adjustments"][trait_name] = evaluation.value_adjustment
                    
            elif evaluation.recommended_action.value in ["suggest_revision", "flag_for_review"]:
                outcomes["actions_taken"][trait_name] = evaluation.recommended_action.value
                outcomes["confidence_changes"][trait_name] = evaluation.confidence_adjustment
                if evaluation.revision_suggestion:
                    outcomes["revision_suggestions"].append({
                        "trait": trait_name,
                        "suggestion": evaluation.revision_suggestion,
                        "timestamp": evaluation.evaluation_timestamp
                    })
        
        return outcomes
    
    def _store_drift_evaluation(self, evaluation: Any, memory_store):
        """Store drift evaluation in memory for traceability."""
        import json
        
        evaluation_data = {
            "trait_name": evaluation.trait_name,
            "baseline_value": evaluation.baseline_value,
            "baseline_confidence": evaluation.baseline_confidence,
            "drift_pressure": evaluation.drift_pressure.drift_pressure,
            "evidence_strength": evaluation.drift_pressure.evidence_strength,
            "recommended_action": evaluation.recommended_action.value,
            "action_rationale": evaluation.action_rationale,
            "confidence_adjustment": evaluation.confidence_adjustment,
            "evaluation_timestamp": evaluation.evaluation_timestamp
        }
        
        # Store as memory entry
        memory_id = memory_store.add_memory_entry(
            memory_type="persona_drift_evaluation",
            content=json.dumps(evaluation_data),
            source_module="drift_controller",
            confidence=0.8,  # High confidence in evaluation process
            tags=f"persona,drift,{evaluation.trait_name}"
        )
        
        return memory_id
