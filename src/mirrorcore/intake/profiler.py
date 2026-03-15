"""
Profiler

Analyzes assessment responses to generate explainable, evidence-backed persona traits.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .questionnaire import get_assessment_definition, Response, Question, QuestionType


class SignalType(Enum):
    """Types of signals that contribute to trait inference."""
    OPTION_SELECTION = "option_selection"
    CUSTOM_RESPONSE = "custom_response"
    EXPLANATION = "explanation"
    KEYWORD_THEME = "keyword_theme"


@dataclass
class TraitSignal:
    """Represents a single signal contributing to a trait inference."""
    signal_type: SignalType
    strength: float  # -1.0 to 1.0 (negative = opposing direction)
    source_question_id: str
    source_response_type: str  # selected_option, custom_response, explanation
    theme: Optional[str] = None  # Keyword theme if applicable
    raw_text: Optional[str] = None  # Original text for custom responses


@dataclass
class TraitEvidence:
    """Structured evidence for a trait inference."""
    trait_name: str
    value: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence_count: int
    contributing_question_ids: List[str]
    supporting_signals: List[str]  # Descriptive names of supporting signals
    conflict_signals: List[str]  # Descriptive names of conflicting signals
    evidence_summary: str
    raw_signals: List[TraitSignal]


@dataclass
class PersonaTrait:
    """Represents a single persona trait with evidence backing."""
    name: str
    value: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    evidence_count: int
    contributing_responses: List[str]
    evidence: TraitEvidence  # Rich evidence structure


class Profiler:
    """Analyzes assessment responses to generate baseline persona traits."""
    
    def __init__(self):
        self.definition = get_assessment_definition()
        self.trait_descriptions = self._create_trait_descriptions()
        self.keyword_themes = self._create_keyword_themes()
    
    def _create_keyword_themes(self) -> Dict[str, Dict[str, Any]]:
        """Create deterministic keyword themes for custom response scoring."""
        return {
            "verification_theme": {
                "keywords": ["verify", "check", "confirm", "test", "double-check", "validate", "ensure", "make sure"],
                "trait_impacts": {
                    "evidence_threshold": 0.7,
                    "confidence_calibration": 0.6,
                    "troubleshooting_style": 0.5
                },
                "direction": 1.0  # Positive direction
            },
            "speed_theme": {
                "keywords": ["fast", "quick", "immediate", "urgent", "rapid", "swift", "prompt", "right away"],
                "trait_impacts": {
                    "decision_speed": 0.8,
                    "speed_vs_accuracy_preference": 0.7,
                    "action_bias": 0.4
                },
                "direction": 1.0
            },
            "caution_theme": {
                "keywords": ["safe", "risk", "careful", "reversible", "conservative", "cautious", "wary", "hesitant"],
                "trait_impacts": {
                    "risk_tolerance": -0.8,
                    "decision_speed": -0.6,
                    "evidence_threshold": 0.5
                },
                "direction": -1.0  # Negative direction (risk-averse)
            },
            "evidence_theme": {
                "keywords": ["proof", "evidence", "certainty", "data", "research", "facts", "documentation"],
                "trait_impacts": {
                    "evidence_threshold": 0.9,
                    "trust_verification_style": 0.7,
                    "confidence_calibration": 0.5
                },
                "direction": 1.0
            },
            "intuition_theme": {
                "keywords": ["instinct", "gut", "feels right", "intuition", "feel", "sense", "hunch"],
                "trait_impacts": {
                    "confidence_calibration": -0.6,
                    "self_reliance_level": 0.7,
                    "evidence_threshold": -0.5
                },
                "direction": -1.0
            },
            "tool_theme": {
                "keywords": ["tool", "calculator", "docs", "search", "lookup", "reference", "automate"],
                "trait_impacts": {
                    "tool_dependence": 0.8,
                    "self_reliance_level": -0.3,
                    "confidence_calibration": 0.4
                },
                "direction": 1.0
            },
            "independence_theme": {
                "keywords": ["figure it out", "myself", "independent", "alone", "own", "solo", "without help"],
                "trait_impacts": {
                    "self_reliance_level": 0.9,
                    "trust_verification_style": -0.4,
                    "communication_preference": -0.3
                },
                "direction": 1.0
            },
            "collaboration_theme": {
                "keywords": ["team", "together", "help", "ask", "collaborate", "discuss", "others", "group"],
                "trait_impacts": {
                    "self_reliance_level": -0.8,
                    "communication_preference": 0.7,
                    "trust_verification_style": 0.3
                },
                "direction": -1.0
            }
        }
    
    def _score_custom_text(self, text: str, target_traits: List[str]) -> List[TraitSignal]:
        """Score custom text using deterministic keyword themes."""
        signals = []
        text_lower = text.lower()
        
        for theme_name, theme_config in self.keyword_themes.items():
            keyword_matches = [kw for kw in theme_config["keywords"] if kw in text_lower]
            
            if keyword_matches:
                # Create signals for each impacted trait
                for trait_name, impact_strength in theme_config["trait_impacts"].items():
                    if trait_name in target_traits:
                        # Apply theme direction and reduce strength for custom responses
                        adjusted_strength = impact_strength * theme_config["direction"] * 0.6
                        
                        signal = TraitSignal(
                            signal_type=SignalType.KEYWORD_THEME,
                            strength=adjusted_strength,
                            source_question_id="",  # Will be set by caller
                            source_response_type="custom_response",
                            theme=theme_name,
                            raw_text=text
                        )
                        signals.append(signal)
        
        return signals
    
    def _create_trait_descriptions(self) -> Dict[str, str]:
        """Create human-readable descriptions for traits."""
        return {
            "troubleshooting_style": "How you approach technical problems (0=quick fixes, 1=systematic analysis)",
            "risk_tolerance": "Comfort with uncertainty and risk (0=risk-averse, 1=risk-tolerant)",
            "ambiguity_tolerance": "Comfort with incomplete information (0=needs clarity, 1=comfortable with ambiguity)",
            "decision_speed": "How quickly you make decisions (0=deliberate, 1=rapid)",
            "evidence_threshold": "How much evidence you need before acting (0=low threshold, 1=high threshold)",
            "trust_verification_style": "How you evaluate advice and sources (0=trust authority, 1=verify independently)",
            "communication_preference": "Communication style preference (0=direct/technical, 1=emotional/collaborative)",
            "action_bias": "Tendency toward action vs analysis (0=analytical, 1=action-oriented)",
            "setback_response": "How you respond to failure (0=resilient/quick recovery, 1=cautious/analysis)",
            "priority_resolution_style": "How you resolve competing priorities (0=impact-focused, 1=context/people-focused)",
            "advice_filtering_style": "How you filter conflicting advice (0=empirical testing, 1=consensus/authority)",
            "self_reliance_level": "Reliance on self vs others (0=independent, 1=collaborative)",
            "confidence_calibration": "Accuracy of your self-confidence (0=well-calibrated, 1=overconfident)",
            "speed_vs_accuracy_preference": "Preference for speed vs accuracy (0=accuracy-focused, 1=speed-focused)",
            "tool_dependence": "Reliance on tools vs mental work (0=mental-first, 1=tool-first)"
        }
    
    def analyze_responses(self, responses: List[Response]) -> Dict[str, PersonaTrait]:
        """Convert assessment responses into evidence-backed persona traits."""
        # Initialize trait signal accumulators
        trait_signals = {trait: [] for trait in self.definition["target_traits"]}
        
        # Process each response
        for response in responses:
            self._process_response(response, trait_signals)
        
        # Convert signals to traits with evidence
        persona_traits = {}
        for trait_name, signals in trait_signals.items():
            if signals:
                persona_traits[trait_name] = self._create_trait_from_signals(trait_name, signals)
        
        return persona_traits
    
    def _process_response(self, response: Response, trait_signals: Dict[str, List[TraitSignal]]):
        """Process a single response and extract trait signals."""
        # Get the question definition
        question = self._get_question_by_id(response.question_id)
        if not question:
            return
        
        # Process selected option
        if response.selected_option_id:
            for option in question.answer_options:
                if option.id == response.selected_option_id:
                    for trait_name, strength in option.target_traits.items():
                        # Weight follow-up responses more heavily
                        weight = 1.5 if response.trigger_reason != "core_question" else 1.0
                        adjusted_strength = min(strength * weight, 1.0)
                        
                        signal = TraitSignal(
                            signal_type=SignalType.OPTION_SELECTION,
                            strength=adjusted_strength,
                            source_question_id=response.question_id,
                            source_response_type="selected_option",
                            raw_text=option.text
                        )
                        trait_signals[trait_name].append(signal)
                    break
        
        # Process custom response using keyword themes
        if response.custom_response:
            custom_signals = self._score_custom_text(response.custom_response, question.target_traits)
            for signal in custom_signals:
                signal.source_question_id = response.question_id
                # Apply signal to all relevant traits from this question
                for trait_name in question.target_traits:
                    if any(trait_name in theme["trait_impacts"] and signal.theme == theme_name 
                           for theme_name, theme in self.keyword_themes.items()):
                        trait_signals[trait_name].append(signal)
        
        # Process explanation using keyword themes (lower weight)
        if response.explanation:
            explanation_signals = self._score_custom_text(response.explanation, question.target_traits)
            for signal in explanation_signals:
                signal.source_question_id = response.question_id
                signal.source_response_type = "explanation"
                signal.strength *= 0.5  # Lower weight for explanations
                # Apply signal to all relevant traits from this question
                for trait_name in question.target_traits:
                    if any(trait_name in theme["trait_impacts"] and signal.theme == theme_name 
                           for theme_name, theme in self.keyword_themes.items()):
                        trait_signals[trait_name].append(signal)
    
    def _get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Get question definition by ID."""
        return self.core_questions.get(question_id) or self.followup_questions.get(question_id)
    
    @property
    def core_questions(self) -> Dict[str, Question]:
        """Get core questions dictionary."""
        return {q.id: q for q in self.definition["core_questions"]}
    
    @property
    def followup_questions(self) -> Dict[str, Question]:
        """Get follow-up questions dictionary."""
        return {q.id: q for q in self.definition["followup_questions"]}
    
    def _create_trait_from_signals(self, trait_name: str, signals: List[TraitSignal]) -> PersonaTrait:
        """Create a PersonaTrait from accumulated signals with evidence tracking."""
        if not signals:
            return PersonaTrait(
                name=trait_name,
                value=0.5,  # Default middle value
                confidence=0.0,
                evidence_count=0,
                contributing_responses=[],
                evidence=self._create_empty_evidence(trait_name)
            )
        
        # Separate positive and negative signals
        positive_signals = [s for s in signals if s.strength > 0]
        negative_signals = [s for s in signals if s.strength < 0]
        
        # Calculate weighted average value
        total_weight = sum(abs(s.strength) for s in signals)
        if total_weight > 0:
            weighted_sum = sum(s.strength for s in signals)
            # Convert from -1..1 range to 0..1 range
            raw_value = (weighted_sum / total_weight + 1) / 2
            final_value = max(0.0, min(1.0, raw_value))
        else:
            final_value = 0.5
        
        # Calculate confidence based on evidence count and consistency
        evidence_count = len(signals)
        base_confidence = min(evidence_count * 0.12, 0.75)  # Conservative confidence growth
        
        # Check for conflicts and adjust confidence
        conflict_ratio = len(negative_signals) / len(signals) if signals else 0
        conflict_penalty = conflict_ratio * 0.3  # Reduce confidence for conflicting signals
        
        # Consistency bonus for aligned signals
        if conflict_ratio < 0.2:  # Mostly consistent signals
            consistency_bonus = 0.15
        else:
            consistency_bonus = 0.0
        
        final_confidence = max(0.1, min(0.9, base_confidence - conflict_penalty + consistency_bonus))
        
        # Create evidence summary
        supporting_signals = []
        conflict_signals = []
        
        for signal in signals:
            signal_desc = self._describe_signal(signal)
            if signal.strength > 0:
                supporting_signals.append(signal_desc)
            else:
                conflict_signals.append(signal_desc)
        
        evidence_summary = self._create_evidence_summary(final_value, supporting_signals, conflict_signals)
        
        # Create evidence structure
        evidence = TraitEvidence(
            trait_name=trait_name,
            value=final_value,
            confidence=final_confidence,
            evidence_count=evidence_count,
            contributing_question_ids=list(set(s.source_question_id for s in signals)),
            supporting_signals=supporting_signals,
            conflict_signals=conflict_signals,
            evidence_summary=evidence_summary,
            raw_signals=signals
        )
        
        return PersonaTrait(
            name=trait_name,
            value=final_value,
            confidence=final_confidence,
            evidence_count=evidence_count,
            contributing_responses=list(set(s.source_question_id for s in signals)),
            evidence=evidence
        )
    
    def _create_empty_evidence(self, trait_name: str) -> TraitEvidence:
        """Create empty evidence structure for traits with no data."""
        return TraitEvidence(
            trait_name=trait_name,
            value=0.5,
            confidence=0.0,
            evidence_count=0,
            contributing_question_ids=[],
            supporting_signals=[],
            conflict_signals=[],
            evidence_summary="No evidence collected for this trait.",
            raw_signals=[]
        )
    
    def _describe_signal(self, signal: TraitSignal) -> str:
        """Create a human-readable description of a signal."""
        if signal.signal_type == SignalType.OPTION_SELECTION:
            return f"Selected option: {signal.raw_text[:50]}..."
        elif signal.signal_type == SignalType.KEYWORD_THEME:
            return f"Keyword theme: {signal.theme.replace('_', ' ')}"
        elif signal.signal_type == SignalType.CUSTOM_RESPONSE:
            return f"Custom response: {signal.raw_text[:30]}..."
        elif signal.signal_type == SignalType.EXPLANATION:
            return f"Explanation: {signal.raw_text[:30]}..."
        else:
            return f"Signal: {signal.signal_type.value}"
    
    def _create_evidence_summary(self, value: float, supporting: List[str], conflicts: List[str]) -> str:
        """Create a concise evidence summary."""
        if not supporting and not conflicts:
            return "No clear evidence pattern detected."
        
        if value > 0.7:
            direction = "strongly prefers"
        elif value > 0.6:
            direction = "moderately prefers"
        elif value < 0.3:
            direction = "strongly avoids"
        elif value < 0.4:
            direction = "moderately avoids"
        else:
            direction = "has mixed preferences for"
        
        trait_name = "this trait"  # Generic since we don't have the trait context here
        
        if conflicts:
            if len(conflicts) > len(supporting):
                summary = f"User shows conflicting signals, with {len(conflicts)} conflicting indicators vs {len(supporting)} supporting ones."
            else:
                summary = f"User {direction} {trait_name} despite {len(conflicts)} conflicting signals."
        else:
            summary = f"User consistently {direction} {trait_name} based on {len(supporting)} supporting signals."
        
        return summary
    
    def generate_summary(self, persona_traits: Dict[str, PersonaTrait]) -> str:
        """Generate a human-readable summary of the persona profile."""
        summary_lines = ["BASELINE PERSONA PROFILE SUMMARY", "=" * 40]
        
        # Sort traits by confidence
        sorted_traits = sorted(
            persona_traits.items(), 
            key=lambda x: x[1].confidence, 
            reverse=True
        )
        
        for trait_name, trait in sorted_traits:
            description = self.trait_descriptions.get(trait_name, trait_name)
            
            # Convert numeric value to descriptive label
            if trait.value < 0.3:
                value_label = "Low"
            elif trait.value < 0.7:
                value_label = "Moderate"
            else:
                value_label = "High"
            
            confidence_pct = int(trait.confidence * 100)
            
            summary_lines.append(f"\n{trait_name.replace('_', ' ').title()}: {value_label}")
            summary_lines.append(f"  {description}")
            summary_lines.append(f"  Confidence: {confidence_pct}% (based on {trait.evidence_count} responses)")
        
        return "\n".join(summary_lines)
    
    def generate_assessment_summary(self, persona_traits: Dict[str, PersonaTrait], responses: List) -> str:
        """Generate improved end-of-assessment summary."""
        total_traits = len(persona_traits)
        total_responses = len(responses)
        followup_count = len([r for r in responses if r.trigger_reason != "core_question"])
        
        # Calculate average confidence
        if persona_traits:
            avg_confidence = sum(t.confidence for t in persona_traits.values()) / len(persona_traits)
        else:
            avg_confidence = 0.0
        
        # Get top 3 strongest traits
        sorted_by_confidence = sorted(
            persona_traits.items(), 
            key=lambda x: x[1].confidence, 
            reverse=True
        )[:3]
        
        summary_lines = [
            "ASSESSMENT RESULTS",
            "-" * 18,
            f"Traits inferred: {total_traits}",
            f"Responses collected: {total_responses}",
            f"Follow-up questions asked: {followup_count}",
            f"Average confidence: {avg_confidence:.2f}",
            "",
            "Strongest inferred traits:"
        ]
        
        for i, (trait_name, trait) in enumerate(sorted_by_confidence, 1):
            confidence_pct = int(trait.confidence * 100)
            trait_display = trait_name.replace('_', ' ').title()
            summary_lines.append(f"{i}. {trait_display} ({confidence_pct}% confidence)")
        
        return "\n".join(summary_lines)
