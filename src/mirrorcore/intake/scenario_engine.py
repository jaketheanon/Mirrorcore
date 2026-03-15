"""
Scenario Engine

Runs the interactive branching assessment in the terminal.
"""

import sys
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from .questionnaire import get_assessment_definition, Question, QuestionType, Response


@dataclass
class AssessmentState:
    """Tracks the current state of the assessment."""
    asked_question_ids: Set[str]
    response_count: int
    trait_confidence: Dict[str, float]
    current_order_index: int
    assessment_session_id: str


class ScenarioEngine:
    """Engine for running interactive branching assessments."""
    
    def __init__(self):
        self.definition = get_assessment_definition()
        self.core_questions = {q.id: q for q in self.definition["core_questions"]}
        self.followup_questions = {q.id: q for q in self.definition["followup_questions"]}
        self.state = AssessmentState(
            asked_question_ids=set(),
            response_count=0,
            trait_confidence={trait: 0.0 for trait in self.definition["target_traits"]},
            current_order_index=0,
            assessment_session_id=str(uuid.uuid4())
        )
    
    def run_assessment(self) -> List[Response]:
        """Run the complete interactive assessment."""
        print("\n" + "="*60)
        print("MIRRORCORE INITIAL REASONING ASSESSMENT")
        print("="*60)
        print("\nThis assessment helps Mirrorcore understand how you reason,")
        print("make decisions, and approach problems.")
        print(f"\nWe'll ask approximately {len(self.definition['core_questions'])} core questions")
        print("with some follow-ups based on your answers.")
        print(f"Total questions: max {self.definition['max_total_questions']}")
        print("\n" + "-"*60)
        
        responses = []
        
        # Ask core questions in priority order
        remaining_core = sorted(
            self.definition["core_questions"], 
            key=lambda q: (q.priority, q.id)
        )
        
        for question in remaining_core:
            if self._should_continue_assessment():
                response = self._ask_question(question, "core_question")
                if response:
                    responses.append(response)
                    self._update_trait_confidence(response)
                    
                    # Check for follow-up triggers
                    followup_ids = self._get_followup_triggers(response)
                    for followup_id in followup_ids:
                        if self._should_continue_assessment():
                            followup_question = self.followup_questions.get(followup_id)
                            if followup_question and followup_id not in self.state.asked_question_ids:
                                followup_response = self._ask_question(followup_question, f"followup_to_{response.question_id}")
                                if followup_response:
                                    responses.append(followup_response)
                                    self._update_trait_confidence(followup_response)
            else:
                break
        
        # Check if we need gap-filling follow-ups for low-confidence traits
        if self._should_continue_assessment():
            gap_followups = self._get_gap_filling_followups()
            for followup_id in gap_followups:
                if self._should_continue_assessment():
                    followup_question = self.followup_questions.get(followup_id)
                    if followup_question:
                        response = self._ask_question(followup_question, "gap_filling")
                        if response:
                            responses.append(response)
                            self._update_trait_confidence(response)
        
        print("\n" + "="*60)
        print("ASSESSMENT COMPLETE")
        print("="*60)
        print(f"Total questions asked: {len(responses)}")
        print(f"Core questions: {len([r for r in responses if r.trigger_reason == 'core_question'])}")
        print(f"Follow-up questions: {len([r for r in responses if r.trigger_reason != 'core_question'])}")
        
        return responses
    
    def _should_continue_assessment(self) -> bool:
        """Check if assessment should continue."""
        # Hard cap on total questions
        if self.state.response_count >= self.definition["max_total_questions"]:
            return False
        
        # Check if we have sufficient confidence in all traits
        min_confidence_threshold = 0.6
        high_confidence_traits = sum(1 for conf in self.state.trait_confidence.values() if conf >= min_confidence_threshold)
        
        # If we have good confidence on most traits and asked enough questions, we can stop
        if (high_confidence_traits >= len(self.definition["target_traits"]) * 0.7 and 
            self.state.response_count >= len(self.definition["core_questions"]) * 0.8):
            return False
        
        return True
    
    def _ask_question(self, question: Question, trigger_reason: str) -> Optional[Response]:
        """Ask a single question and get response."""
        self.state.asked_question_ids.add(question.id)
        self.state.current_order_index += 1
        
        print(f"\n[Question {self.state.response_count + 1}]")
        print("-" * 40)
        print(f"\n{question.scenario_text}\n")
        print(f"{question.question_text}\n")
        
        # Display answer options
        for i, option in enumerate(question.answer_options, 1):
            print(f"  {i}. {option.text}")
        
        # Always show custom option
        custom_option_num = len(question.answer_options) + 1
        print(f"  {custom_option_num}. Custom response")
        
        # Get user input
        while True:
            try:
                choice = input(f"\nYour choice (1-{custom_option_num}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(question.answer_options):
                    selected_option = question.answer_options[choice_num - 1]
                    break
                elif choice_num == custom_option_num and question.allow_custom:
                    custom_response = input("Your custom response: ").strip()
                    if custom_response:
                        selected_option = None
                        break
                    else:
                        print("Please enter a custom response.")
                else:
                    print(f"Please enter a number between 1 and {custom_option_num}.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get optional explanation
        explanation = None
        if question.allow_explanation:
            explain_choice = input("\nWould you like to add an explanation? (y/N): ").strip().lower()
            if explain_choice in ['y', 'yes']:
                explanation = input("Explanation: ").strip()
                if not explanation:
                    explanation = None
        
        # Create response
        response = Response(
            question_id=question.id,
            question_type=question.type,
            selected_option_id=selected_option.id if selected_option else None,
            custom_response=custom_response if not selected_option else None,
            explanation=explanation,
            trigger_reason=trigger_reason,
            order_index=self.state.current_order_index,
            assessment_session_id=self.state.assessment_session_id
        )
        
        self.state.response_count += 1
        return response
    
    def _get_followup_triggers(self, response: Response) -> List[str]:
        """Get follow-up question IDs triggered by this response."""
        triggered_ids = []
        
        # Initialize keyword triggers before any conditional logic
        keyword_triggers = {
            "risk": ["risk_assessment"],
            "assume": ["assumption_making"],
            "tool": ["tool_reliance"],
            "fail": ["failure_analysis"],
            "expert": ["authority_trust"],
            "intuit": ["intuition_trust"],
            "sudo": ["sudo_habits"]
        }
        
        if response.selected_option_id:
            # Get the original question to find triggers
            question = self.core_questions.get(response.question_id)
            if not question:
                question = self.followup_questions.get(response.question_id)
            
            if question and question.followup_map:
                triggered_ids.extend(question.followup_map.get(response.selected_option_id, []))
        
        # Check custom response for keyword triggers
        if response.custom_response:
            custom_lower = response.custom_response.lower()
            
            for keyword, followup_ids in keyword_triggers.items():
                if keyword in custom_lower:
                    triggered_ids.extend(followup_ids)
        
        # Check explanation for triggers
        if response.explanation:
            # Safe fallback handling if explanation is empty
            if response.explanation.strip():
                explanation_lower = response.explanation.lower()
                for keyword, followup_ids in keyword_triggers.items():
                    if keyword in explanation_lower:
                        triggered_ids.extend(followup_ids)
        
        # Remove duplicates and already asked questions
        return list(set(triggered_ids) - self.state.asked_question_ids)
    
    def _get_gap_filling_followups(self) -> List[str]:
        """Get follow-up questions for traits with low confidence."""
        low_confidence_traits = [
            trait for trait, conf in self.state.trait_confidence.items() 
            if conf < 0.5
        ]
        
        # Map traits to relevant follow-up questions
        trait_followups = {
            "troubleshooting_style": ["sudo_habits"],
            "risk_tolerance": ["risk_assessment"],
            "ambiguity_tolerance": ["assumption_making"],
            "tool_dependence": ["tool_reliance"],
            "setback_response": ["failure_analysis"],
            "trust_verification_style": ["authority_trust"],
            "confidence_calibration": ["intuition_trust"]
        }
        
        gap_followups = []
        for trait in low_confidence_traits:
            if trait in trait_followups:
                gap_followups.extend(trait_followups[trait])
        
        # Remove duplicates and already asked questions
        return list(set(gap_followups) - self.state.asked_question_ids)
    
    def _update_trait_confidence(self, response: Response):
        """Update trait confidence based on response."""
        # Get the question to determine trait impacts
        question = self.core_questions.get(response.question_id)
        if not question:
            question = self.followup_questions.get(response.question_id)
        
        if not question or not question.target_traits:
            return
        
        # Weight follow-up responses more heavily
        weight = 1.5 if response.trigger_reason != "core_question" else 1.0
        
        if response.selected_option_id:
            # Find the selected option
            for option in question.answer_options:
                if option.id == response.selected_option_id:
                    # Update trait confidence
                    for trait, strength in option.target_traits.items():
                        current_conf = self.state.trait_confidence.get(trait, 0.0)
                        # Weighted average with confidence decay for multiple signals
                        new_conf = (current_conf + (strength * weight)) / (1 + weight * 0.5)
                        self.state.trait_confidence[trait] = min(new_conf, 1.0)
                    break
        
        # Consider custom responses and explanations
        if response.custom_response or response.explanation:
            # Small confidence boost for providing detailed responses
            for trait in question.target_traits:
                current_conf = self.state.trait_confidence.get(trait, 0.0)
                self.state.trait_confidence[trait] = min(current_conf + 0.1, 1.0)
