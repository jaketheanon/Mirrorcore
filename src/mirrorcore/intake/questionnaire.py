"""
Questionnaire Engine

This module manages the interactive branching questionnaire for user assessment.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    """Types of questions in the assessment."""
    SCENARIO_MULTIPLE_CHOICE = "scenario_multiple_choice"
    SCENARIO_CUSTOM = "scenario_custom"
    FOLLOWUP_MULTIPLE_CHOICE = "followup_multiple_choice"
    FOLLOWUP_OPEN = "followup_open"


@dataclass
class AnswerOption:
    """Represents an answer option."""
    id: str
    text: str
    target_traits: Dict[str, float]  # trait_name -> inference_strength
    followup_triggers: List[str]  # question_ids to trigger


@dataclass
class Question:
    """Represents a question in the assessment."""
    id: str
    scenario_text: str
    question_text: str
    type: QuestionType
    answer_options: List[AnswerOption]
    allow_custom: bool = True
    allow_explanation: bool = False
    target_traits: List[str] = None  # Primary traits this question assesses
    priority: int = 1  # 1=high, 2=medium, 3=low
    followup_map: Dict[str, List[str]] = None  # answer_id -> followup_question_ids


@dataclass
class Response:
    """Represents a user response."""
    question_id: str
    question_type: QuestionType
    selected_option_id: Optional[str] = None
    custom_response: Optional[str] = None
    explanation: Optional[str] = None
    trigger_reason: str = "core_question"
    order_index: int = 0
    assessment_session_id: Optional[str] = None


def get_assessment_definition() -> Dict[str, Any]:
    """Get the complete assessment definition with core and follow-up questions."""
    
    core_questions = [
        Question(
            id="troubleshooting_permissions",
            scenario_text="You run 'docker ps' and get 'permission denied' after a recent system update.",
            question_text="What's your most likely first approach?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="check_sudo",
                    text="Try 'sudo docker ps' to see if it's a permissions issue",
                    target_traits={"troubleshooting_style": 0.8, "risk_tolerance": 0.3},
                    followup_triggers=["sudo_habits"]
                ),
                AnswerOption(
                    id="check_service",
                    text="Check if docker service is running with 'systemctl status docker'",
                    target_traits={"troubleshooting_style": 0.6, "evidence_threshold": 0.7},
                    followup_triggers=["service_troubleshooting"]
                ),
                AnswerOption(
                    id="check_logs",
                    text="Look at system logs to see what changed during the update",
                    target_traits={"troubleshooting_style": 0.9, "evidence_threshold": 0.8},
                    followup_triggers=["log_analysis"]
                ),
                AnswerOption(
                    id="search_web",
                    text="Search online for 'docker permission denied after update'",
                    target_traits={"self_reliance_level": 0.3, "trust_verification_style": 0.6},
                    followup_triggers=["online_research"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["troubleshooting_style", "risk_tolerance", "evidence_threshold"],
            priority=1,
            followup_map={
                "check_sudo": ["sudo_habits"],
                "check_service": ["service_troubleshooting"], 
                "check_logs": ["log_analysis"],
                "search_web": ["online_research"]
            }
        ),
        
        Question(
            id="risky_system_fix",
            scenario_text="A critical production service is failing. You found two potential fixes:",
            question_text="Option A: Quick fix that usually works but could cause data corruption. Option B: Slower, safer method that takes 2 hours. Which do you choose?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="quick_fix",
                    text="Try the quick fix first - speed is critical in production",
                    target_traits={"risk_tolerance": 0.9, "decision_speed": 0.8, "speed_vs_accuracy_preference": 0.8},
                    followup_triggers=["risk_assessment"]
                ),
                AnswerOption(
                    id="safe_method",
                    text="Use the safe method - data integrity is worth the time",
                    target_traits={"risk_tolerance": 0.2, "speed_vs_accuracy_preference": 0.2},
                    followup_triggers=["safety_priority"]
                ),
                AnswerOption(
                    id="hybrid_approach",
                    text="Test the quick fix on staging first, then decide",
                    target_traits={"risk_tolerance": 0.5, "evidence_threshold": 0.8, "decision_speed": 0.5},
                    followup_triggers=["staging_testing"]
                ),
                AnswerOption(
                    id="seek_help",
                    text="Ask senior engineers for their recommendation",
                    target_traits={"self_reliance_level": 0.3, "trust_verification_style": 0.7},
                    followup_triggers=["expert_consultation"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["risk_tolerance", "decision_speed", "speed_vs_accuracy_preference", "self_reliance_level"],
            priority=1
        ),
        
        Question(
            id="career_path_decision",
            scenario_text="You have two job offers: Offer A is stable at current company with 10% raise. Offer B is startup with 50% more pay but 40% chance of failure in 2 years.",
            question_text="Which offer do you lean toward and why?",
            type=QuestionType.SCENARIO_CUSTOM,
            answer_options=[
                AnswerOption(
                    id="stable_offer",
                    text="Take the stable offer - predictability matters more than potential upside",
                    target_traits={"risk_tolerance": 0.2, "priority_resolution_style": 0.7},
                    followup_triggers=["stability_values"]
                ),
                AnswerOption(
                    id="startup_offer", 
                    text="Take the startup offer - high risk, high reward is exciting",
                    target_traits={"risk_tolerance": 0.9, "action_bias": 0.8},
                    followup_triggers=["risk_appetite"]
                ),
                AnswerOption(
                    id="negotiate_stable",
                    text="Try to negotiate the stable offer for more money",
                    target_traits={"priority_resolution_style": 0.8, "action_bias": 0.6},
                    followup_triggers=["negotiation_style"]
                ),
                AnswerOption(
                    id="seek_advice",
                    text="Ask mentors and industry contacts for their perspective",
                    target_traits={"trust_verification_style": 0.7, "self_reliance_level": 0.4},
                    followup_triggers=["mentor_trust"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["risk_tolerance", "priority_resolution_style", "action_bias", "trust_verification_style"],
            priority=1
        ),
        
        Question(
            id="incomplete_information",
            scenario_text="You need to make a decision by end of day, but you're missing key data from another team that's unresponsive.",
            question_text="What's your approach?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="make_assumption",
                    text="Make reasonable assumptions and proceed - deadlines matter",
                    target_traits={"ambiguity_tolerance": 0.8, "decision_speed": 0.7},
                    followup_triggers=["assumption_making"]
                ),
                AnswerOption(
                    id="escalate",
                    text="Escalate to their manager to get the data",
                    target_traits={"priority_resolution_style": 0.8, "action_bias": 0.6},
                    followup_triggers=["escalation_style"]
                ),
                AnswerOption(
                    id="delay_decision",
                    text="Request deadline extension until data arrives",
                    target_traits={"ambiguity_tolerance": 0.3, "decision_speed": 0.3},
                    followup_triggers=["deadline_negotiation"]
                ),
                AnswerOption(
                    id="partial_decision",
                    text="Make a provisional decision that can be adjusted later",
                    target_traits={"ambiguity_tolerance": 0.6, "evidence_threshold": 0.7},
                    followup_triggers=["iterative_decisions"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["ambiguity_tolerance", "decision_speed", "priority_resolution_style", "evidence_threshold"],
            priority=1
        ),
        
        Question(
            id="friend_advice",
            scenario_text="Your close friend gives you emotionally supportive advice about a technical problem, but their solution seems technically weak.",
            question_text="How do you handle this situation?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="follow_advice",
                    text="Follow their advice - the emotional support matters more",
                    target_traits={"trust_verification_style": 0.3, "communication_preference": 0.8},
                    followup_triggers=["emotional_vs_technical"]
                ),
                AnswerOption(
                    id="politely_ignore",
                    text="Thank them but find your own technical solution",
                    target_traits={"trust_verification_style": 0.8, "self_reliance_level": 0.7},
                    followup_triggers=["social_navigation"]
                ),
                AnswerOption(
                    id="hybrid_approach",
                    text="Use their emotional support but implement a technical solution",
                    target_traits={"trust_verification_style": 0.6, "communication_preference": 0.6},
                    followup_triggers=["emotional_technical_balance"]
                ),
                AnswerOption(
                    id="educate_friend",
                    text="Explain the technical issues and collaborate on better solution",
                    target_traits={"communication_preference": 0.8, "action_bias": 0.7},
                    followup_triggers=["teaching_style"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["trust_verification_style", "communication_preference", "self_reliance_level", "action_bias"],
            priority=1
        ),
        
        Question(
            id="expert_disagreement",
            scenario_text="A respected expert recommends solution A, but your research suggests solution B is better. The expert has 20 years more experience.",
            question_text="What do you do?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="trust_expert",
                    text="Go with the expert's recommendation - experience trumps research",
                    target_traits={"trust_verification_style": 0.2, "self_reliance_level": 0.3},
                    followup_triggers=["authority_trust"]
                ),
                AnswerOption(
                    id="trust_research",
                    text="Follow your research - data beats authority",
                    target_traits={"trust_verification_style": 0.9, "self_reliance_level": 0.8},
                    followup_triggers=["research_confidence"]
                ),
                AnswerOption(
                    id="seek_more_data",
                    text="Gather more evidence to resolve the conflict",
                    target_traits={"evidence_threshold": 0.9, "decision_speed": 0.4},
                    followup_triggers=["conflict_resolution"]
                ),
                AnswerOption(
                    id="combine_approaches",
                    text="Try to combine both approaches for best of both",
                    target_traits={"ambiguity_tolerance": 0.7, "priority_resolution_style": 0.6},
                    followup_triggers=["synthesis_skills"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["trust_verification_style", "self_reliance_level", "evidence_threshold", "decision_speed"],
            priority=1
        ),
        
        Question(
            id="mental_math_vs_tools",
            scenario_text="You need to calculate something complex for a project. You could do it mentally/on paper or use a calculator/spreadsheet.",
            question_text="What's your default approach?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="mental_math",
                    text="Do it mentally - keeps your mind sharp and it's faster for simple cases",
                    target_traits={"tool_dependence": 0.2, "confidence_calibration": 0.6},
                    followup_triggers=["mental_calculation"]
                ),
                AnswerOption(
                    id="use_tools",
                    text="Always use tools - reduces error and saves mental energy",
                    target_traits={"tool_dependence": 0.9, "confidence_calibration": 0.4},
                    followup_triggers=["tool_reliance"]
                ),
                AnswerOption(
                    id="hybrid_approach",
                    text="Mental estimate first, then verify with tools",
                    target_traits={"tool_dependence": 0.5, "confidence_calibration": 0.8},
                    followup_triggers=["verification_habits"]
                ),
                AnswerOption(
                    id="context_dependent",
                    text="Depends on complexity and consequences of error",
                    target_traits={"ambiguity_tolerance": 0.7, "evidence_threshold": 0.8},
                    followup_triggers=["contextual_tool_use"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["tool_dependence", "confidence_calibration", "ambiguity_tolerance", "evidence_threshold"],
            priority=1
        ),
        
        Question(
            id="speed_vs_precision",
            scenario_text="You're debugging an issue. You could try quick fixes until something works, or systematically analyze until you find the root cause.",
            question_text="Which approach do you prefer?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="quick_fixes",
                    text="Try quick fixes first - speed matters more than understanding",
                    target_traits={"speed_vs_accuracy_preference": 0.8, "decision_speed": 0.7},
                    followup_triggers=["trial_error"]
                ),
                AnswerOption(
                    id="systematic_analysis",
                    text="Systematic analysis - understanding prevents future issues",
                    target_traits={"speed_vs_accuracy_preference": 0.2, "troubleshooting_style": 0.8},
                    followup_triggers=["analytical_approach"]
                ),
                AnswerOption(
                    id="balanced_approach",
                    text="Quick fixes for simple issues, analysis for complex ones",
                    target_traits={"speed_vs_accuracy_preference": 0.5, "ambiguity_tolerance": 0.7},
                    followup_triggers=["situational_approaches"]
                ),
                AnswerOption(
                    id="parallel_approach",
                    text="Try a quick fix while analyzing in parallel",
                    target_traits={"speed_vs_accuracy_preference": 0.6, "action_bias": 0.8},
                    followup_triggers=["parallel_processing"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["speed_vs_accuracy_preference", "decision_speed", "troubleshooting_style", "ambiguity_tolerance"],
            priority=1
        ),
        
        Question(
            id="failure_response",
            scenario_text="You were confident in your approach, but it failed completely. Your team is waiting for results.",
            question_text="What's your immediate reaction?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="quick_pivot",
                    text="Immediately try a different approach - no time to dwell on failure",
                    target_traits={"setback_response": 0.8, "decision_speed": 0.9},
                    followup_triggers=["resilience"]
                ),
                AnswerOption(
                    id="analyze_failure",
                    text="Stop and analyze what went wrong before proceeding",
                    target_traits={"setback_response": 0.3, "evidence_threshold": 0.8},
                    followup_triggers=["failure_analysis"]
                ),
                AnswerOption(
                    id="seek_input",
                    text="Ask team members for their perspective on what went wrong",
                    target_traits={"setback_response": 0.5, "trust_verification_style": 0.6},
                    followup_triggers=["collaborative_recovery"]
                ),
                AnswerOption(
                    id="communicate_delay",
                    text="Inform team of delay and take time to regroup",
                    target_traits={"setback_response": 0.4, "communication_preference": 0.7},
                    followup_triggers=["transparency"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["setback_response", "decision_speed", "evidence_threshold", "trust_verification_style", "communication_preference"],
            priority=1
        ),
        
        Question(
            id="advice_filtering",
            scenario_text="You're getting advice from multiple sources: senior developer, documentation, Stack Overflow, and a blog post. They conflict.",
            question_text="How do you filter and decide which advice to follow?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="trust_documentation",
                    text="Trust official documentation most - it's authoritative",
                    target_traits={"trust_verification_style": 0.8, "evidence_threshold": 0.7},
                    followup_triggers=["authority_preference"]
                ),
                AnswerOption(
                    id="trust_experience",
                    text="Go with the senior developer's advice - experience matters most",
                    target_traits={"trust_verification_style": 0.4, "self_reliance_level": 0.5},
                    followup_triggers=["experience_trust"]
                ),
                AnswerOption(
                    id="trust_consensus",
                    text="Look for the approach that appears most frequently across sources",
                    target_traits={"advice_filtering_style": 0.8, "evidence_threshold": 0.6},
                    followup_triggers=["consensus_building"]
                ),
                AnswerOption(
                    id="test_all_options",
                    text="Test multiple approaches on small scale to see what works",
                    target_traits={"advice_filtering_style": 0.2, "self_reliance_level": 0.9},
                    followup_triggers=["empirical_testing"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["trust_verification_style", "advice_filtering_style", "evidence_threshold", "self_reliance_level"],
            priority=1
        ),
        
        Question(
            id="instinct_vs_verification",
            scenario_text="You have a strong instinct about the right approach, but it contradicts obvious evidence.",
            question_text="What do you trust more?",
            type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="trust_instinct",
                    text="Trust your instinct - intuition often sees what evidence misses",
                    target_traits={"confidence_calibration": 0.3, "self_reliance_level": 0.8},
                    followup_triggers=["intuition_trust"]
                ),
                AnswerOption(
                    id="trust_evidence",
                    text="Follow the evidence - instincts can be wrong",
                    target_traits={"confidence_calibration": 0.8, "evidence_threshold": 0.9},
                    followup_triggers=["evidence_priority"]
                ),
                AnswerOption(
                    id="investigate_conflict",
                    text="Investigate why instinct and evidence conflict",
                    target_traits={"confidence_calibration": 0.6, "evidence_threshold": 0.8},
                    followup_triggers=["conflict_investigation"]
                ),
                AnswerOption(
                    id="small_test",
                    text="Test your instinct on a small scale before committing",
                    target_traits={"confidence_calibration": 0.7, "risk_tolerance": 0.5},
                    followup_triggers=["controlled_testing"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["confidence_calibration", "self_reliance_level", "evidence_threshold", "risk_tolerance"],
            priority=1
        ),
        
        Question(
            id="priority_conflict",
            scenario_text="You have three urgent tasks: fixing a critical bug (impact: high), finishing a feature for a deadline (impact: medium), and helping a struggling teammate (impact: variable).",
            question_text="How do you prioritize?",
            type=QuestionType.SCENARIO_CUSTOM,
            answer_options=[
                AnswerOption(
                    id="impact_priority",
                    text="Fix the critical bug first - highest impact wins",
                    target_traits={"priority_resolution_style": 0.8, "evidence_threshold": 0.7},
                    followup_triggers=["impact_assessment"]
                ),
                AnswerOption(
                    id="deadline_priority",
                    text="Finish the feature - deadlines are non-negotiable",
                    target_traits={"priority_resolution_style": 0.6, "decision_speed": 0.8},
                    followup_triggers=["deadline_focus"]
                ),
                AnswerOption(
                    id="people_priority",
                    text="Help the teammate first - team health enables everything else",
                    target_traits={"priority_resolution_style": 0.3, "communication_preference": 0.8},
                    followup_triggers=["team_focus"]
                ),
                AnswerOption(
                    id="contextual_approach",
                    text="Depends on specific context and timelines of each task",
                    target_traits={"priority_resolution_style": 0.5, "ambiguity_tolerance": 0.7},
                    followup_triggers=["contextual_priority"]
                )
            ],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["priority_resolution_style", "evidence_threshold", "decision_speed", "communication_preference", "ambiguity_tolerance"],
            priority=1
        )
    ]
    
    followup_questions = [
        Question(
            id="sudo_habits",
            scenario_text="Regarding your use of sudo for permissions issues...",
            question_text="When do you typically use sudo versus fixing permissions properly?",
            type=QuestionType.FOLLOWUP_MULTIPLE_CHOICE,
            answer_options=[
                AnswerOption(
                    id="sudo_first",
                    text="Always try sudo first - it's fastest",
                    target_traits={"troubleshooting_style": 0.2, "decision_speed": 0.9},
                    followup_triggers=[]
                ),
                AnswerOption(
                    id="sudo_last",
                    text="Only use sudo as last resort",
                    target_traits={"troubleshooting_style": 0.9, "evidence_threshold": 0.8},
                    followup_triggers=[]
                ),
                AnswerOption(
                    id="context_dependent",
                    text="Depends on whether it's my system vs production",
                    target_traits={"risk_tolerance": 0.6, "ambiguity_tolerance": 0.7},
                    followup_triggers=[]
                )
            ],
            allow_custom=True,
            target_traits=["troubleshooting_style", "decision_speed", "risk_tolerance", "ambiguity_tolerance"],
            priority=2
        ),
        
        Question(
            id="risk_assessment",
            scenario_text="About choosing quick fixes in production...",
            question_text="What factors make you accept or reject risky solutions?",
            type=QuestionType.FOLLOWUP_OPEN,
            answer_options=[],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["risk_tolerance", "evidence_threshold", "decision_speed"],
            priority=2
        ),
        
        Question(
            id="assumption_making",
            scenario_text="Regarding making assumptions with incomplete data...",
            question_text="What types of assumptions do you feel comfortable making?",
            type=QuestionType.FOLLOWUP_OPEN,
            answer_options=[],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["ambiguity_tolerance", "confidence_calibration", "risk_tolerance"],
            priority=2
        ),
        
        Question(
            id="tool_reliance",
            scenario_text="About your preference for using tools...",
            question_text="At what point do you switch from mental work to tools?",
            type=QuestionType.FOLLOWUP_OPEN,
            answer_options=[],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["tool_dependence", "confidence_calibration", "evidence_threshold"],
            priority=2
        ),
        
        Question(
            id="failure_analysis",
            scenario_text="Regarding how you handle failure...",
            question_text="What do you typically learn from your failures?",
            type=QuestionType.FOLLOWUP_OPEN,
            answer_options=[],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["setback_response", "evidence_threshold", "confidence_calibration"],
            priority=2
        ),
        
        Question(
            id="authority_trust",
            scenario_text="About trusting experts vs your research...",
            question_text="What makes an expert trustworthy versus questionable?",
            type=QuestionType.FOLLOWUP_OPEN,
            answer_options=[],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["trust_verification_style", "evidence_threshold", "self_reliance_level"],
            priority=2
        ),
        
        Question(
            id="intuition_trust",
            scenario_text="Regarding trusting your instincts...",
            question_text="When has your intuition been most right or wrong?",
            type=QuestionType.FOLLOWUP_OPEN,
            answer_options=[],
            allow_custom=True,
            allow_explanation=True,
            target_traits=["confidence_calibration", "self_reliance_level", "evidence_threshold"],
            priority=2
        )
    ]
    
    return {
        "core_questions": core_questions,
        "followup_questions": followup_questions,
        "max_total_questions": 24,
        "target_traits": [
            "troubleshooting_style", "risk_tolerance", "ambiguity_tolerance",
            "decision_speed", "evidence_threshold", "trust_verification_style",
            "communication_preference", "action_bias", "setback_response",
            "priority_resolution_style", "advice_filtering_style",
            "self_reliance_level", "confidence_calibration",
            "speed_vs_accuracy_preference", "tool_dependence"
        ]
    }
