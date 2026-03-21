"""
Decision Interview Module (Phase 26 + Phase 27)

Presents structured decision scenarios, collects user choices,
and extracts deterministic trait/value signals from responses.

Phase 27 additions:
- Human-friendly everyday scenarios with simple answer choices
- Multi-scenario interview sessions (3-5 per run)
- Mid-interview reflection generation
- Correction loop with structured correction metadata
- Deterministic response variation

Phase 30.1+:
- Follow-up “why” options are keyed by the selected main choice (no shared pool
  per scenario) so reasons match the action taken.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class InterviewOption:
    """A selectable option within a scenario question."""
    id: str
    label: str
    value_tags: List[str] = field(default_factory=list)
    trait_signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class InterviewQuestion:
    """A single question in an interview scenario."""
    prompt: str
    options: List[InterviewOption] = field(default_factory=list)


@dataclass
class InterviewScenario:
    """Scenario with a main question and follow-up reasons per main choice."""
    id: str
    text: str
    main_question: InterviewQuestion
    followup_by_main_choice: Dict[str, InterviewQuestion]


@dataclass
class InterviewResult:
    """Collected result from running one interview scenario."""
    scenario_id: str
    scenario_text: str
    choice_label: str
    choice_value: str
    reasoning_label: str
    reasoning_value: str
    value_tags: List[str]
    trait_signals: Dict[str, float]
    confidence_score: float


@dataclass
class ReflectionLine:
    """A single reflection observation with the trait it came from."""
    text: str
    trait: str
    strength: float


@dataclass
class CorrectionResult:
    """Outcome of the correction loop."""
    status: str  # "accurate", "partially_true", "not_really"
    accepted_traits: List[str] = field(default_factory=list)
    rejected_traits: List[str] = field(default_factory=list)
    replacement_choices: Dict[str, str] = field(default_factory=dict)


@dataclass
class SessionResult:
    """Full result of a multi-scenario interview session."""
    results: List[InterviewResult]
    reflections: List[ReflectionLine]
    correction: Optional[CorrectionResult]


# ---------------------------------------------------------------------------
# Scenario definitions — human-friendly everyday scenarios
# ---------------------------------------------------------------------------

SCENARIOS: List[InterviewScenario] = [
    InterviewScenario(
        id="expensive_purchase_v1",
        text="You're thinking about buying something expensive that you want but don't strictly need.",
        main_question=InterviewQuestion(
            prompt="What would you do?",
            options=[
                InterviewOption(
                    id="buy_now",
                    label="Buy it now",
                    value_tags=["impulsiveness", "reward_seeking"],
                    trait_signals={"patience": 0.2, "risk_tolerance": 0.8, "self_reward": 0.9},
                ),
                InterviewOption(
                    id="wait_and_think",
                    label="Wait a few days and think it over",
                    value_tags=["caution", "deliberation"],
                    trait_signals={"patience": 0.9, "risk_tolerance": 0.3, "self_reward": 0.4},
                ),
                InterviewOption(
                    id="find_cheaper",
                    label="Look for a cheaper alternative first",
                    value_tags=["frugality", "pragmatism"],
                    trait_signals={"patience": 0.6, "risk_tolerance": 0.3, "resourcefulness": 0.8},
                ),
            ],
        ),
        followup_by_main_choice={
            "buy_now": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="exp_buy_wanted_now",
                        label="Wanted it now",
                        value_tags=["impulse", "speed", "reward_seeking"],
                        trait_signals={
                            "analytical_thinking": 0.25,
                            "intuitive_leaning": 0.75,
                            "efficiency": 0.7,
                            "self_reward": 0.8,
                        },
                    ),
                    InterviewOption(
                        id="exp_buy_treat_myself",
                        label="Wanted to treat myself",
                        value_tags=["reward_seeking", "self_care"],
                        trait_signals={
                            "self_reward": 0.85,
                            "analytical_thinking": 0.35,
                            "intuitive_leaning": 0.65,
                        },
                    ),
                    InterviewOption(
                        id="exp_buy_no_delay",
                        label="Didn't want to wait and overthink",
                        value_tags=["decisiveness", "momentum"],
                        trait_signals={
                            "decisiveness": 0.75,
                            "patience": 0.25,
                            "analytical_thinking": 0.4,
                            "intuitive_leaning": 0.6,
                        },
                    ),
                ],
            ),
            "wait_and_think": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="exp_wait_sleep_on_it",
                        label="Needed time to think it through",
                        value_tags=["caution", "deliberation"],
                        trait_signals={
                            "patience": 0.9,
                            "risk_tolerance": 0.2,
                            "analytical_thinking": 0.85,
                            "intuitive_leaning": 0.2,
                            "future_orientation": 0.7,
                        },
                    ),
                    InterviewOption(
                        id="exp_wait_avoid_impulse",
                        label="Wanted to avoid an impulse buy",
                        value_tags=["caution", "foresight"],
                        trait_signals={
                            "future_orientation": 0.8,
                            "financial_caution": 0.75,
                            "analytical_thinking": 0.8,
                            "intuitive_leaning": 0.25,
                        },
                    ),
                    InterviewOption(
                        id="exp_wait_compare",
                        label="Wanted to compare options calmly",
                        value_tags=["deliberation", "pragmatism"],
                        trait_signals={
                            "analytical_thinking": 0.85,
                            "resourcefulness": 0.65,
                            "patience": 0.75,
                            "risk_tolerance": 0.3,
                        },
                    ),
                ],
            ),
            "find_cheaper": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="exp_cheap_save",
                        label="Wanted to save money",
                        value_tags=["frugality", "security"],
                        trait_signals={
                            "financial_caution": 0.9,
                            "resourcefulness": 0.85,
                            "self_direction": 0.65,
                            "risk_tolerance": 0.25,
                        },
                    ),
                    InterviewOption(
                        id="exp_cheap_value",
                        label="Wanted better value for the price",
                        value_tags=["pragmatism", "resourcefulness"],
                        trait_signals={
                            "resourcefulness": 0.85,
                            "financial_caution": 0.7,
                            "analytical_thinking": 0.75,
                        },
                    ),
                    InterviewOption(
                        id="exp_cheap_still_get",
                        label="Still wanted it, just smarter",
                        value_tags=["pragmatism", "reward_seeking"],
                        trait_signals={
                            "self_reward": 0.55,
                            "resourcefulness": 0.8,
                            "patience": 0.65,
                            "analytical_thinking": 0.6,
                        },
                    ),
                ],
            ),
        },
    ),
    InterviewScenario(
        id="conflict_handling_v1",
        text="Someone you know says something that bothers you during a conversation.",
        main_question=InterviewQuestion(
            prompt="What would you do?",
            options=[
                InterviewOption(
                    id="speak_up",
                    label="Say something about it right away",
                    value_tags=["directness", "assertiveness"],
                    trait_signals={"conflict_comfort": 0.8, "directness": 0.9, "patience": 0.3},
                ),
                InterviewOption(
                    id="let_it_go",
                    label="Let it go and move on",
                    value_tags=["avoidance", "peace_keeping"],
                    trait_signals={"conflict_comfort": 0.2, "directness": 0.2, "patience": 0.7},
                ),
                InterviewOption(
                    id="bring_up_later",
                    label="Bring it up later when things are calmer",
                    value_tags=["diplomacy", "timing"],
                    trait_signals={"conflict_comfort": 0.5, "directness": 0.6, "patience": 0.8},
                ),
            ],
        ),
        followup_by_main_choice={
            "speak_up": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="conf_speak_honest",
                        label="Wanted to be honest in the moment",
                        value_tags=["honesty", "directness"],
                        trait_signals={
                            "directness": 0.85,
                            "conflict_comfort": 0.75,
                            "patience": 0.35,
                            "analytical_thinking": 0.55,
                            "intuitive_leaning": 0.45,
                        },
                    ),
                    InterviewOption(
                        id="conf_speak_stop_bothering",
                        label="Wanted it to stop bothering me",
                        value_tags=["directness", "self_care"],
                        trait_signals={
                            "directness": 0.8,
                            "conflict_comfort": 0.7,
                            "anxiety_avoidance": 0.55,
                        },
                    ),
                    InterviewOption(
                        id="conf_speak_clear_air",
                        label="Wanted to clear the air now",
                        value_tags=["directness", "assertiveness"],
                        trait_signals={
                            "directness": 0.85,
                            "decisiveness": 0.65,
                            "conflict_comfort": 0.75,
                        },
                    ),
                ],
            ),
            "let_it_go": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="conf_letgo_not_worth",
                        label="It wasn't worth the energy",
                        value_tags=["avoidance", "pragmatism"],
                        trait_signals={
                            "conflict_comfort": 0.25,
                            "patience": 0.65,
                            "analytical_thinking": 0.55,
                            "risk_tolerance": 0.35,
                        },
                    ),
                    InterviewOption(
                        id="conf_letgo_keep_peace",
                        label="Wanted to keep the peace",
                        value_tags=["peace_keeping", "avoidance"],
                        trait_signals={
                            "conflict_comfort": 0.2,
                            "anxiety_avoidance": 0.75,
                            "empathy": 0.55,
                            "patience": 0.7,
                        },
                    ),
                    InterviewOption(
                        id="conf_letgo_pick_battles",
                        label="Pick my battles",
                        value_tags=["pragmatism", "timing"],
                        trait_signals={
                            "future_orientation": 0.65,
                            "patience": 0.75,
                            "analytical_thinking": 0.6,
                            "conflict_comfort": 0.35,
                        },
                    ),
                ],
            ),
            "bring_up_later": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="conf_later_better_moment",
                        label="Waited for a better moment",
                        value_tags=["diplomacy", "timing"],
                        trait_signals={
                            "patience": 0.85,
                            "directness": 0.55,
                            "conflict_comfort": 0.55,
                            "analytical_thinking": 0.8,
                            "future_orientation": 0.75,
                        },
                    ),
                    InterviewOption(
                        id="conf_later_cooler_heads",
                        label="Cooler heads would help",
                        value_tags=["composure", "caution"],
                        trait_signals={
                            "patience": 0.8,
                            "anxiety_avoidance": 0.6,
                            "analytical_thinking": 0.75,
                        },
                    ),
                    InterviewOption(
                        id="conf_later_thought_first",
                        label="Wanted to think before I spoke",
                        value_tags=["deliberation", "planning"],
                        trait_signals={
                            "analytical_thinking": 0.85,
                            "patience": 0.8,
                            "directness": 0.5,
                            "intuitive_leaning": 0.3,
                        },
                    ),
                ],
            ),
        },
    ),
    InterviewScenario(
        id="help_when_overwhelmed_v1",
        text="Someone asks you for help, but you're already overwhelmed with your own stuff.",
        main_question=InterviewQuestion(
            prompt="What would you do?",
            options=[
                InterviewOption(
                    id="help_anyway",
                    label="Help them anyway",
                    value_tags=["selflessness", "people_pleasing"],
                    trait_signals={"boundary_setting": 0.2, "empathy": 0.9, "self_care": 0.3},
                ),
                InterviewOption(
                    id="say_no",
                    label="Say no, explain you can't right now",
                    value_tags=["boundaries", "honesty"],
                    trait_signals={"boundary_setting": 0.9, "empathy": 0.5, "self_care": 0.8},
                ),
                InterviewOption(
                    id="help_a_little",
                    label="Help a little, but not as much as they need",
                    value_tags=["compromise", "balance"],
                    trait_signals={"boundary_setting": 0.5, "empathy": 0.7, "self_care": 0.5},
                ),
            ],
        ),
        followup_by_main_choice={
            "help_anyway": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="help_anyway_couldnt_say_no",
                        label="Couldn't bring myself to say no",
                        value_tags=["selflessness", "people_pleasing"],
                        trait_signals={
                            "empathy": 0.85,
                            "boundary_setting": 0.25,
                            "self_care": 0.35,
                            "analytical_thinking": 0.4,
                            "intuitive_leaning": 0.55,
                        },
                    ),
                    InterviewOption(
                        id="help_anyway_they_needed",
                        label="They really needed the help",
                        value_tags=["empathy", "responsibility"],
                        trait_signals={
                            "empathy": 0.9,
                            "boundary_setting": 0.3,
                            "self_care": 0.35,
                        },
                    ),
                    InterviewOption(
                        id="help_anyway_guilt",
                        label="Would have felt guilty saying no",
                        value_tags=["guilt", "people_pleasing"],
                        trait_signals={
                            "empathy": 0.75,
                            "anxiety_avoidance": 0.65,
                            "boundary_setting": 0.3,
                            "intuitive_leaning": 0.5,
                        },
                    ),
                ],
            ),
            "say_no": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="help_no_protect_time",
                        label="Needed to protect my own time or energy",
                        value_tags=["boundaries", "self_care"],
                        trait_signals={
                            "boundary_setting": 0.9,
                            "self_care": 0.85,
                            "empathy": 0.45,
                            "analytical_thinking": 0.75,
                            "intuitive_leaning": 0.25,
                        },
                    ),
                    InterviewOption(
                        id="help_no_overloaded",
                        label="I was already overloaded",
                        value_tags=["stress", "honesty"],
                        trait_signals={
                            "self_care": 0.85,
                            "stress_tolerance": 0.45,
                            "boundary_setting": 0.85,
                            "empathy": 0.45,
                        },
                    ),
                    InterviewOption(
                        id="help_no_regret_yes",
                        label="Didn't want to say yes and regret it later",
                        value_tags=["foresight", "caution"],
                        trait_signals={
                            "future_orientation": 0.8,
                            "boundary_setting": 0.85,
                            "anxiety_avoidance": 0.6,
                            "analytical_thinking": 0.75,
                        },
                    ),
                ],
            ),
            "help_a_little": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="help_little_what_i_could",
                        label="It was what I could realistically offer",
                        value_tags=["compromise", "balance"],
                        trait_signals={
                            "boundary_setting": 0.55,
                            "empathy": 0.65,
                            "self_care": 0.55,
                            "analytical_thinking": 0.65,
                        },
                    ),
                    InterviewOption(
                        id="help_little_better_than_nothing",
                        label="A little felt better than nothing",
                        value_tags=["compromise", "empathy"],
                        trait_signals={
                            "empathy": 0.7,
                            "boundary_setting": 0.5,
                            "self_care": 0.5,
                        },
                    ),
                    InterviewOption(
                        id="help_little_test_waters",
                        label="Wanted to help without giving everything",
                        value_tags=["boundaries", "balance"],
                        trait_signals={
                            "boundary_setting": 0.65,
                            "empathy": 0.6,
                            "self_care": 0.55,
                            "analytical_thinking": 0.6,
                        },
                    ),
                ],
            ),
        },
    ),
    InterviewScenario(
        id="speed_vs_safety_v1",
        text="You need to get something done and there's a quick way that cuts some corners.",
        main_question=InterviewQuestion(
            prompt="What would you do?",
            options=[
                InterviewOption(
                    id="take_shortcut",
                    label="Take the quick way",
                    value_tags=["speed", "pragmatism"],
                    trait_signals={"risk_tolerance": 0.7, "thoroughness": 0.3, "efficiency": 0.9},
                ),
                InterviewOption(
                    id="do_it_right",
                    label="Do it the proper way even if it takes longer",
                    value_tags=["quality", "caution"],
                    trait_signals={"risk_tolerance": 0.2, "thoroughness": 0.9, "efficiency": 0.4},
                ),
                InterviewOption(
                    id="mix_both",
                    label="Do a bit of both",
                    value_tags=["balance", "pragmatism"],
                    trait_signals={"risk_tolerance": 0.5, "thoroughness": 0.6, "efficiency": 0.7},
                ),
            ],
        ),
        followup_by_main_choice={
            "take_shortcut": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="speed_short_move_fast",
                        label="Needed to move fast",
                        value_tags=["speed", "pragmatism", "momentum"],
                        trait_signals={
                            "efficiency": 0.9,
                            "risk_tolerance": 0.65,
                            "thoroughness": 0.25,
                            "analytical_thinking": 0.3,
                            "intuitive_leaning": 0.75,
                            "decisiveness": 0.6,
                        },
                    ),
                    InterviewOption(
                        id="speed_short_deadline",
                        label="Had a deadline or pressure",
                        value_tags=["speed", "stress"],
                        trait_signals={
                            "efficiency": 0.85,
                            "stress_tolerance": 0.55,
                            "risk_tolerance": 0.6,
                        },
                    ),
                    InterviewOption(
                        id="speed_short_good_enough",
                        label="Good enough was good enough",
                        value_tags=["pragmatism", "efficiency"],
                        trait_signals={
                            "efficiency": 0.8,
                            "thoroughness": 0.35,
                            "risk_tolerance": 0.55,
                        },
                    ),
                ],
            ),
            "do_it_right": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="speed_right_proper",
                        label="Wanted it done the proper way",
                        value_tags=["quality", "caution", "safety"],
                        trait_signals={
                            "thoroughness": 0.85,
                            "risk_tolerance": 0.2,
                            "anxiety_avoidance": 0.7,
                            "future_orientation": 0.6,
                            "analytical_thinking": 0.85,
                            "intuitive_leaning": 0.2,
                            "efficiency": 0.4,
                        },
                    ),
                    InterviewOption(
                        id="speed_right_avoid_mess",
                        label="Wanted to avoid a mess later",
                        value_tags=["foresight", "caution"],
                        trait_signals={
                            "future_orientation": 0.85,
                            "anxiety_avoidance": 0.65,
                            "thoroughness": 0.75,
                            "risk_tolerance": 0.2,
                        },
                    ),
                    InterviewOption(
                        id="speed_right_standards",
                        label="My standards mattered here",
                        value_tags=["quality", "integrity"],
                        trait_signals={
                            "thoroughness": 0.85,
                            "self_direction": 0.7,
                            "risk_tolerance": 0.25,
                        },
                    ),
                ],
            ),
            "mix_both": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="speed_mix_balance",
                        label="Wanted a practical balance",
                        value_tags=["balance", "pragmatism"],
                        trait_signals={
                            "risk_tolerance": 0.5,
                            "thoroughness": 0.6,
                            "efficiency": 0.7,
                            "analytical_thinking": 0.65,
                        },
                    ),
                    InterviewOption(
                        id="speed_mix_control_risk",
                        label="Kept risk under control",
                        value_tags=["caution", "pragmatism"],
                        trait_signals={
                            "risk_tolerance": 0.45,
                            "future_orientation": 0.6,
                            "thoroughness": 0.55,
                        },
                    ),
                    InterviewOption(
                        id="speed_mix_middle_ground",
                        label="Middle ground felt safest",
                        value_tags=["caution", "balance"],
                        trait_signals={
                            "anxiety_avoidance": 0.55,
                            "risk_tolerance": 0.45,
                            "patience": 0.55,
                        },
                    ),
                ],
            ),
        },
    ),
    InterviewScenario(
        id="wait_or_act_v1",
        text="You're unsure about a decision and could wait for more information or just go with what you know.",
        main_question=InterviewQuestion(
            prompt="What would you do?",
            options=[
                InterviewOption(
                    id="act_now",
                    label="Go ahead with what you know",
                    value_tags=["decisiveness", "action_bias"],
                    trait_signals={"patience": 0.2, "decisiveness": 0.9, "risk_tolerance": 0.7},
                ),
                InterviewOption(
                    id="wait_more",
                    label="Wait until you know more",
                    value_tags=["caution", "information_seeking"],
                    trait_signals={"patience": 0.9, "decisiveness": 0.3, "risk_tolerance": 0.2},
                ),
                InterviewOption(
                    id="small_step",
                    label="Take a small step and see what happens",
                    value_tags=["incremental", "pragmatism"],
                    trait_signals={"patience": 0.5, "decisiveness": 0.6, "risk_tolerance": 0.5},
                ),
            ],
        ),
        followup_by_main_choice={
            "act_now": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="wait_act_ready",
                        label="Felt ready to act",
                        value_tags=["decisiveness", "action_bias", "momentum"],
                        trait_signals={
                            "decisiveness": 0.9,
                            "risk_tolerance": 0.7,
                            "patience": 0.25,
                            "analytical_thinking": 0.35,
                            "intuitive_leaning": 0.7,
                            "efficiency": 0.65,
                        },
                    ),
                    InterviewOption(
                        id="wait_act_enough",
                        label="Had enough to go on",
                        value_tags=["decisiveness", "confidence"],
                        trait_signals={
                            "decisiveness": 0.85,
                            "self_direction": 0.7,
                            "risk_tolerance": 0.6,
                        },
                    ),
                    InterviewOption(
                        id="wait_act_momentum",
                        label="Wanted momentum instead of stalling",
                        value_tags=["action_bias", "efficiency"],
                        trait_signals={
                            "decisiveness": 0.8,
                            "patience": 0.3,
                            "efficiency": 0.65,
                        },
                    ),
                ],
            ),
            "wait_more": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="wait_more_info",
                        label="Wanted more information first",
                        value_tags=["caution", "information_seeking", "deliberation"],
                        trait_signals={
                            "patience": 0.9,
                            "decisiveness": 0.3,
                            "risk_tolerance": 0.2,
                            "analytical_thinking": 0.85,
                            "intuitive_leaning": 0.25,
                            "future_orientation": 0.7,
                        },
                    ),
                    InterviewOption(
                        id="wait_more_risk",
                        label="Wanted to reduce risk before committing",
                        value_tags=["caution", "foresight"],
                        trait_signals={
                            "risk_tolerance": 0.2,
                            "future_orientation": 0.8,
                            "analytical_thinking": 0.85,
                            "patience": 0.85,
                        },
                    ),
                    InterviewOption(
                        id="wait_more_uncertain",
                        label="Uncertainty felt too high to decide",
                        value_tags=["caution", "information_seeking"],
                        trait_signals={
                            "anxiety_avoidance": 0.65,
                            "patience": 0.85,
                            "analytical_thinking": 0.75,
                        },
                    ),
                ],
            ),
            "small_step": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="wait_step_trial",
                        label="Wanted a small trial first",
                        value_tags=["incremental", "pragmatism", "balance"],
                        trait_signals={
                            "patience": 0.55,
                            "decisiveness": 0.6,
                            "risk_tolerance": 0.5,
                            "analytical_thinking": 0.65,
                            "intuitive_leaning": 0.4,
                            "future_orientation": 0.5,
                        },
                    ),
                    InterviewOption(
                        id="wait_step_learn",
                        label="Learn more before going all in",
                        value_tags=["incremental", "caution"],
                        trait_signals={
                            "analytical_thinking": 0.75,
                            "patience": 0.6,
                            "risk_tolerance": 0.45,
                        },
                    ),
                    InterviewOption(
                        id="wait_step_test",
                        label="Test the waters safely",
                        value_tags=["caution", "pragmatism"],
                        trait_signals={
                            "risk_tolerance": 0.45,
                            "future_orientation": 0.55,
                            "patience": 0.55,
                        },
                    ),
                ],
            ),
        },
    ),
    InterviewScenario(
        id="unexpected_change_v1",
        text="Plans you were counting on suddenly fall through at the last minute.",
        main_question=InterviewQuestion(
            prompt="What would you do?",
            options=[
                InterviewOption(
                    id="make_new_plan",
                    label="Quickly make a new plan",
                    value_tags=["adaptability", "action_bias"],
                    trait_signals={"adaptability": 0.9, "stress_tolerance": 0.7, "patience": 0.3},
                ),
                InterviewOption(
                    id="take_a_break",
                    label="Take a moment to regroup first",
                    value_tags=["self_care", "composure"],
                    trait_signals={"adaptability": 0.5, "stress_tolerance": 0.6, "patience": 0.8},
                ),
                InterviewOption(
                    id="frustrated",
                    label="Feel frustrated and need time to process",
                    value_tags=["emotional_honesty", "processing"],
                    trait_signals={"adaptability": 0.3, "stress_tolerance": 0.3, "patience": 0.5},
                ),
            ],
        ),
        followup_by_main_choice={
            "make_new_plan": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="change_plan_now",
                        label="Needed a new plan right away",
                        value_tags=["adaptability", "action_bias"],
                        trait_signals={
                            "adaptability": 0.9,
                            "stress_tolerance": 0.7,
                            "patience": 0.3,
                            "analytical_thinking": 0.35,
                            "intuitive_leaning": 0.65,
                            "decisiveness": 0.6,
                        },
                    ),
                    InterviewOption(
                        id="change_plan_keep_moving",
                        label="Couldn't afford to freeze",
                        value_tags=["resilience", "action_bias"],
                        trait_signals={
                            "decisiveness": 0.7,
                            "stress_tolerance": 0.65,
                            "adaptability": 0.85,
                        },
                    ),
                    InterviewOption(
                        id="change_plan_adapt",
                        label="I'm used to adapting when plans break",
                        value_tags=["adaptability", "experience"],
                        trait_signals={
                            "adaptability": 0.85,
                            "stress_tolerance": 0.65,
                            "self_direction": 0.6,
                        },
                    ),
                ],
            ),
            "take_a_break": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="change_break_regroup",
                        label="Needed a moment to regroup",
                        value_tags=["self_care", "composure"],
                        trait_signals={
                            "adaptability": 0.5,
                            "stress_tolerance": 0.6,
                            "patience": 0.85,
                            "analytical_thinking": 0.8,
                            "intuitive_leaning": 0.2,
                            "risk_tolerance": 0.25,
                        },
                    ),
                    InterviewOption(
                        id="change_break_clear_head",
                        label="Needed a clearer head before deciding",
                        value_tags=["self_care", "deliberation"],
                        trait_signals={
                            "self_care": 0.8,
                            "patience": 0.8,
                            "analytical_thinking": 0.75,
                        },
                    ),
                    InterviewOption(
                        id="change_break_nerves",
                        label="My nerves needed a reset first",
                        value_tags=["self_care", "stress"],
                        trait_signals={
                            "stress_tolerance": 0.45,
                            "self_care": 0.75,
                            "anxiety_avoidance": 0.55,
                        },
                    ),
                ],
            ),
            "frustrated": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="change_proc_feelings",
                        label="Let myself feel the frustration first",
                        value_tags=["processing", "emotional_honesty"],
                        trait_signals={
                            "adaptability": 0.35,
                            "stress_tolerance": 0.4,
                            "patience": 0.6,
                            "self_care": 0.75,
                            "analytical_thinking": 0.5,
                            "intuitive_leaning": 0.45,
                        },
                    ),
                    InterviewOption(
                        id="change_proc_honest",
                        label="Bad news hits me hard before I pivot",
                        value_tags=["emotional_honesty", "processing"],
                        trait_signals={
                            "stress_tolerance": 0.35,
                            "self_care": 0.65,
                            "empathy": 0.45,
                        },
                    ),
                    InterviewOption(
                        id="change_proc_space",
                        label="Needed space before I could think straight",
                        value_tags=["processing", "self_care"],
                        trait_signals={
                            "patience": 0.65,
                            "self_care": 0.7,
                            "analytical_thinking": 0.55,
                        },
                    ),
                ],
            ),
        },
    ),
    InterviewScenario(
        id="favor_request_v1",
        text="Someone asks you to do them a favour that's inconvenient but not impossible.",
        main_question=InterviewQuestion(
            prompt="What would you do?",
            options=[
                InterviewOption(
                    id="say_yes",
                    label="Say yes without hesitation",
                    value_tags=["agreeableness", "generosity"],
                    trait_signals={"boundary_setting": 0.2, "empathy": 0.8, "agreeableness": 0.9},
                ),
                InterviewOption(
                    id="negotiate",
                    label="Agree but suggest a different way or time",
                    value_tags=["diplomacy", "balance"],
                    trait_signals={"boundary_setting": 0.6, "empathy": 0.6, "agreeableness": 0.6},
                ),
                InterviewOption(
                    id="decline",
                    label="Politely say no",
                    value_tags=["boundaries", "self_care"],
                    trait_signals={"boundary_setting": 0.9, "empathy": 0.4, "agreeableness": 0.3},
                ),
            ],
        ),
        followup_by_main_choice={
            "say_yes": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="favor_yes_easy",
                        label="It felt easy to say yes",
                        value_tags=["agreeableness", "generosity"],
                        trait_signals={
                            "agreeableness": 0.9,
                            "empathy": 0.75,
                            "boundary_setting": 0.25,
                            "analytical_thinking": 0.35,
                            "intuitive_leaning": 0.6,
                        },
                    ),
                    InterviewOption(
                        id="favor_yes_disappoint",
                        label="Didn't want to disappoint them",
                        value_tags=["agreeableness", "empathy"],
                        trait_signals={
                            "empathy": 0.85,
                            "agreeableness": 0.85,
                            "anxiety_avoidance": 0.55,
                        },
                    ),
                    InterviewOption(
                        id="favor_yes_right_thing",
                        label="Felt like the right thing to do",
                        value_tags=["generosity", "integrity"],
                        trait_signals={
                            "empathy": 0.75,
                            "self_direction": 0.65,
                            "agreeableness": 0.8,
                        },
                    ),
                ],
            ),
            "negotiate": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="favor_neg_time",
                        label="Wanted to help on my terms",
                        value_tags=["diplomacy", "balance"],
                        trait_signals={
                            "boundary_setting": 0.65,
                            "empathy": 0.6,
                            "agreeableness": 0.6,
                            "analytical_thinking": 0.75,
                            "future_orientation": 0.55,
                        },
                    ),
                    InterviewOption(
                        id="favor_neg_partial",
                        label="Could only manage part of it",
                        value_tags=["compromise", "honesty"],
                        trait_signals={
                            "boundary_setting": 0.6,
                            "empathy": 0.6,
                            "self_care": 0.55,
                        },
                    ),
                    InterviewOption(
                        id="favor_neg_fair",
                        label="Wanted something fair for both of us",
                        value_tags=["diplomacy", "fairness"],
                        trait_signals={
                            "empathy": 0.65,
                            "analytical_thinking": 0.7,
                            "agreeableness": 0.55,
                        },
                    ),
                ],
            ),
            "decline": InterviewQuestion(
                prompt="Why that approach?",
                options=[
                    InterviewOption(
                        id="favor_decl_protect",
                        label="Needed to protect my time or energy",
                        value_tags=["boundaries", "self_care"],
                        trait_signals={
                            "boundary_setting": 0.9,
                            "self_care": 0.85,
                            "empathy": 0.4,
                            "agreeableness": 0.3,
                            "analytical_thinking": 0.85,
                            "intuitive_leaning": 0.2,
                        },
                    ),
                    InterviewOption(
                        id="favor_decl_stretched",
                        label="Already stretched too thin",
                        value_tags=["honesty", "self_care"],
                        trait_signals={
                            "self_care": 0.85,
                            "stress_tolerance": 0.4,
                            "boundary_setting": 0.85,
                        },
                    ),
                    InterviewOption(
                        id="favor_decl_not_fit",
                        label="Wasn't the right fit for me to take on",
                        value_tags=["boundaries", "self_direction"],
                        trait_signals={
                            "boundary_setting": 0.85,
                            "self_direction": 0.7,
                            "empathy": 0.45,
                        },
                    ),
                ],
            ),
        },
    ),
]


def get_followup_question_for_main(
    scenario: InterviewScenario, main_choice_id: str
) -> InterviewQuestion:
    """Return the follow-up question for the selected main option."""
    q = scenario.followup_by_main_choice.get(main_choice_id)
    if q is None:
        raise KeyError(
            f"No follow-up question for main choice {main_choice_id!r} "
            f"in scenario {scenario.id!r}"
        )
    return q


def get_scenario(scenario_id: Optional[str] = None) -> InterviewScenario:
    """Return a scenario by id, or the first available scenario."""
    if scenario_id:
        for s in SCENARIOS:
            if s.id == scenario_id:
                return s
        raise ValueError(f"Unknown scenario: {scenario_id}")
    return SCENARIOS[0]


def get_scenarios_for_session(count: int = 4,
                              session_index: int = 0) -> List[InterviewScenario]:
    """Return a deterministic rotating selection of scenarios.

    ``session_index`` controls which slice of the scenario pool is used.
    Each successive index shifts the starting position by ``count`` so
    that repeated sessions cycle through different scenarios before
    repeating.  When all scenarios have been covered, the cycle wraps.
    """
    count = max(3, min(count, 5))
    pool = list(SCENARIOS)
    n = len(pool)
    if count >= n:
        return pool[:count]
    # Shift the window start by 1 each session so consecutive runs are visibly
    # different, even when ``count`` divides the scenario pool size.
    offset = session_index % n
    selected: List[InterviewScenario] = []
    for i in range(count):
        selected.append(pool[(offset + i) % n])
    return selected


# ---------------------------------------------------------------------------
# Signal extraction (deterministic, no LLM)
# ---------------------------------------------------------------------------

def extract_signals(
    main_choice: InterviewOption,
    followup_choice: InterviewOption,
) -> Tuple[List[str], Dict[str, float]]:
    """Merge value tags and trait signals from both choices.

    Trait signals are merged deterministically with a main-choice bias:
    - If both choices contribute to the same trait, main gets higher weight.
    - If only the follow-up contributes to a trait, that signal is softened
      toward a neutral baseline so it can't overpower the main choice.
    """
    merged_tags = sorted(set(main_choice.value_tags + followup_choice.value_tags))

    MAIN_WEIGHT = 0.7
    FOLLOW_WEIGHT = 1.0 - MAIN_WEIGHT
    NEUTRAL = 0.5

    merged_signals: Dict[str, float] = {}
    all_traits = set(main_choice.trait_signals.keys()) | set(followup_choice.trait_signals.keys())
    for trait in all_traits:
        in_main = trait in main_choice.trait_signals
        in_follow = trait in followup_choice.trait_signals

        if in_main and in_follow:
            m = main_choice.trait_signals[trait]
            f = followup_choice.trait_signals[trait]
            merged = MAIN_WEIGHT * m + FOLLOW_WEIGHT * f
        elif in_main:
            # If follow-up has no view on this trait, don't dilute the signal.
            merged = main_choice.trait_signals[trait]
        else:
            # Follow-up-only traits get softened to prevent "reason wording"
            # from overpowering the actual action choice.
            merged = MAIN_WEIGHT * NEUTRAL + FOLLOW_WEIGHT * followup_choice.trait_signals[trait]

        merged_signals[trait] = round(float(merged), 2)

    return merged_tags, merged_signals


def compute_confidence(main_choice: InterviewOption,
                       followup_choice: InterviewOption) -> float:
    """Compute a simple confidence score for the combined response.

    Higher when the main and follow-up choices share overlapping
    value tags (i.e. the user's reasoning is internally consistent).
    """
    main_tags = set(main_choice.value_tags)
    followup_tags = set(followup_choice.value_tags)
    union = main_tags | followup_tags
    if not union:
        return 0.5
    overlap = len(main_tags & followup_tags) / len(union)
    return round(0.6 + 0.4 * overlap, 2)


# ---------------------------------------------------------------------------
# Deterministic response variation
# ---------------------------------------------------------------------------

_CONFIRM_PHRASES = [
    "Got it.",
    "Noted.",
    "Understood.",
    "Thanks for sharing that.",
    "Alright.",
]

_TRANSITION_PHRASES = [
    "Here's another one.",
    "Next scenario.",
    "Moving on.",
    "One more.",
    "Let's keep going.",
]

_ACCURATE_PHRASES = [
    "Good to know.",
    "Thanks for confirming.",
    "That helps.",
    "Noted, glad it's on track.",
    "Alright, keeping those as-is.",
]

_PARTIAL_PHRASES = [
    "Thanks for the correction.",
    "Good catch, updating that.",
    "Appreciate the adjustment.",
    "Noted, that's helpful.",
    "Thanks, refining the model.",
]

_FULL_CORRECTION_PHRASES = [
    "Thanks for setting the record straight.",
    "Good to know, adjusting.",
    "Appreciate the honesty.",
    "Noted, that changes things.",
    "Got it, will recalibrate.",
]


def _rotate_phrase(phrases: List[str], counter: int) -> str:
    """Pick a phrase by rotating through the list deterministically."""
    return phrases[counter % len(phrases)]


# ---------------------------------------------------------------------------
# Reflection generation (deterministic)
# ---------------------------------------------------------------------------

_TRAIT_REFLECTION_MAP: Dict[str, str] = {
    "patience": "You seem to slow down before acting",
    "risk_tolerance": "You may accept some risk when it helps you move forward",
    "thoroughness": "You usually prefer doing things carefully",
    "empathy": "You tend to consider how others feel",
    "boundary_setting": "You seem to know when to say no",
    "directness": "You lean toward being straightforward",
    "conflict_comfort": "You don't seem to shy away from difficult conversations",
    "analytical_thinking": "You tend to think things through",
    "intuitive_leaning": "When time is tight, you sometimes rely on instinct",
    "adaptability": "You adjust pretty quickly when plans change",
    "self_care": "You seem to look after yourself",
    "future_orientation": "You usually try to avoid unnecessary problems",
    "decisiveness": "You tend to make decisions without a lot of back-and-forth",
    "self_direction": "You seem to trust your own judgment",
    "financial_caution": "You seem careful with money",
    "anxiety_avoidance": "You tend to steer clear of situations that might be stressful",
    "agreeableness": "You tend to go along with what others need",
    "resourcefulness": "You seem to find practical ways around problems",
    "stress_tolerance": "You seem to handle pressure reasonably well",
    "self_reward": "You let yourself enjoy things",
    "efficiency": "You value getting things done quickly",
}

_TRAIT_LOW_REFLECTION_MAP: Dict[str, str] = {
    "patience": "You tend to move quickly rather than wait",
    "risk_tolerance": "You seem to avoid unnecessary risk and prefer lower-regret choices",
    "thoroughness": "You don't get bogged down in details",
    "empathy": "You focus more on the practical side than feelings",
    "boundary_setting": "You tend to say yes even when it's hard",
    "directness": "You tend to be more diplomatic than blunt",
    "conflict_comfort": "You tend to avoid confrontation",
    "analytical_thinking": "When you're deciding, you may lean on what feels right instead of careful analysis",
    "adaptability": "Changes of plan tend to bother you",
    "decisiveness": "You prefer to take your time before deciding",
}

_REFLECTION_THRESHOLD_HIGH = 0.65
_REFLECTION_THRESHOLD_LOW = 0.35
_MIN_EVIDENCE_FOR_REFLECTION = 2

_TRAIT_HIGH_THRESHOLD_OVERRIDES: Dict[str, float] = {
    # Require stronger evidence before surfacing "gut/instinct" style.
    "intuitive_leaning": 0.7,
}


def generate_reflections(results: List[InterviewResult]) -> List[ReflectionLine]:
    """Generate lightweight reflection lines from accumulated trait signals.

    Only reflects traits with enough evidence and reasonably strong signal.
    """
    trait_accum: Dict[str, List[float]] = {}
    for result in results:
        for trait, value in result.trait_signals.items():
            trait_accum.setdefault(trait, []).append(value)

    reflections: List[ReflectionLine] = []
    for trait, values in sorted(trait_accum.items()):
        if len(values) < _MIN_EVIDENCE_FOR_REFLECTION:
            continue
        avg = sum(values) / len(values)
        high_threshold = _TRAIT_HIGH_THRESHOLD_OVERRIDES.get(trait, _REFLECTION_THRESHOLD_HIGH)
        if avg >= high_threshold and trait in _TRAIT_REFLECTION_MAP:
            reflections.append(ReflectionLine(
                text=_TRAIT_REFLECTION_MAP[trait],
                trait=trait,
                strength=round(avg, 2),
            ))
        elif avg <= _REFLECTION_THRESHOLD_LOW and trait in _TRAIT_LOW_REFLECTION_MAP:
            reflections.append(ReflectionLine(
                text=_TRAIT_LOW_REFLECTION_MAP[trait],
                trait=trait,
                strength=round(avg, 2),
            ))

    reflections.sort(key=lambda r: abs(r.strength - 0.5), reverse=True)
    return reflections[:5]


# ---------------------------------------------------------------------------
# Correction loop
# ---------------------------------------------------------------------------

def _prompt_correction(reflections: List[ReflectionLine], phrase_counter: int) -> CorrectionResult:
    """Run the correction loop after reflection display."""
    print("\n  How does that sound?\n")
    print("    1. Accurate")
    print("    2. Partially true")
    print("    3. Not really")

    while True:
        try:
            raw = input("\n  Your answer: ").strip()
            if not raw:
                continue
            choice = int(raw)
            if choice in (1, 2, 3):
                break
            print("  Please enter 1, 2, or 3.")
        except ValueError:
            print("  Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            raise

    if choice == 1:
        print(f"\n  {_rotate_phrase(_ACCURATE_PHRASES, phrase_counter)}")
        return CorrectionResult(
            status="accurate",
            accepted_traits=[r.trait for r in reflections],
        )

    if choice == 2:
        return _handle_partial_correction(reflections, phrase_counter)

    return _handle_full_correction(reflections, phrase_counter)


def _handle_partial_correction(
    reflections: List[ReflectionLine], phrase_counter: int
) -> CorrectionResult:
    """Let user mark which traits were right and which were wrong."""
    accepted: List[str] = []
    rejected: List[str] = []

    print("\n  Which of these felt right? (enter numbers separated by spaces)")
    for idx, ref in enumerate(reflections, 1):
        print(f"    {idx}. {ref.text}")

    while True:
        try:
            raw = input("\n  Right ones: ").strip()
            if not raw:
                continue
            nums = [int(x) for x in raw.split()]
            if all(1 <= n <= len(reflections) for n in nums):
                break
            print(f"  Please enter numbers between 1 and {len(reflections)}.")
        except ValueError:
            print("  Please enter numbers separated by spaces.")
        except (EOFError, KeyboardInterrupt):
            raise

    right_set = set(nums)
    for idx, ref in enumerate(reflections, 1):
        if idx in right_set:
            accepted.append(ref.trait)
        else:
            rejected.append(ref.trait)

    print(f"\n  {_rotate_phrase(_PARTIAL_PHRASES, phrase_counter)}")
    return CorrectionResult(
        status="partially_true",
        accepted_traits=accepted,
        rejected_traits=rejected,
    )


def _handle_full_correction(
    reflections: List[ReflectionLine], phrase_counter: int
) -> CorrectionResult:
    """Provide a low-effort correction path when reflection is mostly wrong."""
    replacements: Dict[str, str] = {}

    print("\n  No problem. Pick a better fit for each one:")
    for idx, ref in enumerate(reflections, 1):
        print(f"\n  {idx}. Instead of \"{ref.text}\", would you say:")
        print("     a. The opposite is true")
        print("     b. It depends on the situation")
        print("     c. I'm not sure")

        while True:
            try:
                raw = input("     Your pick: ").strip().lower()
                if raw in ("a", "b", "c"):
                    break
                print("     Please enter a, b, or c.")
            except ValueError:
                print("     Please enter a, b, or c.")
            except (EOFError, KeyboardInterrupt):
                raise

        replacements[ref.trait] = {"a": "opposite", "b": "contextual", "c": "unsure"}[raw]

    print(f"\n  {_rotate_phrase(_FULL_CORRECTION_PHRASES, phrase_counter)}")
    return CorrectionResult(
        status="not_really",
        rejected_traits=[r.trait for r in reflections],
        replacement_choices=replacements,
    )


# ---------------------------------------------------------------------------
# Interview runner
# ---------------------------------------------------------------------------

def _prompt_choice(question: InterviewQuestion) -> InterviewOption:
    """Display a question and collect the user's numbered choice."""
    print(f"\n  {question.prompt}\n")
    for idx, option in enumerate(question.options, 1):
        print(f"    {idx}. {option.label}")

    while True:
        try:
            raw = input("\n  Your choice: ").strip()
            if not raw:
                continue
            choice_idx = int(raw)
            if 1 <= choice_idx <= len(question.options):
                return question.options[choice_idx - 1]
            print(f"  Please enter a number between 1 and {len(question.options)}.")
        except ValueError:
            print("  Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            raise


def run_interview(scenario: Optional[InterviewScenario] = None) -> InterviewResult:
    """Run a single interview scenario interactively and return the result.

    Kept for backwards compatibility with Phase 26.
    Raises EOFError / KeyboardInterrupt if the user cancels.
    """
    if scenario is None:
        scenario = get_scenario()

    print("\n" + "=" * 56)
    print("  DECISION INTERVIEW")
    print("=" * 56)
    print(f"\n  {scenario.text}")

    main_choice = _prompt_choice(scenario.main_question)
    followup_q = get_followup_question_for_main(scenario, main_choice.id)
    followup_choice = _prompt_choice(followup_q)

    value_tags, trait_signals = extract_signals(main_choice, followup_choice)
    confidence = compute_confidence(main_choice, followup_choice)

    return InterviewResult(
        scenario_id=scenario.id,
        scenario_text=scenario.text,
        choice_label=main_choice.label,
        choice_value=main_choice.id,
        reasoning_label=followup_choice.label,
        reasoning_value=followup_choice.id,
        value_tags=value_tags,
        trait_signals=trait_signals,
        confidence_score=confidence,
    )


def run_interview_session(scenario_count: int = 4,
                          session_index: int = 0) -> SessionResult:
    """Run a multi-scenario interview session with reflection and correction.

    ``session_index`` is forwarded to :func:`get_scenarios_for_session`
    so that repeated runs cycle through different scenario subsets.

    Steps:
    1. Present 3-5 scenarios, collecting one main + one reasoning choice each
    2. After all scenarios, generate a lightweight reflection
    3. Run the correction loop on the reflection
    4. Return the full session result

    Raises EOFError / KeyboardInterrupt if the user cancels mid-session.
    """
    scenarios = get_scenarios_for_session(scenario_count, session_index)
    results: List[InterviewResult] = []
    phrase_counter = 0

    print("\n" + "=" * 56)
    print("  DECISION INTERVIEW")
    print("=" * 56)
    print(f"\n  This will be a short session — {len(scenarios)} quick scenarios.")
    print("  Just pick whatever feels closest. There are no wrong answers.")

    for i, scenario in enumerate(scenarios):
        if i > 0:
            print(f"\n  {_rotate_phrase(_TRANSITION_PHRASES, phrase_counter)}")
            phrase_counter += 1

        print(f"\n  --- Scenario {i + 1} of {len(scenarios)} ---")
        print(f"\n  {scenario.text}")

        main_choice = _prompt_choice(scenario.main_question)
        followup_q = get_followup_question_for_main(scenario, main_choice.id)
        followup_choice = _prompt_choice(followup_q)

        value_tags, trait_signals = extract_signals(main_choice, followup_choice)
        confidence = compute_confidence(main_choice, followup_choice)

        result = InterviewResult(
            scenario_id=scenario.id,
            scenario_text=scenario.text,
            choice_label=main_choice.label,
            choice_value=main_choice.id,
            reasoning_label=followup_choice.label,
            reasoning_value=followup_choice.id,
            value_tags=value_tags,
            trait_signals=trait_signals,
            confidence_score=confidence,
        )
        results.append(result)

        print(f"\n  {_rotate_phrase(_CONFIRM_PHRASES, phrase_counter)}")
        phrase_counter += 1

    # Reflection
    reflections = generate_reflections(results)
    correction: Optional[CorrectionResult] = None

    if reflections:
        print("\n" + "-" * 56)
        print("  Based on your answers, here's what I'm noticing:\n")
        for ref in reflections:
            print(f"    - {ref.text}")

        correction = _prompt_correction(reflections, phrase_counter)
    else:
        print("\n  Not enough data yet for a clear reflection.")

    return SessionResult(
        results=results,
        reflections=reflections,
        correction=correction,
    )
