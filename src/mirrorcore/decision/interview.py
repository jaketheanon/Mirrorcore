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
    """A complete interview scenario with a main choice and a follow-up."""
    id: str
    text: str
    main_question: InterviewQuestion
    followup_question: InterviewQuestion


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
        followup_question=InterviewQuestion(
            prompt="Why that approach?",
            options=[
                InterviewOption(
                    id="felt_right",
                    label="Felt right",
                    value_tags=["intuition", "gut_feeling"],
                    trait_signals={"analytical_thinking": 0.3, "intuitive_leaning": 0.8},
                ),
                InterviewOption(
                    id="save_money",
                    label="Wanted to save money",
                    value_tags=["frugality", "security"],
                    trait_signals={"financial_caution": 0.9, "self_direction": 0.6},
                ),
                InterviewOption(
                    id="no_problems_later",
                    label="Didn't want problems later",
                    value_tags=["foresight", "caution"],
                    trait_signals={"future_orientation": 0.8, "anxiety_avoidance": 0.6},
                ),
            ],
        ),
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
        followup_question=InterviewQuestion(
            prompt="Why that approach?",
            options=[
                InterviewOption(
                    id="didnt_think_much",
                    label="Didn't think much about it",
                    value_tags=["spontaneity", "low_deliberation"],
                    trait_signals={"analytical_thinking": 0.2, "intuitive_leaning": 0.7},
                ),
                InterviewOption(
                    id="seemed_safest",
                    label="Seemed safest",
                    value_tags=["safety", "caution"],
                    trait_signals={"risk_tolerance": 0.2, "anxiety_avoidance": 0.8},
                ),
                InterviewOption(
                    id="thought_it_through",
                    label="Thought it through",
                    value_tags=["deliberation", "planning"],
                    trait_signals={"analytical_thinking": 0.8, "self_direction": 0.7},
                ),
            ],
        ),
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
        followup_question=InterviewQuestion(
            prompt="Why that approach?",
            options=[
                InterviewOption(
                    id="felt_right",
                    label="Felt right",
                    value_tags=["intuition", "gut_feeling"],
                    trait_signals={"analytical_thinking": 0.3, "intuitive_leaning": 0.8},
                ),
                InterviewOption(
                    id="no_problems_later",
                    label="Didn't want problems later",
                    value_tags=["foresight", "caution"],
                    trait_signals={"future_orientation": 0.8, "anxiety_avoidance": 0.6},
                ),
                InterviewOption(
                    id="thought_it_through",
                    label="Thought it through",
                    value_tags=["deliberation", "planning"],
                    trait_signals={"analytical_thinking": 0.8, "self_direction": 0.7},
                ),
            ],
        ),
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
        followup_question=InterviewQuestion(
            prompt="Why that approach?",
            options=[
                InterviewOption(
                    id="didnt_think_much",
                    label="Didn't think much about it",
                    value_tags=["spontaneity", "low_deliberation"],
                    trait_signals={"analytical_thinking": 0.2, "intuitive_leaning": 0.7},
                ),
                InterviewOption(
                    id="seemed_safest",
                    label="Seemed safest",
                    value_tags=["safety", "caution"],
                    trait_signals={"risk_tolerance": 0.2, "anxiety_avoidance": 0.8},
                ),
                InterviewOption(
                    id="save_money",
                    label="Wanted to save money",
                    value_tags=["frugality", "security"],
                    trait_signals={"financial_caution": 0.9, "self_direction": 0.6},
                ),
            ],
        ),
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
        followup_question=InterviewQuestion(
            prompt="Why that approach?",
            options=[
                InterviewOption(
                    id="felt_right",
                    label="Felt right",
                    value_tags=["intuition", "gut_feeling"],
                    trait_signals={"analytical_thinking": 0.3, "intuitive_leaning": 0.8},
                ),
                InterviewOption(
                    id="thought_it_through",
                    label="Thought it through",
                    value_tags=["deliberation", "planning"],
                    trait_signals={"analytical_thinking": 0.8, "self_direction": 0.7},
                ),
                InterviewOption(
                    id="no_problems_later",
                    label="Didn't want problems later",
                    value_tags=["foresight", "caution"],
                    trait_signals={"future_orientation": 0.8, "anxiety_avoidance": 0.6},
                ),
            ],
        ),
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
        followup_question=InterviewQuestion(
            prompt="Why that approach?",
            options=[
                InterviewOption(
                    id="felt_right",
                    label="Felt right",
                    value_tags=["intuition", "gut_feeling"],
                    trait_signals={"analytical_thinking": 0.3, "intuitive_leaning": 0.8},
                ),
                InterviewOption(
                    id="seemed_safest",
                    label="Seemed safest",
                    value_tags=["safety", "caution"],
                    trait_signals={"risk_tolerance": 0.2, "anxiety_avoidance": 0.8},
                ),
                InterviewOption(
                    id="didnt_think_much",
                    label="Didn't think much about it",
                    value_tags=["spontaneity", "low_deliberation"],
                    trait_signals={"analytical_thinking": 0.2, "intuitive_leaning": 0.7},
                ),
            ],
        ),
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
        followup_question=InterviewQuestion(
            prompt="Why that approach?",
            options=[
                InterviewOption(
                    id="no_problems_later",
                    label="Didn't want problems later",
                    value_tags=["foresight", "caution"],
                    trait_signals={"future_orientation": 0.8, "anxiety_avoidance": 0.6},
                ),
                InterviewOption(
                    id="felt_right",
                    label="Felt right",
                    value_tags=["intuition", "gut_feeling"],
                    trait_signals={"analytical_thinking": 0.3, "intuitive_leaning": 0.8},
                ),
                InterviewOption(
                    id="thought_it_through",
                    label="Thought it through",
                    value_tags=["deliberation", "planning"],
                    trait_signals={"analytical_thinking": 0.8, "self_direction": 0.7},
                ),
            ],
        ),
    ),
]


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
    offset = (session_index * count) % n
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

    Trait signals from main and follow-up are averaged when both
    contribute to the same trait; otherwise they are kept as-is.
    """
    merged_tags = sorted(set(main_choice.value_tags + followup_choice.value_tags))

    all_traits: Dict[str, List[float]] = {}
    for signals in (main_choice.trait_signals, followup_choice.trait_signals):
        for trait, value in signals.items():
            all_traits.setdefault(trait, []).append(value)

    merged_signals = {
        trait: round(sum(vals) / len(vals), 2)
        for trait, vals in all_traits.items()
    }

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
    "patience": "You seem to think before acting",
    "risk_tolerance": "You tend to be comfortable with some risk",
    "thoroughness": "You usually prefer doing things carefully",
    "empathy": "You tend to consider how others feel",
    "boundary_setting": "You seem to know when to say no",
    "directness": "You lean toward being straightforward",
    "conflict_comfort": "You don't seem to shy away from difficult conversations",
    "analytical_thinking": "You tend to think things through",
    "intuitive_leaning": "You often go with your gut",
    "adaptability": "You seem to adjust quickly when things change",
    "self_care": "You seem to look after yourself",
    "future_orientation": "You usually try to avoid unnecessary problems",
    "decisiveness": "You tend to make decisions without a lot of back-and-forth",
    "self_direction": "You seem to trust your own judgment",
    "financial_caution": "You seem careful with money",
    "anxiety_avoidance": "You tend to steer clear of situations that might cause stress",
    "agreeableness": "You tend to go along with what others need",
    "resourcefulness": "You seem to find practical ways around problems",
    "stress_tolerance": "You seem to handle pressure reasonably well",
    "self_reward": "You give yourself permission to enjoy things",
    "efficiency": "You value getting things done quickly",
}

_TRAIT_LOW_REFLECTION_MAP: Dict[str, str] = {
    "patience": "You tend to act quickly rather than wait",
    "risk_tolerance": "You seem careful about risk",
    "thoroughness": "You don't get bogged down in details",
    "empathy": "You focus more on the practical side than feelings",
    "boundary_setting": "You tend to say yes even when it's hard",
    "directness": "You tend to be diplomatic rather than blunt",
    "conflict_comfort": "You tend to avoid confrontation",
    "analytical_thinking": "You tend to decide intuitively rather than analytically",
    "adaptability": "Changes of plan tend to bother you",
    "decisiveness": "You prefer to take your time before deciding",
}

_REFLECTION_THRESHOLD_HIGH = 0.65
_REFLECTION_THRESHOLD_LOW = 0.35
_MIN_EVIDENCE_FOR_REFLECTION = 2


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
        if avg >= _REFLECTION_THRESHOLD_HIGH and trait in _TRAIT_REFLECTION_MAP:
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
    followup_choice = _prompt_choice(scenario.followup_question)

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
        followup_choice = _prompt_choice(scenario.followup_question)

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
