"""
Decision Interview Module (Phase 26)

Presents structured decision scenarios, collects user choices,
and extracts deterministic trait/value signals from responses.
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


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: List[InterviewScenario] = [
    InterviewScenario(
        id="deadline_tradeoff_v1",
        text=(
            "You're working on a project and the deadline is tomorrow.\n"
            "You discover a shortcut that would save time, but it skips\n"
            "some testing you'd normally do."
        ),
        main_question=InterviewQuestion(
            prompt="What would you do?",
            options=[
                InterviewOption(
                    id="take_shortcut",
                    label="Take the shortcut and ship on time",
                    value_tags=["speed", "pragmatism"],
                    trait_signals={
                        "risk_tolerance": 0.7,
                        "thoroughness": 0.3,
                        "deadline_orientation": 0.9,
                    },
                ),
                InterviewOption(
                    id="skip_shortcut",
                    label="Skip the shortcut and do the full testing",
                    value_tags=["quality", "caution"],
                    trait_signals={
                        "risk_tolerance": 0.2,
                        "thoroughness": 0.9,
                        "deadline_orientation": 0.3,
                    },
                ),
                InterviewOption(
                    id="partial_test",
                    label="Do a quick partial test, then ship",
                    value_tags=["balance", "pragmatism"],
                    trait_signals={
                        "risk_tolerance": 0.5,
                        "thoroughness": 0.6,
                        "deadline_orientation": 0.7,
                    },
                ),
            ],
        ),
        followup_question=InterviewQuestion(
            prompt="Why did you pick that approach?",
            options=[
                InterviewOption(
                    id="reason_reputation",
                    label="I don't want to look unreliable",
                    value_tags=["reputation", "external_motivation"],
                    trait_signals={
                        "external_validation": 0.8,
                        "self_direction": 0.3,
                    },
                ),
                InterviewOption(
                    id="reason_quality",
                    label="I care more about getting it right",
                    value_tags=["quality", "internal_motivation"],
                    trait_signals={
                        "external_validation": 0.2,
                        "self_direction": 0.8,
                    },
                ),
                InterviewOption(
                    id="reason_practical",
                    label="It felt like the most practical option",
                    value_tags=["pragmatism", "efficiency"],
                    trait_signals={
                        "external_validation": 0.4,
                        "self_direction": 0.6,
                    },
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
