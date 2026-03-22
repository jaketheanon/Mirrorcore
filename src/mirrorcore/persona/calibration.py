"""
Style / Persona Calibration Module (Phase 28)

Deterministic, low-effort style calibration with structured prompts,
reflection generation, and correction loop support.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StyleOption:
    id: str
    label: str
    style_tags: List[str] = field(default_factory=list)
    tone_signals: Dict[str, float] = field(default_factory=dict)


@dataclass
class StylePrompt:
    id: str
    text: str
    options: List[StyleOption] = field(default_factory=list)


@dataclass
class StyleCalibrationResult:
    prompt_id: str
    prompt_text: str
    selected_label: str
    selected_value: str
    style_tags: List[str]
    tone_signals: Dict[str, float]
    confidence_score: float
    optional_notes: Optional[str] = None


@dataclass
class StyleReflectionLine:
    text: str
    trait: str
    strength: float


@dataclass
class StyleCorrectionResult:
    status: str  # accurate, partially_true, not_really
    accepted_traits: List[str] = field(default_factory=list)
    rejected_traits: List[str] = field(default_factory=list)
    replacement_choices: Dict[str, str] = field(default_factory=dict)


@dataclass
class StyleCalibrationSession:
    results: List[StyleCalibrationResult]
    reflections: List[StyleReflectionLine]
    correction: Optional[StyleCorrectionResult]


STYLE_PROMPTS: List[StylePrompt] = [
    StylePrompt(
        id="blunt_vs_gentle_v1",
        text="If someone annoyed you, which reply is more like you?",
        options=[
            StyleOption(
                id="straight_callout",
                label="Straight to the point",
                style_tags=["blunt", "direct"],
                tone_signals={"bluntness": 0.9, "diplomacy": 0.2, "warmth": 0.3},
            ),
            StyleOption(
                id="soft_callout",
                label="Softer and polite",
                style_tags=["gentle", "polite"],
                tone_signals={"bluntness": 0.2, "diplomacy": 0.8, "warmth": 0.7},
            ),
            StyleOption(
                id="light_sarcastic_callout",
                label="A little sarcastic",
                style_tags=["sarcastic", "direct"],
                tone_signals={"sarcasm": 0.8, "bluntness": 0.6, "warmth": 0.3},
            ),
        ],
    ),
    StylePrompt(
        id="short_vs_detailed_v1",
        text="When you explain something, what usually sounds more natural?",
        options=[
            StyleOption(
                id="short_direct",
                label="Short and direct",
                style_tags=["concise", "direct"],
                tone_signals={"verbosity": 0.2, "bluntness": 0.7, "seriousness": 0.7},
            ),
            StyleOption(
                id="balanced_length",
                label="Clear with some detail",
                style_tags=["balanced", "clear"],
                tone_signals={"verbosity": 0.5, "diplomacy": 0.5, "seriousness": 0.6},
            ),
            StyleOption(
                id="full_explanation",
                label="More detailed",
                style_tags=["detailed", "thorough"],
                tone_signals={"verbosity": 0.85, "diplomacy": 0.6, "seriousness": 0.7},
            ),
        ],
    ),
    StylePrompt(
        id="direct_vs_diplomatic_v1",
        text="Which one feels closer to your normal tone?",
        options=[
            StyleOption(
                id="say_exactly",
                label="Say exactly what I mean",
                style_tags=["direct", "explicit"],
                tone_signals={"diplomacy": 0.2, "bluntness": 0.8, "seriousness": 0.7},
            ),
            StyleOption(
                id="tone_it_down",
                label="Tone it down a bit",
                style_tags=["diplomatic", "considered"],
                tone_signals={"diplomacy": 0.8, "warmth": 0.7, "bluntness": 0.3},
            ),
            StyleOption(
                id="depends_context",
                label="Depends on who I'm talking to",
                style_tags=["contextual", "adaptive"],
                tone_signals={"diplomacy": 0.6, "bluntness": 0.5, "seriousness": 0.6},
            ),
        ],
    ),
    StylePrompt(
        id="casual_vs_serious_v1",
        text="How would you usually say this in everyday chat?",
        options=[
            StyleOption(
                id="keep_it_chill",
                label="Keep it chill",
                style_tags=["casual", "relaxed"],
                tone_signals={"casualness": 0.9, "seriousness": 0.2, "warmth": 0.7},
            ),
            StyleOption(
                id="neutral_focused",
                label="Neutral and focused",
                style_tags=["neutral", "focused"],
                tone_signals={"casualness": 0.5, "seriousness": 0.6, "warmth": 0.5},
            ),
            StyleOption(
                id="formal_serious",
                label="Serious and formal",
                style_tags=["serious", "formal"],
                tone_signals={"casualness": 0.1, "seriousness": 0.9, "warmth": 0.4},
            ),
        ],
    ),
    StylePrompt(
        id="sarcastic_vs_straight_v1",
        text="Which reply sounds more like you?",
        options=[
            StyleOption(
                id="no_sarcasm",
                label="Straight, no sarcasm",
                style_tags=["straight", "literal"],
                tone_signals={"sarcasm": 0.1, "seriousness": 0.8, "diplomacy": 0.6},
            ),
            StyleOption(
                id="light_sarcasm",
                label="Light sarcasm sometimes",
                style_tags=["light_sarcasm", "playful"],
                tone_signals={"sarcasm": 0.6, "casualness": 0.7, "warmth": 0.5},
            ),
            StyleOption(
                id="heavy_sarcasm",
                label="Pretty sarcastic",
                style_tags=["sarcastic", "edgy"],
                tone_signals={"sarcasm": 0.9, "bluntness": 0.7, "warmth": 0.2},
            ),
        ],
    ),
]


_SESSION_INTRO = [
    "Quick style check. Pick what sounds most like you.",
    "Short calibration round. Just choose the closest option.",
    "Fast tone setup. Go with what feels natural.",
]

_TRANSITIONS = [
    "Next one.",
    "Moving on.",
    "One more.",
    "Okay — next.",
]

_CONFIRMS = [
    "Alright.",
    "Got you.",
    "Makes sense.",
    "Okay.",
    "That helps.",
    "Saved that choice.",
]

_SUMMARY_TRANSITIONS = [
    "Here is what your choices suggest:",
    "Based on those picks:",
    "Quick read of your style:",
]

_ACCURATE_PHRASES = [
    "Nice, keeping that profile as-is.",
    "Great, that lines up cleanly.",
    "Perfect, logging it that way.",
]

_PARTIAL_PHRASES = [
    "Thanks, that helps tighten it.",
    "Good correction, updating it.",
    "Helpful tweak, applying it.",
]

_NOT_REALLY_PHRASES = [
    "No problem, recalibrating it.",
    "All good, adjusting the profile.",
    "Thanks, that gives a better fit.",
]

_REFLECT_HIGH = {
    "bluntness": "You seem pretty direct",
    "verbosity": "You usually add more detail when you explain",
    "diplomacy": "You usually soften how you say it instead of going blunt",
    "sarcasm": "You seem to use sarcasm fairly often",
    "warmth": "You come across as warm and approachable",
    "seriousness": "You lean more serious than playful",
    "casualness": "You usually lean casual instead of formal",
}

_REFLECT_LOW = {
    "bluntness": "You do not come across as blunt",
    "verbosity": "You do not seem to over-explain much",
    "diplomacy": "You tend to be more direct than careful with wording",
    "sarcasm": "You mostly keep it straight, not sarcastic",
    "warmth": "Your tone seems more matter-of-fact than warm",
    "seriousness": "You do not sound overly formal or serious",
    "casualness": "You do not usually keep things very casual",
}

_REFLECT_HIGH_THRESHOLD = 0.65
_REFLECT_LOW_THRESHOLD = 0.35
_MIN_EVIDENCE_FOR_REFLECTION = 2


def _rotate_phrase(phrases: List[str], counter: int) -> str:
    return phrases[counter % len(phrases)]


def get_prompts_for_session(prompt_count: int = 4, session_index: int = 0) -> List[StylePrompt]:
    """Return a deterministic rotating slice of prompts."""
    prompt_count = max(3, min(prompt_count, 5))
    pool = list(STYLE_PROMPTS)
    n = len(pool)
    if prompt_count >= n:
        return pool[:prompt_count]
    offset = (session_index * prompt_count) % n
    return [pool[(offset + i) % n] for i in range(prompt_count)]


def compute_style_confidence(option: StyleOption) -> float:
    """Simple deterministic confidence from signal strength."""
    if not option.tone_signals:
        return 0.6
    avg_strength = sum(option.tone_signals.values()) / len(option.tone_signals)
    return round(0.55 + min(0.4, abs(avg_strength - 0.5)), 2)


def generate_style_reflections(results: List[StyleCalibrationResult]) -> List[StyleReflectionLine]:
    """Generate lightweight reflections for traits with enough evidence."""
    trait_values: Dict[str, List[float]] = {}
    for result in results:
        for trait, value in result.tone_signals.items():
            trait_values.setdefault(trait, []).append(value)

    reflections: List[StyleReflectionLine] = []
    for trait, values in sorted(trait_values.items()):
        if len(values) < _MIN_EVIDENCE_FOR_REFLECTION:
            continue
        avg = round(sum(values) / len(values), 2)
        if avg >= _REFLECT_HIGH_THRESHOLD and trait in _REFLECT_HIGH:
            reflections.append(StyleReflectionLine(_REFLECT_HIGH[trait], trait, avg))
        elif avg <= _REFLECT_LOW_THRESHOLD and trait in _REFLECT_LOW:
            reflections.append(StyleReflectionLine(_REFLECT_LOW[trait], trait, avg))

    reflections.sort(key=lambda r: abs(r.strength - 0.5), reverse=True)
    return reflections[:5]


def _prompt_choice(prompt: StylePrompt) -> StyleOption:
    print(f"\n  {prompt.text}\n")
    for i, option in enumerate(prompt.options, 1):
        print(f"    {i}. {option.label}")

    while True:
        try:
            raw = input("\n  Your choice: ").strip()
            if not raw:
                continue
            idx = int(raw)
            if 1 <= idx <= len(prompt.options):
                return prompt.options[idx - 1]
            print(f"  Please enter a number between 1 and {len(prompt.options)}.")
        except ValueError:
            print("  Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            raise


def _handle_partial_correction(reflections: List[StyleReflectionLine], phrase_counter: int) -> StyleCorrectionResult:
    accepted: List[str] = []
    rejected: List[str] = []
    replacements: Dict[str, str] = {}

    print("\n  Which ones felt right? (numbers separated by spaces)")
    for idx, ref in enumerate(reflections, 1):
        print(f"    {idx}. {ref.text}")

    while True:
        try:
            raw = input("\n  Right ones: ").strip()
            if not raw:
                continue
            numbers = [int(x) for x in raw.split()]
            if all(1 <= n <= len(reflections) for n in numbers):
                break
            print(f"  Use numbers between 1 and {len(reflections)}.")
        except ValueError:
            print("  Enter numbers separated by spaces.")
        except (EOFError, KeyboardInterrupt):
            raise

    accepted_set = set(numbers)
    for idx, ref in enumerate(reflections, 1):
        if idx in accepted_set:
            accepted.append(ref.trait)
        else:
            rejected.append(ref.trait)

    if rejected:
        print("\n  For the ones that were off, pick a better fit:")
        for trait in rejected:
            print(f"    {trait}:")
            print("      a. opposite")
            print("      b. depends on context")
            print("      c. not sure")
            while True:
                try:
                    raw = input("      Pick (a/b/c): ").strip().lower()
                    if raw in ("a", "b", "c"):
                        break
                    print("      Please enter a, b, or c.")
                except (EOFError, KeyboardInterrupt):
                    raise
            replacements[trait] = {"a": "opposite", "b": "contextual", "c": "unsure"}[raw]

    print(f"\n  {_rotate_phrase(_PARTIAL_PHRASES, phrase_counter)}")
    return StyleCorrectionResult(
        status="partially_true",
        accepted_traits=accepted,
        rejected_traits=rejected,
        replacement_choices=replacements,
    )


def _handle_not_really(reflections: List[StyleReflectionLine], phrase_counter: int) -> StyleCorrectionResult:
    replacements: Dict[str, str] = {}
    print("\n  No stress. Pick a quick correction for each line:")
    for idx, ref in enumerate(reflections, 1):
        print(f"\n  {idx}. {ref.text}")
        print("     a. opposite")
        print("     b. depends on context")
        print("     c. not sure")
        while True:
            try:
                raw = input("     Pick (a/b/c): ").strip().lower()
                if raw in ("a", "b", "c"):
                    break
                print("     Please enter a, b, or c.")
            except (EOFError, KeyboardInterrupt):
                raise
        replacements[ref.trait] = {"a": "opposite", "b": "contextual", "c": "unsure"}[raw]

    print(f"\n  {_rotate_phrase(_NOT_REALLY_PHRASES, phrase_counter)}")
    return StyleCorrectionResult(
        status="not_really",
        accepted_traits=[],
        rejected_traits=[r.trait for r in reflections],
        replacement_choices=replacements,
    )


def _run_correction_loop(reflections: List[StyleReflectionLine], phrase_counter: int) -> StyleCorrectionResult:
    print("\n  Is that accurate?")
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
        return StyleCorrectionResult(
            status="accurate",
            accepted_traits=[r.trait for r in reflections],
            rejected_traits=[],
            replacement_choices={},
        )
    if choice == 2:
        return _handle_partial_correction(reflections, phrase_counter)
    return _handle_not_really(reflections, phrase_counter)


def run_style_calibration_session(prompt_count: int = 4, session_index: int = 0) -> StyleCalibrationSession:
    """Run a deterministic style calibration session."""
    prompts = get_prompts_for_session(prompt_count=prompt_count, session_index=session_index)
    results: List[StyleCalibrationResult] = []
    phrase_counter = 0

    print("\n" + "=" * 56)
    print("  STYLE CALIBRATION")
    print("=" * 56)
    print(f"\n  {_rotate_phrase(_SESSION_INTRO, session_index)}")

    for idx, prompt in enumerate(prompts):
        if idx > 0:
            print(f"\n  {_rotate_phrase(_TRANSITIONS, phrase_counter)}")
            phrase_counter += 1

        print(f"\n  --- Prompt {idx + 1} of {len(prompts)} ---")
        selected = _prompt_choice(prompt)
        results.append(
            StyleCalibrationResult(
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                selected_label=selected.label,
                selected_value=selected.id,
                style_tags=sorted(set(selected.style_tags)),
                tone_signals=dict(selected.tone_signals),
                confidence_score=compute_style_confidence(selected),
            )
        )
        print(f"\n  {_rotate_phrase(_CONFIRMS, phrase_counter)}")
        phrase_counter += 1

    reflections = generate_style_reflections(results)
    correction: Optional[StyleCorrectionResult] = None

    if reflections:
        print("\n" + "-" * 56)
        print(f"  {_rotate_phrase(_SUMMARY_TRANSITIONS, phrase_counter)}\n")
        for ref in reflections:
            print(f"    - {ref.text}")
        correction = _run_correction_loop(reflections, phrase_counter)
    else:
        print("\n  Not enough evidence yet for a clean style read.")

    return StyleCalibrationSession(results=results, reflections=reflections, correction=correction)

