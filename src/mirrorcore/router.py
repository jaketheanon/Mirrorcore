"""
Deterministic conversational router (Phase 31 + Phase 32 quiet routing).

Single-input intent classification and lightweight disambiguation — no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, MutableMapping, Optional, Tuple

# --- Thresholds (inspectable, deterministic) ---

# Second-place must be within this of the winner to trigger two-way ambiguity.
GAP_AMBIGUITY = 1.25
# Absolute score below this → treat as weak / unclear.
WEAK_TOTAL = 1.5

# Category ids (stable strings for tests and CLI)
DECISION_HELP = "decision_help"
PERSONAL_RESPONSE = "personal_response"
DEBUG_HELP = "debug_help"
PROFILE_BUILDING = "profile_building"
ONBOARDING_OR_HELP = "onboarding_or_help"

# Sub-targets for profile_building
PROFILE_INTERVIEW = "interview"
PROFILE_STYLE = "style"


@dataclass(frozen=True)
class IntentClassification:
    """Result of ``classify_intent`` — scores and structural ambiguity flags."""

    scores: Dict[str, float]
    ordered: List[Tuple[str, float]]  # descending by score, then name
    profile_style_score: float
    profile_interview_score: float
    needs_profile_split: bool
    needs_intent_disambiguation: bool
    ambiguous_options: Tuple[str, ...] = field(default_factory=tuple)
    weak_input: bool = False


@dataclass(frozen=True)
class ResolvedRoute:
    """Where the CLI should send the user after optional disambiguation."""

    category: str
    profile_target: Optional[str]  # PROFILE_INTERVIEW | PROFILE_STYLE | None
    from_disambiguation: bool = False


# (substring, category, weight) — matched on padded lowercase text " {norm} "
_PHRASE_RULES: List[Tuple[str, str, float]] = [
    # Personal / likely-you (strong)
    (" what would i probably say ", PERSONAL_RESPONSE, 5.0),
    (" what would i say ", PERSONAL_RESPONSE, 4.5),
    (" how would i respond ", PERSONAL_RESPONSE, 5.0),
    (" how would i reply ", PERSONAL_RESPONSE, 5.0),
    (" answer like me ", PERSONAL_RESPONSE, 4.0),
    (" respond like me ", PERSONAL_RESPONSE, 4.0),
    (" like me ", PERSONAL_RESPONSE, 2.0),
    (" my voice ", PERSONAL_RESPONSE, 2.5),
    (" wording ", PERSONAL_RESPONSE, 2.0),
    (" phrasing ", PERSONAL_RESPONSE, 2.0),
    (" probably say ", PERSONAL_RESPONSE, 3.0),
    (" sound like me ", PERSONAL_RESPONSE, 3.5),
    (" what would i do ", PERSONAL_RESPONSE, 4.0),
    # Decision
    (" should i ", DECISION_HELP, 4.0),
    (" should we ", DECISION_HELP, 3.0),
    (" help me decide ", DECISION_HELP, 4.5),
    (" what should i do ", DECISION_HELP, 4.5),
    (" which should i ", DECISION_HELP, 4.0),
    (" think through ", DECISION_HELP, 3.5),
    (" stuck choosing ", DECISION_HELP, 3.0),
    (" can't decide ", DECISION_HELP, 3.0),
    (" cant decide ", DECISION_HELP, 3.0),
    (" worth it ", DECISION_HELP, 2.5),
    (" buy this ", DECISION_HELP, 2.5),
    (" right now ", DECISION_HELP, 1.5),
    (" help me choose ", DECISION_HELP, 4.0),
    (" which option ", DECISION_HELP, 3.5),
    (" what to do ", DECISION_HELP, 3.0),
    (" what do i do ", DECISION_HELP, 4.5),
    (" what do we do ", DECISION_HELP, 4.0),
    (" figure out what to do ", DECISION_HELP, 3.5),
    (" should i do it ", DECISION_HELP, 4.0),
    (" should i say yes ", DECISION_HELP, 3.8),
    (" do i say yes ", DECISION_HELP, 3.5),
    # Timing / wait vs act (normalized text strips apostrophes → "don t" etc.)
    (" wait or ", DECISION_HELP, 4.0),
    (" wait or act ", DECISION_HELP, 4.5),
    (" whether to wait ", DECISION_HELP, 4.5),
    (" whether to ", DECISION_HELP, 3.2),
    (" know whether ", DECISION_HELP, 3.8),
    (" act now ", DECISION_HELP, 3.5),
    (" hold off ", DECISION_HELP, 3.5),
    (" move now ", DECISION_HELP, 3.2),
    (" not sure yet ", DECISION_HELP, 3.2),
    (" more information ", DECISION_HELP, 3.0),
    (" gather more info ", DECISION_HELP, 3.0),
    (" don t know whether ", DECISION_HELP, 4.2),
    (" don t know if ", DECISION_HELP, 3.4),
    # Debug / troubleshooting
    (" stack trace ", DEBUG_HELP, 4.5),
    (" traceback ", DEBUG_HELP, 4.5),
    (" permission denied ", DEBUG_HELP, 4.5),
    (" won't start ", DEBUG_HELP, 3.5),
    (" wont start ", DEBUG_HELP, 3.5),
    (" does not start ", DEBUG_HELP, 3.5),
    (" deployment failed ", DEBUG_HELP, 4.0),
    (" deploy failed ", DEBUG_HELP, 3.5),
    (" analyze this log ", DEBUG_HELP, 4.0),
    (" this log ", DEBUG_HELP, 2.5),
    (" crashing ", DEBUG_HELP, 3.5),
    (" keeps failing ", DEBUG_HELP, 3.5),
    (" service ", DEBUG_HELP, 1.2),
    (" not working ", DEBUG_HELP, 3.0),
    (" isn't working ", DEBUG_HELP, 3.0),
    (" isnt working ", DEBUG_HELP, 3.0),
    (" connection refused ", DEBUG_HELP, 3.5),
    (" timed out ", DEBUG_HELP, 3.0),
    (" timeout ", DEBUG_HELP, 2.5),
    (" exception ", DEBUG_HELP, 3.0),
    (" segfault ", DEBUG_HELP, 3.5),
    (" seg fault ", DEBUG_HELP, 3.5),
    # Profile / calibration
    (" calibrate ", PROFILE_BUILDING, 4.0),
    (" calibration ", PROFILE_BUILDING, 3.5),
    (" learn how i talk ", PROFILE_BUILDING, 4.5),
    (" how i talk ", PROFILE_BUILDING, 3.0),
    (" interview me ", PROFILE_BUILDING, 4.5),
    (" learn how i decide ", PROFILE_BUILDING, 4.5),
    (" decision interview ", PROFILE_BUILDING, 3.5),
    (" build my profile ", PROFILE_BUILDING, 3.5),
    (" building my profile ", PROFILE_BUILDING, 3.5),
    (" improve my profile ", PROFILE_BUILDING, 3.0),
    (" my profile ", PROFILE_BUILDING, 2.0),
    # Onboarding / meta
    (" what can you do ", ONBOARDING_OR_HELP, 4.5),
    (" where do i start ", ONBOARDING_OR_HELP, 4.5),
    (" how do i use ", ONBOARDING_OR_HELP, 3.5),
    (" continue where ", ONBOARDING_OR_HELP, 3.5),
    (" left off ", ONBOARDING_OR_HELP, 2.5),
    (" get started ", ONBOARDING_OR_HELP, 3.0),
]

# Extra profile sub-signals (do not add to PROFILE_BUILDING total directly;
# used to pick interview vs style.)
_STYLE_MARKERS: List[Tuple[str, float]] = [
    (" how i talk ", 3.0),
    (" calibrate ", 3.0),
    (" style ", 2.0),
    (" tone ", 2.0),
    (" voice ", 2.0),
    (" sound ", 1.5),
]

_INTERVIEW_MARKERS: List[Tuple[str, float]] = [
    (" learn how i decide ", 3.0),
    (" interview ", 2.5),
    (" decide ", 2.0),
    (" decision ", 2.0),
    (" values ", 1.5),
]

_DEBUG_EXTRA_WORDS = (
    "error",
    "failed",
    "failure",
    "crash",
    "broken",
    "bug",
    "fix",
    "debug",
    "logs",
    "log",
    "failing",
)

_ONBOARDING_EXACT = frozenset(
    {
        "help",
        "help me",
        "?",
        "hi",
        "hello",
        "hey",
    }
)


def normalize_input(text: str) -> str:
    """Lowercase, collapse whitespace, strip attached punctuation so phrases still match."""
    t = text.strip().lower()
    for ch in "?!.,;:()[]{}\"'":
        t = t.replace(ch, " ")
    return " ".join(t.split())


def _padded(norm: str) -> str:
    return f" {norm} "


def _apply_shift_obligation_decision_boost(norm: str, padded: str, scores: MutableMapping[str, float]) -> None:
    """Strong decision_help for shift/cover asks + fatigue / workplace — avoids false weak_input."""
    peer = any(
        s in padded
        for s in (
            " coworker ",
            " colleague ",
            " boss ",
            " manager ",
            " teammate ",
        )
    )
    if norm.startswith("coworker ") or norm.startswith("colleague "):
        peer = True

    shift_like = (
        " cover " in padded
        or " covering " in padded
        or " shift " in padded
        or " shifts " in padded
        or " overtime " in padded
        or " pick up " in padded
        or "pick up a shift" in norm
        or "pick up another shift" in norm
        or " cover for" in norm
        or "cover their" in norm
        or "cover his" in norm
        or "cover her" in norm
    )

    ask_signal = (
        " wants me " in padded
        or " want me to " in padded
        or " asked me " in padded
        or " asking me " in padded
    )

    fatigue = any(
        w in norm
        for w in (
            "exhausted",
            "tired",
            "wiped",
            "drained",
            "overwhelmed",
            "burnout",
            "burned out",
            "burnt out",
            "no energy",
            "too tired",
        )
    )

    favor = " favor " in padded or " favour " in padded

    if fatigue and shift_like and (peer or ask_signal):
        scores[DECISION_HELP] += 5.25
    elif fatigue and shift_like:
        scores[DECISION_HELP] += 3.75
    elif shift_like and ask_signal and (" help " in padded or favor):
        scores[DECISION_HELP] += 4.25


def _apply_peer_boundary_obligation_boost(
    norm: str, padded: str, scores: MutableMapping[str, float]
) -> None:
    """Decision help when a peer keeps pressing / ignores a clear no — avoids false weak_input."""
    peer = any(
        s in padded
        for s in (
            " coworker ",
            " colleague ",
            " boss ",
            " manager ",
            " teammate ",
        )
    )
    if norm.startswith("coworker ") or norm.startswith("colleague "):
        peer = True

    same_peer_thread = (
        (
            " same coworker " in padded
            or " same colleague " in padded
            or " this coworker " in padded
            or " this colleague " in padded
            or norm.startswith("same coworker ")
            or norm.startswith("same colleague ")
        )
        and any(
            x in padded
            for x in (
                " still ",
                " keeps ",
                " pushing ",
                " asking ",
                " again ",
            )
        )
    )

    boundary_refusal = any(
        x in padded
        for x in (
            " said no ",
            " i said no ",
            " already said no ",
            " already told ",
            " told them no ",
            " told her no ",
            " told him no ",
            " after i said ",
            " after i told ",
        )
    )

    wont_take_no = any(
        x in padded
        for x in (
            " wont take no ",
            " won t take no ",
            " take no for an answer ",
        )
    )

    push_pressure = any(
        x in padded
        for x in (
            " keeps pushing ",
            " keep pushing ",
            " still pushing ",
            " pushing after ",
            " keeps asking ",
            " keep asking ",
            " won t stop ",
            " wont stop ",
            " not taking no ",
        )
    )

    if same_peer_thread:
        scores[DECISION_HELP] += 4.75
    if peer and (boundary_refusal or wont_take_no or push_pressure):
        scores[DECISION_HELP] += 5.25


def _apply_multiline_debug_boost(raw: str, norm: str, scores: Dict[str, float]) -> None:
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 3 and any(
        any(k in ln.lower() for k in ("error", "exception", "trace", "fatal", "denied", "failed"))
        for ln in lines
    ):
        scores[DEBUG_HELP] += 3.0
    if " at " in norm and ("line" in norm or "error" in norm):
        scores[DEBUG_HELP] += 2.5


def _profile_subscores(padded: str) -> Tuple[float, float]:
    st = 0.0
    iv = 0.0
    for sub, w in _STYLE_MARKERS:
        if sub in padded:
            st += w
    for sub, w in _INTERVIEW_MARKERS:
        if sub in padded:
            iv += w
    return st, iv


def classify_intent(text: str) -> IntentClassification:
    """
    Deterministic weighted classification. Inspectable rules only.
    """
    raw = text or ""
    norm = normalize_input(raw)
    padded = _padded(norm) if norm else "  "

    scores = {
        DECISION_HELP: 0.0,
        PERSONAL_RESPONSE: 0.0,
        DEBUG_HELP: 0.0,
        PROFILE_BUILDING: 0.0,
        ONBOARDING_OR_HELP: 0.0,
    }
    for sub, cat, w in _PHRASE_RULES:
        if sub in padded:
            scores[cat] += w

    for w in _DEBUG_EXTRA_WORDS:
        if w in norm:
            scores[DEBUG_HELP] += 0.85

    _apply_multiline_debug_boost(raw, norm, scores)
    _apply_shift_obligation_decision_boost(norm, padded, scores)
    _apply_peer_boundary_obligation_boost(norm, padded, scores)

    if norm in _ONBOARDING_EXACT:
        scores[ONBOARDING_OR_HELP] += 4.0

    # Whole-string “help” variants
    if norm == "help" or norm == "help me" or norm.startswith("help "):
        scores[ONBOARDING_OR_HELP] += 2.0

    profile_style_score, profile_interview_score = _profile_subscores(padded)

    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    top_name, top_v = ordered[0]

    weak_input = top_v < WEAK_TOTAL

    needs_profile_split = False
    if top_name == PROFILE_BUILDING:
        if profile_style_score >= 2.0 and profile_interview_score >= 2.0:
            if abs(profile_style_score - profile_interview_score) < 1.1:
                needs_profile_split = True
        elif profile_style_score < 1.0 and profile_interview_score < 1.0:
            needs_profile_split = True

    ambiguous_options: Tuple[str, ...] = ()
    needs_intent_disambiguation = False

    if not weak_input and len(ordered) >= 2:
        second_name, second_v2 = ordered[1]
        if (
            top_name != second_name
            and second_v2 >= WEAK_TOTAL
            and (top_v - second_v2) < GAP_AMBIGUITY
        ):
            needs_intent_disambiguation = True
            ambiguous_options = tuple(sorted({top_name, second_name}))

    return IntentClassification(
        scores=scores,
        ordered=ordered,
        profile_style_score=profile_style_score,
        profile_interview_score=profile_interview_score,
        needs_profile_split=needs_profile_split,
        needs_intent_disambiguation=needs_intent_disambiguation,
        ambiguous_options=ambiguous_options,
        weak_input=weak_input,
    )


def _pick_profile_target(c: IntentClassification) -> Optional[str]:
    if c.profile_style_score > c.profile_interview_score + 0.5:
        return PROFILE_STYLE
    if c.profile_interview_score > c.profile_style_score + 0.5:
        return PROFILE_INTERVIEW
    return None


def resolve_route_after_classification(c: IntentClassification) -> ResolvedRoute:
    """
    Turn a classification into a route without interactive disambiguation.
    May return profile_target None when caller must disambiguate.
    """
    if c.weak_input:
        return ResolvedRoute(category=ONBOARDING_OR_HELP, profile_target=None)

    top = c.ordered[0][0]

    if c.needs_profile_split and top == PROFILE_BUILDING:
        tgt = _pick_profile_target(c)
        if tgt:
            return ResolvedRoute(category=PROFILE_BUILDING, profile_target=tgt)
        return ResolvedRoute(category=PROFILE_BUILDING, profile_target=None)

    if top == PROFILE_BUILDING:
        tgt = _pick_profile_target(c) or PROFILE_INTERVIEW
        return ResolvedRoute(category=PROFILE_BUILDING, profile_target=tgt)

    return ResolvedRoute(category=top, profile_target=None)


def disambiguate_profile(
    c: IntentClassification,
    read_choice: Callable[[str], str],
) -> str:
    """Ask one short question; return PROFILE_INTERVIEW or PROFILE_STYLE."""
    print()
    print("This looks like profile stuff. Which do you want to do right now?")
    print("  1) How I decide (short interview)")
    print("  2) How I talk (style calibration)")
    choice = (read_choice("Pick 1 or 2: ").strip() or "").lower()
    if choice in ("2", "two", "talk", "style", "sound", "voice"):
        return PROFILE_STYLE
    return PROFILE_INTERVIEW


def disambiguate_intent(
    c: IntentClassification,
    read_choice: Callable[[str], str],
) -> ResolvedRoute:
    """
    Present a small menu for two-way intent ambiguity; deterministic mapping.
    """
    opts = list(c.ambiguous_options)
    if len(opts) < 2:
        r = resolve_route_after_classification(c)
        return r

    print()
    print("That could mean a few different things — which one's closest?")
    labels = []
    for i, cat in enumerate(opts, start=1):
        labels.append((str(i), cat))
        print(f"  {i}) {_human_label(cat)}")
    print(f"  {len(opts) + 1}) Show the main menu instead")
    raw = (read_choice("Pick a number: ").strip() or "").lower()
    if raw == str(len(opts) + 1) or raw in ("m", "menu", "start"):
        return ResolvedRoute(category=ONBOARDING_OR_HELP, profile_target=None, from_disambiguation=True)
    for key, cat in labels:
        if raw == key:
            if cat == PROFILE_BUILDING:
                tgt = _pick_profile_target(c) or disambiguate_profile(c, read_choice)
                return ResolvedRoute(
                    category=PROFILE_BUILDING,
                    profile_target=tgt,
                    from_disambiguation=True,
                )
            return ResolvedRoute(category=cat, profile_target=None, from_disambiguation=True)
    # Fallback: main menu
    return ResolvedRoute(category=ONBOARDING_OR_HELP, profile_target=None, from_disambiguation=True)


def _human_label(cat: str) -> str:
    return {
        DECISION_HELP: "A choice I'm trying to make",
        PERSONAL_RESPONSE: "How I'd probably answer out loud",
        DEBUG_HELP: "Something broke — logs or errors",
        PROFILE_BUILDING: "Teaching MirrorCore how I work",
        ONBOARDING_OR_HELP: "What MirrorCore can do / where to start",
    }.get(cat, cat)


def routing_feedback(route: ResolvedRoute) -> str:
    """Optional line before dispatch (Phase 32: quiet by default)."""
    if route.from_disambiguation and route.category == ONBOARDING_OR_HELP:
        return "Alright — here's the main menu."

    # Ambiguous inputs already went through a menu; stay quiet after they pick.
    if route.from_disambiguation:
        return ""

    if route.category == DEBUG_HELP:
        # One plain hint so it’s obvious what to paste next.
        return "Paste the error or log output when you’re ready."

    if route.category == ONBOARDING_OR_HELP:
        return "Here’s what you can do and where to start."

    return ""


def resolve_full_route(
    text: str,
    read_choice: Callable[[str], str],
) -> Tuple[IntentClassification, ResolvedRoute]:
    """
    Classify, then apply interactive disambiguation when needed.
    """
    c = classify_intent(text)

    if c.weak_input:
        return c, ResolvedRoute(category=ONBOARDING_OR_HELP, profile_target=None)

    if c.needs_intent_disambiguation:
        r = disambiguate_intent(c, read_choice)
        return c, r

    r0 = resolve_route_after_classification(c)
    if r0.category == PROFILE_BUILDING and r0.profile_target is None:
        tgt = disambiguate_profile(c, read_choice)
        return c, ResolvedRoute(category=PROFILE_BUILDING, profile_target=tgt)

    return c, r0


def route_request(
    text: str,
    read_choice: Callable[[str], str],
) -> Tuple[IntentClassification, ResolvedRoute]:
    """Classify input and resolve the route (alias for :func:`resolve_full_route`)."""
    return resolve_full_route(text, read_choice)
