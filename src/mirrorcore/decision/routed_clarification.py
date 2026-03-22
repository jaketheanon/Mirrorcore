"""
Routed decision engine: ontology (families + dimensions), slot-based clarification,
and structured situation/tendency memory (Phase 31.2 refactor of 31.1).

Deterministic, inspectable, one question at a time. Used from ``mirrorcore ask``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, FrozenSet, List, MutableMapping, Optional, Sequence, Tuple

from ..persona.profile import PersonalProfile, build_personal_profile
from ..router import normalize_input

MAX_CLARIFICATION_ROUNDS = 2


def _stable_index(key: str, modulo: int) -> int:
    """Deterministic pick in ``0..modulo-1`` (no ``hash()`` salt)."""
    if modulo <= 1:
        return 0
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % modulo


# --- Decision families (broad ontology) ---
SPENDING = "spending"
OBLIGATION_OVERLOAD = "obligation_overload"
CONFLICT_FAMILY = "conflict"
RISK_TIMING = "risk_timing"
LOYALTY_BOUNDARY = "loyalty_boundary"
CONVENIENCE_QUALITY = "convenience_quality"
GENERAL = "general"

# Backwards-compatible aliases (Phase 31.1 tests and callers)
MONEY = SPENDING
CONFLICT = CONFLICT_FAMILY
TIMING = RISK_TIMING
OBLIGATION = OBLIGATION_OVERLOAD
RISK = RISK_TIMING


# --- Dimensions (weighted signals; drive family scores + slot boosts) ---
# Note: do not use bare "should i" under obligation — it fires on every decision question.
_DIMENSION_RULES: Tuple[Tuple[str, Tuple[str, ...], float], ...] = (
    ("money_pressure", ("rent", "broke", "afford", "bill", "debt", "salary", "budget", "pay", "late", "$", "dollar", "money tight", "skint"), 1.0),
    ("need_vs_want_signal", ("need", "want", "luxury", "essential", "nice to have"), 0.85),
    ("urgency", ("now", "today", "asap", "deadline", "rush", "right away", "immediately"), 0.9),
    ("uncertainty", ("unsure", "don t know", "dont know", "what if", "maybe", "confused", "unclear", "not sure yet"), 0.85),
    (
        "obligation",
        (
            "owe",
            "expected to",
            "duty",
            "guilt",
            "asked me to",
            "have to help",
            " should i help",
            "pick up a shift",
            "cover for",
            "cover their",
            "cover a shift",
            "cover my shift",
            "cover the shift",
            "cover shift",
            "covering a shift",
            "covering for",
            "needs me to cover",
            "want me to cover",
            " extra shift",
            " extra hours",
            "do them a favor",
        ),
        0.95,
    ),
    (
        "overload",
        (
            "exhausted",
            "burnout",
            "burned out",
            "burnt out",
            "too much",
            "overwhelmed",
            "overloaded",
            "no energy",
            "at capacity",
            "double shift",
            "extra shift",
            "extra hours",
            "drained",
            "wiped out",
        ),
        1.1,
    ),
    ("boundary_strain", ("boundary", "say no", "can't keep", "cant keep", "people pleasing", "walk all over"), 0.95),
    (
        "interpersonal_hurt",
        (
            "bothered",
            "bother me",
            "bothers me",
            "upset",
            "disrespect",
            "disrespected",
            "rude",
            "talking shit",
            " talk shit",
            "let it go",
            "say something",
            "hurt my feelings",
            "insulted",
            "crossed the line",
            "offended",
            "offensive",
            "argument",
            "confrontation",
            "confront",
            "unfair",
            "snubbed",
            "dismissed me",
        ),
        1.15,
    ),
    ("conflict_intensity", ("argue", "fight", "angry", "resent", "toxic", "unfair", "silent treatment"), 1.0),
    (
        "wait_vs_act",
        (
            "wait or",
            "whether to wait",
            "act now",
            "hold off",
            "move now",
            "not sure yet",
            " timing",
            "pull the trigger",
            "more information",
            "gather more info",
            "sooner or later",
            "know whether",
        ),
        1.2,
    ),
    ("relationship_stakes", ("friend", "partner", "family", "coworker", "boss", "team", "marriage"), 0.75),
    ("regret_risk", ("regret", "resent", "wish i hadn't", "worse if i say yes", "worse if i say no"), 0.9),
    ("risk_level", ("risk", "safe", "gamble", "nervous", "scared"), 0.8),
    ("convenience_vs_correctness", ("quick", "shortcut", "fast", "good enough", "proper", "right way", "correct"), 0.85),
    ("short_vs_long", ("short term", "long term", "later me", "future", "temporary fix"), 0.8),
)

# Dimension -> (family, bonus score)
_DIMENSION_FAMILY_BOOST: Tuple[Tuple[str, str, float], ...] = (
    ("money_pressure", SPENDING, 2.2),
    ("need_vs_want_signal", SPENDING, 1.0),
    ("overload", OBLIGATION_OVERLOAD, 2.0),
    ("obligation", OBLIGATION_OVERLOAD, 1.6),
    ("boundary_strain", OBLIGATION_OVERLOAD, 1.2),
    ("boundary_strain", LOYALTY_BOUNDARY, 1.0),
    ("interpersonal_hurt", CONFLICT_FAMILY, 3.2),
    ("conflict_intensity", CONFLICT_FAMILY, 2.2),
    ("relationship_stakes", CONFLICT_FAMILY, 1.0),
    ("relationship_stakes", LOYALTY_BOUNDARY, 1.2),
    ("regret_risk", RISK_TIMING, 1.0),
    ("regret_risk", LOYALTY_BOUNDARY, 1.0),
    ("uncertainty", RISK_TIMING, 1.5),
    ("wait_vs_act", RISK_TIMING, 2.9),
    ("risk_level", RISK_TIMING, 1.6),
    ("urgency", RISK_TIMING, 1.2),
    ("convenience_vs_correctness", CONVENIENCE_QUALITY, 2.0),
    ("short_vs_long", CONVENIENCE_QUALITY, 1.2),
)

# Base family keywords (weight per hit)
_FAMILY_KEYWORDS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    SPENDING: (
        ("buy", 1.0), ("purchase", 1.0), ("spend", 1.0), ("afford", 1.1), ("price", 0.9),
        ("rent", 1.2), ("bill", 1.0), ("loan", 0.9), ("save", 0.7), ("cheap", 0.8), ("expensive", 0.9),
    ),
    OBLIGATION_OVERLOAD: (
        (" help ", 0.9),
        ("help me", 0.8),
        ("favor", 1.0),
        ("shift", 1.0),
        ("overtime", 1.0),
        ("exhausted", 1.2),
        ("burnout", 1.3),
        ("burnt out", 1.25),
        ("too much", 1.1),
        ("extra hours", 1.15),
        ("cover for", 1.0),
        ("cover a shift", 1.1),
        ("cover shift", 1.0),
        ("pick up", 0.65),
    ),
    CONFLICT_FAMILY: (
        ("confront", 1.1),
        ("argument", 1.0),
        ("boundary", 0.9),
        ("boss", 0.8),
        # Not listed: bare "coworker" — that alone pulled shift/favor asks into conflict;
        # relationship_stakes still nudges when a peer term appears.
        ("apologize", 0.7),
        ("resent", 1.0),
        ("unfair", 0.9),
        ("bothered", 1.3),
        ("bother me", 1.2),
        ("upset", 1.0),
        ("disrespect", 1.2),
        ("rude", 1.0),
        ("let it go", 1.1),
        ("say something", 1.1),
    ),
    RISK_TIMING: (
        ("wait", 0.9),
        ("act now", 1.4),
        ("hold off", 1.2),
        ("whether", 0.9),
        ("timing", 1.0),
        ("now", 0.55),
        ("deadline", 1.1),
        ("risk", 1.0),
        ("uncertain", 1.0),
        ("reversible", 0.8),
        ("one-way", 0.9),
        ("permanent", 0.9),
    ),
    LOYALTY_BOUNDARY: (
        ("loyal", 1.0), ("betray", 0.9), ("family", 0.8), ("guilt", 1.0), ("selfish", 0.8),
        ("protect myself", 1.0), ("people pleaser", 1.1),
    ),
    CONVENIENCE_QUALITY: (
        ("quick", 1.0), ("shortcut", 1.1), ("hack", 0.7), ("proper", 0.9), ("good enough", 1.0),
    ),
    GENERAL: (("choice", 0.3), ("decide", 0.4), ("should i", 0.5)),
}


def score_dimensions(norm_text: str) -> Dict[str, float]:
    padded = f" {norm_text} "
    out: Dict[str, float] = {}
    for dim, phrases, w in _DIMENSION_RULES:
        s = 0.0
        for p in phrases:
            if p in padded or p in norm_text:
                s += w
        if s > 0:
            out[dim] = s
    return out


def infer_family_scores(norm_text: str, dimensions: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    dimensions = dimensions or score_dimensions(norm_text)
    scores: Dict[str, float] = {f: 0.05 for f in _FAMILY_KEYWORDS}
    scores[GENERAL] = 0.15
    padded = f" {norm_text} "
    for fam, kws in _FAMILY_KEYWORDS.items():
        for kw, wt in kws:
            if kw in padded or kw in norm_text:
                scores[fam] = scores.get(fam, 0) + wt
    for dim, fam, bonus in _DIMENSION_FAMILY_BOOST:
        if dim in dimensions:
            scores[fam] = scores.get(fam, 0) + bonus * min(1.2, dimensions[dim] / 2.5)
    return scores


# Strong obligation/overload cues — required before we demote conflict into obligation slots.
_OBLIGATION_OVERLOAD_CUES: Tuple[str, ...] = (
    " exhausted",
    " burnout",
    " burnt out",
    "burned out",
    " shift",
    " overtime",
    " favor",
    " help them",
    " help her",
    " help him",
    " cover for",
    " cover a shift",
    "needs me to cover",
    " pick up",
    " asked me to",
    " too much on",
    " extra shift",
    " extra hours",
    " owe ",
    "guilt trip",
    "can t cover",
    "can't cover",
)


def _boost_obligation_for_cover_shift_fatigue(
    norm_text: str,
    dimensions: Dict[str, float],
    scores: MutableMapping[str, float],
) -> None:
    """Workplace cover/shift asks + fatigue/strain → obligation-overload, not conflict."""
    padded = f" {norm_text} "
    peer = any(
        x in padded
        for x in (
            " coworker ",
            " colleague ",
            " boss ",
            " teammate ",
            " manager ",
        )
    ) or norm_text.startswith(("coworker ", "colleague "))
    shift_like = any(
        x in padded or x in norm_text
        for x in (
            " shift ",
            " shifts ",
            " cover ",
            " covering ",
            " overtime ",
            " extra hours ",
            "pick up a shift",
            "pick up another shift",
            "cover a shift",
            "cover my shift",
            "cover shift",
            "covering a shift",
            "covering for",
        )
    )
    strain = dimensions.get("overload", 0) >= 0.95 or dimensions.get("obligation", 0) >= 0.55
    if not (peer and shift_like and strain):
        return
    scores[OBLIGATION_OVERLOAD] = scores.get(OBLIGATION_OVERLOAD, 0) + 2.75
    if dimensions.get("interpersonal_hurt", 0) < 0.55:
        scores[CONFLICT_FAMILY] = scores.get(CONFLICT_FAMILY, 0) * 0.62


def _social_conflict_without_obligation(norm_text: str, dimensions: Dict[str, float]) -> bool:
    """Interpersonal upset / confrontation without favors, shifts, or exhaustion context."""
    if dimensions.get("interpersonal_hurt", 0) <= 0:
        return False
    padded = f" {norm_text} "
    if not any(m in padded or m in norm_text for m in _OBLIGATION_OVERLOAD_CUES):
        if dimensions.get("overload", 0) < 0.55 and dimensions.get("obligation", 0) < 0.55:
            return True
    return False


def _adjust_family_scores_for_social_conflict(
    norm_text: str, dimensions: Dict[str, float], scores: MutableMapping[str, float],
) -> None:
    """Keep 'someone bothered me' style questions out of obligation/overload unless cues match."""
    if not _social_conflict_without_obligation(norm_text, dimensions):
        return
    scores[CONFLICT_FAMILY] = scores.get(CONFLICT_FAMILY, 0) + 5.0
    scores[OBLIGATION_OVERLOAD] = scores.get(OBLIGATION_OVERLOAD, 0) * 0.32


def rank_families(norm_text: str) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    dims = score_dimensions(norm_text)
    raw = infer_family_scores(norm_text, dims)
    _boost_obligation_for_cover_shift_fatigue(norm_text, dims, raw)
    _adjust_family_scores_for_social_conflict(norm_text, dims, raw)
    ordered = sorted(raw.items(), key=lambda x: (-x[1], x[0]))
    return ordered, dims


def score_decision_domains(norm_text: str) -> List[Tuple[str, float]]:
    """Back-compat: ranked (family, score) list."""
    ordered, _ = rank_families(norm_text)
    return ordered


@dataclass(frozen=True)
class ClarificationSlot:
    """A missing-information slot: one inspectable question."""

    slot_id: str
    families: FrozenSet[str]
    priority: int  # lower = higher priority within the same family tier
    question: str
    filled_markers: Tuple[str, ...] = ()
    # Dimensions that lower effective priority (more urgent to ask) when active
    boost_dims: Tuple[str, ...] = ()
    # Same slot / logic; surface wording only (Phase 32).
    question_variants: Tuple[str, ...] = ()

    @property
    def qid(self) -> str:
        return self.slot_id

    @property
    def text(self) -> str:
        return self.question

    def wording(self, session_seed: str) -> str:
        variants = (self.question,) + self.question_variants
        i = _stable_index(f"{session_seed}:{self.slot_id}", len(variants))
        return variants[i]

    @property
    def skip_if_any(self) -> Tuple[str, ...]:
        return self.filled_markers


def _all_slots() -> Tuple[ClarificationSlot, ...]:
    return (
        # Spending
        ClarificationSlot(
            "bills_basics",
            frozenset({SPENDING}),
            10,
            "First thing: are basics like rent or bills already handled this month, or still hanging over you?",
            filled_markers=(
                "paid rent", "rent paid", "bills covered", "already paid", "handled rent",
                "still unpaid", "not paid", "havent paid", "haven't paid", "behind on rent",
                "rent covered", "rent sorted",
            ),
            boost_dims=("money_pressure",),
            question_variants=(
                "Are rent and the big monthly bills squared away yet, or still hanging over you?",
                "Have the important bills been handled yet, or not really?",
            ),
        ),
        ClarificationSlot(
            "need_vs_want",
            frozenset({SPENDING}),
            20,
            "Is this mostly something you actually need right now, or more of a want?",
            filled_markers=(
                " need ", " want ", "mostly need", "mostly want", "for work", "for school",
                "for my job", "essential", "luxury",
            ),
            boost_dims=("need_vs_want_signal",),
            question_variants=(
                "Honestly — real need right now, or mostly a want?",
                "Is this a must-have for you at the moment, or a nice-to-have?",
            ),
        ),
        ClarificationSlot(
            "purpose_purchase",
            frozenset({SPENDING}),
            35,
            "What would you mainly use it for day to day?",
            filled_markers=(
                "use it for", "for gaming", "for coding", "for classes", "for studying",
                "for school", "work laptop", "school laptop", "day to day",
            ),
        ),
        ClarificationSlot(
            "can_wait_purchase",
            frozenset({SPENDING}),
            40,
            "If you waited two weeks, what would actually change for you?",
            filled_markers=(
                "wait two weeks", "wait a week", "in two weeks", "can wait", "cannot wait",
                "can't wait", "cant wait", "no rush",
            ),
            boost_dims=("urgency",),
        ),
        ClarificationSlot(
            "cheaper_ok",
            frozenset({SPENDING}),
            45,
            "Is there a cheaper option that would still be good enough for now?",
            filled_markers=("cheaper", "used", "refurb", "borrow", "rent one", "good enough"),
        ),
        # Obligation / overload
        ClarificationSlot(
            "energy_capacity",
            frozenset({OBLIGATION_OVERLOAD}),
            10,
            "Do you actually have the energy for this without screwing yourself over?",
            filled_markers=("energy", "exhausted", "at capacity", "no bandwidth", "burnt out", "burned out"),
            boost_dims=("overload",),
            question_variants=(
                "Do you have enough gas left for this without wiping yourself out?",
                "Realistically — do you have the bandwidth for this right now?",
            ),
        ),
        ClarificationSlot(
            "guilt_axis",
            frozenset({OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY}),
            15,
            "Are you more worried about saying no, or about saying yes and regretting it later?",
            filled_markers=(
                "say no", "saying no", "say yes", "resent", "regret", "let them down",
                "people pleasing",
            ),
            boost_dims=("regret_risk", "obligation"),
            question_variants=(
                "What scares you more — saying no, or saying yes and wishing you hadn’t?",
                "Is the hard part letting them down, or ending up bitter if you agree?",
            ),
        ),
        ClarificationSlot(
            "consequence_no",
            frozenset({OBLIGATION_OVERLOAD}),
            25,
            "If you said no, what's the realistic downside — not the scary story, the likely one?",
            filled_markers=("if i said no", "if i say no", "downside", "realistic", "they'd be fine"),
        ),
        ClarificationSlot(
            "pressure_source",
            frozenset({OBLIGATION_OVERLOAD}),
            30,
            "Is the pressure mostly guilt, or something concrete like money or your job on the line?",
            filled_markers=("guilt", "job on the line", "money on the line", "concrete", "expectations"),
        ),
        # Conflict
        ClarificationSlot(
            "peace_vs_clarity",
            frozenset({CONFLICT_FAMILY}),
            10,
            "Trying to keep the peace, or say what needs to be said?",
            filled_markers=(
                "keep the peace", "clear the air", "speak up", "stay quiet", "peace",
            ),
            boost_dims=("conflict_intensity",),
            question_variants=(
                "Want to keep things calm, or say the thing that’s on your mind?",
                "More important right now — smooth things over, or get it out in the open?",
            ),
        ),
        ClarificationSlot(
            "pattern_vs_once",
            frozenset({CONFLICT_FAMILY}),
            18,
            "One-time flare-up, or something that keeps repeating?",
            filled_markers=(
                "one time", "one-time", "pattern", "keeps happening", "again and again", "first time",
            ),
        ),
        ClarificationSlot(
            "stakes_real",
            frozenset({CONFLICT_FAMILY}),
            26,
            "Does this touch safety, job, or housing in a real way, or mostly pride and feelings?",
            filled_markers=("safety", "job", "housing", "mostly feelings", "not really", "pride"),
        ),
        ClarificationSlot(
            "timing_conflict",
            frozenset({CONFLICT_FAMILY, RISK_TIMING}),
            32,
            "Does this need words today, or can it wait until you're calmer?",
            filled_markers=("today", "wait until", "calmer", "cool off", "tomorrow"),
            boost_dims=("urgency",),
        ),
        # Risk / timing
        ClarificationSlot(
            "real_deadline",
            frozenset({RISK_TIMING}),
            12,
            "Is there a real deadline, or does it mostly feel urgent?",
            filled_markers=("deadline", "real deadline", "no deadline", "feel urgent", "emotionally"),
            boost_dims=("urgency", "uncertainty"),
            question_variants=(
                "Is there an actual cutoff, or is it mostly nerves talking?",
                "Hard deadline somewhere, or just feels like it has to be now?",
            ),
        ),
        ClarificationSlot(
            "reversibility",
            frozenset({RISK_TIMING}),
            22,
            "Could you undo or adjust this later, or is it mostly one-way once you choose?",
            filled_markers=("undo", "reversible", "one-way", "permanent", "change my mind"),
        ),
        ClarificationSlot(
            "worst_hurt",
            frozenset({RISK_TIMING}),
            30,
            "If it went wrong, what would hurt most — money, relationships, reputation, or something else?",
            filled_markers=("money", "relationship", "reputation", "career", "health"),
            boost_dims=("risk_level", "uncertainty"),
        ),
        # Loyalty / boundary
        ClarificationSlot(
            "loyalty_cost",
            frozenset({LOYALTY_BOUNDARY}),
            14,
            "Does this ask you to protect someone else at a real cost to you?",
            filled_markers=("cost to me", "protect them", "sacrifice", "my expense"),
            boost_dims=("relationship_stakes", "regret_risk"),
        ),
        ClarificationSlot(
            "always_yes_pattern",
            frozenset({LOYALTY_BOUNDARY, OBLIGATION_OVERLOAD}),
            24,
            "What happens if you always say yes here — does it train people to expect more than you have?",
            filled_markers=("always say yes", "train", "expect more", "pattern"),
        ),
        # Convenience vs correctness
        ClarificationSlot(
            "speed_vs_right",
            frozenset({CONVENIENCE_QUALITY}),
            12,
            "Is the pull mostly 'get it done quick,' or 'get it right'?",
            filled_markers=("quick", "fast", "get it right", "proper", "shortcut"),
            boost_dims=("convenience_vs_correctness",),
        ),
        ClarificationSlot(
            "good_enough_bar",
            frozenset({CONVENIENCE_QUALITY}),
            22,
            "What's the minimum good-enough version you'd actually accept?",
            filled_markers=("good enough", "minimum", "bare minimum", "acceptable"),
        ),
        # General fallbacks
        ClarificationSlot(
            "two_paths",
            frozenset({GENERAL}),
            50,
            "In one line each, what are the two paths you're choosing between?",
            filled_markers=("two paths", "two options", "either", "on one hand"),
            question_variants=(
                "What are the two real options in one short line each?",
                "Name the two choices you’re actually stuck between — one line per choice.",
            ),
        ),
        ClarificationSlot(
            "week_okay",
            frozenset({GENERAL}),
            60,
            "A week later, what would make you feel okay about how you chose?",
            filled_markers=("week later", "feel okay", "at peace", "no regrets"),
            question_variants=(
                "Picture next week — what would make you glad you picked the way you did?",
                "A week from now, what would ‘glad I chose that’ look like for you?",
            ),
        ),
    )


@lru_cache(maxsize=1)
def _slots_tuple() -> Tuple[ClarificationSlot, ...]:
    return _all_slots()


def _padded_ctx(parts: Sequence[str]) -> str:
    merged = normalize_input(" ".join(p for p in parts if p))
    return f" {merged} "


def _effective_priority(slot: ClarificationSlot, dimensions: Dict[str, float]) -> float:
    p = float(slot.priority)
    for d in slot.boost_dims:
        if d in dimensions:
            p -= min(6.0, dimensions[d] * 0.55)
    return p


def pick_next_question(
    *,
    context_parts: Sequence[str],
    asked_ids: Sequence[str],
    domain_order: Sequence[str],
    dimensions: Optional[Dict[str, float]] = None,
) -> Optional[ClarificationSlot]:
    """Pick the next single slot: family order × priority × missing markers × dimension boosts."""
    asked = set(asked_ids)
    ctx = _padded_ctx(context_parts)
    dims = dimensions if dimensions is not None else score_dimensions(
        normalize_input(" ".join(context_parts))
    )
    slots = _slots_tuple()
    for fam in domain_order:
        tier = [s for s in slots if fam in s.families]
        tier.sort(key=lambda s: (_effective_priority(s, dims), s.slot_id))
        for slot in tier:
            if slot.slot_id in asked:
                continue
            if any(m in ctx for m in slot.filled_markers):
                continue
            return slot
    return None


# --- Evidence extraction (situation facts vs tendencies) ---

def get_slot_by_id(slot_id: str) -> Optional[ClarificationSlot]:
    for s in _slots_tuple():
        if s.slot_id == slot_id:
            return s
    return None


def extract_clarification_evidence(
    slot_id: str, answer: str
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, float]]]:
    """Public helper for tests: situation facts vs tendency signals from one answer."""
    slot = get_slot_by_id(slot_id)
    if not slot:
        return [], []
    return _extract_situation_and_tendency(slot, normalize_input(answer))


def _extract_situation_and_tendency(
    slot: ClarificationSlot, answer_norm: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, float]]]:
    """Return (situation_key_value pairs, tendency_key_signal pairs)."""
    sit: List[Tuple[str, str]] = []
    tend: List[Tuple[str, float]] = []
    a = f" {answer_norm} "

    def has(*words: str) -> bool:
        return any(w in answer_norm for w in words)

    sid = slot.slot_id
    if sid == "bills_basics":
        if has("not paid", "unpaid", "behind", "late", "still owe", "havent", "haven't", "outstanding"):
            sit.append(("housing_bill_pressure", "open"))
        elif has("paid", "handled", "covered", "sorted", "square", "done"):
            sit.append(("housing_bill_pressure", "handled"))
    elif sid == "need_vs_want":
        if has("need", "essential", "work", "school", "job", "must"):
            sit.append(("purchase_need_level", "need"))
        elif has("want", "luxury", "nice"):
            sit.append(("purchase_need_level", "want"))
    elif sid == "energy_capacity":
        if has("no energy", "exhausted", "burnt", "burned", "tapped", "empty", "can't", "cant"):
            sit.append(("energy_available", "low"))
            tend.append(("tendency_low_energy_guard", 0.35))
        elif has("fine", "okay", "enough energy", "manage"):
            sit.append(("energy_available", "ok"))
    elif sid == "guilt_axis":
        if has("say no", "no ", "boundary", "let down", "disappoint"):
            tend.append(("tendency_guilt_about_no", 0.4))
        if has("regret", "resent", "yes "):
            tend.append(("tendency_regret_if_yes", 0.4))
    elif sid == "peace_vs_clarity":
        if has("peace", "quiet", "avoid"):
            tend.append(("tendency_peace_over_confrontation", 0.35))
        if has("say", "clear", "honest", "address"):
            tend.append(("tendency_clarity_priority", 0.35))
    elif sid == "reversibility":
        if has("undo", "change", "reversible", "adjust"):
            sit.append(("choice_reversibility", "high"))
        if has("permanent", "one-way", "cant take back", "can't take back"):
            sit.append(("choice_reversibility", "low"))

    return sit, tend


def _persist_evidence(store, situation: Sequence[Tuple[str, str]], tendencies: Sequence[Tuple[str, float]]) -> None:
    rec = getattr(store, "record_router_situation_fact", None)
    merge = getattr(store, "merge_router_tendency", None)
    if callable(rec):
        for k, v in situation:
            try:
                rec(k, v)
            except Exception:
                pass
    if callable(merge):
        for k, sig in tendencies:
            try:
                merge(k, sig)
            except Exception:
                pass


def _situation_counts_recent(store, limit: int = 30) -> Dict[str, int]:
    fn = getattr(store, "get_recent_router_situation_facts", None)
    if not callable(fn):
        return {}
    try:
        rows = fn(limit)
    except Exception:
        return {}
    counts: Dict[str, int] = {}
    for r in rows:
        key = f"{r.get('slot_key')}={r.get('slot_value')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


# --- Guidance (data-driven checks on merged context + memory) ---

def _profile_lines(profile: Optional[PersonalProfile], seed: str) -> List[str]:
    """At most one rotated line; higher evidence bar (Phase 32)."""
    if not profile or profile.total_evidence_weight < 1.15:
        return []
    candidates: List[str] = []
    risk = profile.decision_risk_summary()
    risk_ok = any(
        t.name == "risk_tolerance" and t.confidence >= 0.48
        for t in profile.trait_estimates
    )
    if risk and risk_ok:
        if risk == "leans cautious":
            risk_phrase = "often slow down when a call feels heavy"
        elif risk == "leans risk-tolerant":
            risk_phrase = "been okay taking bigger swings before"
        else:
            risk_phrase = risk
        candidates.append(
            f"From older saves, you {risk_phrase} — only you know if this fits today."
        )
    ds = next((t for t in profile.trait_estimates if t.name == "decision_speed"), None)
    if ds and ds.confidence >= 0.48 and ds.weighted_mean <= 0.38:
        candidates.append(
            "Older answers suggest you like a beat before big calls — "
            "that’s fine if nothing is actually on fire."
        )
    vt = profile.value_tag_weights[:6]
    if any(
        ("regret" in x[0].lower() or "security" in x[0].lower()) and x[1] >= 1.15
        for x in vt
    ):
        candidates.append(
            "You’ve leaned toward playing it safe before — weigh that against what you’d give up here."
        )
    if not candidates:
        return []
    idx = _stable_index(f"{seed}:profile_line", len(candidates))
    return [candidates[idx]]


def _current_money_context_strong(norm_text: str) -> bool:
    """Gate housing/bill memory: only when this turn clearly involves money pressure."""
    d = score_dimensions(norm_text)
    if d.get("money_pressure", 0) >= 1.15:
        return True
    if d.get("money_pressure", 0) >= 0.95 and any(
        x in norm_text for x in ("rent", "bill", "afford", "buy", "purchase", "pay", "debt", "salary", "broke")
    ):
        return True
    hits = sum(
        1
        for t in ("rent", "bill", "afford", "purchase", "salary", "budget", "loan", "spend", "$", "money", "broke")
        if t in norm_text
    )
    return hits >= 3


def _obligation_overlap_strong(norm_text: str, dims: Dict[str, float]) -> bool:
    """Helping / overload / boundary signal strong enough to reuse obligation-style memory."""
    combo = (
        dims.get("obligation", 0)
        + dims.get("overload", 0)
        + dims.get("boundary_strain", 0) * 0.9
    )
    if combo >= 1.25:
        return True
    padded = f" {norm_text} "
    cues = (
        " exhausted",
        " burnout",
        " burnt out",
        "burned out",
        " shift",
        " overtime",
        " favor",
        " cover for",
        " cover a shift",
        "needs me to cover",
        " asked me",
        " too much",
        " extra shift",
        " extra hours",
        " owe ",
        "help them",
        "help her",
        "help him",
        "pick up a shift",
        "say no",
    )
    return sum(1 for c in cues if c in padded or c.strip() in norm_text) >= 2


def _tendency_lines(
    tmap: Dict[str, float],
    *,
    primary_family: str,
    initial_norm: str,
    seed: str,
) -> List[str]:
    lines: List[str] = []
    dims0 = score_dimensions(initial_norm)
    t_guilt = 0.42
    t_regret = 0.42
    t_energy = 0.42

    if tmap.get("tendency_guilt_about_no", 0) >= t_guilt:
        if _obligation_overlap_strong(initial_norm, dims0) and (
            primary_family in (OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY)
            or (
                primary_family == GENERAL
                and dims0.get("obligation", 0) + dims0.get("overload", 0) >= 1.35
            )
        ):
            lines.append(
                "You’ve worried a lot about saying no before — check if that’s guilt or real fallout."
            )
    if tmap.get("tendency_regret_if_yes", 0) >= t_regret:
        if primary_family in (OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY):
            lines.append(
                "You’ve said yes before and felt bitter after — if that feeling’s back, take it seriously."
            )
        elif primary_family == CONFLICT_FAMILY and dims0.get("regret_risk", 0) >= 0.95:
            lines.append(
                "You’ve said yes before and felt bitter after — if that feeling’s back, take it seriously."
            )
        elif _obligation_overlap_strong(initial_norm, dims0) and dims0.get("regret_risk", 0) >= 0.55:
            lines.append(
                "You’ve said yes before and felt bitter after — if that feeling’s back, take it seriously."
            )
    if tmap.get("tendency_low_energy_guard", 0) >= t_energy:
        if primary_family == CONFLICT_FAMILY and dims0.get("overload", 0) < 0.85:
            pass
        elif primary_family == OBLIGATION_OVERLOAD or dims0.get("overload", 0) >= 0.95:
            lines.append(
                "You’ve been wiped in similar spots — piling on another full yes usually doesn’t age well."
            )
    if len(lines) > 1:
        pick = _stable_index(f"{seed}:tendency", len(lines))
        return [lines[pick]]
    return lines


def _pattern_repeat_line(counts: Dict[str, int], primary_family: str) -> Optional[str]:
    if primary_family != SPENDING:
        return None
    if counts.get("housing_bill_pressure=open", 0) >= 3:
        return (
            "Rent or bill stress has shown up a few times in past clarifications — "
            "worth treating that as a pattern, not a one-off bad week."
        )
    return None


def build_routed_decision_guidance(
    *,
    original_question: str,
    qa_pairs: Sequence[Tuple[str, str]],
    domain_order: Sequence[str],
    profile: Optional[PersonalProfile],
    tendency_map: Optional[Dict[str, float]] = None,
    situation_repeat_counts: Optional[Dict[str, int]] = None,
) -> str:
    parts_ctx = [original_question] + [a for _, a in qa_pairs]
    ctx = _padded_ctx(parts_ctx)
    primary = domain_order[0] if domain_order else GENERAL
    tendency_map = tendency_map or {}
    counts = situation_repeat_counts or {}
    phrase_seed = hashlib.sha256(
        normalize_input(original_question).encode("utf-8")
    ).hexdigest()[:24]

    bodies: List[str] = []

    if primary == SPENDING:
        unpaid = any(
            x in ctx
            for x in (
                "not paid", "unpaid", "haven't paid", "havent paid", "isn't paid", "isnt paid",
                "still unpaid", "behind on rent", "still owe", "outstanding",
            )
        )
        if unpaid and ("rent" in ctx or "bill" in ctx or "housing" in ctx):
            bodies.append(
                "If rent or core bills are still open, a big buy hits the sorest spot first. "
                "That doesn’t mean never — it means get basics steadier first, unless this buy is how you keep income or pass something you can’t move."
            )
        elif " need " in ctx or "need for" in ctx or "mostly need" in ctx or ("school" in ctx and "want" not in ctx):
            bodies.append(
                "If it’s a real need for work or school, treat it like gear: what’s the cheapest setup that still works, what’s one step up, and is the extra cash worth it."
            )
        elif " want " in ctx or "mostly want" in ctx or "luxury" in ctx:
            bodies.append(
                "If it’s mostly a want while cash is tight, waiting isn’t weak — it’s space to choose without boxing yourself in. Pick a date to check again."
            )
        else:
            bodies.append(
                "Money stuff is easier when food, rent, and getting around are honest first. If those wobble, trim the spend or wait until one layer feels steadier."
            )
    elif primary == OBLIGATION_OVERLOAD:
        if "low" in ctx and ("energy" in ctx or "exhaust" in ctx):
            bodies.append(
                "If you’re out of gas, the kind move is a smaller yes, a later yes, or a short honest no — not a hero yes you’ll hate later."
            )
        else:
            bodies.append(
                "Helping people works better when you know your real line before you answer. A plain ‘here’s what I can do’ beats a full yes you’ll resent."
            )
    elif primary == CONFLICT_FAMILY:
        if "peace" in ctx or "quiet" in ctx or "avoid" in ctx:
            bodies.append(
                "If calm matters most, small steady limits usually beat one huge blow-up. You can stay decent and still say what you won’t take."
            )
        elif "say" in ctx or "clear" in ctx or "honest" in ctx:
            bodies.append(
                "If something needs saying, one clear point and one example beats a long speech. Say what you want next time, not your whole life story."
            )
        else:
            bodies.append(
                "This is mostly about what you can live with afterward. Decide if you want things fixed, some distance, or just straight talk — those take different moves."
            )
    elif primary == RISK_TIMING:
        bodies.append(
            "Split real deadlines from nerves. If waiting doesn’t break anything, use the pause to grab one missing fact. If there’s a real cutoff, count backward from it."
        )
        if "low" in ctx and "revers" in ctx:
            bodies.append(
                "Hard-to-undo choices deserve a slower yes; easy-to-undo ones can be small tries."
            )
    elif primary == LOYALTY_BOUNDARY:
        bodies.append(
            "Being loyal doesn’t have to mean wiping yourself out. If yes costs sleep, money, or self-respect every time, the habit is the issue — not only this one ask."
        )
    elif primary == CONVENIENCE_QUALITY:
        bodies.append(
            "When fast fights ‘do it right,’ try the smallest step that still shows if the careful path is worth it — don’t let rush lock you into fix-it-twice work."
        )
    else:
        bodies.append(
            "Pick what you’d stand by with a friend who’s on your side — not the story that only sounds good when you’re tired. If both choices hurt, guard what’s costly to undo."
        )

    initial_norm = normalize_input(original_question)
    extra = (
        _pattern_repeat_line(counts, primary)
        if _current_money_context_strong(initial_norm)
        else None
    )
    if extra:
        bodies.insert(0, extra)

    bodies.extend(
        _tendency_lines(
            tendency_map,
            primary_family=primary,
            initial_norm=initial_norm,
            seed=phrase_seed,
        )
    )
    bodies.extend(_profile_lines(profile, phrase_seed))
    return "\n\n".join(bodies)


def run_routed_decision_guidance(
    *,
    initial_text: str,
    read_line: Callable[[str], str],
    db_store,
) -> str:
    norm = normalize_input(initial_text)
    ranked, dimensions = rank_families(norm)

    primary = ranked[0][0]
    order: List[str] = []
    if ranked[0][1] < 0.55:
        primary = GENERAL
    order.append(primary)
    for fam, sc in ranked[1:]:
        if fam not in order and sc >= 1.25:
            order.append(fam)
    if GENERAL not in order:
        order.append(GENERAL)

    try:
        profile = build_personal_profile(db_store)
        if profile.total_evidence_weight < 0.85:
            profile = None
    except Exception:
        profile = None

    try:
        tendency_map = db_store.get_router_tendency_map()
    except Exception:
        tendency_map = {}

    context_parts: List[str] = [initial_text]
    asked_ids: List[str] = []
    qa_pairs: List[Tuple[str, str]] = []
    session_sit: List[Tuple[str, str]] = []
    session_tend: List[Tuple[str, float]] = []

    session_seed = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]

    for _ in range(MAX_CLARIFICATION_ROUNDS):
        slot = pick_next_question(
            context_parts=context_parts,
            asked_ids=asked_ids,
            domain_order=order,
            dimensions=dimensions,
        )
        if slot is None:
            break
        qtext = slot.wording(session_seed)
        print(qtext)
        try:
            ans = (read_line("Your answer: ") or "").strip()
        except EOFError:
            ans = ""
        asked_ids.append(slot.slot_id)
        if not ans:
            break
        context_parts.append(ans)
        qa_pairs.append((qtext, ans))
        an = normalize_input(ans)
        sit, tend = _extract_situation_and_tendency(slot, an)
        session_sit.extend(sit)
        session_tend.extend(tend)
        # Refresh dimensions lightly with new text (deterministic follow-up signal)
        dimensions = score_dimensions(normalize_input(" ".join(context_parts)))

    _persist_evidence(db_store, session_sit, session_tend)

    try:
        tendency_map = db_store.get_router_tendency_map()
    except Exception:
        tendency_map = tendency_map or {}

    repeat_counts = _situation_counts_recent(db_store)

    return build_routed_decision_guidance(
        original_question=initial_text,
        qa_pairs=qa_pairs,
        domain_order=order,
        profile=profile,
        tendency_map=tendency_map,
        situation_repeat_counts=repeat_counts,
    )


LEGACY_GENERIC_PHRASES = (
    "what are the available options or approaches",
    "this framework will help us make a more informed choice",
)
