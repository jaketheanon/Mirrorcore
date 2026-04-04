"""
Phase 33: deterministic decision ontology + multi-label axis scoring.

Inspectable, data-driven rules (keyword groups + phrase patterns + explicit
family boosts). No randomness. Family bucket ids stay stable for storage/tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, MutableMapping, Optional, Sequence, Tuple

# --- Family buckets (stable ids; used by slots, guidance, tests) ---
SPENDING = "spending"
OBLIGATION_OVERLOAD = "obligation_overload"
CONFLICT_FAMILY = "conflict"
RISK_TIMING = "risk_timing"
LOYALTY_BOUNDARY = "loyalty_boundary"
CONVENIENCE_QUALITY = "convenience_quality"
GENERAL = "general"

# Backwards-compatible aliases (Phase 31+)
MONEY = SPENDING
CONFLICT = CONFLICT_FAMILY
TIMING = RISK_TIMING
OBLIGATION = OBLIGATION_OVERLOAD
RISK = RISK_TIMING

# --- Phase 33 ontology axes (multi-label; inspectable in tests) ---
# Each axis is a coarse “decision shape” signal; one utterance can fire several.
AXIS_MONEY_SPENDING = "money_spending"
AXIS_BILLS_FINANCIAL_PRESSURE = "bills_financial_pressure"
AXIS_WORK_OBLIGATION = "work_obligation"
AXIS_HELPING_FAVOR = "helping_favor"
AXIS_OVERLOAD_BURNOUT = "overload_burnout"
AXIS_BOUNDARY_SETTING = "boundary_setting"
AXIS_CONFLICT_CONFRONTATION = "conflict_confrontation"
AXIS_BACKCHANNEL_HURT = "backchannel_hurt"
AXIS_TIMING_WAIT_VS_ACT = "timing_wait_vs_act"
AXIS_UNCERTAINTY_RISK = "uncertainty_risk"
AXIS_NEED_VS_WANT = "need_vs_want"
AXIS_LOYALTY_VS_SELF = "loyalty_vs_self_protection"
AXIS_CONVENIENCE_VS_CORRECT = "convenience_vs_correctness"
AXIS_SHORT_VS_LONG = "short_term_relief_vs_long_term_cost"

# (axis_id, phrase tuple, weight per hit)
_ONTOLOGY_AXIS_RULES: Tuple[Tuple[str, Tuple[str, ...], float], ...] = (
    (
        AXIS_BILLS_FINANCIAL_PRESSURE,
        (
            "rent is late",
            "rent late",
            "late rent",
            "late on rent",
            "behind on rent",
            "bills late",
            "bills unpaid",
            "not paid rent",
            "havent paid rent",
            "haven't paid rent",
            "cant afford",
            "can't afford",
            "broke",
            "overdraft",
            "collections",
            "eviction",
        ),
        1.05,
    ),
    (
        AXIS_MONEY_SPENDING,
        (
            "buy",
            "purchase",
            "spend",
            "price",
            "$",
            "dollar",
            "afford",
            "expensive",
            "cheap",
            "budget",
            "save money",
            "waste money",
        ),
        0.85,
    ),
    (
        AXIS_WORK_OBLIGATION,
        (
            "shift",
            "overtime",
            "cover a shift",
            "cover shift",
            "cover for",
            "covering for",
            "pick up a shift",
            "extra hours",
            "on call",
            "coworker",
            "colleague",
            "boss",
            "manager",
            "workplace",
            "job on the line",
        ),
        0.95,
    ),
    (
        AXIS_HELPING_FAVOR,
        (
            "favor",
            "asked me to",
            "needs me to",
            "want me to",
            "help them",
            "help her",
            "help him",
            "do them a solid",
            "cover for them",
        ),
        1.0,
    ),
    (
        AXIS_OVERLOAD_BURNOUT,
        (
            "exhausted",
            "burnout",
            "burned out",
            "burnt out",
            "wiped out",
            "drained",
            "no energy",
            "too much",
            "overwhelmed",
            "overloaded",
            "at capacity",
            "already drained",
            "cant take more",
            "can't take more",
        ),
        1.1,
    ),
    (
        AXIS_BOUNDARY_SETTING,
        (
            "say no",
            "boundary",
            "people pleas",
            "walk all over",
            "cant keep saying yes",
            "can't keep saying yes",
            "protect myself",
            "my limit",
        ),
        0.95,
    ),
    (
        AXIS_BACKCHANNEL_HURT,
        (
            "behind my back",
            "behind your back",
            "talking shit",
            " talk shit",
            "rumor",
            "rumour",
            "two faced",
            "two-faced",
            "badmouth",
            "trash talk",
            "gossip",
            " passive aggressive",
            "passive-aggressive",
            " snide ",
            "underhanded",
            "whisper",
            " speaking ill",
        ),
        1.25,
    ),
    (
        AXIS_CONFLICT_CONFRONTATION,
        (
            "confront",
            "say something",
            "let it go",
            "address it",
            "clear the air",
            "unfair",
            "disrespect",
            "rude",
            "bothered",
            "upset",
            "hurt my feelings",
            "crossed the line",
            "argument",
            "fight",
            "passive aggressive",
            "passive-aggressive",
            "snide",
            "dig at",
            "digs at",
            "testing my patience",
            "pushing boundaries",
            "push my buttons",
            "walk all over",
        ),
        0.9,
    ),
    (
        AXIS_TIMING_WAIT_VS_ACT,
        (
            "wait or",
            "whether to wait",
            "act now",
            "hold off",
            "move now",
            "sooner or later",
            "pull the trigger",
            "dont know whether",
            "don't know whether",
            "dont know if i should wait",
            "timing",
        ),
        1.15,
    ),
    (
        AXIS_UNCERTAINTY_RISK,
        (
            "risky",
            "feels risky",
            "not sure",
            "unsure",
            "uncertain",
            "what if",
            "scared",
            "nervous",
            "gamble",
            "reversible",
            "one-way",
            "missing information",
            "dont know enough",
            "don't know enough",
        ),
        0.9,
    ),
    (
        AXIS_NEED_VS_WANT,
        (
            "need it",
            "dont need",
            "don't need",
            "want it",
            "need vs want",
            "nice to have",
            "essential",
            "luxury",
        ),
        0.95,
    ),
    (
        AXIS_LOYALTY_VS_SELF,
        (
            "my mom",
            "my dad",
            "my mother",
            "my father",
            "parent",
            "family expects",
            "loyal",
            "guilt",
            "selfish",
            "let them down",
        ),
        0.9,
    ),
    (
        AXIS_CONVENIENCE_VS_CORRECT,
        (
            "quick",
            "shortcut",
            "fast",
            "proper",
            "right way",
            "good enough",
            "hack",
        ),
        0.85,
    ),
    (
        AXIS_SHORT_VS_LONG,
        (
            "short term",
            "long term",
            "temporary fix",
            "later me",
            "future me",
            "pay now or later",
        ),
        0.85,
    ),
)

# Full Phase 31–32 dimension phrase rules (preserved); ontology axes augment these additively.
_LEGACY_DIMENSION_RULES: Tuple[Tuple[str, Tuple[str, ...], float], ...] = (
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
            "passive aggressive",
            "passive-aggressive",
            " snide ",
            "underhanded",
            "dig at me",
            "keeps disrespect",
            "pushing it",
            "pushes boundaries",
        ),
        1.15,
    ),
    (
        "conflict_intensity",
        (
            "argue",
            "fight",
            "angry",
            "resent",
            "toxic",
            "unfair",
            "silent treatment",
            "every time",
            "keeps doing",
            "pattern",
        ),
        1.0,
    ),
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

# axis_id -> (family, boost) — multi-label: one axis can feed multiple families
_AXIS_FAMILY_BOOST: Tuple[Tuple[str, str, float], ...] = (
    (AXIS_BILLS_FINANCIAL_PRESSURE, SPENDING, 2.35),
    (AXIS_MONEY_SPENDING, SPENDING, 1.15),
    (AXIS_NEED_VS_WANT, SPENDING, 1.05),
    (AXIS_OVERLOAD_BURNOUT, OBLIGATION_OVERLOAD, 2.15),
    (AXIS_WORK_OBLIGATION, OBLIGATION_OVERLOAD, 1.55),
    (AXIS_HELPING_FAVOR, OBLIGATION_OVERLOAD, 1.45),
    (AXIS_BOUNDARY_SETTING, OBLIGATION_OVERLOAD, 1.15),
    (AXIS_BOUNDARY_SETTING, LOYALTY_BOUNDARY, 1.05),
    (AXIS_LOYALTY_VS_SELF, LOYALTY_BOUNDARY, 1.35),
    (AXIS_LOYALTY_VS_SELF, OBLIGATION_OVERLOAD, 1.05),
    (AXIS_BACKCHANNEL_HURT, CONFLICT_FAMILY, 3.4),
    (AXIS_CONFLICT_CONFRONTATION, CONFLICT_FAMILY, 2.35),
    (AXIS_TIMING_WAIT_VS_ACT, RISK_TIMING, 2.95),
    (AXIS_UNCERTAINTY_RISK, RISK_TIMING, 1.65),
    (AXIS_SHORT_VS_LONG, CONVENIENCE_QUALITY, 1.25),
    (AXIS_CONVENIENCE_VS_CORRECT, CONVENIENCE_QUALITY, 2.05),
)

# Base family keywords (substring hit, weight) — padded matching in scorer
_FAMILY_KEYWORDS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    SPENDING: (
        ("buy", 1.0),
        ("purchase", 1.0),
        ("spend", 1.0),
        ("afford", 1.1),
        ("price", 0.9),
        ("rent", 1.2),
        ("bill", 1.0),
        ("loan", 0.9),
        ("save", 0.7),
        ("cheap", 0.8),
        ("expensive", 0.9),
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
        ("passive aggressive", 1.05),
        ("behind my back", 1.0),
        ("boundary", 0.95),
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
        ("loyal", 1.0),
        ("betray", 0.9),
        ("family", 0.8),
        ("guilt", 1.0),
        ("selfish", 0.8),
        ("protect myself", 1.0),
        ("people pleaser", 1.1),
    ),
    CONVENIENCE_QUALITY: (
        ("quick", 1.0),
        ("shortcut", 1.1),
        ("hack", 0.7),
        ("proper", 0.9),
        ("good enough", 1.0),
    ),
    GENERAL: (("choice", 0.3), ("decide", 0.4), ("should i", 0.5)),
}

# Explicit tie-break when scores are within epsilon (stable sort uses name too)
_FAMILY_TIE_ORDER: Tuple[str, ...] = (
    SPENDING,
    OBLIGATION_OVERLOAD,
    CONFLICT_FAMILY,
    RISK_TIMING,
    LOYALTY_BOUNDARY,
    CONVENIENCE_QUALITY,
    GENERAL,
)



def score_ontology_axes(norm_text: str) -> Dict[str, float]:
    """Multi-label axis scores for the utterance (inspectable, testable)."""
    padded = f" {norm_text} "
    out: Dict[str, float] = {}
    for axis_id, phrases, w in _ONTOLOGY_AXIS_RULES:
        s = 0.0
        for p in phrases:
            if p in padded or p in norm_text:
                s += w
        if s > 0:
            out[axis_id] = s
    return out


def _augment_dimensions_from_ontology_axes(
    axes: Dict[str, float], out: Dict[str, float], norm_text: str
) -> None:
    """Add Phase 33 axis signal on top of legacy phrase dimensions (no replacement)."""

    def add(key: str, val: float) -> None:
        if val <= 0:
            return
        out[key] = out.get(key, 0.0) + val

    add(
        "money_pressure",
        axes.get(AXIS_BILLS_FINANCIAL_PRESSURE, 0.0) * 0.55 + axes.get(AXIS_MONEY_SPENDING, 0.0) * 0.2,
    )
    add("need_vs_want_signal", axes.get(AXIS_NEED_VS_WANT, 0.0) * 0.45)
    add("overload", axes.get(AXIS_OVERLOAD_BURNOUT, 0.0) * 0.35)
    add(
        "obligation",
        axes.get(AXIS_WORK_OBLIGATION, 0.0) * 0.35 + axes.get(AXIS_HELPING_FAVOR, 0.0) * 0.35,
    )
    add(
        "boundary_strain",
        axes.get(AXIS_BOUNDARY_SETTING, 0.0) * 0.4 + axes.get(AXIS_LOYALTY_VS_SELF, 0.0) * 0.25,
    )
    add(
        "interpersonal_hurt",
        axes.get(AXIS_BACKCHANNEL_HURT, 0.0) * 0.85 + axes.get(AXIS_CONFLICT_CONFRONTATION, 0.0) * 0.25,
    )
    add("wait_vs_act", axes.get(AXIS_TIMING_WAIT_VS_ACT, 0.0) * 0.45)
    add("uncertainty", axes.get(AXIS_UNCERTAINTY_RISK, 0.0) * 0.35)
    add("risk_level", axes.get(AXIS_UNCERTAINTY_RISK, 0.0) * 0.25)
    add("convenience_vs_correctness", axes.get(AXIS_CONVENIENCE_VS_CORRECT, 0.0) * 0.35)
    add("short_vs_long", axes.get(AXIS_SHORT_VS_LONG, 0.0) * 0.35)
    # Strong gossip signal even if legacy interpersonal phrases missed wording
    padded = f" {norm_text} "
    if axes.get(AXIS_BACKCHANNEL_HURT, 0) >= 1.0:
        add("interpersonal_hurt", 0.9)
    if "someone keeps" in padded or "keeps talking" in padded:
        add("interpersonal_hurt", 0.55)
    if "passive aggressive" in padded or "passive-aggressive" in norm_text:
        add("interpersonal_hurt", 0.65)
        add("conflict_intensity", 0.45)


def score_dimensions(norm_text: str) -> Dict[str, float]:
    """Legacy dimension scores for slot boosts and guidance (Phase 31+ API)."""
    padded = f" {norm_text} "
    out: Dict[str, float] = {}
    for dim, phrases, w in _LEGACY_DIMENSION_RULES:
        s = 0.0
        for p in phrases:
            if p in padded or p in norm_text:
                s += w
        if s > 0:
            out[dim] = s
    axes = score_ontology_axes(norm_text)
    _augment_dimensions_from_ontology_axes(axes, out, norm_text)
    return out


def infer_family_scores_from_keywords_and_axes(
    norm_text: str,
    axes: Optional[Dict[str, float]] = None,
    dimensions: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Weighted family scores from keywords + ontology axis boosts."""
    axes = axes if axes is not None else score_ontology_axes(norm_text)
    dimensions = dimensions if dimensions is not None else score_dimensions(norm_text)

    scores: Dict[str, float] = {f: 0.05 for f in _FAMILY_KEYWORDS}
    scores[GENERAL] = 0.15
    padded = f" {norm_text} "
    for fam, kws in _FAMILY_KEYWORDS.items():
        for kw, wt in kws:
            if kw in padded or kw in norm_text:
                scores[fam] = scores.get(fam, 0) + wt

    # Ontology axes → families
    for axis_id, fam, boost in _AXIS_FAMILY_BOOST:
        if axis_id in axes:
            scores[fam] = scores.get(fam, 0) + boost * min(1.2, axes[axis_id] / 2.5)

    # Legacy dimension → family boosts (kept for slot/guidance parity)
    _LEGACY_DIM_FAMILY_BOOST: Tuple[Tuple[str, str, float], ...] = (
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
    for dim, fam, bonus in _LEGACY_DIM_FAMILY_BOOST:
        if dim in dimensions:
            scores[fam] = scores.get(fam, 0) + bonus * min(1.2, dimensions[dim] / 2.5)

    return scores


def _tie_break_index(family: str) -> int:
    try:
        return _FAMILY_TIE_ORDER.index(family)
    except ValueError:
        return len(_FAMILY_TIE_ORDER)


def order_family_scores(raw: Dict[str, float]) -> List[Tuple[str, float]]:
    """Sort families by score (desc), then explicit tie-break order, then name."""
    items = list(raw.items())

    def sort_key(item: Tuple[str, float]) -> Tuple[float, int, str]:
        fam, sc = item
        return (-round(sc, 5), _tie_break_index(fam), fam)

    items.sort(key=sort_key)
    return items


@dataclass(frozen=True)
class DecisionSignals:
    """Inspectable bundle for one normalized user utterance."""

    axes: Dict[str, float]
    dimensions: Dict[str, float]
    family_scores: Dict[str, float]


def evaluate_decision_signals(norm_text: str) -> DecisionSignals:
    """Full scoring pass (axes + legacy dimensions + family scores before post-rules)."""
    axes = score_ontology_axes(norm_text)
    dimensions = score_dimensions(norm_text)
    families = infer_family_scores_from_keywords_and_axes(
        norm_text, axes=axes, dimensions=dimensions
    )
    return DecisionSignals(axes=axes, dimensions=dimensions, family_scores=families)


# --- Post-rules (kept callable from routed_clarification) ---

# Only treat as interpersonal *conflict* when these show up — not mere "coworker" / peer reference.
_INTERPERSONAL_CONFLICT_MARKERS: Tuple[str, ...] = (
    " rude ",
    " rudely ",
    " disrespect",
    " unfair",
    " argument",
    " fight",
    " bothered",
    " bother me",
    " talk shit",
    " gossip",
    " let it go",
    " say something",
    " cross the line",
    " crossed the line",
    " hurt my feelings",
    " insult",
    "two faced",
    "two-faced",
    " trash talk",
    " badmouth",
    " passive aggressive",
    "passive-aggressive",
    " snide",
    " underhanded",
    "dig at me",
    " behind my back",
    "behind your back",
    "pushing my buttons",
    "pushes boundaries",
    "testing my patience",
)


def interpersonal_conflict_markers_present(norm_text: str) -> bool:
    """Explicit upset / confrontation wording — not workplace obligation alone."""
    padded = f" {norm_text} "
    for m in _INTERPERSONAL_CONFLICT_MARKERS:
        if m in padded or m.strip() in norm_text:
            return True
    return False


def work_obligation_peer_shape(norm_text: str, dimensions: Dict[str, float]) -> bool:
    """
    Peer/family + cover/shift/favor ask + fatigue — should stay in obligation/overload
    routing, not conflict, unless interpersonal conflict markers are present.
    """
    padded = f" {norm_text} "
    strain = (
        dimensions.get("overload", 0) >= 0.55
        or dimensions.get("obligation", 0) >= 0.45
        or any(
            x in padded
            for x in (
                " exhausted",
                " burnt out",
                " burned out",
                " drained",
                " overwhelmed",
                " no energy",
                " too tired",
                " burnout",
            )
        )
    )
    if not strain:
        return False

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
    work_ask = any(
        x in padded or x in norm_text
        for x in (
            " shift ",
            " cover ",
            " covering ",
            " overtime ",
            " extra hours ",
            "pick up a shift",
            "cover a shift",
            "cover for",
            "covering for",
            " asked me ",
            " asking me ",
            " needs me ",
            " want me to ",
            " wants me ",
            " favor ",
            " favour ",
        )
    )
    path_peer = bool(peer and work_ask)

    family_favor = any(
        x in padded
        for x in (
            " my mom ",
            " my mother ",
            " my dad ",
            " my father ",
            " mom wants",
            " mother wants",
            " dad wants",
            " parent ",
        )
    )
    favor_verb = any(
        x in padded
        for x in (
            " favor ",
            " favour ",
            " asking me ",
            " needs me to ",
            " want me to ",
            " wants me ",
            " ask me to ",
        )
    )
    path_family = bool(family_favor and favor_verb)

    return path_peer or path_family


def apply_work_obligation_demotion_to_conflict(
    norm_text: str, dimensions: Dict[str, float], scores: MutableMapping[str, float],
) -> None:
    """Strip conflict mass from coworker + shift/favor + fatigue without conflict cues."""
    if interpersonal_conflict_markers_present(norm_text):
        return
    if not work_obligation_peer_shape(norm_text, dimensions):
        return
    c = scores.get(CONFLICT_FAMILY, 0.0)
    if c <= 0:
        return
    scores[CONFLICT_FAMILY] = c * 0.08
    scores[OBLIGATION_OVERLOAD] = scores.get(OBLIGATION_OVERLOAD, 0) + 1.85


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
    "my mom",
    "my mother",
    "parent wants",
)


def boost_obligation_for_cover_shift_fatigue(
    norm_text: str,
    dimensions: Dict[str, float],
    axes: Dict[str, float],
    scores: MutableMapping[str, float],
) -> None:
    """Workplace cover/shift + fatigue → obligation-overload, not raw conflict."""
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
    strain = (
        dimensions.get("overload", 0) >= 0.85
        or dimensions.get("obligation", 0) >= 0.45
        or axes.get(AXIS_OVERLOAD_BURNOUT, 0) >= 0.95
        or axes.get(AXIS_WORK_OBLIGATION, 0) >= 0.85
    )
    if not (peer and shift_like and strain):
        return
    scores[OBLIGATION_OVERLOAD] = scores.get(OBLIGATION_OVERLOAD, 0) + 3.35
    if dimensions.get("interpersonal_hurt", 0) < 0.55:
        scores[CONFLICT_FAMILY] = scores.get(CONFLICT_FAMILY, 0) * 0.55


def social_conflict_without_obligation(norm_text: str, dimensions: Dict[str, float]) -> bool:
    """Interpersonal upset without favor/shift/exhaustion context."""
    if dimensions.get("interpersonal_hurt", 0) <= 0:
        return False
    padded = f" {norm_text} "
    if not any(m in padded or m in norm_text for m in _OBLIGATION_OVERLOAD_CUES):
        if dimensions.get("overload", 0) < 0.55 and dimensions.get("obligation", 0) < 0.55:
            return True
    return False


def adjust_family_scores_for_social_conflict(
    norm_text: str, dimensions: Dict[str, float], scores: MutableMapping[str, float],
) -> None:
    """Keep pure social hurt out of obligation unless cues match."""
    if not social_conflict_without_obligation(norm_text, dimensions):
        return
    scores[CONFLICT_FAMILY] = scores.get(CONFLICT_FAMILY, 0) + 5.0
    scores[OBLIGATION_OVERLOAD] = scores.get(OBLIGATION_OVERLOAD, 0) * 0.32


def mom_favor_drained_boost(norm_text: str, scores: MutableMapping[str, float]) -> None:
    """Family favor + drained → obligation/loyalty, not generic conflict."""
    padded = f" {norm_text} "
    family = any(x in padded for x in (" my mom ", " my mother ", " my dad ", " my father ", " parent "))
    drained = any(
        x in padded or x in norm_text
        for x in ("drained", "burnt out", "burned out", "exhausted", "no energy", "too much already")
    )
    favor = any(x in padded for x in (" favor ", " wants ", " asking me ", " ask me "))
    if family and drained and favor:
        scores[OBLIGATION_OVERLOAD] = scores.get(OBLIGATION_OVERLOAD, 0) + 2.1
        scores[LOYALTY_BOUNDARY] = scores.get(LOYALTY_BOUNDARY, 0) + 1.25
        scores[CONFLICT_FAMILY] = scores.get(CONFLICT_FAMILY, 0) * 0.72


def risk_wait_tension_boost(norm_text: str, scores: MutableMapping[str, float]) -> None:
    """Risk + wait/act tradeoff phrasing → timing family."""
    padded = f" {norm_text} "
    if ("risk" in padded or "risky" in norm_text) and any(
        x in padded for x in (" wait ", "waiting", " act ", "act now", "cost", "too long")
    ):
        scores[RISK_TIMING] = scores.get(RISK_TIMING, 0) + 1.45


def demote_shallow_timing_for_peer_continuation(
    norm_text: str, scores: MutableMapping[str, float]
) -> None:
    """
    ``today`` / ``now`` on a peer-continuation line (same/still/again…) is usually
    scene-setting, not a wait-vs-act decision.  Demote timing only when no strong
    timing question is present (Phase 44 ask routing).
    """
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
    if not peer:
        return
    ref = any(
        x in padded
        for x in (
            " same ",
            " still ",
            " again ",
            " another ",
            " keeps ",
            "same coworker",
            "same person",
            "still pushing",
            "still asking",
            "still trying",
        )
    )
    if not ref:
        return
    strong_timing = any(
        x in padded
        for x in (
            " wait or ",
            " whether to wait ",
            " whether to ",
            " act now or ",
            " hold off ",
            " know whether ",
            " dont know whether ",
            "don't know whether ",
            " not sure whether ",
            " too soon ",
            " too late ",
            " reversible ",
            " point of no return ",
            " real deadline ",
            " hard deadline ",
            " wait until ",
            " until monday",
            " until next",
            " worth waiting",
            "should i wait",
            "should we wait",
        )
    )
    if strong_timing:
        return
    rt = scores.get(RISK_TIMING, 0) or 0.0
    if rt <= 0.0:
        return
    scores[RISK_TIMING] = rt * 0.48


def rank_families_full(
    norm_text: str,
) -> Tuple[List[Tuple[str, float]], Dict[str, float], Dict[str, float]]:
    """
    Rank decision families with ontology + post-rules.

    Returns:
        ordered: (family, score) descending
        dimensions: legacy slot-boost dict
        axes: inspectable ontology axis scores
    """
    sig = evaluate_decision_signals(norm_text)
    raw = dict(sig.family_scores)
    boost_obligation_for_cover_shift_fatigue(norm_text, sig.dimensions, sig.axes, raw)
    adjust_family_scores_for_social_conflict(norm_text, sig.dimensions, raw)
    mom_favor_drained_boost(norm_text, raw)
    risk_wait_tension_boost(norm_text, raw)
    demote_shallow_timing_for_peer_continuation(norm_text, raw)
    apply_work_obligation_demotion_to_conflict(norm_text, sig.dimensions, raw)
    ordered = order_family_scores(raw)
    return ordered, sig.dimensions, sig.axes


def score_decision_domains(norm_text: str) -> List[Tuple[str, float]]:
    """Back-compat: ranked (family, score) list."""
    ordered, _, _ = rank_families_full(norm_text)
    return ordered


_SLOT_COVERAGE_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "bills_basics",
        (
            "rent is late",
            "rent late",
            "late rent",
            "late on rent",
            "behind on rent",
            "rent not paid",
            "havent paid rent",
            "haven't paid rent",
            "bills not paid",
            "cant pay rent",
            "can't pay rent",
        ),
    ),
    (
        "need_vs_want",
        (
            "dont really need",
            "don't really need",
            "dont need",
            "don't need",
            "just a want",
            "mostly a want",
            "mostly want",
            "nice to have",
            "really need",
            "mostly need",
        ),
    ),
    (
        "energy_capacity",
        (
            "no energy",
            "no bandwidth",
            "at capacity",
            "burnt out",
            "burned out",
            "burnout",
            "wiped out",
            "exhausted",
        ),
    ),
    (
        "real_deadline",
        (
            "no real deadline",
            "no deadline",
            "real deadline",
            "hard deadline",
        ),
    ),
    (
        "pattern_vs_once",
        (
            "keeps happening",
            "again and again",
            "every time",
            "pattern",
            "not the first time",
        ),
    ),
    (
        "conflict_aim",
        (
            "let it go",
            "let it ride",
            "drop it",
            "move on",
            "speak up",
            "say something",
            "set a boundary",
            "draw a line",
            "pull back",
            "take space",
            "less contact",
            "distance myself",
        ),
    ),
)


def slot_ids_covered_by_context(norm_text: str) -> FrozenSet[str]:
    """
    Slots already implied by the user text — skip redundant first questions.

    Conservative: only skip when the signal is explicit (Phase 33).
    """
    padded = f" {norm_text} "
    covered: set[str] = set()
    for slot_id, phrases in _SLOT_COVERAGE_RULES:
        if any(p in padded or p in norm_text for p in phrases):
            covered.add(slot_id)
    return frozenset(covered)
