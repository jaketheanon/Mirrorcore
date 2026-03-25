"""
Routed decision engine: slot-based clarification and structured situation/tendency
memory. Core family scoring and ontology axes live in ``decision.ontology`` (Phase 33).

Deterministic, inspectable, one question at a time. Used from ``mirrorcore ask``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..persona.profile import PersonalProfile, build_personal_profile
from ..router import normalize_input
from .cross_system_knowledge import build_ask_interview_memory_line_candidates
from .memory_relevance import (
    profile_decision_speed_snippet_allowed,
    profile_memory_fit_score,
    profile_risk_snippet_allowed,
    profile_value_tag_snippet_allowed,
)
from .ontology import (
    CONFLICT_FAMILY,
    CONVENIENCE_QUALITY,
    GENERAL,
    LOYALTY_BOUNDARY,
    OBLIGATION_OVERLOAD,
    RISK_TIMING,
    SPENDING,
    interpersonal_conflict_markers_present,
    rank_families_full,
    score_decision_domains as _ontology_score_decision_domains,
    score_dimensions,
    score_ontology_axes,
    slot_ids_covered_by_context,
    work_obligation_peer_shape,
)

# Phase 31 API alias (same as ontology.MONEY)
MONEY = SPENDING

MAX_CLARIFICATION_ROUNDS = 2

# Buckets all “saved trait” profile blurbs for aggressive repetition control (Phase 34+).
_PROFILE_SURFACE_BUCKET = "surface_profile_trait_snippet"


def _stable_index(key: str, modulo: int) -> int:
    """Deterministic pick in ``0..modulo-1`` (no ``hash()`` salt)."""
    if modulo <= 1:
        return 0
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % modulo


def rank_families(norm_text: str) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
    """Rank families + legacy dimensions (Phase 33 ontology core)."""
    ordered, dims, _axes = rank_families_full(norm_text)
    return ordered, dims


def score_decision_domains(norm_text: str) -> List[Tuple[str, float]]:
    """Back-compat: ranked (family, score) list."""
    return _ontology_score_decision_domains(norm_text)


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
                "Realistically — do you have the room for this right now?",
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
    merged_norm = normalize_input(" ".join(context_parts))
    covered = slot_ids_covered_by_context(merged_norm)
    dims = dimensions if dimensions is not None else score_dimensions(merged_norm)
    slots = _slots_tuple()
    for fam in domain_order:
        tier = [s for s in slots if fam in s.families]
        tier.sort(key=lambda s: (_effective_priority(s, dims), s.slot_id))
        for slot in tier:
            if slot.slot_id in asked:
                continue
            if slot.slot_id in covered:
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
    seen_sit: set = set()
    if callable(rec):
        for k, v in situation:
            kv = (k, v)
            if kv in seen_sit:
                continue
            seen_sit.add(kv)
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

def sanitize_domain_order_for_obligation(
    norm_text: str,
    dimensions: Dict[str, float],
    order: Sequence[str],
) -> List[str]:
    """Defer conflict slots when the ask is workplace/favor overload without conflict cues."""
    if interpersonal_conflict_markers_present(norm_text):
        return list(order)
    if not work_obligation_peer_shape(norm_text, dimensions):
        return list(order)
    out: List[str] = []
    deferred: List[str] = []
    for fam in order:
        if fam == CONFLICT_FAMILY:
            deferred.append(fam)
        else:
            out.append(fam)
    return out + deferred


def _canonical_memory_surface_key(line_key: str) -> str:
    if line_key.startswith("profile_"):
        return _PROFILE_SURFACE_BUCKET
    return line_key


def _pick_best_keyed_line(
    candidates: Sequence[Tuple[str, float, str]],
    seed: str,
    min_score: float,
) -> Optional[Tuple[str, str]]:
    """Highest relevance wins; deterministic tie-break. Returns (key, text) or None."""
    strong = [(k, s, t) for k, s, t in candidates if s >= min_score]
    if not strong:
        return None
    strong.sort(key=lambda x: (-x[1], x[0]))
    top_s = strong[0][1]
    tied = [x for x in strong if abs(x[1] - top_s) < 1e-9]
    pick = tied[_stable_index(f"{seed}:tie", len(tied))]
    return pick[0], pick[2]


def _pick_merged_memory_lines(
    tend_cands: Sequence[Tuple[str, float, str]],
    prof_cands: Sequence[Tuple[str, float, str]],
    *,
    seed: str,
    min_tend: float,
    min_prof: float,
    interview_cands: Sequence[Tuple[str, float, str]] = (),
    min_interview: float = 0.52,
) -> List[Tuple[str, str]]:
    """
    Single strongest habit/profile/interview-memory line (Phase 35–36): one closers block.
    """
    pool: List[Tuple[str, float, str]] = []
    for k, s, t in tend_cands:
        if s >= min_tend:
            pool.append((k, s, t))
    for k, s, t in prof_cands:
        if s >= min_prof:
            pool.append((k, s, t))
    for k, s, t in interview_cands:
        if s >= min_interview:
            pool.append((k, s, t))
    if not pool:
        return []
    pool.sort(key=lambda x: (-x[1], x[0]))
    top_s = pool[0][1]
    tied = [x for x in pool if abs(x[1] - top_s) < 1e-9]
    pick = tied[_stable_index(f"{seed}:mem_merge", len(tied))]
    return [(pick[0], pick[2])]


def _profile_line_candidates(
    profile: Optional[PersonalProfile],
    *,
    primary_family: str,
    initial_norm: str,
    seed: str,
) -> List[Tuple[str, float, str]]:
    """Phase 34: keyed snippets with relevance scores (traits only when family/dims fit)."""
    if not profile or profile.total_evidence_weight < 1.28:
        return []
    dims = score_dimensions(initial_norm)
    axes = score_ontology_axes(initial_norm)
    fit = profile_memory_fit_score(primary_family, initial_norm, dims, axes)
    if fit < 0.58:
        return []

    out: List[Tuple[str, float, str]] = []

    risk = profile.decision_risk_summary()
    risk_ok = any(
        t.name == "risk_tolerance" and t.confidence >= 0.52
        for t in profile.trait_estimates
    )
    if risk and risk_ok and profile_risk_snippet_allowed(primary_family, initial_norm, dims, axes):
        rel = fit * 0.96
        tails = (
            "Might be off today; trust how you feel.",
            "Use it only if it still rings true.",
            "Ignore it if this case feels different.",
            "Hold it lightly, not as a verdict.",
        )
        tail = tails[_stable_index(f"{seed}:risk_tail", len(tails))]
        if risk == "leans cautious":
            phrases = (
                (
                    "profile_risk_cautious_a",
                    "you often want a beat before a big call",
                ),
                (
                    "profile_risk_cautious_b",
                    "you usually look for a clear risk read before you move",
                ),
                (
                    "profile_risk_cautious_c",
                    "you’ve tended to pause before locking something in",
                ),
            )
            kid, phrase = phrases[_stable_index(f"{seed}:risk_c", len(phrases))]
            frames = (
                f"From older saves, {phrase} — {tail}",
                f"Past answers: {phrase}. {tail}",
                f"Patterns on file: {phrase}. {tail}",
            )
            line = frames[_stable_index(f"{seed}:risk_fr", len(frames))]
            out.append((kid, rel, line))
        elif risk == "leans risk-tolerant":
            tol = (
                (
                    "profile_risk_tolerant_a",
                    rel * 0.98,
                    "Older saves show you’ve taken bigger swings before — {tail}",
                ),
                (
                    "profile_risk_tolerant_b",
                    rel * 0.97,
                    "Past picks leaned bolder than average for you — {tail}",
                ),
            )
            k, sc, pat = tol[_stable_index(f"{seed}:risk_tol", len(tol))]
            out.append((k, sc, pat.format(tail=tail)))
        else:
            if fit >= 0.68:
                plain = (
                    "From older saves, you {risk}. {tail}",
                    "Past answers: you {risk}. {tail}",
                )
                p = plain[_stable_index(f"{seed}:risk_plain", len(plain))]
                out.append(
                    (
                        "profile_risk_plain",
                        rel * 0.88,
                        p.format(risk=risk, tail=tail),
                    )
                )

    ds = next((t for t in profile.trait_estimates if t.name == "decision_speed"), None)
    if ds and ds.confidence >= 0.52 and ds.weighted_mean <= 0.38:
        if profile_decision_speed_snippet_allowed(primary_family, initial_norm, dims):
            speed_variants = (
                (
                    "profile_decision_speed_a",
                    fit * 0.9,
                    "Older answers show you like a pause before big calls — "
                    "that’s fine if nothing is actually on fire.",
                ),
                (
                    "profile_decision_speed_b",
                    fit * 0.9,
                    "Past picks suggest you step back before you lock something in — "
                    "fine unless something’s truly urgent.",
                ),
                (
                    "profile_decision_speed_c",
                    fit * 0.89,
                    "You’ve usually wanted time before a hard commit — "
                    "skip the wait only if the clock is real.",
                ),
            )
            k, sc, tx = speed_variants[_stable_index(f"{seed}:ds", len(speed_variants))]
            out.append((k, sc, tx))

    vt = profile.value_tag_weights[:6]
    if any(
        ("regret" in x[0].lower() or "security" in x[0].lower()) and x[1] >= 1.22
        for x in vt
    ):
        if profile_value_tag_snippet_allowed(primary_family) and fit >= 0.52:
            safe_v = (
                (
                    "profile_value_safe_a",
                    fit * 0.84,
                    "You’ve often picked the safer path — check what you’d lose by doing that again.",
                ),
                (
                    "profile_value_safe_b",
                    fit * 0.83,
                    "Past choices leaned careful — balance that with what you want out of this one.",
                ),
            )
            sk, sf, stx = safe_v[_stable_index(f"{seed}:pvs", len(safe_v))]
            out.append((sk, sf, stx))

    return out


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


def _tendency_line_candidates(
    tmap: Dict[str, float],
    *,
    primary_family: str,
    initial_norm: str,
    seed: str,
) -> List[Tuple[str, float, str]]:
    """Phase 34: scored tendency snippets — only when family + strength fit."""
    dims0 = score_dimensions(initial_norm)
    axes0 = score_ontology_axes(initial_norm)
    fit = profile_memory_fit_score(primary_family, initial_norm, dims0, axes0)
    out: List[Tuple[str, float, str]] = []

    g = float(tmap.get("tendency_guilt_about_no", 0.0))
    t_guilt = 0.5
    if g >= t_guilt:
        if _obligation_overlap_strong(initial_norm, dims0) and (
            primary_family in (OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY)
            or (
                primary_family == GENERAL
                and dims0.get("obligation", 0) + dims0.get("overload", 0) >= 1.35
            )
        ):
            guilt_v = (
                (
                    "tendency_guilt_no",
                    g * fit * 1.05,
                    "Saying no has nagged at you before — sort real fallout from plain guilt.",
                ),
                (
                    "tendency_guilt_no_b",
                    g * fit * 1.04,
                    "You’ve stressed over turning people down — ask if the cost is mostly in your head.",
                ),
            )
            gk, gs, gt = guilt_v[_stable_index(f"{seed}:tguilt", len(guilt_v))]
            out.append((gk, gs, gt))

    r = float(tmap.get("tendency_regret_if_yes", 0.0))
    t_regret = 0.5
    if r >= t_regret:
        if primary_family in (OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY):
            out.append(
                (
                    "tendency_regret_yes",
                    r * fit * 1.05,
                    "Yes-then-bitter has happened for you — if that taste is back, treat it as a warning light.",
                )
            )
        elif primary_family == CONFLICT_FAMILY and dims0.get("regret_risk", 0) >= 0.95:
            out.append(
                (
                    "tendency_regret_yes_cf",
                    r * fit * 1.02,
                    "You’ve swallowed it in tense moments and felt sour after — don’t ignore that if it’s here again.",
                )
            )
        elif _obligation_overlap_strong(initial_norm, dims0) and dims0.get("regret_risk", 0) >= 0.62:
            out.append(
                (
                    "tendency_regret_yes_ob",
                    r * fit * 0.95,
                    "Stacked yeses have left you bitter before — if you feel that setup forming, slow down.",
                )
            )

    e = float(tmap.get("tendency_low_energy_guard", 0.0))
    t_energy = 0.5
    if e >= t_energy:
        if primary_family == CONFLICT_FAMILY and dims0.get("overload", 0) < 0.85:
            pass
        elif primary_family == OBLIGATION_OVERLOAD or dims0.get("overload", 0) >= 0.95:
            low_e = (
                (
                    "tendency_low_energy",
                    e * fit * 1.02,
                    "You’ve been wiped in spots like this — another full yes often ages badly.",
                ),
                (
                    "tendency_low_energy_b",
                    e * fit * 1.01,
                    "Running on empty has shown up before — stacking a big yes on top rarely helps.",
                ),
            )
            ek, es, et = low_e[_stable_index(f"{seed}:tlowe", len(low_e))]
            out.append((ek, es, et))

    return out


def _pattern_repeat_line_keyed(
    counts: Dict[str, int], primary_family: str, seed: str
) -> Optional[Tuple[str, str]]:
    if primary_family != SPENDING:
        return None
    if counts.get("housing_bill_pressure=open", 0) >= 3:
        opts = (
            (
                "pattern_rent_stress",
                "Rent or bill stress has popped up in a few past clarifications — "
                "treat it like a pattern, not a single bad week.",
            ),
            (
                "pattern_rent_stress_b",
                "Housing or bill pressure keeps surfacing in your past clarifications — "
                "worth naming it as a repeat theme, not noise.",
            ),
        )
        return opts[_stable_index(f"{seed}:patrent", len(opts))]
    return None


def _append_surface_line(
    bodies: List[str],
    line_key: str,
    text: str,
    surface_store,
) -> None:
    if not text.strip():
        return
    canon = _canonical_memory_surface_key(line_key)
    if surface_store is None:
        bodies.append(text)
        return
    should_fn = getattr(surface_store, "should_surface_memory_line", None)
    if callable(should_fn) and not should_fn(canon):
        return
    bodies.append(text)
    rec = getattr(surface_store, "record_memory_line_surface", None)
    if callable(rec):
        try:
            rec(canon)
        except Exception:
            pass


def build_routed_decision_guidance(
    *,
    original_question: str,
    qa_pairs: Sequence[Tuple[str, str]],
    domain_order: Sequence[str],
    profile: Optional[PersonalProfile],
    tendency_map: Optional[Dict[str, float]] = None,
    situation_repeat_counts: Optional[Dict[str, int]] = None,
    surface_store: Optional[Any] = None,
    interview_memory_candidates: Sequence[Tuple[str, float, str]] = (),
) -> str:
    parts_ctx = [original_question] + [a for _, a in qa_pairs]
    ctx = _padded_ctx(parts_ctx)
    merged_norm = normalize_input(" ".join(parts_ctx))
    dims_order = score_dimensions(merged_norm)
    order_list = sanitize_domain_order_for_obligation(
        merged_norm, dims_order, list(domain_order)
    )
    primary = order_list[0] if order_list else GENERAL
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
            v = (
                (
                    "If rent or core bills are still open, a big buy hits the sorest spot first. "
                    "That doesn’t mean never — it means get basics steadier first, unless this buy is how you keep income or pass something you can’t move."
                ),
                (
                    "When rent or core bills are still open, a big purchase lands on the tenderest spot. "
                    "Steady the basics first unless this buy protects income or clears a hard blocker you can’t route around."
                ),
                (
                    "Open rent or core bills mean a big spend stings where you’re already thin. "
                    "Stabilize the must-pays first unless this purchase is what keeps you earning or removes a wall you can’t walk around."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:sp_unpaid", len(v))])
        elif " need " in ctx or "need for" in ctx or "mostly need" in ctx or ("school" in ctx and "want" not in ctx):
            v = (
                (
                    "If it’s a real need for work or school, treat it like gear: what’s the cheapest setup that still works, what’s one step up, and is the extra cash worth it."
                ),
                (
                    "If it’s a real need for work or school, think like tools: what’s the minimum that works, what’s one upgrade, and is the extra money worth it."
                ),
                (
                    "Real work-or-school need? Line up three prices: bare minimum, solid, and nice — then ask if the jump past solid is worth it."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:sp_need", len(v))])
        elif " want " in ctx or "mostly want" in ctx or "luxury" in ctx:
            v = (
                (
                    "If it’s mostly a want while cash is tight, waiting isn’t weak — it’s space to choose without boxing yourself in. Pick a date to check again."
                ),
                (
                    "If it’s mostly a want while money is tight, waiting buys room to choose without trapping yourself. Set a date to revisit it."
                ),
                (
                    "Mostly a want and funds are tight? A pause is just breathing room — put a calendar note on it so it doesn’t drift forever."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:sp_want", len(v))])
        else:
            v = (
                (
                    "Money stuff is easier when food, rent, and getting around are honest first. If those wobble, trim the spend or wait until one layer feels steadier."
                ),
                (
                    "Money feels clearer when food, rent, and transport are sorted first. If those wobble, trim the spend or wait until one layer feels steadier."
                ),
                (
                    "Start with food, shelter, and how you get places — when those feel shaky, big extras usually wait unless they fix one of those three."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:sp_def", len(v))])
    elif primary == OBLIGATION_OVERLOAD:
        if "low" in ctx and ("energy" in ctx or "exhaust" in ctx):
            v = (
                (
                    "If you’re out of gas, the kind move is a smaller yes, a later yes, or a short honest no — not a hero yes you’ll hate later."
                ),
                (
                    "If you’re out of gas, the kind move is a smaller yes, a later yes, or a straight no — not a hero yes you’ll resent."
                ),
                (
                    "Running on empty? Offer what you can actually give — a partial help, a later slot, or a clean no — instead of a full yes you’ll choke on."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:ob_low", len(v))])
        else:
            v = (
                (
                    "Helping people works better when you know your real line before you answer. A plain ‘here’s what I can do’ beats a full yes you’ll resent."
                ),
                (
                    "Helping lands cleaner when you know your line before you answer. A simple ‘here’s what I can do’ beats a full yes you’ll resent."
                ),
                (
                    "Figure out your real limit before you pick up the phone. A clear partial yes beats a whole yes you’ll want to undo."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:ob_def", len(v))])
    elif primary == CONFLICT_FAMILY:
        if "peace" in ctx or "quiet" in ctx or "avoid" in ctx:
            v = (
                (
                    "If calm matters most, small steady limits usually beat one huge blow-up. You can stay decent and still say what you won’t take."
                ),
                (
                    "If keeping the peace matters most, small steady limits usually beat one huge blow-up. You can stay decent and still say what you won’t take."
                ),
                (
                    "If you want calm more than drama, repeat small boundaries instead of saving it all for one blast — you can be kind and still draw a line."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:cf_peace", len(v))])
        elif "say" in ctx or "clear" in ctx or "honest" in ctx:
            v = (
                (
                    "If something needs saying, one clear point and one example beats a long speech. Say what you want next time, not your whole life story."
                ),
                (
                    "If something needs saying, one clear point and one example beats a long speech. Name what you want next time, not your whole history."
                ),
                (
                    "When you need words out, lead with one fact and one ask — short and concrete beats a long vent."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:cf_say", len(v))])
        else:
            v = (
                (
                    "This is mostly about what you can live with afterward. Decide if you want things fixed, some distance, or just straight talk — those take different moves."
                ),
                (
                    "This is mostly about what you can live with afterward. Decide if you want repair, distance, or straight talk — those take different moves."
                ),
                (
                    "Ask what you want on the other side: patch it up, back away, or say it plain. Each goal needs a different playbook."
                ),
            )
            bodies.append(v[_stable_index(f"{phrase_seed}:cf_def", len(v))])
    elif primary == RISK_TIMING:
        v = (
            (
                "Split real deadlines from nerves. If waiting doesn’t break anything, use the pause to grab one missing fact. If there’s a real cutoff, count backward from it."
            ),
            (
                "Separate ‘must decide by’ from ‘I’m antsy.’ If the clock is soft, use the time to fetch one fact you’re missing. If the clock is hard, work backward from the date."
            ),
            (
                "Name what actually forces the timing. If nothing real breaks when you wait, slow down and fill one gap. If something real breaks, plan backward from that point."
            ),
        )
        bodies.append(v[_stable_index(f"{phrase_seed}:rt_main", len(v))])
        if "low" in ctx and "revers" in ctx:
            v2 = (
                (
                    "Hard-to-undo choices deserve a slower yes; easy-to-undo ones can be small tries."
                ),
                (
                    "Hard-to-undo choices deserve a slower yes; easy-to-undo ones can be small experiments."
                ),
                (
                    "If backing out is a mess, drag your feet on the commit. If you can unwind it cheaply, a small test run is fair."
                ),
            )
            bodies.append(v2[_stable_index(f"{phrase_seed}:rt_rev", len(v2))])
    elif primary == LOYALTY_BOUNDARY:
        v = (
            (
                "Being loyal doesn’t have to mean wiping yourself out. If yes costs sleep, money, or self-respect every time, the habit is the issue — not only this one ask."
            ),
            (
                "Being loyal doesn’t have to mean wiping yourself out. If yes costs sleep, money, or self-respect every time, the pattern is the issue — not only this one ask."
            ),
            (
                "Loyalty isn’t self-destruction on repeat. If every yes steals rest, cash, or dignity, fix the pattern — not just this single request."
            ),
        )
        bodies.append(v[_stable_index(f"{phrase_seed}:lb", len(v))])
    elif primary == CONVENIENCE_QUALITY:
        v = (
            (
                "When fast fights ‘do it right,’ try the smallest step that still shows if the careful path is worth it — don’t let rush lock you into fix-it-twice work."
            ),
            (
                "When fast fights ‘do it right,’ try the smallest step that still shows if the careful path is worth it — don’t let hurry lock you into doing it twice."
            ),
            (
                "Speed vs quality? Run a tiny slice the careful way once — enough to see if the slow path pays off — so you don’t redo the whole job."
            ),
        )
        bodies.append(v[_stable_index(f"{phrase_seed}:cq", len(v))])
    else:
        v = (
            (
                "Pick what you’d stand by with a friend who’s on your side — not the story that only sounds good when you’re tired. If both choices hurt, guard what’s costly to undo."
            ),
            (
                "Pick what you’d stand by with a friend who’s on your side — not the story that only sounds good when you’re tired. If both choices hurt, protect what’s hard to take back."
            ),
            (
                "Choose what you could explain to someone who wants you okay — not the version that only works at 2 a.m. If every option stings, keep what’s hardest to reverse."
            ),
        )
        bodies.append(v[_stable_index(f"{phrase_seed}:gen", len(v))])

    initial_norm = normalize_input(original_question)
    pattern_prefix: List[str] = []
    pattern_keyed = (
        _pattern_repeat_line_keyed(counts, primary, phrase_seed)
        if _current_money_context_strong(initial_norm)
        else None
    )
    if pattern_keyed:
        _append_surface_line(
            pattern_prefix, pattern_keyed[0], pattern_keyed[1], surface_store
        )

    tend_cands = _tendency_line_candidates(
        tendency_map,
        primary_family=primary,
        initial_norm=initial_norm,
        seed=phrase_seed,
    )
    prof_cands = _profile_line_candidates(
        profile,
        primary_family=primary,
        initial_norm=initial_norm,
        seed=phrase_seed,
    )
    memory_tail: List[str] = []
    for mk, mtxt in _pick_merged_memory_lines(
        tend_cands,
        prof_cands,
        seed=phrase_seed,
        min_tend=0.28,
        min_prof=0.55,
        interview_cands=interview_memory_candidates,
        min_interview=0.52,
    ):
        _append_surface_line(memory_tail, mk, mtxt, surface_store)

    merged = pattern_prefix + bodies + memory_tail
    return "\n\n".join(merged)


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

    order = sanitize_domain_order_for_obligation(norm, dimensions, order)

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

    parts_for_merge = [initial_text] + [a for _, a in qa_pairs]
    merged_norm = normalize_input(" ".join(parts_for_merge))
    phrase_seed = hashlib.sha256(
        normalize_input(initial_text).encode("utf-8")
    ).hexdigest()[:24]
    interview_cands: List[Tuple[str, float, str]] = []
    try:
        drows = db_store.get_recent_decision_memory(limit=80)
        primary_g = order[0] if order else GENERAL
        interview_cands = build_ask_interview_memory_line_candidates(
            drows,
            merged_norm=merged_norm,
            initial_norm=norm,
            primary_family=primary_g,
            seed=phrase_seed,
        )
    except Exception:
        interview_cands = []

    return build_routed_decision_guidance(
        original_question=initial_text,
        qa_pairs=qa_pairs,
        domain_order=order,
        profile=profile,
        tendency_map=tendency_map,
        situation_repeat_counts=repeat_counts,
        surface_store=db_store,
        interview_memory_candidates=interview_cands,
    )


LEGACY_GENERIC_PHRASES = (
    "what are the available options or approaches",
    "this framework will help us make a more informed choice",
)
