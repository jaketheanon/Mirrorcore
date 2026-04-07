"""
Unified personal response layer (Phase 29).

Retrieves relevant decision/style memory with deterministic ranking and
composes a grounded, template-guided likely-you response with explicit
confidence and memory basis.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..db.store import DatabaseStore
from ..decision.cross_system_knowledge import (
    clarification_cross_evidence_boost,
    effective_primary_for_cross_filter,
    filter_relevant_situation_facts,
)
from ..decision.memory_relevance import (
    _spending_pressure_prompt,
    conflict_situational_cues,
    decision_row_text_blob,
    decision_memory_relevance_multiplier,
    gossip_or_backchannel_user_prompt,
    personal_response_decision_families_aligned,
    profile_memory_fit_score,
    public_audience_disrespect_prompt,
    respond_main_decision_passes_shape_gate,
    style_memory_passes_respond_conflict_shape,
    style_memory_relevance_multiplier,
    style_row_text_blob,
)
from ..decision.ontology import (
    CONFLICT_FAMILY,
    GENERAL,
    OBLIGATION_OVERLOAD,
    SPENDING,
    interpersonal_conflict_markers_present,
    rank_families_full,
    score_dimensions,
)
from ..decision.routed_clarification import rank_families
from ..decision.situation_carryover import (
    PHASE46_REPLACEMENT_FEEDBACK_MERGE_STRENGTH_MIN,
    boundary_carryover_aligned,
    carryover_safe_stance_fallback,
    carryover_shape_key,
    carryover_slots_prefix,
    combined_shape_key,
    conflict_escalation_carryover_thread_ok,
    diagnose_ask_carryover_candidates,
    feedback_replacement_same_thread_gate,
    pa_carryover_aligned,
    persistence_after_declined_shaped,
    reference_continuation_cues,
    respond_route_keys_skip_short_term_situation_record,
    same_person_conflict_thread_carryover_aligned,
    short_term_row_recent_enough,
    still_persisting_wording,
)
from ..router import normalize_input
from .profile import PersonalProfile, build_personal_profile_from_rows

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Phase 44: ask pick threshold is 0.25; respond *influence* uses stricter floors below.
RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN = 0.28
RESPOND_CARRYOVER_NOTE_HIGH_MIN = 0.58
RESPOND_CARRYOVER_NOTE_SOFT_MIN = 0.50
# Phase 46: deterministic lexical nudge when adopting carryover-thread replacement wording
RESPOND_PHASE46_REPLACEMENT_PREF_TOKEN_SEED = 0.18
# Phase 44/45: GENERAL routing escape when conflict carryover is very strong + continuation cues.
RESPOND_CONFLICT_GENERAL_ESCAPE_CARRY_STRENGTH_MIN = 0.45
# Phase 45: gentle retrieval tilt toward direct/boundary-shaped saves (deterministic cap).
RESPOND_ESCALATION_DIRECT_RETRIEVAL_MULT = 1.09
# Same-thread boundary persistence can route obligation_overload; include it narrowly for Phase 45.
_PHASE45_ESCALATION_EFFECTIVE_FAMILIES = frozenset({CONFLICT_FAMILY, OBLIGATION_OVERLOAD})


def _respond_path_multipliers(
    path: "RespondEvidencePath", mmap: Mapping[str, float]
) -> List[float]:
    """Collect multipliers for evidence keys on this answer path (Phase 38)."""
    out: List[float] = []
    for did in path.decision_ids:
        out.append(float(mmap.get(f"decision:{did}", 1.0)))
    for sid in path.style_ids:
        out.append(float(mmap.get(f"style:{sid}", 1.0)))
    for rk in path.route_keys:
        out.append(float(mmap.get(f"route:{rk}", 1.0)))
    for ck in path.clarif_slot_keys:
        out.append(float(mmap.get(f"clarif_slot:{ck}", 1.0)))
    return out


def _stable_index(key: str, modulo: int) -> int:
    """Deterministic index in ``0..modulo-1`` (no salted ``hash()``)."""
    if modulo <= 1:
        return 0
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(h[:12], 16) % modulo


def tokenize_prompt(text: str) -> List[str]:
    if not text:
        return []
    seen = set()
    out: List[str] = []
    for m in _TOKEN_RE.findall(text.lower()):
        if len(m) > 2 and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def classify_answer_focus(prompt_norm: str) -> str:
    """Detect whether the user is mainly asking for action, wording, or both (Phase 39)."""
    pn = (prompt_norm or "").strip().lower()
    if not pn:
        return "both"

    # Explicit combined asks ("what would I do and say") must stay both — otherwise
    # action phrase hooks alone win and we drop the wording half (Phase 39.1).
    combined_asks = (
        "do and say",
        "say and do",
        "to do and say",
        "what would i do and say",
        "what should i do and say",
        "what would you do and say",
        "what should you do and say",
        "would i do and say",
        "should i do and say",
    )
    for h in combined_asks:
        if h in pn:
            return "both"

    w = 0.0
    a = 0.0

    wording_phrases = (
        "what would i say",
        "what would you say",
        "how would i say",
        "how would you say",
        "how should i say",
        "how would i phrase",
        "what should i say",
        "what to say",
        "what do i say",
        "say to them",
        "say to him",
        "say to her",
        "tell them",
        "tell her",
        "tell him",
        "tell my ",
        "tell your ",
        "reply with",
        "respond with",
        "probably say",
        "would i say",
        "would i tell",
        "wording",
        "phrasing",
        "out loud",
        "talking shit",
        " if i said",
        " i say if",
    )
    for h in wording_phrases:
        if h in pn:
            w += 1.15

    action_phrases = (
        "what would i do",
        "what should i do",
        "what would you do",
        "would i probably do",
        "probably do",
        "my move",
        "what do i do",
        "what to do",
    )
    for h in action_phrases:
        if h in pn:
            a += 1.05
    if "should i say" in pn or "should i tell" in pn:
        w += 1.1
    elif "should i " in pn:
        a += 0.95

    wording_tokens = {
        "say",
        "tell",
        "phrase",
        "reply",
        "wording",
        "respond",
        "script",
    }
    action_tokens = {
        "do",
        "handle",
        "decide",
        "approach",
        "act",
        "move",
        "choose",
    }
    for tok in tokenize_prompt(pn):
        if tok in wording_tokens:
            w += 0.38
        if tok in action_tokens:
            a += 0.38

    if w >= 0.85 and a >= 0.85:
        return "both"
    if w > a + 0.45:
        return "wording"
    if a > w + 0.45:
        return "action"
    if max(w, a) < 0.35:
        return "both"
    if abs(w - a) < 0.4:
        return "both"
    return "wording" if w > a else "action"


def _correction_retrieval_factor(status: str) -> float:
    return {
        "accurate": 1.22,
        "uncorrected": 1.0,
        "partially_true": 0.78,
        "not_really": 0.12,
    }.get((status or "").strip(), 0.85)


def _count_occurrences(keys: Sequence[str]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for k in keys:
        ks = (k or "").strip()
        if not ks:
            continue
        c[ks] = c.get(ks, 0) + 1
    return c


def score_decision_memory_row(
    row: Dict[str, Any],
    keywords: Sequence[str],
    scenario_counts: Mapping[str, int],
    *,
    evidence_row_mult: float = 1.0,
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if not keywords:
        overlap = 0
    else:
        blob = " ".join(
            [
                str(row.get("scenario_text") or ""),
                str(row.get("choice_label") or ""),
                str(row.get("reasoning_label") or ""),
                " ".join(row.get("value_tags") or []),
                " ".join((row.get("trait_signals") or {}).keys()),
            ]
        ).lower()
        overlap = sum(1 for kw in keywords if kw in blob)
    tag_hits = sum(
        1 for t in row.get("value_tags") or [] if str(t).lower() in keywords
    )
    trait_hits = sum(
        1 for k in (row.get("trait_signals") or {}) if str(k).lower() in keywords
    )
    score = overlap * 1.15 + tag_hits * 1.35 + trait_hits * 1.0
    sid = row.get("scenario_id") or ""
    rep = max(0, scenario_counts.get(sid, 0) - 1)
    if rep:
        score += 0.45 * rep
        if rep >= 1:
            reasons.append("repeated scenario pattern")
    score *= _correction_retrieval_factor(row.get("correction_status"))
    entry_conf = float(row.get("confidence_score") or 0.75)
    entry_conf = max(0.2, min(1.0, entry_conf))
    score *= 0.75 + 0.25 * entry_conf

    if overlap:
        reasons.append("word overlap with what you typed")
    if tag_hits:
        reasons.append("matching value tags")
    if trait_hits:
        reasons.append("matching trait names")
    st = row.get("correction_status")
    if st == "accurate":
        reasons.append("you confirmed this batch as accurate")

    em = float(evidence_row_mult or 1.0)
    score *= max(0.15, min(1.25, em))

    return score, reasons


def score_style_memory_row(
    row: Dict[str, Any],
    keywords: Sequence[str],
    prompt_counts: Mapping[str, int],
    *,
    evidence_row_mult: float = 1.0,
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if not keywords:
        overlap = 0
    else:
        blob = " ".join(
            [
                str(row.get("prompt_text") or ""),
                str(row.get("selected_label") or ""),
                " ".join(row.get("style_tags") or []),
                " ".join((row.get("tone_signals") or {}).keys()),
            ]
        ).lower()
        overlap = sum(1 for kw in keywords if kw in blob)
    tag_hits = sum(
        1 for t in row.get("style_tags") or [] if str(t).lower() in keywords
    )
    tone_hits = sum(
        1 for k in (row.get("tone_signals") or {}) if str(k).lower() in keywords
    )
    score = overlap * 1.05 + tag_hits * 1.25 + tone_hits * 0.95
    pid = row.get("prompt_id") or ""
    rep = max(0, prompt_counts.get(pid, 0) - 1)
    if rep:
        score += 0.4 * rep
    score *= _correction_retrieval_factor(row.get("correction_status"))
    entry_conf = float(row.get("confidence_score") or 0.75)
    entry_conf = max(0.2, min(1.0, entry_conf))
    score *= 0.75 + 0.25 * entry_conf

    if overlap:
        reasons.append("word overlap with style prompts")
    if tag_hits:
        reasons.append("matching style tags")
    if tone_hits:
        reasons.append("matching tone dimensions")
    st = row.get("correction_status")
    if st == "accurate":
        reasons.append("confirmed style calibration")

    em = float(evidence_row_mult or 1.0)
    score *= max(0.15, min(1.25, em))

    return score, reasons


@dataclass(frozen=True)
class RespondFeedbackInfluence:
    """Aggregated same-prompt feedback for ranking (Phase 40)."""

    pref_token_weight: Mapping[str, float]
    avoidance_demote: float
    direct_calm_signal: float
    wrong_replacement_count: int = 0
    replacement_inject_line: str = ""
    # Phase 43: same-prompt replacements disagree; damp confidence / overlays.
    replacement_direction_mixed: float = 0.0
    # Phase 45: partly + action ok / wording off — do not anchor carryover on surface line.
    action_ok_wording_off: bool = False
    # Newest-first replacement from partly + action_ok_word_bad (not the wrong-only inject path).
    action_ok_wording_replacement_line: str = ""
    # Phase 46: action-ok wording line was merged from carryover row prompt hash (narrow gate).
    phase46_merged_carryover_feedback: bool = False


@dataclass(frozen=True)
class RespondExampleInfluence:
    """Promoted reusable examples derived from repeated corrections (Phase 42)."""

    token_weight: Mapping[str, float]
    action_line: str = ""
    wording_line: str = ""
    both_line: str = ""
    strongest_strength: float = 0.0
    winning_effective_strength: float = 0.0
    contradiction_level: float = 0.0
    consistency: float = 1.0
    winning_gap: float = 0.0
    uses_fallback_only: bool = False


_FEEDBACK_TOK_STOP = frozenset(
    {
        "the",
        "and",
        "but",
        "for",
        "you",
        "your",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "had",
        "would",
        "could",
        "should",
        "what",
        "how",
        "say",
        "out",
        "loud",
        "just",
        "like",
        "than",
        "then",
        "them",
        "they",
        "very",
        "also",
        "into",
    }
)


def _feedback_family_compatible(stored: str, current: str) -> bool:
    a = (stored or "").strip().lower()
    b = (current or "").strip().lower()
    if not a or not b:
        return True
    if a == b:
        return True
    if "conflict" in a and "conflict" in b:
        return True
    if a == "general" or b == "general":
        return True
    return False


def _tokenize_feedback_phrase(text: str) -> List[str]:
    out: List[str] = []
    for m in _TOKEN_RE.findall((text or "").lower()):
        if len(m) > 2 and m not in _FEEDBACK_TOK_STOP:
            out.append(m)
    return out


def _feedback_snippet_signals_avoidance(snippet: str) -> bool:
    s = (snippet or "").lower()
    needles = (
        "let it go",
        "let it slide",
        "move on",
        "ignore it",
        "just ignore",
        "not worth the fight",
        "drop it",
        "let it ride",
    )
    return any(n in s for n in needles)


def _memory_blob_avoidance_hit(blob: str) -> bool:
    b = (blob or "").lower()
    needles = (
        "let it go",
        "let it slide",
        "move on",
        "ignore it",
        "just ignore",
        "not worth",
        "don't engage",
        "dont engage",
        "walk away",
        "rise above",
        "let it ride",
    )
    return any(n in b for n in needles)


def _stance_favors_engagement_over_avoidance(snippet: str) -> bool:
    """Recent stance line leans direct/calm vs passive-avoid (Phase 44 respond)."""
    low = (snippet or "").lower()
    avoid = (
        "let it go",
        "let it slide",
        "move on",
        "ignore it",
        "just ignore",
        "drop it",
        "let it ride",
        "walk away",
        "rise above",
    )
    if any(a in low for a in avoid):
        return False
    direct = (
        "direct",
        "address",
        "name ",
        "plain",
        "boundary",
        "straight",
        "calm",
        "clear",
        "conversation",
        "honest",
        "steady",
        "spoken",
    )
    return any(d in low for d in direct)


def _respond_situation_carryover_effective_family_aligned(
    eff_pf: str,
    situation_carryover: Optional[Dict[str, Any]],
    *,
    prompt_norm: str,
    carry_strength: float,
) -> Optional[Dict[str, Any]]:
    """
    Respond-like-me: only reuse short-term rows that match the routed decision family.

    Prevents a conflict-thread carryover from affecting spending-shaped answers (and
    the continuity note) while allowing a narrow escape when routing stays GENERAL
    but continuation phrasing + strong conflict carry clearly match.
    """
    if not situation_carryover or not situation_carryover.get("match"):
        return None
    cfam = str(situation_carryover.get("carry_family") or "").strip().lower()
    if not cfam:
        cfam = (
            str(situation_carryover["match"].get("effective_family") or "")
            .strip()
            .lower()
        )
    eff = str(eff_pf or "general").strip().lower()
    if cfam == eff:
        return situation_carryover
    if (
        eff == GENERAL
        and cfam == CONFLICT_FAMILY
        and carry_strength >= float(RESPOND_CONFLICT_GENERAL_ESCAPE_CARRY_STRENGTH_MIN)
        and reference_continuation_cues(prompt_norm)
    ):
        return situation_carryover
    return None


def _respond_carryover_reasoning_line_audit(
    situation_carryover: Optional[Dict[str, Any]],
    prompt_norm: str,
    carry_strength: float,
    prompt_norm_hash: str,
    eff_pf: str,
) -> Dict[str, Any]:
    """Structured gate trace for Phase 44 continuity wording (deterministic)."""
    soft_min = float(RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN)
    audit: Dict[str, Any] = {
        "allowed": False,
        "blocked_by": "",
        "carry_strength": round(float(carry_strength), 4),
        "same_prompt_hash": False,
        "shape_core_match": False,
        "explicit_continuation_cues": False,
        "pa_carryover_aligned": False,
        "conflict_pa_soft_continuity_path": False,
        "cur_core": "",
        "row_core": "",
        "eff_family": str(eff_pf or "general").strip().lower(),
    }
    if not situation_carryover or not situation_carryover.get("match"):
        audit["blocked_by"] = "no_situation_carryover_match"
        return audit
    match = situation_carryover["match"]
    row_prompt_norm = str(match.get("prompt_norm") or "")
    explicit = reference_continuation_cues(prompt_norm)
    pa_row_align = pa_carryover_aligned(prompt_norm, row_prompt_norm)
    audit["explicit_continuation_cues"] = explicit
    audit["pa_carryover_aligned"] = pa_row_align
    if str(match.get("state") or "").strip().lower() != "unresolved":
        audit["blocked_by"] = "match_state_not_unresolved"
        return audit
    if not short_term_row_recent_enough(match):
        audit["blocked_by"] = "match_not_recent_enough_42h"
        return audit

    eff = audit["eff_family"]
    row_hash = str(match.get("prompt_norm_hash") or "").strip()
    row_shape = str(match.get("shape_key") or "")
    cur_core = carryover_slots_prefix(
        carryover_shape_key(prompt_norm, str(eff_pf or "general"))
    )
    row_core = carryover_slots_prefix(row_shape)
    shape_core_match = bool(cur_core) and cur_core == row_core
    same_prompt = bool(row_hash and prompt_norm_hash and row_hash == prompt_norm_hash)
    audit["same_prompt_hash"] = same_prompt
    audit["shape_core_match"] = shape_core_match
    audit["cur_core"] = cur_core
    audit["row_core"] = row_core

    if eff == SPENDING:
        if carry_strength < 0.50:
            audit["blocked_by"] = "spending_carry_strength_below_0_50"
            return audit
        if not same_prompt:
            audit["blocked_by"] = "spending_requires_same_prompt_hash"
            return audit
        audit["allowed"] = True
        audit["blocked_by"] = ""
        return audit

    if carry_strength < soft_min:
        audit["blocked_by"] = "carry_strength_below_influence_soft_min"
        return audit

    if same_prompt:
        audit["allowed"] = True
        audit["blocked_by"] = ""
        return audit

    if carry_strength >= 0.50:
        if carry_strength >= 0.62 and shape_core_match:
            audit["allowed"] = True
            audit["blocked_by"] = ""
            return audit
        if explicit and carry_strength >= 0.56 and shape_core_match:
            audit["allowed"] = True
            audit["blocked_by"] = ""
            return audit
        if explicit and carry_strength >= 0.58:
            audit["allowed"] = True
            audit["blocked_by"] = ""
            return audit
        audit["blocked_by"] = "continuity_branch_gates_failed"
        return audit

    # Narrow soft band [soft_min, 0.50): conflict + explicit continuation + PA thread
    # alignment only (real CLI scores ~0.33–0.36; spending/general leaks stay out).
    if eff != CONFLICT_FAMILY:
        audit["blocked_by"] = "soft_band_requires_conflict_family"
        return audit
    if not explicit:
        audit["blocked_by"] = "soft_band_requires_explicit_continuation_cues"
        return audit
    if not pa_row_align:
        audit["blocked_by"] = "soft_band_requires_pa_carryover_aligned"
        return audit
    audit["allowed"] = True
    audit["blocked_by"] = ""
    audit["conflict_pa_soft_continuity_path"] = True
    return audit


def _respond_carryover_reasoning_line_allowed(
    situation_carryover: Optional[Dict[str, Any]],
    prompt_norm: str,
    carry_strength: float,
    prompt_norm_hash: str,
    eff_pf: str,
) -> bool:
    """Continuity in Why (brief) only for a fresh, unresolved, well-matched short-term row."""
    return bool(
        _respond_carryover_reasoning_line_audit(
            situation_carryover,
            prompt_norm,
            carry_strength,
            prompt_norm_hash,
            eff_pf,
        )["allowed"]
    )


def _respond_carryover_suppress_avoidance(
    carry_payload: Optional[Dict[str, Any]],
    prompt_norm: str,
) -> bool:
    """
    Strong unresolved conflict carryover + passive-aggressive continuation:
    demote avoidance-shaped decision/style saves so recent engaged stance wins.
    """
    if not carry_payload or not carry_payload.get("match"):
        return False
    strength = float(carry_payload.get("strength") or 0.0)
    stance = str(carry_payload["match"].get("stance_snippet") or "")
    row_pn = str(carry_payload["match"].get("prompt_norm") or "")
    pa_align = pa_carryover_aligned(prompt_norm, row_pn)
    soft_min = float(RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN)
    if _stance_favors_engagement_over_avoidance(stance) or pa_align:
        min_s = soft_min
    else:
        min_s = 0.41
    if strength < min_s:
        return False
    fam = str(carry_payload.get("carry_family") or "").strip().lower()
    if "conflict" not in fam:
        return False
    if not reference_continuation_cues(prompt_norm):
        return False
    cues = conflict_situational_cues(prompt_norm)
    boundary_suppress_path = bool(
        boundary_carryover_aligned(prompt_norm, row_pn)
        and (
            cues.get("boundary_push")
            or persistence_after_declined_shaped(prompt_norm)
        )
        and (
            cues.get("repeat_pattern")
            or still_persisting_wording(prompt_norm)
        )
    )
    if not (
        cues.get("passive_slight")
        or (cues.get("repeat_pattern") and reference_continuation_cues(prompt_norm))
        or boundary_suppress_path
    ):
        return False
    if _stance_favors_engagement_over_avoidance(stance):
        return True
    if pa_align:
        return strength >= soft_min
    if boundary_suppress_path:
        return strength >= soft_min
    return strength >= 0.52


def _decision_row_phase45_escalation_retrieval_multiplier(row: Dict[str, Any]) -> float:
    """Slight score tilt toward direct/boundary-shaped decision rows (Phase 45)."""
    blob = decision_row_text_blob(row)
    if _memory_blob_avoidance_hit(blob):
        return 1.0
    low = blob.lower()
    needles = (
        "boundary",
        "direct",
        "address",
        "name ",
        "plain",
        "calm",
        "conversation",
        "honest",
        "clear line",
        "one clear",
        "pattern",
        "noticed",
        "sideways",
    )
    if any(n in low for n in needles):
        return float(RESPOND_ESCALATION_DIRECT_RETRIEVAL_MULT)
    return 1.0


def _apply_phase45_escalation_retrieval_boost(
    ranked: List[Tuple[Dict[str, Any], float, List[str]]],
) -> List[Tuple[Dict[str, Any], float, List[str]]]:
    out: List[Tuple[Dict[str, Any], float, List[str]]] = []
    for row, sc, rs in ranked:
        mult = _decision_row_phase45_escalation_retrieval_multiplier(row)
        if mult > 1.0:
            rs2 = list(rs) + ["phase45_escalation_direct_boundary_boost"]
            out.append((row, float(sc) * mult, rs2))
        else:
            out.append((row, sc, rs))
    return out


def _evaluate_phase45_conflict_escalation(
    *,
    eff_pf: str,
    prompt_norm: str,
    prompt_norm_hash: str,
    situation_carryover: Optional[Dict[str, Any]],
    carry_strength: float,
) -> Dict[str, Any]:
    """
    Phase 45: deterministic escalation state for repeated same-thread conflict
    continuation (debug + retrieval + avoidance demotion).
    """
    eff = str(eff_pf or "").strip().lower()
    rejects: List[str] = []
    base: Dict[str, Any] = {
        "phase45_escalation_evaluated": False,
        "phase45_escalation_active": False,
        "phase45_escalation_subtype": "",
        "phase45_escalation_reject_reasons": rejects,
        "phase45_demote_avoidance_due_to_escalation": False,
        "phase45_boost_direct_boundary_retrieval": False,
    }
    if eff not in _PHASE45_ESCALATION_EFFECTIVE_FAMILIES:
        rejects.append("not_escalation_eligible_family")
        return base
    base["phase45_escalation_evaluated"] = True
    soft_floor = float(RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN)
    if not situation_carryover or not situation_carryover.get("match"):
        rejects.append("no_situation_carryover_match")
        return base
    match = situation_carryover["match"]
    strength = float(carry_strength or 0.0)
    if strength < soft_floor:
        rejects.append("carry_strength_below_escalation_floor")
    st = str(match.get("state") or "").strip().lower()
    if st != "unresolved":
        rejects.append("carryover_match_not_unresolved")
    if not short_term_row_recent_enough(match):
        rejects.append("carryover_match_not_recent_enough_42h")
    if not reference_continuation_cues(prompt_norm):
        rejects.append("no_explicit_continuation_cues")
    if rejects:
        return base
    row_pn = str(match.get("prompt_norm") or "")
    row_h = str(match.get("prompt_norm_hash") or "").strip()
    cur_h = str(prompt_norm_hash or "").strip()
    thread_ok = conflict_escalation_carryover_thread_ok(
        prompt_norm, row_pn, cur_h, row_h
    )
    if not thread_ok:
        rejects.append("escalation_thread_alignment_failed")
        return base
    cues = conflict_situational_cues(prompt_norm)
    subtype = ""
    if (
        pa_carryover_aligned(prompt_norm, row_pn)
        and cues.get("passive_slight")
        and cues.get("repeat_pattern")
    ):
        subtype = "passive_aggressive_repeat"
    elif boundary_carryover_aligned(prompt_norm, row_pn) and (
        cues.get("boundary_push") or persistence_after_declined_shaped(prompt_norm)
    ) and (cues.get("repeat_pattern") or still_persisting_wording(prompt_norm)):
        subtype = "boundary_push_repeat"
    elif same_person_conflict_thread_carryover_aligned(
        prompt_norm, row_pn
    ) and (cues.get("repeat_pattern") or still_persisting_wording(prompt_norm)):
        subtype = "interpersonal_persistence_repeat"
    else:
        rejects.append("no_escalation_subtype_matched")
        return base
    base["phase45_escalation_active"] = True
    base["phase45_escalation_subtype"] = subtype
    base["phase45_demote_avoidance_due_to_escalation"] = True
    base["phase45_boost_direct_boundary_retrieval"] = True
    base["phase45_escalation_reject_reasons"] = []
    return base


def _respond_repeated_passive_aggressive_escalation_active(
    *,
    eff_pf: str,
    prompt_norm: str,
    prompt_norm_hash: str,
    situation_carryover: Optional[Dict[str, Any]],
    carry_strength: float,
) -> bool:
    """
    Phase 44 flag: only the passive-aggressive + repeat subtype (debug field compat).
    """
    st = _evaluate_phase45_conflict_escalation(
        eff_pf=eff_pf,
        prompt_norm=prompt_norm,
        prompt_norm_hash=prompt_norm_hash,
        situation_carryover=situation_carryover,
        carry_strength=carry_strength,
    )
    return bool(
        st.get("phase45_escalation_active")
        and st.get("phase45_escalation_subtype") == "passive_aggressive_repeat"
    )


def _normalize_replacement_direction(text: str) -> str:
    t = " ".join((text or "").strip().lower().split())
    return t[:400]


def build_respond_feedback_influence(
    store: DatabaseStore,
    prompt_norm_hash: str,
    effective_family: str,
) -> RespondFeedbackInfluence:
    """Derive lexical preference + avoidance demotion from recent same-prompt feedback."""
    if not (prompt_norm_hash or "").strip():
        return RespondFeedbackInfluence({}, 0.0, 0.0, 0, "", 0.0)
    if not hasattr(store, "list_personal_response_feedback_for_prompt"):
        return RespondFeedbackInfluence({}, 0.0, 0.0, 0, "", 0.0)
    rows = store.list_personal_response_feedback_for_prompt(
        prompt_norm_hash, limit=40
    )
    action_ok_wording_off = any(
        _feedback_family_compatible(str(r.get("effective_family") or ""), effective_family)
        and (r.get("partial_aspect") or "").strip() == "action_ok_word_bad"
        for r in rows
    )
    action_ok_wording_replacement_line = ""
    for row in rows:
        if not _feedback_family_compatible(
            str(row.get("effective_family") or ""), effective_family
        ):
            continue
        if (row.get("partial_aspect") or "").strip() != "action_ok_word_bad":
            continue
        rep_w = (row.get("replacement_text") or "").strip()
        if len(rep_w) >= 8:
            action_ok_wording_replacement_line = rep_w[:400]
            break
    pref: Dict[str, float] = {}
    wrong_avoid = 0
    direct_hits = 0.0
    wrong_rep_count = 0
    inject_line = ""
    replacement_direction_mixed = 0.0

    # Phase 43: aggregate wrong+replacement by direction (support beats one-off recency).
    rep_counts: Dict[str, int] = {}
    # Newest-first list → first stored text per norm is the latest wording for that direction.
    rep_newest_text: Dict[str, str] = {}
    dominant_norm = ""

    for row in rows:
        if not _feedback_family_compatible(
            str(row.get("effective_family") or ""), effective_family
        ):
            continue
        rt = (row.get("rating") or "").strip().lower()
        rep_t = (row.get("replacement_text") or "").strip()
        snip = str(row.get("likely_answer_snippet") or "")
        if rt == "wrong" and rep_t:
            wrong_rep_count += 1
        if rt == "wrong" and rep_t:
            rn = _normalize_replacement_direction(rep_t)
            if len(rn) >= 8:
                rep_counts[rn] = rep_counts.get(rn, 0) + 1
                if rn not in rep_newest_text:
                    rep_newest_text[rn] = rep_t[:400]

    establish_index: Dict[str, int] = {}
    _ei = 0
    for row in reversed(rows):
        if not _feedback_family_compatible(
            str(row.get("effective_family") or ""), effective_family
        ):
            continue
        if (row.get("rating") or "").strip().lower() != "wrong":
            continue
        rep_t = (row.get("replacement_text") or "").strip()
        rn = _normalize_replacement_direction(rep_t)
        if len(rn) < 8:
            continue
        if rn not in establish_index:
            establish_index[rn] = _ei
            _ei += 1

    if rep_counts:
        sorted_norms = sorted(
            rep_counts.keys(),
            key=lambda k: (-rep_counts[k], establish_index.get(k, 999), k),
        )
        dominant_norm = sorted_norms[0]
        inject_line = (rep_newest_text.get(dominant_norm) or "").strip()
        dom_c = rep_counts[dominant_norm]
        second_c = rep_counts[sorted_norms[1]] if len(sorted_norms) > 1 else 0

        newest_contra_norm: Optional[str] = None
        for row in rows:
            if not _feedback_family_compatible(
                str(row.get("effective_family") or ""), effective_family
            ):
                continue
            if (row.get("rating") or "").strip().lower() != "wrong":
                continue
            rep_t = (row.get("replacement_text") or "").strip()
            if not rep_t:
                continue
            rn = _normalize_replacement_direction(rep_t)
            if len(rn) < 8:
                continue
            newest_contra_norm = rn
            break

        if newest_contra_norm is not None:
            n_new = rep_counts.get(newest_contra_norm, 0)
            if dom_c >= 2 and n_new == 1 and newest_contra_norm != dominant_norm:
                replacement_direction_mixed = max(replacement_direction_mixed, 0.48)
            elif dom_c >= 2 and second_c >= 2 and second_c >= dom_c - 1:
                replacement_direction_mixed = max(replacement_direction_mixed, 0.36)
    else:
        for row in rows:
            if not _feedback_family_compatible(
                str(row.get("effective_family") or ""), effective_family
            ):
                continue
            rt = (row.get("rating") or "").strip().lower()
            rep_t = (row.get("replacement_text") or "").strip()
            if rt == "wrong" and rep_t and not inject_line:
                inject_line = rep_t[:400]
                break

    for row in rows:
        if not _feedback_family_compatible(
            str(row.get("effective_family") or ""), effective_family
        ):
            continue
        rt = (row.get("rating") or "").strip().lower()
        rep_t = (row.get("replacement_text") or "").strip()
        snip = str(row.get("likely_answer_snippet") or "")
        rn = _normalize_replacement_direction(rep_t) if rep_t else ""
        dom_c = rep_counts.get(dominant_norm, 0) if dominant_norm else 0
        if rep_t and rt in ("wrong", "partly"):
            mult = 1.0
            if dominant_norm and rn and rn != dominant_norm:
                cn = rep_counts.get(rn, 0)
                if dom_c >= 2 and cn == 1:
                    mult = 0.20
                elif dom_c >= 2 and cn < dom_c:
                    mult = min(1.0, 0.38 + 0.14 * float(cn))
            add = (0.20 if rt == "wrong" else 0.10) * mult
            for tok in _tokenize_feedback_phrase(rep_t):
                pref[tok] = min(0.62, pref.get(tok, 0.0) + add)
            rl = rep_t.lower()
            if any(x in rl for x in ("address", "direct", "calm", "straightforward")):
                direct_hits = min(1.0, direct_hits + 0.32 * mult)
        if rt == "wrong" and _feedback_snippet_signals_avoidance(snip):
            wrong_avoid += 1
    if wrong_rep_count >= 2:
        bump = 0.05 * min(4, wrong_rep_count - 1)
        for k in list(pref.keys()):
            pref[k] = min(0.72, pref[k] + bump)
    demote = 0.0
    if wrong_avoid >= 1 and direct_hits >= 0.22:
        demote = min(1.0, 0.44 + 0.16 * max(0, wrong_avoid - 1))
    if wrong_avoid >= 2 and direct_hits > 0:
        demote = min(1.0, max(demote, 0.62))
    if wrong_rep_count >= 2 and direct_hits >= 0.22:
        demote = min(1.0, max(demote, 0.55 + 0.06 * min(3, wrong_rep_count - 2)))
    return RespondFeedbackInfluence(
        pref_token_weight=pref,
        avoidance_demote=demote,
        direct_calm_signal=direct_hits,
        wrong_replacement_count=wrong_rep_count,
        replacement_inject_line=inject_line.strip(),
        replacement_direction_mixed=replacement_direction_mixed,
        action_ok_wording_off=action_ok_wording_off,
        action_ok_wording_replacement_line=action_ok_wording_replacement_line.strip(),
    )


def _phase46_has_action_ok_replacement_line(infl: RespondFeedbackInfluence) -> bool:
    return bool(
        infl.action_ok_wording_off
        and len((infl.action_ok_wording_replacement_line or "").strip()) >= 8
    )


# Phrases from obligation stitched templates (see _obligation_boundary_blunt_reply_line) —
# if a carryover stance_snippet contains these, it is not a user-approved direct line.
_PHASE47_OBLIGATION_STITCHED_STANCE_MARKERS: Tuple[str, ...] = (
    "same way as before",
    "clear and plain",
    "same short no, not a new debate",
    "repeat the same words if they push",
    "like: \"say no",
    'like: "say no',
    "like: \"say",
    " — like:",
    ' — like: "',
    " - like: \"",
    ' - like: "',
    "mostly because i was already overloaded",
    "i'd say no the same way",
    "i'd hold the line on what i already said",
    "i'd keep my boundary simple",
)

# Prefixes aligned with carryover_safe_stance_fallback (narrow reject list).
_PHASE47_CARRYOVER_SAFE_FALLBACK_STANCE_PREFIXES: Tuple[str, ...] = (
    "same clear no as before",
    "same boundary as before",
    "same direct calm stance as before",
    "same stance as before",
    "same direction as before",
)


def _phase47_stance_snippet_has_stitched_obligation_markers(low: str) -> bool:
    return any(m in low for m in _PHASE47_OBLIGATION_STITCHED_STANCE_MARKERS)


def _phase47_stance_snippet_matches_safe_fallback_prefix(low: str) -> bool:
    return any(low.startswith(p) for p in _PHASE47_CARRYOVER_SAFE_FALLBACK_STANCE_PREFIXES)


def _phase47_stance_snippet_has_first_person_refusal_anchor(low: str) -> bool:
    """Narrow: direct surface line the user would say, not generic imperative coaching."""
    if low.startswith(("i ", "i'd ", "i've ", "i'm ")):
        return True
    if "i already" in low or "i told you" in low or "i've already" in low:
        return True
    if "told you no" in low:
        return True
    return False


def _phase47_obligation_persistence_stance_merge_prompt_shaped(prompt_norm: str) -> bool:
    """
    Phase 47 stance merge applies only to obligation / boundary-persistence threads,
    not generic interpersonal-conflict coaching (wrong+replacement inject still applies).
    """
    pn = normalize_input(prompt_norm or "")
    if persistence_after_declined_shaped(pn) or still_persisting_wording(pn):
        return True
    cues = conflict_situational_cues(pn)
    if cues.get("boundary_push"):
        return True
    return False


def carryover_row_stance_eligible_for_phase47_corrected_merge(
    snippet: str,
    *,
    prompt_norm: str,
) -> bool:
    """
    Phase 47: when a carryover row's prompt hash has no action_ok feedback rows, the
    short-term stance_snippet may still hold the user-approved replacement (STM rewrite).

    Only accept snippets that look like a direct spoken line, not stitched templates
    or deterministic safe fallbacks. Narrow to obligation/boundary-persistence prompts.
    """
    if not _phase47_obligation_persistence_stance_merge_prompt_shaped(prompt_norm):
        return False
    s = (snippet or "").strip()
    if len(s) < 12 or len(s) > 400:
        return False
    low = s.lower()
    if _phase47_stance_snippet_has_stitched_obligation_markers(low):
        return False
    if _phase47_stance_snippet_matches_safe_fallback_prefix(low):
        return False
    if not _phase47_stance_snippet_has_first_person_refusal_anchor(low):
        return False
    return True


def _phase46_resolved_replacement_line(infl: RespondFeedbackInfluence) -> str:
    if infl.action_ok_wording_off:
        w = (infl.action_ok_wording_replacement_line or "").strip()
        if len(w) >= 8:
            return w
    w2 = (infl.replacement_inject_line or "").strip()
    if len(w2) >= 8:
        return w2
    return ""


def phase46_merge_carryover_feedback_influence(
    store: DatabaseStore,
    base: RespondFeedbackInfluence,
    *,
    prompt_norm: str,
    prompt_norm_hash: str,
    eff_pf: str,
    situation_carryover: Optional[Dict[str, Any]],
    carry_strength: float,
) -> Tuple[RespondFeedbackInfluence, Dict[str, Any]]:
    """
    When the current prompt hash has no usable action-ok replacement line, pull
    wording-off replacement feedback from the active carryover row's hash only
    under Phase 46 same-thread gates (deterministic, narrow).
    """
    info: Dict[str, Any] = {
        "merge_attempted": False,
        "merge_applied": False,
        "merge_block_reason": "",
        "merge_source": "",
        "carryover_row_hash": "",
        "replacement_thread_gate_ok": False,
        "replacement_thread_gate_reason": "",
        "stance_snippet_carryover_merge": False,
        "corrected_stance_inheritance_reason": "",
    }
    if _phase46_has_action_ok_replacement_line(base):
        info["merge_block_reason"] = "current_prompt_has_action_ok_replacement"
        return base, info
    if not situation_carryover or not situation_carryover.get("match"):
        info["merge_block_reason"] = "no_situation_carryover_match"
        return base, info
    if carry_strength < float(PHASE46_REPLACEMENT_FEEDBACK_MERGE_STRENGTH_MIN):
        info["merge_block_reason"] = "carry_strength_below_phase46_merge_min"
        return base, info
    match = situation_carryover["match"]
    row_hash = str(match.get("prompt_norm_hash") or "").strip()
    info["carryover_row_hash"] = row_hash
    if not row_hash:
        info["merge_block_reason"] = "empty_carryover_row_hash"
        return base, info
    cur_h = (prompt_norm_hash or "").strip()
    if row_hash == cur_h:
        info["merge_block_reason"] = "same_prompt_hash_no_carryover_merge"
        return base, info
    info["merge_attempted"] = True
    row_norm = str(match.get("prompt_norm") or "")
    row_shape = str(match.get("shape_key") or "")
    cur_shape = carryover_shape_key(prompt_norm, str(eff_pf or "general"))
    ok, reason = feedback_replacement_same_thread_gate(
        prompt_norm,
        row_norm,
        cur_h,
        row_hash,
        cur_shape,
        row_shape,
        match,
    )
    info["replacement_thread_gate_ok"] = bool(ok)
    info["replacement_thread_gate_reason"] = str(reason or "")
    if not ok:
        info["merge_block_reason"] = reason
        return base, info
    alt = build_respond_feedback_influence(store, row_hash, str(eff_pf or "general"))
    rep = ""
    merge_source = ""
    if _phase46_has_action_ok_replacement_line(alt):
        rep = (alt.action_ok_wording_replacement_line or "").strip()
        merge_source = "carryover_prompt_action_ok_feedback"
        info["corrected_stance_inheritance_reason"] = (
            "carryover_row_prompt_hash_has_action_ok_wording_feedback"
        )
    else:
        stance_snip = str(match.get("stance_snippet") or "").strip()
        if carryover_row_stance_eligible_for_phase47_corrected_merge(
            stance_snip, prompt_norm=prompt_norm
        ):
            rep = stance_snip[:400]
            merge_source = "carryover_row_stored_stance_snippet_phase47"
            info["stance_snippet_carryover_merge"] = True
            info["corrected_stance_inheritance_reason"] = (
                "same_thread_gate_ok_carryover_row_stance_matches_phase47_direct_line_rules"
            )
        else:
            info["merge_block_reason"] = (
                "carryover_prompt_no_action_ok_wording_feedback_or_eligible_stance"
            )
            return base, info
    if len(rep) < 8:
        info["merge_block_reason"] = "carryover_replacement_text_too_short"
        return base, info
    info["merge_applied"] = True
    info["merge_source"] = merge_source
    new_pref = dict(base.pref_token_weight)
    seed = float(RESPOND_PHASE46_REPLACEMENT_PREF_TOKEN_SEED)
    for tok in _tokenize_feedback_phrase(rep):
        new_pref[tok] = min(0.62, new_pref.get(tok, 0.0) + seed)
    merged = RespondFeedbackInfluence(
        pref_token_weight=new_pref,
        avoidance_demote=max(base.avoidance_demote, alt.avoidance_demote),
        direct_calm_signal=max(base.direct_calm_signal, alt.direct_calm_signal),
        wrong_replacement_count=base.wrong_replacement_count,
        replacement_inject_line=base.replacement_inject_line,
        replacement_direction_mixed=max(
            base.replacement_direction_mixed,
            alt.replacement_direction_mixed,
        ),
        action_ok_wording_off=True,
        action_ok_wording_replacement_line=rep[:400],
        phase46_merged_carryover_feedback=True,
    )
    return merged, info


def _phase46_feedback_same_thread_relevant(
    feedback_influence: RespondFeedbackInfluence,
    merge_info: Dict[str, Any],
) -> bool:
    if merge_info.get("merge_applied"):
        return True
    if _phase46_has_action_ok_replacement_line(feedback_influence):
        return True
    wr = int(feedback_influence.wrong_replacement_count or 0)
    inj = (feedback_influence.replacement_inject_line or "").strip()
    if wr >= 1 and len(inj) >= 8:
        return True
    return False


def _apply_feedback_influence_to_score(
    s: float,
    *,
    blob: str,
    feedback_influence: Optional[RespondFeedbackInfluence],
) -> float:
    if not feedback_influence:
        return s
    out = float(s)
    blob_l = (blob or "").lower()
    wrn = int(feedback_influence.wrong_replacement_count or 0)
    tok_cap = 0.30 if wrn >= 2 else 0.24
    tok_mul = 0.36 if wrn >= 2 else 0.30
    pref_cap = 2.85 if wrn >= 2 else 2.45
    pref_factor = 1.0
    for tok, wt in feedback_influence.pref_token_weight.items():
        if len(tok) > 2 and tok in blob_l:
            pref_factor *= 1.0 + min(tok_cap, float(wt) * tok_mul)
    out *= min(pref_cap, pref_factor)
    if (
        feedback_influence.avoidance_demote > 0
        and _memory_blob_avoidance_hit(blob_l)
    ):
        dm = float(feedback_influence.avoidance_demote)
        pen = 0.72 if wrn >= 2 else 0.65
        out *= max(0.06, 1.0 - pen * dm)
    return out


def _build_example_influence(
    store: DatabaseStore,
    *,
    effective_family: str,
    answer_focus: str,
    route_hints: Sequence[str],
) -> RespondExampleInfluence:
    if not hasattr(store, "list_reusable_response_examples"):
        return RespondExampleInfluence({})
    rows = store.list_reusable_response_examples(
        effective_family=effective_family,
        route_keys=list(route_hints),
        answer_focus=answer_focus,
        limit=10,
        min_strength=0.2,
    )
    if not rows:
        return RespondExampleInfluence({})
    # Deterministic contradiction handling (Phase 43):
    # group reusable examples by normalized text and compare support-weighted strength.
    candidates: List[Tuple[str, str, float, float, int, float]] = []
    # (norm_text, example_type, effective_strength, contradiction_level, support_count, recency_weight)
    for r in rows:
        txt = str(r.get("example_text") or "").strip()
        if not txt:
            continue
        norm = " ".join(txt.lower().split())
        candidates.append(
            (
                norm,
                str(r.get("example_type") or "").strip().lower() or "both",
                float(r.get("effective_strength") or r.get("strength") or 0.0),
                max(0.0, min(1.0, float(r.get("contradiction_level") or 0.0))),
                max(0, int(r.get("support_count") or 0)),
                max(0.0, min(1.0, float(r.get("recency_weight") or 0.0))),
            )
        )
    if not candidates:
        return RespondExampleInfluence({})
    by_text: Dict[str, Dict[str, float]] = {}
    for norm, _, eff, contra, support, rec in candidates:
        slot = by_text.setdefault(
            norm,
            {"score": 0.0, "support": 0.0, "contra": 0.0, "recency": 0.0, "count": 0.0},
        )
        slot["score"] += max(0.0, eff)
        slot["support"] += float(support)
        slot["contra"] += float(contra)
        slot["recency"] += float(rec)
        slot["count"] += 1.0
    text_ranked = sorted(
        by_text.items(),
        key=lambda kv: (
            -float(kv[1]["score"]),
            -float(kv[1]["support"]),
            float(kv[1]["contra"]) / max(1.0, float(kv[1]["count"])),
            kv[0],
        ),
    )
    top_key, top_stat = text_ranked[0]
    runner_score = float(text_ranked[1][1]["score"]) if len(text_ranked) > 1 else 0.0
    total_score = sum(float(v["score"]) for _, v in text_ranked)
    dominance = float(top_stat["score"]) / max(0.0001, total_score)
    contradiction_level = 1.0 - max(0.0, min(1.0, dominance))
    gap = max(0.0, float(top_stat["score"]) - runner_score)
    top_sup = float(top_stat["support"])
    if len(text_ranked) > 1:
        r1_stat = text_ranked[1][1]
        runner_sup = float(r1_stat["support"])
        r_score = float(r1_stat["score"])
        t_score = float(top_stat["score"])
        if top_sup >= 2.0 and runner_sup <= 1.0 and r_score >= 0.17 * max(0.001, t_score):
            contradiction_level = max(contradiction_level, 0.46)
        if top_sup >= 2.0 and runner_sup >= 2.0 and runner_sup + 0.4 >= top_sup:
            contradiction_level = max(contradiction_level, 0.39)
    tok: Dict[str, float] = {}
    action_line = ""
    wording_line = ""
    both_line = ""
    top = 0.0
    for r in rows:
        txt = str(r.get("example_text") or "").strip()
        et = str(r.get("example_type") or "").strip().lower()
        strength = max(
            0.0,
            min(1.1, float(r.get("effective_strength") or r.get("strength") or 0.0)),
        )
        if not txt:
            continue
        norm_txt = " ".join(txt.lower().split())
        if norm_txt != top_key:
            continue
        top = max(top, strength)
        add = min(0.28, 0.08 + strength * 0.18)
        for t in _tokenize_feedback_phrase(txt):
            tok[t] = min(0.9, tok.get(t, 0.0) + add)
        if et == "action" and not action_line:
            action_line = txt
        elif et == "wording" and not wording_line:
            wording_line = txt
        elif et == "both" and not both_line:
            both_line = txt
    return RespondExampleInfluence(
        token_weight=tok,
        action_line=action_line,
        wording_line=wording_line,
        both_line=both_line,
        strongest_strength=top,
        winning_effective_strength=max(0.0, min(1.1, float(top_stat["score"]))),
        contradiction_level=max(0.0, min(1.0, contradiction_level)),
        consistency=max(0.0, min(1.0, 1.0 - contradiction_level)),
        winning_gap=max(0.0, min(1.0, gap)),
        uses_fallback_only=top < 0.56,
    )


def _apply_example_influence_to_score(
    s: float,
    *,
    blob: str,
    example_influence: Optional[RespondExampleInfluence],
) -> float:
    if not example_influence or not example_influence.token_weight:
        return s
    out = float(s)
    blob_l = (blob or "").lower()
    bump = 1.0
    for tok, wt in example_influence.token_weight.items():
        if len(tok) > 2 and tok in blob_l:
            bump *= 1.0 + min(0.18, float(wt) * 0.22)
    out *= min(2.15, bump)
    return out


def _sanitize_replacement_for_overlay(raw: str) -> str:
    t = " ".join((raw or "").strip().split())
    if len(t) > 220:
        t = t[:217].rsplit(" ", 1)[0] + "…"
    return t.strip()


def _answer_covers_injection_tokens(answer: str, inj: str) -> bool:
    toks = [t for t in _tokenize_feedback_phrase(inj) if len(t) > 3]
    if len(toks) < 2:
        toks = _tokenize_feedback_phrase(inj)
    if not toks:
        return True
    a = (answer or "").lower()
    hit = sum(1 for t in toks[:6] if t in a)
    return hit >= max(2, (len(toks) + 1) // 2)


def _apply_feedback_replacement_overlay(
    answer: str,
    reasoning: str,
    *,
    feedback_influence: RespondFeedbackInfluence,
    phrase_seed: str,
    answer_focus: str,
    eff_pf: str,
) -> Tuple[str, str]:
    """
    Bias final text toward stored replacement lines.

    Phase 40: repeated wrong + replacement on conflict-shaped prompts.
    Phase 46: action-ok / wording-off replacement applies in one shot for any
    routed family (not only conflict), using the user-approved line directly.
    """
    inj_ao = ""
    if feedback_influence.action_ok_wording_off:
        inj_ao = _sanitize_replacement_for_overlay(
            feedback_influence.action_ok_wording_replacement_line or ""
        )
    n_wr = int(feedback_influence.wrong_replacement_count or 0)
    inj_wr = _sanitize_replacement_for_overlay(
        feedback_influence.replacement_inject_line or ""
    )
    action_ok_inj = len(inj_ao) >= 8
    if action_ok_inj:
        inj = inj_ao
    else:
        inj = inj_wr
        if "conflict" not in (eff_pf or "").lower():
            return answer, reasoning
        if n_wr < 2 or len(inj) < 8:
            return answer, reasoning
    a = (answer or "").strip()
    low = a.lower()
    covered = _answer_covers_injection_tokens(a, inj)
    avoidance_ans = _memory_blob_avoidance_hit(low) or "let it go" in low
    af = (answer_focus or "both").strip().lower()
    fb_mix = max(0.0, min(1.0, float(feedback_influence.replacement_direction_mixed or 0.0)))
    cautious_fb = fb_mix >= 0.36
    if action_ok_inj and not covered and af == "wording":
        tail = " Wording pulled from your recent same-thread correction."
        rs = (reasoning or "").rstrip()
        return inj, (rs + tail) if rs else tail.strip()
    if not covered and (avoidance_ans or n_wr >= 3 or action_ok_inj):
        if af == "both":
            if cautious_fb:
                act_opts = (
                    "I'd still lean calmer and clearer — the stronger saved pattern points that way.",
                    "I'd step more direct, but I'm not ignoring that one newer note pulled another way.",
                )
                tail = (
                    " The repeat pattern says one line; one recent correction disagreed, so I'm holding this lighter."
                )
            else:
                act_opts = (
                    "I'd step in calmer but clearer — closer to what I've been asking for on this kind of prompt.",
                    "I'd handle it more directly after the corrections I've stacked on this one.",
                )
                tail = (
                    " Recent feedback nudged the say-line that way."
                    if not action_ok_inj
                    else " Same-thread wording correction applied."
                )
            act_line = act_opts[
                _stable_index(f"{phrase_seed}:fbinj_both_act", len(act_opts))
            ]
            w_inj = inj.strip()
            if w_inj and not (w_inj.startswith('"') and w_inj.endswith('"')):
                w_inj = f'"{w_inj}"'
            merged = _merge_action_wording_paragraphs(
                act_line, w_inj, seed=f"{phrase_seed}:fbinj_m"
            )
            rs = (reasoning or "").rstrip()
            return merged, (rs + tail) if rs else tail.strip()
        if cautious_fb:
            pool = (
                f"I'd still lean more like: {inj}, but one newer correction pointed a different way.",
                f"Stronger repeat pattern lands closer to: {inj} — I'm weighing one recent disagree lightly.",
            )
            tail = " Holding the line softer until the disagreement repeats."
        else:
            pool = (
                f"I'd handle it more like: {inj}",
                f"I'd land here after the corrections I've given on this: {inj}",
                f"I'm pushing toward something closer to: {inj}",
            )
            tail = (
                " Recent feedback on this prompt nudges the line that way."
                if not action_ok_inj
                else " Same-thread wording correction applied."
            )
        new_a = pool[_stable_index(f"{phrase_seed}:fbinj", len(pool))]
        rs = (reasoning or "").rstrip()
        return new_a, (rs + tail) if rs else tail.strip()
    if not covered:
        if af == "wording":
            if cautious_fb:
                b = f"I'd still phrase it closer to: {inj}, though one recent note didn't match that pattern."
            else:
                b = f"I'd phrase it closer to: {inj}"
        else:
            if cautious_fb:
                bridges = (
                    f"Said plainly, I'd still lean toward: {inj}, with one newer correction pulling another way.",
                    f"Out loud, closer to: {inj} — but I'm not treating one-off noise as the new default.",
                )
            else:
                bridges = (
                    f"Said plainly, more like: {inj}",
                    f"Out loud, closer to: {inj}",
                )
            b = bridges[_stable_index(f"{phrase_seed}:fbapp", len(bridges))]
        sep = "\n\n" if a else ""
        return f"{a}{sep}{b}", reasoning
    return answer, reasoning


def _apply_example_overlay(
    answer: str,
    reasoning: str,
    *,
    example_influence: Optional[RespondExampleInfluence],
    answer_focus: str,
    phrase_seed: str,
) -> Tuple[str, str, bool]:
    """When strong promoted examples exist, gently pull final line toward them."""
    if not example_influence:
        return answer, reasoning, False
    if float(example_influence.strongest_strength or 0.0) < 0.55:
        return answer, reasoning, False
    mix_ex = float(example_influence.contradiction_level or 0.0)
    # Conflicting same-shape examples: avoid overconfident direct overlay.
    if mix_ex >= 0.48:
        return answer, reasoning, False
    cautious_ex = 0.34 <= mix_ex < 0.48
    af = (answer_focus or "both").strip().lower()
    line = ""
    if af == "action":
        line = example_influence.action_line or example_influence.both_line
    elif af == "wording":
        line = example_influence.wording_line or example_influence.both_line
    else:
        line = (
            example_influence.both_line
            or example_influence.action_line
            or example_influence.wording_line
        )
    line = _sanitize_replacement_for_overlay(line)
    if len(line) < 8:
        return answer, reasoning, False
    if _answer_covers_injection_tokens(answer, line):
        return answer, reasoning, False
    if af == "both":
        if cautious_ex:
            hedges = (
                f"I might keep the say-line closer to: {line}, but past saves don't fully agree yet.",
                f"If I had to pick wording: maybe {line} — still mixed against older saves.",
            )
            hx = hedges[_stable_index(f"{phrase_seed}:excau", len(hedges))]
            base_a = (answer or "").strip()
            sep = "\n" if base_a else ""
            out = f"{base_a}{sep}{hx}"
        else:
            act = _both_mode_action_line(
                line,
                "",
                primary_family="general",
                seed=f"{phrase_seed}:exact",
                cautious=False,
                blunt=0.5,
                aggressive_short=False,
            )
            out = _merge_action_wording_paragraphs(act, line, seed=f"{phrase_seed}:exm")
    else:
        if cautious_ex:
            opts = (
                f"I might lean toward: {line}, though saved corrections still disagree some.",
                f"Rough direction: {line} — I'm not locking it while examples pull two ways.",
            )
        else:
            opts = (
                f"I'd keep it closer to this: {line}",
                f"This lines up with what I've corrected before: {line}",
            )
        out = opts[_stable_index(f"{phrase_seed}:exov", len(opts))]
    rb = (reasoning or "").rstrip()
    tail = (
        " Past saved lines don't fully line up, so I'm keeping the wording softer."
        if cautious_ex
        else " This also lines up with a saved repeated correction."
    )
    return out, (rb + tail) if rb else tail.strip(), True


def retrieve_relevant_decision_memories(
    rows: Sequence[Dict[str, Any]],
    prompt: str,
    top_k: int = 5,
    min_score: float = 0.28,
    evidence_mult_map: Optional[Mapping[str, float]] = None,
    *,
    score_bias: float = 1.0,
    feedback_influence: Optional[RespondFeedbackInfluence] = None,
    example_influence: Optional[RespondExampleInfluence] = None,
) -> List[Tuple[Dict[str, Any], float, List[str]]]:
    keywords = tokenize_prompt(prompt)
    prompt_norm = normalize_input(prompt)
    scenario_counts = _count_occurrences([str(r.get("scenario_id") or "") for r in rows])
    mmap = evidence_mult_map or {}
    sb = max(0.5, min(1.35, float(score_bias or 1.0)))
    is_public_disrespect = public_audience_disrespect_prompt(prompt_norm)
    scored: List[Tuple[Dict[str, Any], float, List[str], str, str]] = []
    for row in rows:
        rid = str(row.get("id") or "")
        em = float(mmap.get(f"decision:{rid}", 1.0))
        s, reasons = score_decision_memory_row(
            row, keywords, scenario_counts, evidence_row_mult=em
        )
        blob = decision_row_text_blob(row)
        s *= decision_memory_relevance_multiplier(prompt_norm, row, s)
        if is_public_disrespect and _memory_blob_avoidance_hit(blob):
            # Public disrespect is not a low-stakes "ignore it" shape.
            s *= 0.22
        s *= sb
        s = _apply_feedback_influence_to_score(
            s,
            blob=blob,
            feedback_influence=feedback_influence,
        )
        s = _apply_example_influence_to_score(
            s,
            blob=blob,
            example_influence=example_influence,
        )
        scored.append(
            (
                row,
                s,
                reasons,
                str(row.get("timestamp") or ""),
                str(row.get("id") or ""),
            )
        )
    scored.sort(key=lambda x: (-x[1], -len(x[3]), x[4]))
    filtered = [r for r in scored if r[1] >= min_score]
    return [(r[0], r[1], r[2]) for r in filtered[:top_k]]


def retrieve_relevant_style_memories(
    rows: Sequence[Dict[str, Any]],
    prompt: str,
    top_k: int = 4,
    evidence_mult_map: Optional[Mapping[str, float]] = None,
    *,
    score_bias: float = 1.0,
    feedback_influence: Optional[RespondFeedbackInfluence] = None,
    example_influence: Optional[RespondExampleInfluence] = None,
) -> List[Tuple[Dict[str, Any], float, List[str]]]:
    keywords = tokenize_prompt(prompt)
    prompt_counts = _count_occurrences([str(r.get("prompt_id") or "") for r in rows])
    scored: List[Tuple[Dict[str, Any], float, List[str], str, str]] = []
    prompt_norm = normalize_input(prompt)
    mmap = evidence_mult_map or {}
    sb = max(0.5, min(1.35, float(score_bias or 1.0)))
    for row in rows:
        sid = str(row.get("id") or "")
        em = float(mmap.get(f"style:{sid}", 1.0))
        s, reasons = score_style_memory_row(
            row, keywords, prompt_counts, evidence_row_mult=em
        )
        s *= style_memory_relevance_multiplier(prompt_norm, row, s)
        s *= sb
        s = _apply_feedback_influence_to_score(
            s,
            blob=style_row_text_blob(row),
            feedback_influence=feedback_influence,
        )
        s = _apply_example_influence_to_score(
            s,
            blob=style_row_text_blob(row),
            example_influence=example_influence,
        )
        scored.append(
            (
                row,
                s,
                reasons,
                str(row.get("timestamp") or ""),
                str(row.get("id") or ""),
            )
        )
    scored.sort(key=lambda x: (-x[1], -len(x[3]), x[4]))
    return [(r[0], r[1], r[2]) for r in scored[:top_k]]


def _verbosity_from_profile(profile: PersonalProfile) -> float:
    for t in profile.trait_estimates:
        if t.name == "verbosity":
            return t.weighted_mean
    return 0.5


def _bluntness_from_profile(profile: PersonalProfile) -> float:
    for t in profile.trait_estimates:
        if t.name == "bluntness":
            return t.weighted_mean
    return 0.5


def _confidence_bucket(value: float) -> str:
    if value < 0.35:
        return "low"
    if value < 0.70:
        return "moderate"
    if value < 0.90:
        return "fairly high"
    return "high"


def _compute_response_confidence(
    profile: PersonalProfile,
    top_decision_score: float,
    top_decision: Optional[Dict[str, Any]],
    agreement_boost: float,
    *,
    decision_family_aligned: bool = True,
    path_multipliers: Optional[Sequence[float]] = None,
    example_influence: Optional[RespondExampleInfluence] = None,
    route_keys: Optional[Sequence[str]] = None,
    feedback_direction_mixed: float = 0.0,
) -> float:
    base = 0.32
    ev = min(1.0, profile.total_evidence_weight / 8.0)
    base += 0.18 * ev
    base += 0.14 * min(1.0, top_decision_score / 4.5)
    if profile.has_trait_conflict:
        base -= 0.12
    conf_ag = agreement_boost
    base += 0.12 * min(1.0, conf_ag)
    if top_decision:
        st = top_decision.get("correction_status")
        if st == "accurate":
            base += 0.08
        elif st == "not_really":
            base -= 0.22
        elif st == "partially_true":
            base -= 0.06
    if not decision_family_aligned:
        base -= 0.2
    if profile.total_evidence_weight < 0.85:
        base -= 0.14
    if profile.decision_entries_used == 0 and profile.style_entries_used == 0:
        base -= 0.2
    if path_multipliers:
        deficit = sum(max(0.0, 1.0 - float(m)) for m in path_multipliers if m < 1.0)
        base -= min(0.24, 0.058 * deficit)
    rks = {str(x).strip().lower() for x in (route_keys or ()) if str(x).strip()}
    fallback_route = any(
        x in rks
        for x in (
            "profile_pattern_fallback",
            "profile_pattern_fallback_suppressed",
            "weak_profile_signal",
            "insufficient_evidence",
            "strict_shape_miss",
            "strict_conflict_fallback",
            "strict_spending_fallback",
        )
    )
    if fallback_route:
        base -= 0.08
    if example_influence:
        cx = max(0.0, min(1.0, float(example_influence.contradiction_level or 0.0)))
        base -= 0.18 * cx
        if float(example_influence.winning_gap or 0.0) >= 0.34 and cx <= 0.28:
            base += 0.04
        if float(example_influence.winning_effective_strength or 0.0) >= 0.92 and cx <= 0.2:
            base += 0.03
        if bool(example_influence.uses_fallback_only):
            base -= 0.05
    fb_mix = max(0.0, min(1.0, float(feedback_direction_mixed or 0.0)))
    if fb_mix > 0:
        base -= 0.19 * fb_mix
    return max(0.12, min(0.9, base))


def _shorten_sentence(text: str, aggressive: bool) -> str:
    t = (text or "").strip()
    if not aggressive or len(t) < 90:
        return t
    cut = t[:87].rsplit(" ", 1)[0]
    return cut + "…"


def _phase41_style_realism_pass(text: str, *, answer_focus: str) -> str:
    """Deterministic phrasing cleanup for more natural first-person voice.

    Keep logic/evidence unchanged; only de-meta the surface wording.
    """
    out = (text or "").strip()
    if not out:
        return out

    # Strip explicit retrieval narration from the main response text.
    noise = (
        "If I'm reading your saves right, ",
        "From what's on file, ",
        "on file you tend to ",
        "on file you tend to be ",
        "My read is ",
        "my read is ",
        "My guess is ",
        "my guess is ",
        "given your corrections, ",
        "Given your corrections, ",
    )
    for n in noise:
        out = out.replace(n, "")

    # Shift from "system describing user" toward spoken first-person style.
    swaps = (
        ("you'd probably", "I'd probably"),
        ("you'd likely", "I'd probably"),
        ("you'd most likely", "I'd probably"),
        ("you're most likely to say", "I'd probably say"),
        ("you'd probably say something like:", "I'd probably say:"),
        ("you'd probably say:", "I'd probably say:"),
        ("you'd probably go with:", "I'd probably go with:"),
        ("you'd probably choose:", "I'd probably choose:"),
        ("what you'd probably do", "what I'd do"),
        ("What you'd probably do", "What I'd do"),
        ("you'd", "I'd"),
        ("You'd", "I'd"),
        ("You are", "I am"),
        ("you are", "I am"),
        ("you were", "I was"),
        ("You were", "I was"),
        ("you've", "I've"),
        ("You've", "I've"),
        ("you want", "I want"),
        ("You want", "I want"),
        ("you noticed", "I noticed"),
        ("you heard", "I heard"),
        ("You heard", "I heard"),
    )
    for a, b in swaps:
        out = out.replace(a, b)
        if a != a.title():
            out = out.replace(a.title(), b)

    # Keep combined mode explicit and natural.
    if (answer_focus or "").strip().lower() == "both":
        out = out.replace("If you said it out loud, ", "If I said it out loud, ")
        out = out.replace("In plain words, ", "In plain words, I'd ")
        out = out.replace("If You said it out loud, ", "If I said it out loud, ")

    # Tighter voice: drop redundant "probably" after "I'd".
    out = out.replace("I'd probably ", "I'd ")
    out = out.replace("i'd probably ", "i'd ")

    # Spending / leftovers: second-person out-loud hooks → first person.
    out = out.replace("If you said it out loud, ", "If I said it out loud, ")
    out = out.replace("if you said it out loud, ", "If I said it out loud, ")
    out = out.replace("The line you'd probably use with yourself is ", "The line I'd use with myself is ")

    # Tone/meta clauses sometimes leak from older templates; strip known fragments.
    for _rm in (
        ", with pretty blunt wording",
        ", with fairly soft wording",
        ", with short and direct wording",
        ", with fairly detailed wording",
        "pretty blunt in how you usually phrase things",
        "fairly soft in how you usually phrase things",
        "short and direct in how you usually phrase things",
        "fairly detailed in how you usually phrase things",
    ):
        out = out.replace(_rm, "")
    out = re.sub(
        r"\s*[—:]\s*that's usually me:\s*.+?\.",
        ".",
        out,
        flags=re.IGNORECASE | re.DOTALL,
        count=1,
    )
    # Collapse repeated spaces only; preserve newlines for both-mode paragraphs.
    out = re.sub(r" {2,}", " ", out)

    # Likely-you voice is first-person: fix self-reference (I'd + your → my).
    for old, new in (
        ("what you truly need", "what I truly need"),
        ("What you truly need", "What I truly need"),
        ("what you actually need", "what I actually need"),
        ("What you actually need", "What I actually need"),
        ("what you heard", "what I heard"),
        ("What you heard", "What I heard"),
        ("protect your energy", "protect my energy"),
        ("Protect your energy", "Protect my energy"),
        ("your energy the way", "my energy the way"),
        ("your energy and go", "my energy and go"),
        ("your bandwidth", "my bandwidth"),
        ("Your bandwidth", "My bandwidth"),
        ("say your piece", "say my piece"),
        ("Say your piece", "Say my piece"),
        ("keep your tone", "keep my tone"),
        ("Keep your tone", "Keep my tone"),
        ("keep your voice", "keep my voice"),
        ("Keep your voice", "Keep my voice"),
        ("keep your footing", "keep my footing"),
        ("Keep your footing", "Keep my footing"),
        ("narrow your availability", "narrow my availability"),
        ("Narrow your availability", "Narrow my availability"),
        ("protect your week", "protect my week"),
        ("shrinking your week", "shrinking my week"),
        ("spell out the limit and what you do", "spell out the limit and what I do"),
    ):
        out = out.replace(old, new)

    # Clean obvious filler artifacts after replacements.
    out = out.replace("I'd I'd I'd ", "I'd ")
    out = out.replace("I'd I'd ", "I'd ")
    out = out.replace("  ", " ")
    out = out.replace(" .", ".")
    if (answer_focus or "").strip().lower() == "both":
        out = re.sub(r"\n{2,}", "\n", out)
        if "\n" not in out:
            for sm in (
                " If I said it out loud",
                " if I said it out loud",
                " I'd say out loud",
            ):
                if sm in out:
                    i = out.index(sm)
                    out = out[:i].rstrip() + "\n" + out[i:].lstrip()
                    break
    return out.strip()


def _phase48_fix_sentence_initial_pronoun(chunk: str) -> str:
    """Capitalize sentence-initial ``i`` / common contractions (safe surface fix)."""
    if not chunk:
        return chunk
    m = re.match(r"^(\s*)([\s\S]*)$", chunk)
    if not m:
        return chunk
    ws, body = m.group(1), m.group(2)
    if not body:
        return chunk
    bl = body.lower()
    fixes = (
        ("i'd ", "I'd "),
        ("i'd.", "I'd."),
        ("i'm ", "I'm "),
        ("i've ", "I've "),
        ("i'll ", "I'll "),
        ("i cant ", "I can't "),
        ("i cant.", "I can't."),
        ("i won't ", "I won't "),
        ("i won't.", "I won't."),
        ("i wont ", "I won't "),
        ("i wont.", "I won't."),
        ("i ", "I "),
    )
    for ol, nw in fixes:
        if bl.startswith(ol):
            return ws + nw + body[len(ol) :]
    return chunk


def _phase48_normalize_sentence_starts(segment: str) -> str:
    """Apply pronoun fixes after ``.!?`` boundaries within one line."""
    if not segment:
        return segment
    parts = re.split(r"([.!?]\s+)", segment)
    out: List[str] = []
    for i, p in enumerate(parts):
        if i % 2 == 0:
            out.append(_phase48_fix_sentence_initial_pronoun(p))
        else:
            out.append(p)
    return "".join(out)


def _phase48_dedupe_stacked_out_loud_wrappers(text: str) -> str:
    """
    Drop a redundant ``it might sound like this`` line when the next block
    already opens with the same out-loud hook.
    """
    t = (text or "").strip()
    if not t:
        return t
    return re.sub(
        r"(?is)\bIf I said it out loud, it might sound like this:\s*\n+\s*(If I said it out loud)",
        r"\1",
        t,
    )


def _phase48_fix_pronoun_after_closing_quote(text: str) -> str:
    """``..." i was`` fragments — sentence split misses ``."`` before a space."""
    return re.sub(r'(["\u201d])(\s+)i\b', r"\1\2I", text)


def _phase48_fix_because_i(text: str) -> str:
    """Normalize ``because i`` / ``mostly because i`` after template merges."""
    t = re.sub(r"(?i)\bmostly because i\b", "Mostly because I", text)
    return re.sub(r"(?i)\bbecause i\b", "because I", t)


def _phase48_strip_duplicate_trailing_why(action_line: str, wording_flat: str) -> str:
    """
    When the action paragraph already ends with ``Mostly because {why}``, drop the
    same ``why`` fragment repeated at the end of the wording line (Phase 48).
    """
    a = (action_line or "").strip()
    w = (wording_flat or "").strip()
    if not a or not w:
        return w
    m = re.search(r"(?is)\bmostly\s+because\s+(.+?)\.(\s*)$", a)
    if not m:
        return w
    why_raw = (m.group(1) or "").strip()
    if len(why_raw) < 6:
        return w
    w_st = w.strip()
    w_low = w_st.lower()
    suf = why_raw.lower().rstrip(".") + "."
    if w_low.endswith(suf):
        cut = w_st[: -len(suf)].rstrip()
        return cut if len(cut) >= 12 else w
    return w


def _phase48_final_answer_polish(text: str) -> str:
    """
    Phase 48: presentation-only cleanup for surfaced ``likely_answer``.

    Does not touch ``reasoning_brief``. Deterministic; no retrieval/scoring changes.
    """
    t = (text or "").strip()
    if not t:
        return t
    t = _phase48_dedupe_stacked_out_loud_wrappers(t)
    lines = t.split("\n")
    fixed_lines = []
    for line in lines:
        ln = _phase48_normalize_sentence_starts(line)
        ln = _phase48_fix_pronoun_after_closing_quote(ln)
        ln = _phase48_fix_because_i(ln)
        fixed_lines.append(ln)
    t = "\n".join(fixed_lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _phase49_hedge_level(
    *,
    conf: float,
    route_keys: Sequence[str],
    feedback_direction_mixed: float,
    example_contradiction: float,
) -> str:
    """
    Phase 49: surface-only hedge tier from existing confidence and route signals.

    Does not alter confidence; only selects wording polish strength.
    Returns ``soft``, ``medium``, or ``firm``.
    """
    rks = {str(x).strip().lower() for x in (route_keys or ()) if str(x).strip()}
    soft_routes = {
        "strict_conflict_fallback",
        "strict_shape_miss",
        "insufficient_evidence",
        "profile_pattern_fallback",
        "profile_pattern_fallback_suppressed",
        "weak_profile_signal",
        "family_misalign",
        "family_misalign_ungated",
        "example_conflict_mixed",
    }
    if float(feedback_direction_mixed or 0.0) >= 0.34:
        return "soft"
    if float(example_contradiction or 0.0) >= 0.42:
        return "soft"
    if float(conf or 0.0) < 0.38:
        return "soft"
    if rks & soft_routes:
        return "soft"
    if float(conf or 0.0) >= 0.50:
        return "firm"
    if float(conf or 0.0) >= 0.40:
        return "medium"
    return "soft"


def _phase49_skip_wording_tightening(
    answer: str,
    *,
    carry_used_replacement_stance: bool,
    feedback_influence: RespondFeedbackInfluence,
) -> bool:
    """Preserve carryover / user-approved replacement lines verbatim (Phase 49)."""
    if carry_used_replacement_stance:
        return True
    ao = bool(feedback_influence.action_ok_wording_off)
    if not ao:
        return False
    rep = (_phase46_resolved_replacement_line(feedback_influence) or "").strip()
    if len(rep) < 8:
        return False
    a_norm = " ".join((answer or "").strip().lower().split())
    r_norm = " ".join(rep.lower().split())
    return a_norm == r_norm


def _phase49_wording_strength_polish(
    text: str,
    *,
    hedge_level: str,
) -> str:
    """
    Phase 49: trim redundant hedges on the final surfaced line when grounding is solid.

    Runs after Phase 48; does not change retrieval, caps, or reasoning text.
    """
    t = (text or "").strip()
    if not t:
        return t
    level = (hedge_level or "soft").strip().lower()
    if level not in ("medium", "firm"):
        return t

    cautious_prefs = (
        "If I'm reading your saves right, ",
        "From what's on file, ",
    )

    def _strip_cautious_line_prefixes(segment: str) -> str:
        s = segment
        if level != "firm":
            return s
        for p in cautious_prefs:
            if s.startswith(p):
                s = s[len(p) :]
        return s

    lines = t.split("\n")
    out_lines: List[str] = []
    for raw_ln in lines:
        ln = raw_ln
        ln = _strip_cautious_line_prefixes(ln)
        # Out-loud framing: keep a short hook for both-mode; drop hedgy "maybe"/"like".
        if level == "firm":
            for ol, repl in (
                ("If I said it out loud, maybe: ", "If I said it out loud, "),
                ("If I said it out loud, it'd be something like: ", "If I said it out loud, "),
                ("if i said it out loud, maybe: ", "If I said it out loud, "),
                ("if i said it out loud, it'd be something like: ", "If I said it out loud, "),
            ):
                if ol in ln:
                    ln = ln.replace(ol, repl)
        else:
            ln = re.sub(
                r"(?i)If I said it out loud, maybe:\s*",
                "If I said it out loud, ",
                ln,
                count=1,
            )
            ln = re.sub(
                r"(?i)If I said it out loud, it'd be something like:\s*",
                "If I said it out loud, ",
                ln,
                count=1,
            )
        out_lines.append(ln)
    t = "\n".join(out_lines)

    # Light "likely" trim when not in soft-lock territory.
    if level in ("medium", "firm"):
        t = re.sub(r"(?i)\bI would likely\b", "I would", t)
        t = re.sub(r"(?i)\bI'd likely\b", "I'd", t)
    t = re.sub(r" {2,}", " ", t)
    lines2 = t.split("\n")
    t = "\n".join(_phase48_normalize_sentence_starts(x) for x in lines2)
    return t.strip()


def _family_label(primary_family: str) -> str:
    pf = (primary_family or "").strip().lower()
    if "conflict" in pf:
        return "conflict"
    if "obligation" in pf or "boundary" in pf:
        return "overload"
    if "risk" in pf or "timing" in pf:
        return "timing"
    return "general"


def _to_lower_start(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    return t[0].lower() + t[1:] if len(t) > 1 else t.lower()


def _action_mode_choice_gloss(choice: str) -> str:
    """Short behavioral anchor for action mode — avoids full scripted lines."""
    c = (choice or "").strip()
    if not c:
        return ""
    if len(c) <= 46:
        return _to_lower_start(c)
    split = re.split(r"[.;]", c, maxsplit=1)
    first = (split[0] or "").strip()
    if first and len(first) <= 52:
        return _to_lower_start(first)
    words = c.split()
    lim = 10
    chunk = " ".join(words[:lim]).rstrip(",;:")
    tail = "…" if len(words) > lim else ""
    return _to_lower_start(chunk) + tail


def _both_mode_action_line(
    choice: str,
    why: str,
    *,
    primary_family: str,
    seed: str,
    cautious: bool,
    blunt: float,
    aggressive_short: bool,
) -> str:
    """First paragraph for answer_focus=both: what I'd *do*, grounded in reasoning — not the out-loud script."""
    fam = _family_label(primary_family)
    w = (why or "").strip().lower()
    why_low = _to_lower_start(why) if (why or "").strip() else ""
    yfrag = f" Mostly because {why_low}." if why_low else ""
    cautious_openers = (
        "If I'm reading your saves right, ",
        "From what's on file, ",
        "",
    )
    pref = (
        cautious_openers[_stable_index(f"{seed}:cp", len(cautious_openers))]
        if cautious
        else ""
    )

    soft_why = any(
        x in w for x in ("smooth", "peace", "energy", "calm", "worth", "battle", "recover")
    )
    firm_why = any(
        x in w
        for x in (
            "stop",
            "direct",
            "pattern",
            "boundary",
            "clear the",
            "spiral",
            "firm",
            " say no",
            "said no",
        )
    )

    if fam == "conflict":
        if firm_why and not soft_why:
            pool = (
                f"{pref}I'd step in, name what's sideways, and move it toward a straight talk — calm, but I wouldn't pretend I didn't notice.{yfrag}",
                f"{pref}I'd address it with one clear beat and keep the heat low — still forward, not a performance.{yfrag}",
            )
        elif soft_why and not firm_why:
            pool = (
                f"{pref}I'd keep my footing and not feed the snippy loop — small move, not a big scene.{yfrag}",
                f"{pref}I'd let the sharp moment pass without starting a war, but I wouldn't act like I didn't clock it.{yfrag}",
            )
        else:
            pool = (
                f"{pref}I'd bring it back to a direct conversation instead of letting it hang in the air.{yfrag}",
                f"{pref}I'd say what I noticed once, plainly, and ask for it clean between us.{yfrag}",
            )
    elif fam == "overload":
        if any(
            x in w
            for x in (
                "energy",
                "bandwidth",
                "burn",
                "drain",
                "overload",
                "protect",
                "recover",
                "capacity",
            )
        ):
            pool = (
                f"{pref}I'd guard my bandwidth and keep the no simple — same line if they circle back.{yfrag}",
                f"{pref}I'd stop renegotiating the same ask and park the extras until I'm steadier.{yfrag}",
            )
        else:
            pool = (
                f"{pref}I'd draw the line where it fits what I can actually give.{yfrag}",
                f"{pref}I'd make the smaller yes obvious, or the full no obvious — not both.{yfrag}",
            )
    elif fam == "timing":
        pool = (
            f"{pref}I'd wait for a cleaner beat so the words land the way I mean them.{yfrag}",
            f"{pref}I'd pause for one more signal before I lock the move.{yfrag}",
        )
    else:
        cg = _action_mode_choice_gloss(choice)
        if cg:
            pool = (
                f"{pref}I'd lean toward {cg}.{yfrag}",
                f"{pref}I'd take the practical read: {cg}.{yfrag}",
            )
        else:
            pool = (
                f"{pref}I'd handle it straight without dragging it out.{yfrag}",
                f"{pref}I'd pick the low-drama path that still gets it handled.{yfrag}",
            )

    line = pool[_stable_index(f"{seed}:both_act:{fam}", len(pool))]
    line = line.replace("..", ".").strip()
    if not line.endswith("."):
        line += "."
    return _shorten_sentence(line, aggressive_short)


def _conflict_profile_tone_clause(
    profile_risk: Optional[str], communication_style: Optional[str]
) -> Optional[str]:
    """
    Single readable clause for strict conflict fallback.
    Avoids stacking template words on summaries like "leans cautious"
    (e.g. "skew leans cautious; blunter phrasing").
    """
    risk: Optional[str] = None
    pr = (profile_risk or "").strip()
    if pr == "leans cautious":
        risk = "cautious"
    elif pr == "leans risk-tolerant":
        risk = "more risk-tolerant"

    cs = (communication_style or "").strip()
    style_map = {
        "blunter phrasing": "pretty blunt",
        "softer wording": "fairly soft",
        "shorter, direct explanations": "short and direct",
        "more detailed explanations": "fairly detailed",
    }
    style = style_map.get(cs, cs).strip() if cs else ""

    if risk and style:
        return f"{risk} on risk, with {style} wording"
    if risk:
        return f"{risk} on risk"
    if style:
        return f"{style} in how you usually phrase things"
    return None


def _speed_quality_trap_choice(choice: str, reasoning: str) -> bool:
    """True when saved labels read like work speed vs quality, not bills or wants."""
    c = (choice or "").lower()
    w = (reasoning or "").lower()
    blob = f"{c} {w}"
    if "proper way" in c and "longer" in c:
        return True
    if "corners" in blob and ("proper" in c or "quality" in w or "quality" in c):
        return True
    if "sloppy" in blob and ("proper" in c or "longer" in c):
        return True
    return False


def _respond_strict_decision_shape_prompt(prompt_norm: str, eff_pf: str) -> bool:
    """
    When True, respond-like-me must not treat style/profile blurbs as a stand-in
    for missing same-shape decision memory.
    """
    _, dp, _ = rank_families_full(prompt_norm)
    padded = f" {prompt_norm} "
    if eff_pf == CONFLICT_FAMILY and (
        interpersonal_conflict_markers_present(prompt_norm)
        or gossip_or_backchannel_user_prompt(prompt_norm)
    ):
        return True
    if eff_pf == SPENDING and (
        float(dp.get("money_pressure", 0) or 0) >= 0.65
        or any(
            x in prompt_norm
            for x in (
                "rent",
                "bill",
                "$",
                "afford",
                "broke",
                "debt",
                "salary",
                "laptop",
                "buy",
                "pay ",
            )
        )
    ):
        return True
    if eff_pf == OBLIGATION_OVERLOAD:
        if any(
            x in padded or x in prompt_norm
            for x in (
                " shift ",
                " cover ",
                " favor ",
                " favour ",
                " exhausted",
                " burnt out",
                " burned out",
                "drained",
                " asked me",
                " asking me",
            )
        ):
            return True
        if float(dp.get("overload", 0) or 0) + float(dp.get("obligation", 0) or 0) >= 1.0:
            return True
    return False


def _merge_action_wording_paragraphs(action_line: str, wording_line: str, *, seed: str) -> str:
    """Join do vs say when the ask is both (Phase 41 final).

    One line for what I'd do, one line for what I'd say — single ``\\n`` between
    them so the split survives narrow formatters; wording is flattened to one line.
    """
    a = " ".join((action_line or "").split())
    w = (wording_line or "").strip()
    if not w:
        return a
    if a == w:
        tail = (
            "If I said it out loud, it'd land the same either way.",
            "Same beat whether I'm doing it or saying it — not really two different scripts here.",
        )
        return f"{a}\n{tail[_stable_index(f'{seed}:br_dup', len(tail))]}"
    w_flat = " ".join(w.split())
    w_flat = _phase48_strip_duplicate_trailing_why(a, w_flat)
    low_w = w_flat.lower()
    if low_w.startswith("if i said it out loud"):
        wording_block = w_flat
    else:
        intros = (
            "If I said it out loud, it'd be something like:",
            "If I said it out loud, maybe:",
        )
        wording_block = f"{intros[_stable_index(f'{seed}:wlpr', len(intros))]} {w_flat}"
    return f"{a}\n{wording_block}"


def _obligation_boundary_repeat_both_coherent(
    choice: str,
    why: str,
    *,
    phrase_seed: str,
    cautious: bool,
    blunt: float,
    aggressive_short: bool,
) -> str:
    """
    One paragraph for Phase 45 ``boundary_push_repeat`` + obligation overload + ``both``.

    Replaces stacked action + wording + \"If I said it out loud\" merge, which duplicated
    the same boundary idea and appended \"Mostly because\" twice.

    Rationale from the save stays *before* the quoted spoken line so it never trails
    awkwardly after the closing quote (blunt path used to append a bare fragment).
    """
    ch = (choice or "").strip().rstrip(".,;:!?")
    why_low = _to_lower_start(why) if (why or "").strip() else ""
    if why_low.startswith("i "):
        why_readable = "I " + why_low[2:]
    elif why_low.startswith("i'm "):
        why_readable = "I'm " + why_low[4:]
    elif why_low.startswith("i've "):
        why_readable = "I've " + why_low[5:]
    else:
        why_readable = why_low
    cautious_openers = (
        "If I'm reading your saves right, ",
        "From what's on file, ",
        "",
    )
    pref = (
        cautious_openers[_stable_index(f"{phrase_seed}:obbr_cp", len(cautious_openers))]
        if cautious
        else ""
    )
    if why_readable:
        if blunt >= 0.62:
            lead = f"{why_readable[0].upper()}{why_readable[1:]}. " if len(why_readable) > 1 else f"{why_readable.upper()}. "
        else:
            lead = f"Mostly because {why_readable}, "
    else:
        lead = ""
    if ch:
        pools = (
            f"{pref}{lead}I'd hold the line on what I already said — same short no, not a new debate — like: \"{ch}\".",
            f"{pref}{lead}I'd keep my boundary simple and repeat the same words if they push again — like: \"{ch}\".",
            f"{pref}{lead}I'd say no the same way as before — clear and plain — like: \"{ch}\".",
        )
    else:
        pools = (
            f"{pref}{lead}I'd keep saying no the same short way — no fresh argument round.",
            f"{pref}{lead}I'd hold the boundary steady instead of re-explaining it.",
        )
    line = pools[_stable_index(f"{phrase_seed}:obbr_pool:{blunt}", len(pools))]
    line = line.replace("..", ".").strip()
    if not line.endswith("."):
        line += "."
    return _shorten_sentence(line, aggressive_short)


def _strict_conflict_shape_evidence_fallback(
    *,
    prompt_norm: str,
    phrase_seed: str,
    s_ranked: Sequence[Tuple[Dict[str, Any], float, List[str]]],
    profile_risk: Optional[str],
    communication_style: Optional[str],
    tmap: Mapping[str, float],
    blunt: float,
    aggressive_short: bool,
    store: DatabaseStore,
    answer_focus: str = "both",
    feedback_influence: Optional[RespondFeedbackInfluence] = None,
    skip_avoidance_style_memory: bool = False,
    phase45_escalation_subtype: str = "",
) -> Tuple[str, str, float, List[str], Tuple[str, ...]]:
    """Cautious likely-you line when strict conflict gating finds no decision row.

    The last tuple is style memory row ids that anchored wording (may be empty).
    """
    extra_basis: List[str] = []
    gossip = gossip_or_backchannel_user_prompt(prompt_norm)
    af = (answer_focus or "both").strip().lower()
    if af not in ("action", "wording", "both"):
        af = "both"
    ut_style = "wording" if af in ("wording", "both") else "action"

    for row, sc, _ in s_ranked:
        if (row.get("correction_status") or "") == "not_really":
            continue
        if sc < 0.28:
            continue
        if skip_avoidance_style_memory and _memory_blob_avoidance_hit(
            style_row_text_blob(row)
        ):
            continue
        if (
            feedback_influence
            and float(feedback_influence.avoidance_demote) >= 0.34
            and _memory_blob_avoidance_hit(style_row_text_blob(row))
        ):
            continue
        if not style_memory_passes_respond_conflict_shape(prompt_norm, row):
            continue
        choice = str(row.get("selected_label") or "").strip()
        if not choice:
            continue
        tags = row.get("style_tags") or []
        why = str(tags[0]) if tags else "that is how you would want it to read"
        answer = _natural_likely_line(
            choice,
            why,
            primary_family=CONFLICT_FAMILY,
            seed=f"{phrase_seed}:sf",
            cautious=True,
            blunt=blunt,
            aggressive_short=aggressive_short,
            utterance_mode=ut_style,
        )
        if af == "both":
            act_a = _both_mode_action_line(
                choice,
                why,
                primary_family=CONFLICT_FAMILY,
                seed=f"{phrase_seed}:sf_act",
                cautious=True,
                blunt=blunt,
                aggressive_short=aggressive_short,
            )
            answer = _merge_action_wording_paragraphs(
                act_a, answer, seed=f"{phrase_seed}:sf_m"
            )
        reasoning = (
            "No close matching decision; style calibration that fits this conflict shape still points this way. "
            "Indirect only — confidence stays low."
        )
        conf_cap = 0.41 if sc >= 0.52 else 0.38
        pid = str(row.get("prompt_id") or "").strip() or "style"
        sk = f"respond_mb_style_{row.get('id') or pid or 'x'}"
        if hasattr(store, "should_surface_memory_line") and store.should_surface_memory_line(
            sk
        ):
            stmpl = (
                "Saved style pick ({pid})",
                "Earlier style answer ({pid})",
            )
            extra_basis.append(
                stmpl[_stable_index(f"{phrase_seed}:mbsf", len(stmpl))].format(pid=pid)
            )
            if hasattr(store, "record_memory_line_surface"):
                try:
                    store.record_memory_line_surface(sk)
                except Exception:
                    pass
        sid = str(row.get("id") or "").strip()
        stuple = (sid,) if sid else ()
        return answer, reasoning, conf_cap, extra_basis, stuple

    public_disrespect = public_audience_disrespect_prompt(prompt_norm)
    if public_disrespect:
        if af == "action":
            opts = (
                "You'd probably address it cleanly in the moment or right after — short, steady, and clear about the line.",
                "My read is you'd set a boundary without performing: name it, keep calm, and keep it moving.",
            )
        else:
            opts = (
                "You'd probably keep it calm but direct: that was not okay in front of people, and you want it handled straight.",
                "My read is you'd say it plainly without a scene — you can disagree, but public disrespect is not okay.",
            )
        answer = opts[_stable_index(f"{phrase_seed}:cpub", len(opts))]
        if af == "both":
            act_pub = (
                "You'd probably mark the line right away or right after, then move it to a direct one-on-one.",
                "My read is you'd keep the response controlled but firm — no pile-on, no pretending it was fine.",
            )
            answer = _merge_action_wording_paragraphs(
                act_pub[_stable_index(f"{phrase_seed}:cpub_a", len(act_pub))],
                answer,
                seed=f"{phrase_seed}:cpub_m",
            )
        reasoning = (
            "No close same-shape save; public disrespect cues raise the conflict stakes, so this leans calm/direct over passive ignoring."
        )
        return answer, reasoning, 0.37, extra_basis, ()

    cues = conflict_situational_cues(prompt_norm)
    if phase45_escalation_subtype in (
        "boundary_push_repeat",
        "interpersonal_persistence_repeat",
    ) and not gossip:
        if af == "action":
            opts = (
                "You'd probably hold the line you already drew — short and plain that your answer hasn't changed, and you can't keep rehashing it.",
                "My read is you'd name that they keep coming back after you already said no, and ask for the topic to drop unless something new is on the table.",
            )
        else:
            opts = (
                "You'd probably say it directly: you gave a clear no, they're circling the same ask, and you need them to stop treating it like it's open.",
                "My read is you'd keep it calm but unmistakable — repeat your boundary once, then close the loop instead of debating it again.",
            )
        answer = opts[_stable_index(f"{phrase_seed}:p45bd", len(opts))]
        if af == "both":
            act_m = (
                "You'd probably keep your footing — same boundary as before, no extra justification, and you don't need a long back-and-forth.",
                "My read is you'd treat it as a pattern now: they already heard your answer, so the issue is the push, not the original ask.",
            )
            answer = _merge_action_wording_paragraphs(
                act_m[_stable_index(f"{phrase_seed}:p45bd_a", len(act_m))],
                answer,
                seed=f"{phrase_seed}:p45bd_m",
            )
        reasoning = (
            "No tight conflict save on file; continued boundary pressure on the same thread leans toward calm directness over \"let it go\" — still a template read, not a saved quote."
        )
        return answer, reasoning, 0.36, extra_basis, ()

    if (
        cues["passive_slight"]
        and cues["repeat_pattern"]
        and not gossip
    ):
        if af == "action":
            opts = (
                "You'd probably treat the repeat as the real issue — name the pattern calmly, one clear example, and say you need it to change.",
                "My read is you'd stop giving the sideways shots a pass now that they keep landing — short, steady, and pointed at the behavior, not a character attack.",
            )
        else:
            opts = (
                "You'd probably say you've noticed it more than once, name what they're doing in plain words, and ask for direct talk instead of digs.",
                "My read is you'd keep your voice calm but flat — the point is this keeps happening, not one ambiguous moment.",
            )
        answer = opts[_stable_index(f"{phrase_seed}:parep", len(opts))]
        if af == "both":
            act_m = (
                "You'd probably pick one recent instance, say how it lands, and set that you won't keep absorbing the same sideways move.",
                "My read is you'd keep it work-appropriate but unmistakable — pattern, not mood-reading.",
            )
            answer = _merge_action_wording_paragraphs(
                act_m[_stable_index(f"{phrase_seed}:parep_a", len(act_m))],
                answer,
                seed=f"{phrase_seed}:parep_m",
            )
        reasoning = (
            "No tight conflict save on file; passive-aggressive + repeat cues push away from one-off \"let it go\" reads — still a guess, not a quote from your saves."
        )
        return answer, reasoning, 0.36, extra_basis, ()

    peace = float(tmap.get("tendency_peace_over_confrontation", 0) or 0)
    clar = float(tmap.get("tendency_clarity_priority", 0) or 0)
    if max(peace, clar) >= 0.48:
        if af == "action":
            if clar >= peace + 0.1:
                opts = (
                    "You'd probably move it to a direct conversation: what you heard, why sideways talk is not okay, and what you want instead.",
                    "My read is you'd focus on closing it cleanly with facts and a face-to-face talk, not letting rumors keep rolling.",
                )
            else:
                opts = (
                    "You'd probably keep steady but still handle it in person — short, clear, no dogpile, but the behind-the-back piece stops.",
                    "My guess is you'd end the loop face-to-face without turning it into a speech — still direct about what you noticed.",
                )
        else:
            if clar >= peace + 0.1:
                opts = (
                    "You'd probably name what's going on in plain language, say you don't want sideways talk, "
                    "and ask for a direct conversation instead of rumors.",
                    "My read is you'd keep it factual and forward: what you heard, why it matters, and that you want it to stop.",
                )
            else:
                opts = (
                    "You'd probably keep your tone steady, but still speak up — not a speech, just a clear line that the behind-the-back talk is not okay.",
                    "My guess is you'd aim for calm wording while still closing the loop: you noticed the talk, and you want it handled face-to-face.",
                )
        answer = opts[_stable_index(f"{phrase_seed}:ctend", len(opts))]
        if af == "both":
            if clar >= peace + 0.1:
                act_m = (
                    "You'd probably push for a direct talk and shut down sideways rumors — facts first, then a clean close.",
                    "My read is you'd treat it as a logistics problem for an honest conversation, not something to keep whispering about.",
                )
            else:
                act_m = (
                    "You'd probably close the loop in person with a simple, steady move — no big performance, just ending the sideways part.",
                    "My guess is you'd handle it face-to-face and keep the energy contained while still being clear.",
                )
            answer = _merge_action_wording_paragraphs(
                act_m[_stable_index(f"{phrase_seed}:ctend_a", len(act_m))],
                answer,
                seed=f"{phrase_seed}:ctend_m",
            )
        reasoning = (
            "No same-shape conflict save; check-in tendencies on file nudge toward how you'd phrase this. "
            "Thin evidence — I am not treating this as certain."
        )
        return answer, reasoning, 0.37, extra_basis, ()

    if (
        cues["passive_slight"]
        and not gossip
        and feedback_influence
        and float(feedback_influence.direct_calm_signal) >= 0.25
    ):
        if af == "action":
            opts = (
                "You'd probably address it directly but calmly — one clear example, no pile-on, and space for them to respond.",
                "My read is you'd name the passive shot in plain words, keep your voice steady, and move it toward a direct fix.",
            )
        else:
            opts = (
                "You'd probably say you noticed the sideways jabs, that you want it straight between you, and keep the tone calm.",
                "My read is you'd keep it short and forward — direct words, low heat — because the goal is clarity, not winning a performance.",
            )
        answer = opts[_stable_index(f"{phrase_seed}:cpass_fb", len(opts))]
        if af == "both":
            act_m = (
                "You'd probably take one concrete moment, say what it looked like from your side, and ask for direct talk instead of digs.",
                "My read is you'd keep it work-appropriate but firm — not a lecture — just ending the passive loop.",
            )
            answer = _merge_action_wording_paragraphs(
                act_m[_stable_index(f"{phrase_seed}:cpass_fb_a", len(act_m))],
                answer,
                seed=f"{phrase_seed}:cpass_fb_m",
            )
        reasoning = (
            "No tight conflict save on file; your recent corrections on this kind of ask point toward calm, direct addressing."
        )
        return answer, reasoning, 0.36, extra_basis, ()

    if cues["passive_slight"] and not gossip:
        if af == "action":
            opts = (
                "You'd probably address the indirect piece head-on — name one specific moment in plain words instead of hinting wider.",
                "My read is you'd stop treating the snipes as background noise and say what you noticed, once, calmly.",
            )
        else:
            opts = (
                "You'd probably keep it short: what they did, that the sideways shots hurt, and that you want it straight.",
                "My read is you'd sound matter-of-fact — no long diagnosis — just naming the dig and asking for direct talk.",
            )
        answer = opts[_stable_index(f"{phrase_seed}:cpass", len(opts))]
        if af == "both":
            act_m = (
                "You'd probably move on a single concrete example — not a pattern lecture — so they cannot dodge as easily.",
                "My read is you'd pick one beat to pin down rather than stacking every past slight.",
            )
            answer = _merge_action_wording_paragraphs(
                act_m[_stable_index(f"{phrase_seed}:cpass_a", len(act_m))],
                answer,
                seed=f"{phrase_seed}:cpass_m",
            )
        reasoning = (
            "No tight conflict save on file; this follows passive-aggressive-shaped wording in your prompt, not a stored scenario. "
            "Useful guess only — confidence stays low."
        )
        return answer, reasoning, 0.36, extra_basis, ()

    if cues["boundary_push"] and not gossip:
        if af == "action":
            opts = (
                "You'd probably spell out the limit and what you do if they lean on it again — smaller words, same line each time.",
                "My read is you'd treat it like upkeep: repeat the boundary without turning it into a big performance.",
            )
        else:
            opts = (
                "You'd probably say what is not okay, what you need them to stop, and that you are done re-explaining.",
                "My read is you'd keep the wording steady — the same short sentence — instead of inventing a new speech each round.",
            )
        answer = opts[_stable_index(f"{phrase_seed}:cbpush", len(opts))]
        if af == "both":
            act_m = (
                "You'd probably narrow your availability while you hold the line — actions backing the sentence, not extra debate.",
                "My read is you'd pair fewer openings with the same clean limit until the pattern shifts.",
            )
            answer = _merge_action_wording_paragraphs(
                act_m[_stable_index(f"{phrase_seed}:cbpush_a", len(act_m))],
                answer,
                seed=f"{phrase_seed}:cbpush_m",
            )
        reasoning = (
            "No same-shape save; boundary-push cues in what you typed steer this read. Thin evidence — not a quote from your saves."
        )
        return answer, reasoning, 0.36, extra_basis, ()

    if cues["repeat_pattern"] and cues["direct_blunt"] and not gossip:
        if af == "action":
            opts = (
                "You'd probably stop treating repeat rudeness like a fluke — same short correction each time until it changes or you pull back.",
                "My read is you'd switch from hoping it stops to naming it as a repeat and setting what you do next.",
            )
        else:
            opts = (
                "You'd probably say you have said this before, name the behavior again, and tell them it cannot keep landing.",
                "My read is you'd sound calm but flat — not louder — because the issue is the repeat, not one sharp moment.",
            )
        answer = opts[_stable_index(f"{phrase_seed}:crepdir", len(opts))]
        if af == "both":
            act_m = (
                "You'd probably shrink contact or scope if the blunt disrespect keeps showing up after you name it.",
                "My read is you'd protect your week — fewer hooks for the same insult loop.",
            )
            answer = _merge_action_wording_paragraphs(
                act_m[_stable_index(f"{phrase_seed}:crepdir_a", len(act_m))],
                answer,
                seed=f"{phrase_seed}:crepdir_m",
            )
        reasoning = (
            "No tight save; repeat + direct insult cues in your prompt point this direction. Guess only — confidence stays low."
        )
        return answer, reasoning, 0.35, extra_basis, ()

    fit = profile_memory_fit_score(CONFLICT_FAMILY, prompt_norm)
    clause = _conflict_profile_tone_clause(profile_risk, communication_style)
    if clause and fit >= 0.38:
        if af == "action":
            opts = (
                "You'd probably show up direct without turning it into a scene — say it once, steady, no big performance.",
                "You'd probably handle it head-on and skip the extra packaging around it.",
            )
        else:
            opts = (
                "You'd probably open by naming what you heard, keep your voice steady, and ask for a face-to-face talk.",
                "You'd probably say the behind-the-back part stops here and you want it handled person-to-person, not as chatter.",
            )
        answer = opts[_stable_index(f"{phrase_seed}:cprof", len(opts))]
        if af == "both":
            act_p = (
                "You'd probably handle it head-on without polishing too much — plain intent, low drama.",
                "You'd probably pick the direct channel and keep the heat down while you still say the thing.",
            )
            answer = _merge_action_wording_paragraphs(
                act_p[_stable_index(f"{phrase_seed}:cprof_a", len(act_p))],
                answer,
                seed=f"{phrase_seed}:cprof_m",
            )
        reasoning = (
            "No tight gossip-or-conflict save; this leans on your overall communication pattern from past answers, "
            "not one labeled situation. Indirect only — confidence stays moderate or low."
        )
        reasoning += f" Profile tendency (not a quote): {clause}."
        return answer, reasoning, 0.35, extra_basis, ()

    if gossip and cues["repeat_pattern"]:
        if af == "action":
            opts = (
                "You'd probably treat repeat sideways talk like a pattern — face-to-face, what keeps showing up, and that it stops here.",
                "My read is you'd say what you heard plainly, close the rumor loop, and name that it keeps happening — not like one weird week.",
            )
        else:
            opts = (
                "You'd probably say you have heard versions more than once, keep your voice flat, and ask for direct talk instead of chatter.",
                "My read is you'd sound plain and stop the sideways version — repeat talk is different from one stray comment.",
            )
    elif gossip:
        if af == "action":
            opts = (
                "You'd probably address it directly in person: name what you heard, keep your footing, and shut down the behind-the-back piece.",
                "My read is you'd keep it short and forward — not a long fight — but you would not act like you didn't notice the sideways talk.",
            )
        else:
            opts = (
                "You'd probably address it directly: name what you heard, keep your voice steady, and say you want the behind-the-back talk to stop.",
                "My read is you'd go short and clear — not a long fight — but you would not pretend you did not notice people talking about you behind your back.",
            )
    elif cues["timing_later"] and not cues["timing_now"]:
        if af == "action":
            opts = (
                "You'd probably wait until you have a steady minute, then have the direct talk you already know you need.",
                "My read is you'd time it on purpose — not to dodge, just so the words land the way you mean them.",
            )
        else:
            opts = (
                "You'd probably say less in the heat and more once you can keep it to one clean sentence.",
                "My read is your wording would be calmer with a pause — same point, less static on the line.",
            )
    else:
        if af == "action":
            opts = (
                "You'd probably move toward an honest resolution without dragging the whole crew into it.",
                "My read is you'd keep the lane clean: clear, direct, enough to fix the tension without a pile-on.",
            )
        else:
            opts = (
                "You'd probably say your piece plainly and look for a clean resolution, without dragging lots of people into it.",
                "My read is you'd keep it honest and direct — enough to clear the air without turning it into a pile-on.",
            )
    answer = opts[_stable_index(f"{phrase_seed}:cbase", len(opts))]
    if af == "both" and gossip:
        act_g = (
            "You'd probably take it face-to-face, name what you heard, and end the sideways talk instead of feeding it.",
            "My read is you'd close the loop directly — steady, factual, no pretending it didn't happen.",
        )
        answer = _merge_action_wording_paragraphs(
            act_g[_stable_index(f"{phrase_seed}:cbase_ag", len(act_g))],
            answer,
            seed=f"{phrase_seed}:cbase_mg",
        )
    elif af == "both" and not gossip:
        act_o = (
            "You'd probably handle it straight-on and aim for a clean finish without extra players.",
            "My read is you'd pick the low-drama path that still clears the air.",
        )
        answer = _merge_action_wording_paragraphs(
            act_o[_stable_index(f"{phrase_seed}:cbase_ao", len(act_o))],
            answer,
            seed=f"{phrase_seed}:cbase_mo",
        )
    reasoning = (
        "No same-shape conflict decision on file; this is a cautious pattern read for this kind of situation, not a quote from your saves."
    )
    return answer, reasoning, 0.34, extra_basis, ()


def _strict_spending_pressure_evidence_fallback(
    *,
    phrase_seed: str,
    dims: Mapping[str, float],
    rel_facts: Sequence[Mapping[str, Any]],
    cross_boost: float,
    profile_risk: Optional[str],
    store: DatabaseStore,
    answer_focus: str = "both",
) -> Tuple[str, str, float, List[str], Tuple[str, ...]]:
    """Bills-first / pause-want / trim-spend read when strict spending gate has no row.

    Last tuple: clarification slot keys that were cited in the basis (if any).
    """
    extra_basis: List[str] = []
    nv = float(dims.get("need_vs_want_signal", 0) or 0)

    opts_core = [
        (
            "You'd probably triage rent first, put the big purchase on pause, and avoid making the money squeeze worse — "
            "if it is mostly a want, you'd likely shrink the spend instead of locking in the full price."
        ),
        (
            "My read is you'd hold off on the buy until bills are stable, keep the hole from getting deeper, "
            "and treat something like a laptop as optional until the late-rent situation is handled."
        ),
    ]
    if nv >= 0.52:
        opts_core.append(
            "You'd likely split true needs from wants for right now — cover the roof first, then see if a cheaper option "
            "or more time still works once rent is back on track."
        )
    idx = _stable_index(f"{phrase_seed}:spendfb", len(opts_core))
    answer = opts_core[idx]
    # Phase 48.1: ``opts_core`` already states rent-first / pause-want triage in full.
    # Appending talk-track lines duplicated the same gist with a mismatched “out loud”
    # tone; keep one strong practical surface for all answer_focus values here.

    reasoning = (
        "No same-shape money save matched this prompt, so I am not leaning on quick-vs-careful work habits — "
        "this tracks bills pressure plus a big want, with moderate-to-low confidence."
    )
    if profile_risk and "cautious" in profile_risk.lower():
        reasoning += " Your past answers also skew cautious, which fits waiting out the squeeze."

    conf_cap = 0.39
    if rel_facts and cross_boost >= 0.05:
        conf_cap = min(0.40, conf_cap + 0.02)

    if (
        rel_facts
        and cross_boost >= 0.06
        and (not hasattr(store, "should_surface_memory_line") or store.should_surface_memory_line(
            f"respond_mb_clarif_{rel_facts[0].get('slot_key') or 'x'}_sf"
        ))
    ):
        slot_k = str(rel_facts[0].get("slot_key") or "x")
        xvar = (
            "A recent bills-or-spend check-in you gave lines up with prioritizing rent before extras.",
            "Notes from a past clarification match handling wants after essentials are stable.",
        )
        extra_basis.append(xvar[_stable_index(f"{phrase_seed}:sfbx", len(xvar))])
        if hasattr(store, "record_memory_line_surface"):
            try:
                store.record_memory_line_surface(f"respond_mb_clarif_{slot_k}_sf")
            except Exception:
                pass
        clarif_slots = (slot_k,) if slot_k and slot_k != "x" else ()
    else:
        clarif_slots = ()

    return answer, reasoning, conf_cap, extra_basis, clarif_slots


def _natural_likely_line(
    choice: str,
    why: str,
    *,
    primary_family: str,
    seed: str,
    cautious: bool,
    blunt: float,
    aggressive_short: bool,
    utterance_mode: str = "wording",
) -> str:
    """utterance_mode: ``wording`` (how you'd say it) or ``action`` (what you'd do)."""
    fam = _family_label(primary_family)
    choice = (choice or "").strip()
    why_low = _to_lower_start(why)
    um = (utterance_mode or "wording").strip().lower()
    if um not in ("wording", "action"):
        um = "wording"
    cautious_openers = (
        "If I'm reading your saves right, ",
        "From what's on file, ",
        "",
    )
    pref = cautious_openers[_stable_index(f"{seed}:cp", len(cautious_openers))] if cautious else ""

    if um == "action":
        ch_low = _to_lower_start(choice)
        cg = _action_mode_choice_gloss(choice) if choice else ch_low
        yfrag = f" — mostly because {why_low}" if why_low else ""
        if fam == "conflict":
            if choice:
                variants = (
                    f"{pref}you'd probably handle this head-on — {cg}{yfrag}.",
                    f"{pref}you'd probably step in directly and keep it to one clean move — {cg}{yfrag}.",
                )
            else:
                variants = (
                    f"{pref}you'd probably address it straight-on without letting it fester.",
                    f"{pref}you'd probably move toward a direct fix instead of dodging it.",
                )
        elif fam == "overload":
            if choice:
                variants = (
                    f"{pref}you'd probably draw the line here — {cg}{yfrag}.",
                    f"{pref}you'd probably protect your bandwidth and hold to that limit — {cg}{yfrag}.",
                )
            else:
                variants = (
                    f"{pref}you'd probably set a boundary that protects your bandwidth.",
                    f"{pref}you'd probably offer a smaller yes or a clear no based on energy.",
                )
        elif fam == "timing":
            if choice:
                variants = (
                    f"{pref}you'd probably take the pause-first move — {cg}{yfrag}.",
                    f"{pref}you'd probably wait for a cleaner beat instead of forcing it — {cg}{yfrag}.",
                )
            else:
                variants = (
                    f"{pref}you'd probably pause, check one more fact, then decide.",
                    f"{pref}you'd probably avoid rushing until the tradeoffs are clearer.",
                )
        else:
            if choice:
                variants = (
                    f"{pref}you'd probably {cg}{yfrag}.",
                    f"{pref}you'd probably lean toward {cg}{yfrag}.",
                )
            else:
                variants = (
                    f"{pref}you'd probably lean practical and pick the low-regret move.",
                    f"{pref}you'd probably pick the option that stings least if it goes wrong.",
                )
    elif fam == "conflict":
        if choice:
            variants = (
                f"{pref}\"{choice}.\" — that's about how it comes out.",
                f"{pref}I put it like this: \"{choice}.\"",
            )
        else:
            variants = (
                f"{pref}you'd probably address it directly, but keep it calm.",
                f"{pref}you'd probably say your piece clearly instead of sitting on it.",
            )
    elif fam == "overload":
        if choice:
            variants = (
                f"{pref}you'd probably set a limit and say: \"{choice}.\"",
                f"{pref}you'd probably protect your energy and go with: \"{choice}.\"",
            )
        else:
            variants = (
                f"{pref}you'd probably set a boundary and keep it simple.",
                f"{pref}you'd probably give a smaller yes or a clear no, based on energy.",
            )
    elif fam == "timing":
        if choice:
            variants = (
                f"{pref}you'd probably take the safer timing and go with: \"{choice}.\"",
                f"{pref}you'd probably wait for one more signal, then do: \"{choice}.\"",
            )
        else:
            variants = (
                f"{pref}you'd probably pause, check one more fact, then decide.",
                f"{pref}you'd probably avoid rushing and decide after a quick check.",
            )
    else:
        if choice:
            variants = (
                f"{pref}you'd probably go with: \"{choice}.\"",
                f"{pref}you'd probably choose: \"{choice}.\"",
            )
        else:
            variants = (
                f"{pref}you'd probably lean practical and pick the low-regret move.",
                f"{pref}you'd probably choose the option that costs the least if it turns out wrong.",
            )

    answer = variants[_stable_index(f"{seed}:main:{um}:{fam}", len(variants))]
    if why_low and um != "action":
        if blunt >= 0.62:
            answer += f" {why_low}."
        else:
            answer += f" Mostly because {why_low}."
    answer = answer.replace("..", ".").strip()
    if not answer.endswith("."):
        answer += "."
    return _shorten_sentence(answer, aggressive_short)


@dataclass(frozen=True)
class RespondEvidencePath:
    """Which stored rows / routes shaped this likely-you line (Phase 38)."""

    decision_ids: Tuple[str, ...] = ()
    style_ids: Tuple[str, ...] = ()
    route_keys: Tuple[str, ...] = ()
    clarif_slot_keys: Tuple[str, ...] = ()
    answer_focus: str = "both"

    def to_storage_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "decision_ids": list(self.decision_ids),
            "style_ids": list(self.style_ids),
            "route_keys": list(self.route_keys),
            "clarif_slot_keys": list(self.clarif_slot_keys),
        }
        af = (self.answer_focus or "").strip().lower()
        if af in ("action", "wording", "both"):
            d["answer_focus"] = af
        return d


@dataclass
class PersonalResponse:
    likely_answer: str
    reasoning_brief: str
    confidence: float
    confidence_label: str
    memory_basis: List[str] = field(default_factory=list)
    profile_hint: Optional[str] = None
    evidence_path: RespondEvidencePath = field(default_factory=RespondEvidencePath)
    prompt_norm_hash: str = ""
    effective_family: str = "general"
    answer_focus: str = "both"
    # Temporary Phase 44 observability (set only when ``debug_phase44_carryover``).
    phase44_carryover_debug: Optional[Dict[str, Any]] = None


def generate_personal_response(
    scenario_text: str,
    store: DatabaseStore,
    decision_fetch_limit: int = 800,
    style_fetch_limit: int = 800,
    *,
    debug_phase44_carryover: bool = False,
) -> PersonalResponse:
    """Build a likely-you answer using stored memory and aggregated profile."""
    text = (scenario_text or "").strip()
    d_rows = store.get_recent_decision_memory(limit=decision_fetch_limit)
    s_rows = store.get_recent_style_memory(limit=style_fetch_limit)
    profile = build_personal_profile_from_rows(d_rows, s_rows)

    prompt_norm = normalize_input(text)
    answer_focus = classify_answer_focus(prompt_norm)
    phrase_seed = hashlib.sha256(normalize_input(text).encode("utf-8")).hexdigest()[:24]
    prompt_norm_hash = hashlib.sha256(prompt_norm.encode("utf-8")).hexdigest()

    ordered_pf, _ = rank_families(prompt_norm)
    primary_pf = ordered_pf[0][0] if ordered_pf else "general"
    eff_pf = effective_primary_for_cross_filter(primary_pf, prompt_norm)
    strict_shape = _respond_strict_decision_shape_prompt(prompt_norm, eff_pf)
    dims_for_prompt = score_dimensions(prompt_norm)
    money_pressure_prompt = _spending_pressure_prompt(dims_for_prompt, prompt_norm)

    situation_carryover: Optional[Dict[str, Any]] = None
    try:
        fn_best = getattr(store, "find_situation_carryover_for_ask", None)
        if callable(fn_best):
            situation_carryover = fn_best(prompt_norm, prompt_norm_hash)
        elif hasattr(store, "find_situation_carryover"):
            carry_shape_lookup = carryover_shape_key(
                prompt_norm, str(eff_pf or "general")
            )
            situation_carryover = store.find_situation_carryover(
                prompt_norm,
                prompt_norm_hash,
                str(eff_pf or "general"),
                carry_shape_lookup,
            )
    except Exception:
        situation_carryover = None
    carry_pre = situation_carryover
    carry_strength_pre = (
        float(carry_pre["strength"]) if carry_pre else 0.0
    )
    situation_carryover = _respond_situation_carryover_effective_family_aligned(
        str(eff_pf or "general"),
        situation_carryover,
        prompt_norm=prompt_norm,
        carry_strength=carry_strength_pre,
    )
    carry_strength = (
        float(situation_carryover["strength"])
        if situation_carryover
        else 0.0
    )
    carry_suppress_avoidance = _respond_carryover_suppress_avoidance(
        situation_carryover, prompt_norm
    )
    phase45_escalation = _evaluate_phase45_conflict_escalation(
        eff_pf=str(eff_pf or "general"),
        prompt_norm=prompt_norm,
        prompt_norm_hash=prompt_norm_hash,
        situation_carryover=situation_carryover,
        carry_strength=carry_strength,
    )
    repeated_pa_escalation = bool(
        phase45_escalation.get("phase45_escalation_active")
        and phase45_escalation.get("phase45_escalation_subtype")
        == "passive_aggressive_repeat"
    )
    carry_boost = min(0.11, carry_strength * 0.086) if carry_strength >= 0.38 else 0.0

    try:
        mmap: Mapping[str, float] = store.get_respond_evidence_multiplier_map()
    except Exception:
        mmap = {}

    feedback_influence = build_respond_feedback_influence(
        store, prompt_norm_hash, str(eff_pf or "general")
    )
    feedback_influence, phase46_feedback_merge_info = (
        phase46_merge_carryover_feedback_influence(
            store,
            feedback_influence,
            prompt_norm=prompt_norm,
            prompt_norm_hash=prompt_norm_hash,
            eff_pf=str(eff_pf or "general"),
            situation_carryover=situation_carryover,
            carry_strength=carry_strength,
        )
    )
    ex_route_hints: List[str] = ["strong_decision", "medium_decision"]
    if strict_shape and eff_pf == CONFLICT_FAMILY:
        ex_route_hints.append("strict_conflict_fallback")
    if strict_shape and eff_pf == SPENDING:
        ex_route_hints.append("strict_spending_fallback")
    example_influence = _build_example_influence(
        store,
        effective_family=str(eff_pf or "general"),
        answer_focus=answer_focus,
        route_hints=ex_route_hints,
    )

    d_bias = 1.09 if answer_focus == "action" else (0.91 if answer_focus == "wording" else 1.0)
    s_bias = 1.09 if answer_focus == "wording" else (0.91 if answer_focus == "action" else 1.0)
    d_ranked = retrieve_relevant_decision_memories(
        d_rows,
        text,
        top_k=8,
        min_score=0.38,
        evidence_mult_map=mmap,
        score_bias=d_bias,
        feedback_influence=feedback_influence,
        example_influence=example_influence,
    )
    if phase45_escalation.get("phase45_boost_direct_boundary_retrieval"):
        d_ranked = _apply_phase45_escalation_retrieval_boost(d_ranked)
    s_ranked = retrieve_relevant_style_memories(
        s_rows,
        text,
        top_k=4,
        evidence_mult_map=mmap,
        score_bias=s_bias,
        feedback_influence=feedback_influence,
        example_influence=example_influence,
    )

    d_gated = [
        (r, s, rs)
        for r, s, rs in d_ranked
        if respond_main_decision_passes_shape_gate(
            prompt_norm, r, effective_primary=eff_pf
        )
    ]
    if public_audience_disrespect_prompt(prompt_norm):
        d_gated = [
            (r, s, rs)
            for r, s, rs in d_gated
            if not _memory_blob_avoidance_hit(decision_row_text_blob(r))
        ]
    strip_avoidance_decisions = (
        carry_suppress_avoidance
        or phase45_escalation.get("phase45_escalation_active", False)
    )
    if strip_avoidance_decisions:
        kept_av = [
            (r, s, rs)
            for r, s, rs in d_gated
            if not _memory_blob_avoidance_hit(decision_row_text_blob(r))
        ]
        if kept_av:
            d_gated = kept_av
        else:
            # Do not keep a lone avoidance-shaped save when Phase 44 cues demand
            # demotion but no non-avoidance row survives filtering.
            d_gated = []
    top_d = d_gated[0] if d_gated else None
    top_score = top_d[1] if top_d else 0.0
    top_family_aligned = (
        top_d is not None
        and personal_response_decision_families_aligned(prompt_norm, top_d[0])
    )
    spending_quote_is_speed_trap = (
        top_d is not None
        and eff_pf == SPENDING
        and money_pressure_prompt
        and _speed_quality_trap_choice(
            str(top_d[0].get("choice_label") or ""),
            str(top_d[0].get("reasoning_label") or ""),
        )
    )

    agreement_boost = 0.0
    memory_basis: List[str] = []
    if d_gated:
        top_tags = set(top_d[0].get("value_tags") or [])
        agree = sum(
            1
            for r, sc, _ in d_gated[1:4]
            if sc > 0.4
            and top_tags & set(r.get("value_tags") or [])
            and respond_main_decision_passes_shape_gate(
                prompt_norm, r, effective_primary=eff_pf
            )
        )
        agreement_boost = min(1.0, agree * 0.34)
    try:
        raw_facts = store.get_recent_router_situation_facts(limit=28)
    except Exception:
        raw_facts = []
    try:
        tmap = store.get_router_tendency_map()
    except Exception:
        tmap = {}
    rel_facts = filter_relevant_situation_facts(
        raw_facts, primary_family=eff_pf, prompt_norm=prompt_norm
    )
    cross_boost = clarification_cross_evidence_boost(
        prompt_norm, eff_pf, rel_facts, tmap
    )
    if int(feedback_influence.wrong_replacement_count or 0) >= 2:
        cross_boost *= 0.52
    if cross_boost > 0:
        agreement_boost = min(1.0, agreement_boost + cross_boost)
    if carry_boost > 0:
        agreement_boost = min(1.0, agreement_boost + carry_boost)

    conf_pre = _compute_response_confidence(
        profile,
        top_score,
        top_d[0] if top_d else None,
        agreement_boost,
        decision_family_aligned=top_family_aligned,
        path_multipliers=None,
        example_influence=example_influence,
        feedback_direction_mixed=float(
            feedback_influence.replacement_direction_mixed or 0.0
        ),
    )

    verb = _verbosity_from_profile(profile)
    blunt = _bluntness_from_profile(profile)
    aggressive_short = verb < 0.38

    profile_hint_parts: List[str] = []
    dr = profile.decision_risk_summary()
    cs = profile.communication_style_summary()
    if dr:
        profile_hint_parts.append(dr)
    if cs:
        profile_hint_parts.append(cs)
    profile_hint = "; ".join(profile_hint_parts) if profile_hint_parts else None

    # Memory basis (plain, one scenario + one style max when scores justify it)
    seen_sid = set()
    for row, sc, _ in d_gated:
        if sc < 0.35:
            continue
        if (
            eff_pf == SPENDING
            and money_pressure_prompt
            and _speed_quality_trap_choice(
                str(row.get("choice_label") or ""),
                str(row.get("reasoning_label") or ""),
            )
        ):
            continue
        if not personal_response_decision_families_aligned(prompt_norm, row):
            continue
        sid = str(row.get("scenario_id") or "").strip() or "scenario"
        if sid in seen_sid:
            continue
        mb_key = f"respond_mb_decision_{row.get('id') or sid or 'x'}"
        if hasattr(store, "should_surface_memory_line") and not store.should_surface_memory_line(
            mb_key
        ):
            continue
        seen_sid.add(sid)
        snippet = (row.get("scenario_text") or "")[:52].strip()
        if snippet:
            ellip = "…" if len(row.get("scenario_text") or "") > 52 else ""
            tmpl = (
                'Past decision you saved (“{snippet}{ellip}”, id {sid})',
                'Closest saved decision (“{snippet}{ellip}”, {sid})',
                'Saved choice that lined up (“{snippet}{ellip}”, {sid})',
            )
            memory_basis.append(
                tmpl[_stable_index(f"{phrase_seed}:mbd", len(tmpl))].format(
                    snippet=snippet, ellip=ellip, sid=sid
                )
            )
        else:
            memory_basis.append(
                f"Saved decision entry ({sid})"
            )
        if hasattr(store, "record_memory_line_surface"):
            try:
                store.record_memory_line_surface(mb_key)
            except Exception:
                pass
        if len(memory_basis) >= 1:
            break

    style_ok_for_basis = (not strict_shape) or (top_d is not None)
    seen_pid = set()
    memory_basis_style_id: Optional[str] = None
    for row, sc, _ in s_ranked:
        if sc < 0.32:
            continue
        pid = str(row.get("prompt_id") or "").strip() or "style"
        if pid in seen_pid:
            continue
        if not style_ok_for_basis:
            break
        sk = f"respond_mb_style_{row.get('id') or pid or 'x'}"
        if hasattr(store, "should_surface_memory_line") and not store.should_surface_memory_line(sk):
            continue
        seen_pid.add(pid)
        stmpl = (
            "Saved style pick ({pid})",
            "Earlier style answer ({pid})",
        )
        memory_basis.append(
            stmpl[_stable_index(f"{phrase_seed}:mbs", len(stmpl))].format(pid=pid)
        )
        if hasattr(store, "record_memory_line_surface"):
            try:
                store.record_memory_line_surface(sk)
            except Exception:
                pass
        memory_basis_style_id = str(row.get("id") or "").strip() or None
        break

    if (
        rel_facts
        and cross_boost >= 0.07
        and len(memory_basis) < 3
    ):
        slot_k = str(rel_facts[0].get("slot_key") or "x")
        mb_cross = f"respond_mb_clarif_{slot_k}"
        if not hasattr(store, "should_surface_memory_line") or store.should_surface_memory_line(
            mb_cross
        ):
            xvar = (
                "A recent clarification you gave on a similar kind of question lines up with this one.",
                "Notes from a past decision check-in match the shape of what you’re asking now.",
            )
            memory_basis.append(xvar[_stable_index(f"{phrase_seed}:mbx", len(xvar))])
            if hasattr(store, "record_memory_line_surface"):
                try:
                    store.record_memory_line_surface(mb_cross)
                except Exception:
                    pass

    # Raw retrieval scores are typically ~0.4–6+ (overlap-weighted), not 0–1.
    strong_threshold = 2.55 if strict_shape else 2.05
    weak_or_cross = (
        (not top_family_aligned)
        or top_score < (2.85 if strict_shape else 2.35)
        or conf_pre < 0.52
        or (strict_shape and top_score < 2.35)
    )

    evidence_path = RespondEvidencePath(answer_focus=answer_focus)
    conf_caps: List[float] = []

    # Compose answer
    if spending_quote_is_speed_trap:
        ans, reas, ccap, extra_mb, clar_slots = (
            _strict_spending_pressure_evidence_fallback(
                phrase_seed=phrase_seed,
                dims=dims_for_prompt,
                rel_facts=rel_facts,
                cross_boost=cross_boost,
                profile_risk=dr,
                store=store,
                answer_focus=answer_focus,
            )
        )
        answer = ans
        reasoning = reas
        conf_caps.append(ccap)
        memory_basis.extend(extra_mb)
        evidence_path = RespondEvidencePath(
            route_keys=("spending_speed_trap_fallback",),
            clarif_slot_keys=clar_slots,
            answer_focus=answer_focus,
        )
    elif (
        top_d
        and top_score >= strong_threshold
        and top_family_aligned
        and top_d[0].get("correction_status") != "not_really"
    ):
        row = top_d[0]
        choice = str(row.get("choice_label") or "").strip()
        why = str(row.get("reasoning_label") or "").strip()
        if answer_focus == "both":
            if (
                phase45_escalation.get("phase45_escalation_subtype") == "boundary_push_repeat"
                and str(eff_pf or "").strip().lower() == OBLIGATION_OVERLOAD
            ):
                answer = _obligation_boundary_repeat_both_coherent(
                    choice,
                    why,
                    phrase_seed=f"{phrase_seed}:sd_obbr",
                    cautious=weak_or_cross,
                    blunt=blunt,
                    aggressive_short=aggressive_short,
                )
            else:
                act = _both_mode_action_line(
                    choice,
                    why,
                    primary_family=eff_pf,
                    seed=f"{phrase_seed}:sd_a",
                    cautious=weak_or_cross,
                    blunt=blunt,
                    aggressive_short=aggressive_short,
                )
                wrd = _natural_likely_line(
                    choice,
                    why,
                    primary_family=eff_pf,
                    seed=f"{phrase_seed}:sd_w",
                    cautious=weak_or_cross,
                    blunt=blunt,
                    aggressive_short=aggressive_short,
                    utterance_mode="wording",
                )
                answer = _merge_action_wording_paragraphs(
                    act, wrd, seed=f"{phrase_seed}:sd_m"
                )
        else:
            answer = _natural_likely_line(
                choice,
                why,
                primary_family=eff_pf,
                seed=phrase_seed,
                cautious=weak_or_cross,
                blunt=blunt,
                aggressive_short=aggressive_short,
                utterance_mode="action" if answer_focus == "action" else "wording",
            )
        reasoning = (
            "Closest same-family entry supports this read; your calibration adds a little color."
        )
        if cross_boost >= 0.09:
            reasoning += " Recent same-theme check-ins add a little support."
        if agreement_boost >= 0.3:
            reasoning += " A few entries repeat the same values."
        if profile.has_trait_conflict:
            reasoning += " Your past answers also pull in different directions, so this is only a rough read."
            conf_caps.append(0.62)
        if strict_shape:
            conf_caps.append(0.82)
        did = str(row.get("id") or "").strip()
        clar_slots2: Tuple[str, ...] = ()
        if cross_boost >= 0.09 and rel_facts:
            sk0 = str(rel_facts[0].get("slot_key") or "").strip()
            if sk0:
                clar_slots2 = (sk0,)
        sid_t = (
            (memory_basis_style_id,)
            if memory_basis_style_id
            else ()
        )
        evidence_path = RespondEvidencePath(
            decision_ids=(did,) if did else (),
            style_ids=sid_t,
            route_keys=("strong_decision",),
            clarif_slot_keys=clar_slots2,
            answer_focus=answer_focus,
        )
    elif (
        top_d
        and top_score >= 0.45
        and top_score < strong_threshold
        and top_family_aligned
        and top_d[0].get("correction_status") != "not_really"
    ):
        row = top_d[0]
        choice = str(row.get("choice_label") or "").strip()
        why = str(row.get("reasoning_label") or "").strip()
        if answer_focus == "both":
            if (
                phase45_escalation.get("phase45_escalation_subtype") == "boundary_push_repeat"
                and str(eff_pf or "").strip().lower() == OBLIGATION_OVERLOAD
            ):
                answer = _obligation_boundary_repeat_both_coherent(
                    choice,
                    why,
                    phrase_seed=f"{phrase_seed}:md_obbr",
                    cautious=True,
                    blunt=blunt,
                    aggressive_short=aggressive_short,
                )
            else:
                act = _both_mode_action_line(
                    choice,
                    why,
                    primary_family=eff_pf,
                    seed=f"{phrase_seed}:md_a",
                    cautious=True,
                    blunt=blunt,
                    aggressive_short=aggressive_short,
                )
                wrd = _natural_likely_line(
                    choice,
                    why,
                    primary_family=eff_pf,
                    seed=f"{phrase_seed}:md_w",
                    cautious=True,
                    blunt=blunt,
                    aggressive_short=aggressive_short,
                    utterance_mode="wording",
                )
                answer = _merge_action_wording_paragraphs(
                    act, wrd, seed=f"{phrase_seed}:md_m"
                )
        else:
            answer = _natural_likely_line(
                choice,
                why,
                primary_family=eff_pf,
                seed=phrase_seed,
                cautious=True,
                blunt=blunt,
                aggressive_short=aggressive_short,
                utterance_mode="action" if answer_focus == "action" else "wording",
            )
        reasoning = (
            "There is a related save, but the match is only medium — treat this as a soft read, not a lock."
        )
        if cross_boost >= 0.09:
            reasoning += " Check-ins on file nudge the same direction a little."
        conf_caps.append(0.48 if strict_shape else 0.55)
        if profile.has_trait_conflict:
            conf_caps.append(0.4)
        did = str(row.get("id") or "").strip()
        clar_slots2 = ()
        if cross_boost >= 0.09 and rel_facts:
            sk0 = str(rel_facts[0].get("slot_key") or "").strip()
            if sk0:
                clar_slots2 = (sk0,)
        sid_t = (
            (memory_basis_style_id,)
            if memory_basis_style_id
            else ()
        )
        evidence_path = RespondEvidencePath(
            decision_ids=(did,) if did else (),
            style_ids=sid_t,
            route_keys=("medium_decision",),
            clarif_slot_keys=clar_slots2,
            answer_focus=answer_focus,
        )
    elif (
        not strict_shape
        and not top_d
        and d_ranked
        and d_ranked[0][1] >= 0.38
        and not personal_response_decision_families_aligned(prompt_norm, d_ranked[0][0])
        and d_ranked[0][0].get("correction_status") != "not_really"
    ):
        v = (
            (
                "The labeled saves on file aren’t really the same kind of situation as this one, "
                "so I wouldn’t treat any single past choice as your answer here."
            ),
            (
                "Nothing in your saved decisions matches this shape cleanly — I’d ignore the closest-looking save "
                "for now and think from this moment instead."
            ),
        )
        answer = v[_stable_index(phrase_seed + ":misalign_ungated", len(v))]
        reasoning = (
            "Closest entries are for a different problem family than this prompt, so I’m not mirroring them."
        )
        conf_caps.append(0.36)
        cid = str(d_ranked[0][0].get("id") or "").strip()
        evidence_path = RespondEvidencePath(
            decision_ids=(cid,) if cid else (),
            route_keys=("family_misalign_ungated",),
            answer_focus=answer_focus,
        )
    elif strict_shape and not top_d and eff_pf == CONFLICT_FAMILY:
        skip_style_avoid = (
            carry_suppress_avoidance
            or phase45_escalation.get("phase45_escalation_active", False)
        )
        ans, reas, ccap, extra_mb, style_ids_fb = _strict_conflict_shape_evidence_fallback(
            prompt_norm=prompt_norm,
            phrase_seed=phrase_seed,
            s_ranked=s_ranked,
            profile_risk=dr,
            communication_style=cs,
            tmap=tmap,
            blunt=blunt,
            aggressive_short=aggressive_short,
            store=store,
            answer_focus=answer_focus,
            feedback_influence=feedback_influence,
            skip_avoidance_style_memory=skip_style_avoid,
            phase45_escalation_subtype=str(
                phase45_escalation.get("phase45_escalation_subtype") or ""
            ),
        )
        answer = ans
        reasoning = reas
        conf_caps.append(ccap)
        memory_basis.extend(extra_mb)
        evidence_path = RespondEvidencePath(
            style_ids=style_ids_fb,
            route_keys=("strict_conflict_fallback",),
            answer_focus=answer_focus,
        )
    elif strict_shape and not top_d and eff_pf == SPENDING:
        ans, reas, ccap, extra_mb, clar_slots_sp = (
            _strict_spending_pressure_evidence_fallback(
                phrase_seed=phrase_seed,
                dims=dims_for_prompt,
                rel_facts=rel_facts,
                cross_boost=cross_boost,
                profile_risk=dr,
                store=store,
                answer_focus=answer_focus,
            )
        )
        answer = ans
        reasoning = reas
        conf_caps.append(ccap)
        memory_basis.extend(extra_mb)
        evidence_path = RespondEvidencePath(
            route_keys=("strict_spending_fallback",),
            clarif_slot_keys=clar_slots_sp,
            answer_focus=answer_focus,
        )
    elif strict_shape and not top_d:
        v = (
            (
                "I don’t have a labeled save that really matches this kind of situation on file. "
                "I’d sit with it, name what you want out of it, then decide from there."
            ),
            (
                "Nothing in your saved decisions is the same shape as this ask. "
                "I wouldn’t lean on memory here — I’d go step by step with what you actually want."
            ),
        )
        answer = v[_stable_index(phrase_seed + ":strict_miss", len(v))]
        reasoning = (
            "Strict same-shape gating: no close enough decision entry, so this stays generic on purpose."
        )
        conf_caps.append(0.32)
        evidence_path = RespondEvidencePath(
            route_keys=("strict_shape_miss",),
            answer_focus=answer_focus,
        )
    elif (
        top_d
        and top_score >= 0.38
        and not top_family_aligned
        and top_d[0].get("correction_status") != "not_really"
    ):
        v = (
            (
                "The labeled saves on file aren’t really the same kind of situation as this one, "
                "so I wouldn’t treat any single past choice as your answer here."
            ),
            (
                "Nothing in your saved decisions matches this shape cleanly — I’d ignore the closest-looking save "
                "for now and think from this moment instead."
            ),
        )
        answer = v[_stable_index(phrase_seed + ":misalign", len(v))]
        reasoning = (
            "Closest entries are for a different problem family than this prompt, so I’m not mirroring them."
        )
        conf_caps.append(0.36)
        cid = str(top_d[0].get("id") or "").strip()
        evidence_path = RespondEvidencePath(
            decision_ids=(cid,) if cid else (),
            route_keys=("family_misalign",),
            answer_focus=answer_focus,
        )
    elif (
        not strict_shape
        and profile.total_evidence_weight >= 1.2
        and profile_hint
    ):
        surf_ok = not hasattr(store, "should_surface_memory_line") or store.should_surface_memory_line(
            "respond_profile_fallback"
        )
        if surf_ok:
            variants = (
                (
                    "Nothing on file fits this question tightly. I'd still slow down and grab a few more facts "
                    "before I acted on a guess about how I'd play it."
                ),
                (
                    "I can’t hook this to one past choice. I'd want more detail before I trusted a read."
                ),
                (
                    "No close match for this one. I'd fill in a couple blanks before I called any read solid."
                ),
            )
            answer = variants[_stable_index(f"{phrase_seed}:pf", len(variants))]
            reasoning = (
                "That’s from your saved style and values, not a single labeled decision."
            )
            reasoning += f" Loose tendency read: {profile_hint}."
            conf_caps.append(0.48)
            if hasattr(store, "record_memory_line_surface"):
                try:
                    store.record_memory_line_surface("respond_profile_fallback")
                except Exception:
                    pass
            evidence_path = RespondEvidencePath(
                route_keys=("profile_pattern_fallback",),
                answer_focus=answer_focus,
            )
        else:
            answer = (
                "No tight save for this question. I’d still grab one or two more facts before deciding."
            )
            reasoning = (
                "Playing it quiet — I already surfaced a profile read recently."
            )
            conf_caps.append(0.42)
            evidence_path = RespondEvidencePath(
                route_keys=("profile_pattern_fallback_suppressed",),
                answer_focus=answer_focus,
            )
    elif (
        not strict_shape
        and profile.total_evidence_weight >= 1.2
        and not profile_hint
    ):
        answer = (
            "No tight save for this question. I’d still grab one or two more facts before deciding."
        )
        reasoning = (
            "Some signal on file, but not enough to mirror this cleanly."
        )
        conf_caps.append(0.4)
        evidence_path = RespondEvidencePath(
            route_keys=("weak_profile_signal",),
            answer_focus=answer_focus,
        )
    else:
        answer = (
            "I don't have enough saved decisions or style picks to say what you'd probably do here. "
            "I'd sit with it, grab a fact or two, then pick once the tradeoffs are clearer."
        )
        reasoning = (
            "Almost nothing on file matches this prompt; this is a careful generic take."
        )
        conf_caps.append(0.28)
        evidence_path = RespondEvidencePath(
            route_keys=("insufficient_evidence",),
            answer_focus=answer_focus,
        )

    answer_pre_feedback_overlay = (answer or "").strip()
    answer, reasoning = _apply_feedback_replacement_overlay(
        answer,
        reasoning,
        feedback_influence=feedback_influence,
        phrase_seed=phrase_seed,
        answer_focus=answer_focus,
        eff_pf=str(eff_pf or "general"),
    )
    _fb_mix = float(feedback_influence.replacement_direction_mixed or 0.0)
    if _fb_mix >= 0.42:
        rl = (reasoning or "").lower()
        if "disagree" not in rl and "one newer" not in rl and "one-off" not in rl:
            reasoning = (
                (reasoning or "").rstrip()
                + " One newer correction disagrees with the stronger pattern, so I'm holding this lighter."
            )
    answer, reasoning, used_example_overlay = _apply_example_overlay(
        answer,
        reasoning,
        example_influence=example_influence,
        answer_focus=answer_focus,
        phrase_seed=phrase_seed,
    )
    # Phase 46: promoted examples run after Phase 40 wrong+replacement overlay;
    # re-apply action-ok wording replacement so user-approved lines win examples.
    answer_pre_phase46_example_win = (answer or "").strip()
    if _phase46_has_action_ok_replacement_line(feedback_influence):
        answer, reasoning = _apply_feedback_replacement_overlay(
            answer,
            reasoning,
            feedback_influence=feedback_influence,
            phrase_seed=phrase_seed + ":p46ex",
            answer_focus=answer_focus,
            eff_pf=str(eff_pf or "general"),
        )
    phase46_rep_line = _phase46_resolved_replacement_line(feedback_influence)
    phase46_overlay_demoted = bool(
        phase46_rep_line
        and answer_pre_feedback_overlay
        and answer.strip() != answer_pre_feedback_overlay
        and not _answer_covers_injection_tokens(
            answer_pre_feedback_overlay, phase46_rep_line
        )
    ) or bool(
        phase46_rep_line
        and answer_pre_phase46_example_win
        and answer.strip() != answer_pre_phase46_example_win
        and not _answer_covers_injection_tokens(
            answer_pre_phase46_example_win, phase46_rep_line
        )
    )
    if used_example_overlay:
        rks = tuple(list(evidence_path.route_keys) + ["example_memory_overlay"])
        evidence_path = RespondEvidencePath(
            decision_ids=evidence_path.decision_ids,
            style_ids=evidence_path.style_ids,
            route_keys=rks,
            clarif_slot_keys=evidence_path.clarif_slot_keys,
            answer_focus=evidence_path.answer_focus,
        )
        ex_key = f"respond_mb_example_{str(eff_pf or 'general')}_{answer_focus}"
        if not hasattr(store, "should_surface_memory_line") or store.should_surface_memory_line(
            ex_key
        ):
            memory_basis.append("Saved repeated correction example")
            if hasattr(store, "record_memory_line_surface"):
                try:
                    store.record_memory_line_surface(ex_key)
                except Exception:
                    pass
    if example_influence and float(example_influence.contradiction_level or 0.0) >= 0.34:
        reasoning += " Past examples are mixed here, so confidence stays lower."
        if not used_example_overlay and "example_memory_overlay" not in evidence_path.route_keys:
            evidence_path = RespondEvidencePath(
                decision_ids=evidence_path.decision_ids,
                style_ids=evidence_path.style_ids,
                route_keys=tuple(list(evidence_path.route_keys) + ["example_conflict_mixed"]),
                clarif_slot_keys=evidence_path.clarif_slot_keys,
                answer_focus=evidence_path.answer_focus,
            )
    elif (
        example_influence
        and float(example_influence.winning_effective_strength or 0.0) >= 0.9
        and float(example_influence.contradiction_level or 0.0) <= 0.18
    ):
        reasoning += " Repeated corrections in the same shape point in one direction."

    path_m = _respond_path_multipliers(evidence_path, mmap)
    conf = _compute_response_confidence(
        profile,
        top_score,
        top_d[0] if top_d else None,
        agreement_boost,
        decision_family_aligned=top_family_aligned,
        path_multipliers=path_m or None,
        example_influence=example_influence,
        route_keys=evidence_path.route_keys,
        feedback_direction_mixed=float(
            feedback_influence.replacement_direction_mixed or 0.0
        ),
    )
    if float(feedback_influence.replacement_direction_mixed or 0.0) >= 0.38:
        conf_caps.append(0.54)
    if (
        example_influence
        and float(example_influence.contradiction_level or 0.0) >= 0.52
    ):
        conf_caps.append(0.5)
    elif (
        example_influence
        and float(example_influence.winning_effective_strength or 0.0) >= 0.95
        and float(example_influence.contradiction_level or 0.0) <= 0.14
    ):
        conf = min(0.86, conf + 0.03)
    for cap in conf_caps:
        conf = min(conf, cap)

    answer = _phase41_style_realism_pass(answer, answer_focus=answer_focus)

    _carry_used_replacement_stance = False
    answer_pre_carryover_replace = (answer or "").strip()
    if (
        situation_carryover
        and carry_strength >= float(RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN)
    ):
        _cm = situation_carryover.get("match") or {}
        _cs = (_cm.get("stance_snippet") or "").strip()
        _ch = (_cm.get("prompt_norm_hash") or "").strip()
        _cfam = (
            str(_cm.get("effective_family") or eff_pf or "general")
            .strip()
            .lower()
        )
        if _ch and len(_cs) >= 12:
            _cfb = build_respond_feedback_influence(store, _ch, _cfam)
            _rep_line = (_cfb.action_ok_wording_replacement_line or "").strip()
            _from_db = _cfb.action_ok_wording_off and len(_rep_line) >= 8
            if not _from_db and carryover_row_stance_eligible_for_phase47_corrected_merge(
                _cs, prompt_norm=prompt_norm
            ):
                _rep_line = _cs[:400]
            if len(_rep_line) >= 8:
                row_shape = str(_cm.get("shape_key") or "")
                cur_shape = carryover_shape_key(
                    prompt_norm, str(eff_pf or "general")
                )
                _gate_ok, _ = feedback_replacement_same_thread_gate(
                    prompt_norm,
                    str(_cm.get("prompt_norm") or ""),
                    prompt_norm_hash,
                    _ch,
                    cur_shape,
                    row_shape,
                    _cm,
                )
                if _gate_ok:
                    # Prefer explicit DB replacement text; else Phase 47 eligible
                    # stance_snippet (same-thread STM may hold approved wording without
                    # feedback rows for that variant hash).
                    answer = _rep_line
                    _carry_used_replacement_stance = True

    phase46_carryover_demoted = bool(
        phase46_rep_line
        and answer_pre_carryover_replace
        and (answer or "").strip() != answer_pre_carryover_replace
        and not _answer_covers_injection_tokens(
            answer_pre_carryover_replace, phase46_rep_line
        )
    )

    answer = _phase48_final_answer_polish(answer)

    ex_contra = (
        float(example_influence.contradiction_level or 0.0)
        if example_influence
        else 0.0
    )
    p49_level = _phase49_hedge_level(
        conf=conf,
        route_keys=evidence_path.route_keys,
        feedback_direction_mixed=float(
            feedback_influence.replacement_direction_mixed or 0.0
        ),
        example_contradiction=ex_contra,
    )
    if not _phase49_skip_wording_tightening(
        answer,
        carry_used_replacement_stance=_carry_used_replacement_stance,
        feedback_influence=feedback_influence,
    ):
        answer = _phase49_wording_strength_polish(answer, hedge_level=p49_level)

    continuity_note = ""
    reasoning_line_audit = _respond_carryover_reasoning_line_audit(
        situation_carryover,
        prompt_norm,
        carry_strength,
        prompt_norm_hash,
        str(eff_pf or "general"),
    )
    allow_carry_line = bool(reasoning_line_audit["allowed"])
    if allow_carry_line and situation_carryover and carry_strength >= float(
        RESPOND_CARRYOVER_NOTE_HIGH_MIN
    ):
        cvars = (
            " This looks like the same issue continuing from a recent thread.",
            " This lines up with a recent unresolved situation you were in.",
        )
        continuity_note = cvars[_stable_index(phrase_seed + ":p44c", len(cvars))]
    elif allow_carry_line and situation_carryover and carry_strength >= float(
        RESPOND_CARRYOVER_NOTE_SOFT_MIN
    ):
        cvars2 = (
            " A recent similar thread may still be open for you.",
            " This may connect to something you were just working through.",
        )
        continuity_note = cvars2[_stable_index(phrase_seed + ":p44d", len(cvars2))]
    elif allow_carry_line and situation_carryover and carry_strength >= float(
        RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN
    ):
        cvars3 = (
            " This still looks connected to a recent situation for you.",
            " There is some recent thread context that still applies here.",
        )
        continuity_note = cvars3[_stable_index(phrase_seed + ":p44e", len(cvars3))]
    if continuity_note:
        reasoning = (reasoning or "").rstrip() + continuity_note

    stm_record_skip_reason = respond_route_keys_skip_short_term_situation_record(
        evidence_path.route_keys
    )
    la0_stm = (answer or "").strip().split("\n")[0].strip()
    default_st_snip = (
        la0_stm[:200] if len(la0_stm) > 20 else (reasoning or "")[:200]
    )
    short_term_stance_mode = "surface_first_line"
    planned_st_snip = default_st_snip
    resolved_stm_rep = (_phase46_resolved_replacement_line(feedback_influence) or "").strip()
    # Phase 47: for wording-off feedback, always prefer the resolved user-approved /
    # merged replacement line for STM — not the wrong+replacement inject line alone,
    # which can differ and self-poison replayed stance rows.
    if feedback_influence.action_ok_wording_off:
        if len(resolved_stm_rep) >= 8:
            planned_st_snip = resolved_stm_rep[:220]
            short_term_stance_mode = "replay_resolved_user_approved_replacement"
        else:
            planned_st_snip = carryover_safe_stance_fallback(
                prompt_norm, str(eff_pf or "general")
            )[:200]
            short_term_stance_mode = "carryover_safe_after_wording_feedback"

    rows_for_carryover_debug: Optional[List[Dict[str, Any]]] = None
    if debug_phase44_carryover:
        try:
            if hasattr(store, "list_recent_short_term_situations"):
                rows_for_carryover_debug = store.list_recent_short_term_situations(
                    limit=40
                )
        except Exception:
            rows_for_carryover_debug = []

    replay_recording_mode_dbg = ""
    replay_recording_block_reason_dbg = ""
    replay_protected_from_self_poisoning_dbg = False
    stm_replay_dedupe_touch = False
    stm_row_inserted = False
    try:
        rec = getattr(store, "record_short_term_situation", None)
        if not callable(rec):
            replay_recording_mode_dbg = "no_record_method"
        elif stm_record_skip_reason is not None:
            replay_recording_mode_dbg = "skipped_low_information_route"
            replay_recording_block_reason_dbg = stm_record_skip_reason or ""
        else:
            sk_rec = combined_shape_key(
                prompt_norm,
                str(eff_pf or "general"),
                evidence_path.to_storage_dict(),
            )
            skip_stm_insert = False
            get_latest = getattr(
                store, "get_latest_unresolved_short_term_situation_row", None
            )
            touch_stm = getattr(store, "touch_short_term_situation_updated_at", None)
            if callable(get_latest) and callable(touch_stm):
                prev_stm = get_latest(
                    prompt_norm_hash=prompt_norm_hash,
                    effective_family=str(eff_pf or "general"),
                )
                if prev_stm:
                    prev_st = (prev_stm.get("stance_snippet") or "").strip()
                    plan_st = (planned_st_snip or "").strip()
                    if prev_st and prev_st == plan_st:
                        rid = str(prev_stm.get("id") or "").strip()
                        if rid and touch_stm(rid):
                            skip_stm_insert = True
                            stm_replay_dedupe_touch = True
                            replay_recording_mode_dbg = "replay_refreshed_existing_row"
                            replay_protected_from_self_poisoning_dbg = True
            if not skip_stm_insert:
                rec(
                    prompt_norm=prompt_norm,
                    prompt_norm_hash=prompt_norm_hash,
                    effective_family=str(eff_pf or "general"),
                    shape_key=sk_rec,
                    stance_snippet=planned_st_snip,
                    source="respond_like_me",
                )
                stm_row_inserted = True
                if not replay_recording_mode_dbg:
                    replay_recording_mode_dbg = "inserted_new_row"
    except Exception:
        replay_recording_mode_dbg = replay_recording_mode_dbg or "record_error_skipped"
        replay_recording_block_reason_dbg = (
            replay_recording_block_reason_dbg or "exception_during_short_term_record"
        )

    phase44_carryover_debug: Optional[Dict[str, Any]] = None
    if debug_phase44_carryover:
        rows_dbg: List[Dict[str, Any]] = list(rows_for_carryover_debug or [])
        ask_diag = diagnose_ask_carryover_candidates(
            prompt_norm, prompt_norm_hash, rows_dbg
        )
        cfam_pre = ""
        if carry_pre and carry_pre.get("match"):
            cfam_pre = str(
                carry_pre.get("carry_family")
                or carry_pre["match"].get("effective_family")
                or ""
            ).strip().lower()
        cont_tier = "none"
        if allow_carry_line and situation_carryover and carry_strength >= float(
            RESPOND_CARRYOVER_NOTE_HIGH_MIN
        ):
            cont_tier = "high_template"
        elif allow_carry_line and situation_carryover and carry_strength >= float(
            RESPOND_CARRYOVER_NOTE_SOFT_MIN
        ):
            cont_tier = "soft_template"
        elif allow_carry_line and situation_carryover and carry_strength >= float(
            RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN
        ):
            cont_tier = "very_soft_template"
        mid = ""
        if situation_carryover and situation_carryover.get("match"):
            mid = str(situation_carryover["match"].get("id") or "")
        p46_rep_dbg = _phase46_resolved_replacement_line(feedback_influence)
        p46_same_thr = _phase46_feedback_same_thread_relevant(
            feedback_influence, phase46_feedback_merge_info
        )
        fa_dbg = (answer or "").strip().lower()
        p46_sel_dbg = bool(
            p46_rep_dbg
            and (
                p46_rep_dbg.lower() in fa_dbg
                or _answer_covers_injection_tokens(answer or "", p46_rep_dbg)
            )
        )
        if p46_rep_dbg:
            if p46_sel_dbg:
                p46_block_dbg = ""
            elif not p46_same_thr:
                p46_block_dbg = "replacement_not_same_thread_relevant"
            else:
                p46_block_dbg = "surface_missing_replacement_tokens_after_passes"
        else:
            p46_block_dbg = "no_user_replacement_line_available"

        _mh_dbg = ""
        if situation_carryover and situation_carryover.get("match"):
            _mh_dbg = str(
                situation_carryover["match"].get("prompt_norm_hash") or ""
            ).strip()
        _ch_dbg = (prompt_norm_hash or "").strip()
        replay_exact_prompt_match_dbg = bool(_mh_dbg and _mh_dbg == _ch_dbg)
        replay_same_thread_variant_match_dbg = bool(
            _mh_dbg
            and _ch_dbg
            and _mh_dbg != _ch_dbg
            and carry_strength >= float(RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN)
        )
        replay_used_existing_corrected_stance_dbg = bool(
            _carry_used_replacement_stance
            or (
                bool(p46_rep_dbg)
                and (
                    p46_sel_dbg
                    or (p46_rep_dbg.lower() in fa_dbg)
                    or _answer_covers_injection_tokens(answer or "", p46_rep_dbg)
                )
            )
        )

        phase44_carryover_debug = {
                "prompt_norm_prefix": (prompt_norm or "")[:160],
                "pick_best_for_ask": ask_diag,
                "family_alignment": {
                "routed_effective_family": str(eff_pf or "general").strip().lower(),
                "carry_family_from_pick": cfam_pre,
                "had_carry_pre_align": carry_pre is not None,
                "kept_after_family_gate": situation_carryover is not None,
                "dropped_by_family_align": bool(
                    carry_pre is not None and situation_carryover is None
                ),
            },
            "strength_after_family_align": round(float(carry_strength), 4),
            "active_match_row_id": mid,
            "suppress_avoidance_demotion": carry_suppress_avoidance,
            "repeated_passive_aggressive_escalation": repeated_pa_escalation,
            "strip_avoidance_decision_rows": strip_avoidance_decisions,
            "reasoning_line_audit": reasoning_line_audit,
            "continuity_language_tier": cont_tier,
            "continuity_note_emitted": bool((continuity_note or "").strip()),
            "continuity_note_text": (continuity_note or "").strip(),
            "phase45_escalation_evaluated": bool(
                phase45_escalation.get("phase45_escalation_evaluated")
            ),
            "phase45_escalation_active": bool(
                phase45_escalation.get("phase45_escalation_active")
            ),
            "phase45_escalation_subtype": str(
                phase45_escalation.get("phase45_escalation_subtype") or ""
            ),
            "phase45_escalation_reject_reasons": list(
                phase45_escalation.get("phase45_escalation_reject_reasons") or []
            ),
            "phase45_demote_avoidance_due_to_escalation": bool(
                phase45_escalation.get("phase45_demote_avoidance_due_to_escalation")
            ),
            "phase45_boost_direct_boundary_retrieval": bool(
                phase45_escalation.get("phase45_boost_direct_boundary_retrieval")
            ),
            "short_term_situation_record_skip_reason": stm_record_skip_reason or "",
            "short_term_situation_will_record": bool(
                stm_record_skip_reason is None
                and (stm_row_inserted or stm_replay_dedupe_touch)
            ),
            "short_term_stance_recording_mode": (
                short_term_stance_mode
                if stm_record_skip_reason is None
                else "skipped_no_record"
            ),
            "short_term_stance_snippet_stored_prefix": (
                (planned_st_snip or "")[:120]
                if stm_record_skip_reason is None
                else ""
            ),
            "wording_off_feedback_affects_stance_recording": bool(
                feedback_influence.action_ok_wording_off
            ),
            "surfaced_answer_used_carryover_replacement_stance": _carry_used_replacement_stance,
            "phase46_feedback_merge": dict(phase46_feedback_merge_info),
            "feedback_replacement_available": bool(len(p46_rep_dbg) >= 8),
            "feedback_replacement_same_thread_relevant": p46_same_thr,
            "feedback_replacement_selected_for_surface": p46_sel_dbg,
            "feedback_replacement_block_reason": p46_block_dbg,
            "older_phrasing_demoted_due_to_feedback": bool(
                phase46_overlay_demoted or phase46_carryover_demoted
            ),
            "replay_exact_prompt_match": replay_exact_prompt_match_dbg,
            "replay_same_thread_variant_match": replay_same_thread_variant_match_dbg,
            "replay_used_existing_corrected_stance": (
                replay_used_existing_corrected_stance_dbg
            ),
            "replay_recording_mode": replay_recording_mode_dbg,
            "replay_recording_block_reason": replay_recording_block_reason_dbg,
            "replay_protected_from_self_poisoning": (
                replay_protected_from_self_poisoning_dbg
            ),
        }

        
        
    label = _confidence_bucket(conf)
    return PersonalResponse(
        likely_answer=answer,
        reasoning_brief=reasoning,
        confidence=round(conf, 2),
        confidence_label=label,
        memory_basis=memory_basis,
        profile_hint=profile_hint,
        evidence_path=evidence_path,
        prompt_norm_hash=prompt_norm_hash,
        effective_family=str(eff_pf or "general"),
        answer_focus=answer_focus,
        phase44_carryover_debug=phase44_carryover_debug,
    )

