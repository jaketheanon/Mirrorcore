"""
Phase 36: cross-system integration helpers (deterministic, no LLM).

Unifies relevance gates for decision memory, style memory, router situation
facts, and long-run clarification tendencies so ask / respond / clarification
reuse the same shape-matching rules.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..router import normalize_input
from .memory_relevance import (
    decision_memory_relevance_multiplier,
    profile_memory_fit_score,
)
from .ontology import (
    CONFLICT_FAMILY,
    GENERAL,
    LOYALTY_BOUNDARY,
    OBLIGATION_OVERLOAD,
    RISK_TIMING,
    SPENDING,
    score_dimensions,
)

# Situation keys from clarification → decision families they describe (not tendencies).
SITUATION_KEY_FAMILIES: Dict[str, frozenset] = {
    "housing_bill_pressure": frozenset({SPENDING}),
    "purchase_need_level": frozenset({SPENDING}),
    "energy_available": frozenset({OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY, GENERAL}),
    "choice_reversibility": frozenset(
        {RISK_TIMING, SPENDING, GENERAL, CONFLICT_FAMILY, LOYALTY_BOUNDARY}
    ),
}


def situation_fact_family_match(slot_key: str, primary_family: str) -> bool:
    fams = SITUATION_KEY_FAMILIES.get(slot_key)
    if not fams:
        return False
    if primary_family in fams:
        return True
    if primary_family == GENERAL and GENERAL in fams:
        return True
    return False


def effective_primary_for_cross_filter(
    ranked_top_family: str, prompt_norm: str
) -> str:
    """
    When ontology tops out as GENERAL but dimensions are sharp, use the sharper
    family for cross-system filtering so money/oblig/conflict cues still match.
    """
    dims = score_dimensions(prompt_norm)
    if ranked_top_family != GENERAL:
        return ranked_top_family
    if dims.get("money_pressure", 0) >= 1.05:
        return SPENDING
    if dims.get("obligation", 0) + dims.get("overload", 0) >= 1.15:
        return OBLIGATION_OVERLOAD
    if dims.get("interpersonal_hurt", 0) + dims.get("conflict_intensity", 0) >= 1.05:
        return CONFLICT_FAMILY
    if dims.get("wait_vs_act", 0) + dims.get("uncertainty", 0) >= 1.25:
        return RISK_TIMING
    return GENERAL


def filter_relevant_situation_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    primary_family: str,
    prompt_norm: str,
    max_facts: int = 20,
) -> List[Mapping[str, Any]]:
    """Keep only router situation rows that fit this question’s decision shape."""
    eff = effective_primary_for_cross_filter(primary_family, prompt_norm)
    dims = score_dimensions(prompt_norm)
    out: List[Mapping[str, Any]] = []
    for row in facts:
        sk = str(row.get("slot_key") or "")
        if not situation_fact_family_match(sk, eff):
            continue
        if sk == "housing_bill_pressure":
            if dims.get("money_pressure", 0) < 0.5 and not any(
                x in prompt_norm
                for x in ("rent", "bill", "afford", "pay", "buy", "money", "broke", "debt")
            ):
                continue
        if sk == "purchase_need_level":
            if dims.get("need_vs_want_signal", 0) < 0.45 and not any(
                x in prompt_norm for x in ("need", "want", "buy", "purchase", "afford")
            ):
                continue
        if sk == "energy_available":
            if dims.get("overload", 0) + dims.get("obligation", 0) < 0.65 and not any(
                x in prompt_norm
                for x in ("exhaust", "tired", "shift", "overtime", "cover", "favor", "drain")
            ):
                continue
        if sk == "choice_reversibility":
            if (
                dims.get("wait_vs_act", 0) + dims.get("uncertainty", 0) < 0.55
                and "revers" not in prompt_norm
                and "undo" not in prompt_norm
            ):
                continue
        out.append(row)
        if len(out) >= max_facts:
            break
    return out


def evidence_reuse_priority(score: float, correction_status: str) -> float:
    """Evidence strength / reuse priority for cross-system ordering (deterministic)."""
    cq = {
        "accurate": 1.15,
        "uncorrected": 1.0,
        "partially_true": 0.72,
        "not_really": 0.05,
    }.get((correction_status or "").strip(), 0.82)
    return max(0.0, float(score)) * cq


def cross_system_relevance_score_decision_row(
    row: Mapping[str, Any],
    *,
    prompt_norm: str,
    keywords: Sequence[str],
    scenario_counts: Mapping[str, int],
    evidence_mult_map: Optional[Mapping[str, float]] = None,
) -> float:
    """Single score combining raw retrieval fit and ontology-aware multiplier."""
    from ..persona.respond import score_decision_memory_row

    mmap = evidence_mult_map or {}
    rid = str(row.get("id") or "")
    em = float(mmap.get(f"decision:{rid}", 1.0))
    raw, _ = score_decision_memory_row(
        row, keywords, scenario_counts, evidence_row_mult=em
    )
    return float(raw) * decision_memory_relevance_multiplier(prompt_norm, row, raw)


def build_ask_interview_memory_line_candidates(
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    merged_norm: str,
    initial_norm: str,
    primary_family: str,
    seed: str,
    evidence_mult_map: Optional[Mapping[str, float]] = None,
) -> List[Tuple[str, float, str]]:
    """
    Plain-language lines from decision_memory for the ask / guidance tail.
    Stricter than respond-like-me: requires family fit and a strong combined score.
    """
    if not decision_rows or not merged_norm.strip():
        return []

    from ..persona.respond import _count_occurrences, tokenize_prompt

    fit = profile_memory_fit_score(primary_family, initial_norm)
    if fit < 0.42:
        return []

    keywords = tokenize_prompt(merged_norm)
    scenario_counts = _count_occurrences(
        [str(r.get("scenario_id") or "") for r in decision_rows]
    )
    prompt_norm = normalize_input(merged_norm)

    scored: List[Tuple[str, float, str, str]] = []
    for row in decision_rows:
        if (row.get("correction_status") or "") == "not_really":
            continue
        comb = cross_system_relevance_score_decision_row(
            row,
            prompt_norm=prompt_norm,
            keywords=keywords,
            scenario_counts=scenario_counts,
            evidence_mult_map=evidence_mult_map,
        )
        comb *= 0.82 + 0.18 * fit
        if comb < 0.62:
            continue
        eid = str(row.get("id") or row.get("scenario_id") or "mem")
        choice = (str(row.get("choice_label") or "")).strip()
        why = (str(row.get("reasoning_label") or "")).strip()
        if not choice:
            continue
        why_low = why[0].lower() + why[1:] if len(why) > 1 else why.lower()
        variants = (
            (
                f"ask_ivm_{eid}_a",
                comb,
                f"A saved decision interview lines up here: you picked “{choice}” — "
                f"mostly because {why_low or 'of what you said then'}. "
                f"Use it as one signal, not the whole answer.",
            ),
            (
                f"ask_ivm_{eid}_b",
                comb * 0.99,
                f"Something from your earlier decision exercises fits: you leaned toward “{choice}” "
                f"({why_low or 'your reasons then'}). Let that inform you, not decide for you.",
            ),
        )
        idx = _stable_pick(seed, f"ivm:{eid}", len(variants))
        key, sc, txt = variants[idx]
        pri = evidence_reuse_priority(sc, str(row.get("correction_status") or ""))
        scored.append((key, pri, txt, eid))

    if not scored:
        return []

    scored.sort(key=lambda x: (-x[1], x[3], x[0]))
    top = scored[0]
    return [(top[0], top[1], top[2])]


def _stable_pick(seed: str, salt: str, modulo: int) -> int:
    if modulo <= 1:
        return 0
    h = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) % modulo


def clarification_cross_evidence_boost(
    prompt_norm: str,
    primary_family: str,
    relevant_situation_facts: Sequence[Mapping[str, Any]],
    tendency_map: Optional[Mapping[str, float]],
) -> float:
    """
    Small deterministic boost when router situation facts + tendencies align
    with the same decision shape as the personal-response prompt.
    """
    boost = 0.0
    eff = effective_primary_for_cross_filter(primary_family, prompt_norm)
    dims = score_dimensions(prompt_norm)

    if relevant_situation_facts:
        boost += min(0.14, 0.055 * len(relevant_situation_facts))

    tmap = tendency_map or {}
    g = float(tmap.get("tendency_guilt_about_no", 0.0))
    if g >= 0.48 and eff in (OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY, GENERAL):
        if dims.get("obligation", 0) + dims.get("overload", 0) >= 0.72:
            boost += min(0.1, g * 0.12)

    r = float(tmap.get("tendency_regret_if_yes", 0.0))
    if r >= 0.48 and eff in (OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY):
        if dims.get("regret_risk", 0) >= 0.55 or dims.get("obligation", 0) >= 0.65:
            boost += min(0.08, r * 0.1)

    e = float(tmap.get("tendency_low_energy_guard", 0.0))
    if e >= 0.48 and eff in (OBLIGATION_OVERLOAD, GENERAL):
        if dims.get("overload", 0) >= 0.55:
            boost += min(0.08, e * 0.1)

    if eff == CONFLICT_FAMILY:
        hurt = dims.get("interpersonal_hurt", 0) + dims.get("conflict_intensity", 0)
        peace = float(tmap.get("tendency_peace_over_confrontation", 0.0))
        clar = float(tmap.get("tendency_clarity_priority", 0.0))
        if peace >= 0.48 and hurt >= 0.72:
            boost += min(0.07, peace * 0.09)
        if clar >= 0.48 and hurt >= 0.62:
            boost += min(0.06, clar * 0.08)

    return min(0.24, boost)
