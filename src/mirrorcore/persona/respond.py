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
from ..decision.memory_relevance import decision_memory_relevance_multiplier
from ..decision.routed_clarification import (
    CONFLICT_FAMILY,
    SPENDING,
    rank_families,
    score_dimensions,
)
from ..router import normalize_input
from .profile import PersonalProfile, build_personal_profile_from_rows

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


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

    return score, reasons


def score_style_memory_row(
    row: Dict[str, Any],
    keywords: Sequence[str],
    prompt_counts: Mapping[str, int],
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

    return score, reasons


def retrieve_relevant_decision_memories(
    rows: Sequence[Dict[str, Any]],
    prompt: str,
    top_k: int = 5,
    min_score: float = 0.28,
) -> List[Tuple[Dict[str, Any], float, List[str]]]:
    keywords = tokenize_prompt(prompt)
    prompt_norm = normalize_input(prompt)
    scenario_counts = _count_occurrences([str(r.get("scenario_id") or "") for r in rows])
    scored: List[Tuple[Dict[str, Any], float, List[str], str, str]] = []
    for row in rows:
        s, reasons = score_decision_memory_row(row, keywords, scenario_counts)
        s *= decision_memory_relevance_multiplier(prompt_norm, row, s)
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
) -> List[Tuple[Dict[str, Any], float, List[str]]]:
    keywords = tokenize_prompt(prompt)
    prompt_counts = _count_occurrences([str(r.get("prompt_id") or "") for r in rows])
    scored: List[Tuple[Dict[str, Any], float, List[str], str, str]] = []
    for row in rows:
        s, reasons = score_style_memory_row(row, keywords, prompt_counts)
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
    if profile.total_evidence_weight < 0.85:
        base -= 0.14
    if profile.decision_entries_used == 0 and profile.style_entries_used == 0:
        base -= 0.2
    return max(0.12, min(0.9, base))


def _shorten_sentence(text: str, aggressive: bool) -> str:
    t = (text or "").strip()
    if not aggressive or len(t) < 90:
        return t
    cut = t[:87].rsplit(" ", 1)[0]
    return cut + "…"


@dataclass
class PersonalResponse:
    likely_answer: str
    reasoning_brief: str
    confidence: float
    confidence_label: str
    memory_basis: List[str] = field(default_factory=list)
    profile_hint: Optional[str] = None


def generate_personal_response(
    scenario_text: str,
    store: DatabaseStore,
    decision_fetch_limit: int = 800,
    style_fetch_limit: int = 800,
) -> PersonalResponse:
    """Build a likely-you answer using stored memory and aggregated profile."""
    text = (scenario_text or "").strip()
    d_rows = store.get_recent_decision_memory(limit=decision_fetch_limit)
    s_rows = store.get_recent_style_memory(limit=style_fetch_limit)
    profile = build_personal_profile_from_rows(d_rows, s_rows)

    d_ranked = retrieve_relevant_decision_memories(d_rows, text, top_k=6, min_score=0.35)
    s_ranked = retrieve_relevant_style_memories(s_rows, text, top_k=4)

    top_d = d_ranked[0] if d_ranked else None
    top_score = top_d[1] if top_d else 0.0
    phrase_seed = hashlib.sha256(normalize_input(text).encode("utf-8")).hexdigest()[:24]

    agreement_boost = 0.0
    memory_basis: List[str] = []
    if d_ranked:
        top_tags = set(top_d[0].get("value_tags") or [])
        agree = sum(
            1
            for r, sc, _ in d_ranked[1:4]
            if sc > 0.4 and top_tags & set(r.get("value_tags") or [])
        )
        agreement_boost = min(1.0, agree * 0.34)

    conf = _compute_response_confidence(
        profile, top_score, top_d[0] if top_d else None, agreement_boost
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

    # Memory basis lines (short, user-facing) — only when retrieval is clearly relevant
    for row, sc, _ in d_ranked[:2]:
        if sc < 0.35:
            continue
        mb_key = f"respond_mb_decision_{row.get('id') or row.get('scenario_id') or 'x'}"
        if hasattr(store, "should_surface_memory_line") and not store.should_surface_memory_line(
            mb_key
        ):
            continue
        sid = row.get("scenario_id") or "scenario"
        snippet = (row.get("scenario_text") or "")[:52].strip()
        if snippet:
            ellip = "…" if len(row.get("scenario_text") or "") > 52 else ""
            memory_basis.append(f'Decision memory (“{snippet}{ellip}”, scenario {sid})')
        else:
            memory_basis.append(f"Decision memory (scenario id {sid})")
        if hasattr(store, "record_memory_line_surface"):
            try:
                store.record_memory_line_surface(mb_key)
            except Exception:
                pass

    for row, sc, _ in s_ranked[:2]:
        if sc < 0.32:
            continue
        sk = f"respond_mb_style_{row.get('id') or row.get('prompt_id') or 'x'}"
        if hasattr(store, "should_surface_memory_line") and not store.should_surface_memory_line(sk):
            continue
        pid = row.get("prompt_id") or "style"
        memory_basis.append(f"Style memory ({pid})")
        if hasattr(store, "record_memory_line_surface"):
            try:
                store.record_memory_line_surface(sk)
            except Exception:
                pass

    # Compose answer
    if top_d and top_score >= 0.58 and top_d[0].get("correction_status") != "not_really":
        row = top_d[0]
        choice = str(row.get("choice_label") or "").strip()
        why = str(row.get("reasoning_label") or "").strip()
        if choice:
            answer_core = f"I’d probably choose: {choice}"
        else:
            answer_core = "I’d probably lean the same way as in your saved scenarios"
        if why:
            why_sent = why[0].lower() + why[1:] if len(why) > 1 else why.lower()
            if blunt >= 0.62:
                answer_core += f" — {why_sent}"
            else:
                answer_core += f". Mostly because {why_sent}"
        answer = _shorten_sentence(answer_core.rstrip("."), aggressive_short) + "."
        reasoning = (
            "Pulled from your closest matching saved decision; "
            "it lines up with how you answered similar interview questions."
        )
        if agreement_boost >= 0.3:
            reasoning += " A few saved picks share the same value tags."
        if profile.has_trait_conflict:
            reasoning += " Your saved signals don’t fully match, so treat this as a rough guess."
    elif profile.total_evidence_weight >= 1.2 and profile_hint:
        surf_ok = not hasattr(store, "should_surface_memory_line") or store.should_surface_memory_line(
            "respond_profile_fallback"
        )
        if surf_ok:
            variants = (
                (
                    "From what’s saved, I don’t have a tight match for this exact question. "
                    f"If I had to talk like you usually do, I’d keep it {profile_hint} — "
                    "but I’d want more detail before I’d commit."
                ),
                (
                    "Nothing saved lines up one-to-one with this. If I’m guessing from your usual patterns, "
                    f"you tend toward {profile_hint} — I’d still want more detail before I’d commit."
                ),
                (
                    "I can’t pin this to a single past decision. The rough read from your saves is "
                    f"{profile_hint} — but I’d want more context before I’d trust that."
                ),
            )
            answer = variants[_stable_index(f"{phrase_seed}:pf", len(variants))]
            reasoning = (
                "That read comes from your saved style and values, not one specific past decision."
            )
            conf = min(conf, 0.48)
            if hasattr(store, "record_memory_line_surface"):
                try:
                    store.record_memory_line_surface("respond_profile_fallback")
                except Exception:
                    pass
        else:
            answer = (
                "From what’s saved, I don’t have a tight match for this exact question. "
                "I’d still slow down and grab one or two more facts before I’d commit."
            )
            reasoning = (
                "That’s a cautious fallback — I’m not repeating the same profile read right now."
            )
            conf = min(conf, 0.42)
    elif profile.total_evidence_weight >= 1.2 and not profile_hint:
        answer = (
            "From what’s saved, I don’t have a tight match for this exact question. "
            "I’d still slow down and grab one or two more facts before I’d commit."
        )
        reasoning = (
            "There’s some saved signal, but not enough to mirror this cleanly."
        )
        conf = min(conf, 0.4)
    else:
        answer = (
            "I don’t have enough saved decisions or style picks to say what you’d "
            "probably do here. I’d sit with it, grab one or two more facts, then choose once the tradeoffs are clear."
        )
        reasoning = (
            "There’s little saved memory that fits this prompt; this is a careful generic take."
        )
        conf = min(conf, 0.28)

    label = _confidence_bucket(conf)
    return PersonalResponse(
        likely_answer=answer,
        reasoning_brief=reasoning,
        confidence=round(conf, 2),
        confidence_label=label,
        memory_basis=memory_basis,
        profile_hint=profile_hint,
    )

