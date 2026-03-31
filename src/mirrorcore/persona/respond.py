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
    decision_memory_relevance_multiplier,
    decision_row_text_blob,
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
    OBLIGATION_OVERLOAD,
    SPENDING,
    interpersonal_conflict_markers_present,
    rank_families_full,
    score_dimensions,
)
from ..decision.routed_clarification import rank_families
from ..router import normalize_input
from .profile import PersonalProfile, build_personal_profile_from_rows

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


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


@dataclass(frozen=True)
class RespondExampleInfluence:
    """Promoted reusable examples derived from repeated corrections (Phase 42)."""

    token_weight: Mapping[str, float]
    action_line: str = ""
    wording_line: str = ""
    both_line: str = ""
    strongest_strength: float = 0.0


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


def build_respond_feedback_influence(
    store: DatabaseStore,
    prompt_norm_hash: str,
    effective_family: str,
) -> RespondFeedbackInfluence:
    """Derive lexical preference + avoidance demotion from recent same-prompt feedback."""
    if not (prompt_norm_hash or "").strip():
        return RespondFeedbackInfluence({}, 0.0, 0.0, 0, "")
    if not hasattr(store, "list_personal_response_feedback_for_prompt"):
        return RespondFeedbackInfluence({}, 0.0, 0.0, 0, "")
    rows = store.list_personal_response_feedback_for_prompt(
        prompt_norm_hash, limit=40
    )
    pref: Dict[str, float] = {}
    wrong_avoid = 0
    direct_hits = 0.0
    wrong_rep_count = 0
    inject_line = ""
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
            if not inject_line:
                inject_line = rep_t[:400]
        if rep_t and rt in ("wrong", "partly"):
            add = 0.20 if rt == "wrong" else 0.10
            for tok in _tokenize_feedback_phrase(rep_t):
                pref[tok] = min(0.62, pref.get(tok, 0.0) + add)
            rl = rep_t.lower()
            if any(x in rl for x in ("address", "direct", "calm", "straightforward")):
                direct_hits = min(1.0, direct_hits + 0.32)
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
    )


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
        limit=6,
        min_strength=0.55,
    )
    if not rows:
        return RespondExampleInfluence({})
    tok: Dict[str, float] = {}
    action_line = ""
    wording_line = ""
    both_line = ""
    top = 0.0
    for r in rows:
        txt = str(r.get("example_text") or "").strip()
        et = str(r.get("example_type") or "").strip().lower()
        strength = max(0.0, min(1.0, float(r.get("strength") or 0.0)))
        if not txt:
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
    """Bias final text toward stored replacement lines after repeated wrong + replacement (Phase 40)."""
    if "conflict" not in (eff_pf or "").lower():
        return answer, reasoning
    n_wr = int(feedback_influence.wrong_replacement_count or 0)
    inj = _sanitize_replacement_for_overlay(
        feedback_influence.replacement_inject_line or ""
    )
    if n_wr < 2 or len(inj) < 8:
        return answer, reasoning
    a = (answer or "").strip()
    low = a.lower()
    covered = _answer_covers_injection_tokens(a, inj)
    avoidance_ans = _memory_blob_avoidance_hit(low) or "let it go" in low
    af = (answer_focus or "both").strip().lower()
    if not covered and (avoidance_ans or n_wr >= 3):
        if af == "both":
            act_opts = (
                "I'd step in calmer but clearer — closer to what I've been asking for on this kind of prompt.",
                "I'd handle it more directly after the corrections I've stacked on this one.",
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
            tail = " Recent feedback nudged the say-line that way."
            return merged, (rs + tail) if rs else tail.strip()
        pool = (
            f"I'd handle it more like: {inj}",
            f"I'd land here after the corrections I've given on this: {inj}",
            f"I'm pushing toward something closer to: {inj}",
        )
        new_a = pool[_stable_index(f"{phrase_seed}:fbinj", len(pool))]
        rs = (reasoning or "").rstrip()
        tail = " Recent feedback on this prompt nudges the line that way."
        return new_a, (rs + tail) if rs else tail.strip()
    if not covered:
        if af == "wording":
            b = f"I'd phrase it closer to: {inj}"
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
        opts = (
            f"I'd keep it closer to this: {line}",
            f"This lines up with what I've corrected before: {line}",
        )
        out = opts[_stable_index(f"{phrase_seed}:exov", len(opts))]
    rb = (reasoning or "").rstrip()
    tail = " This also lines up with a saved repeated correction."
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

    cues = conflict_situational_cues(prompt_norm)
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
            "You'd likely separate what you truly need from what you want right now — cover the roof first, then see if a cheaper option "
            "or more time still works once rent is back on track."
        )
    idx = _stable_index(f"{phrase_seed}:spendfb", len(opts_core))
    answer = opts_core[idx]
    af = (answer_focus or "both").strip().lower()
    if af not in ("action", "wording", "both"):
        af = "both"
    if af in ("wording", "both"):
        talk_track = (
            "The line I'd use with myself is blunt: essentials first, treat the big want like it can wait.",
            "If I said it out loud, it would sound like triage — roof and bills stable before the shiny buy.",
        )
        tt = talk_track[_stable_index(f"{phrase_seed}:spendtt", len(talk_track))]
        if af == "both":
            answer = f"{answer}\n\nIf I said it out loud, it might sound like this:\n{tt}"
        else:
            answer = answer + " " + tt

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

    try:
        mmap: Mapping[str, float] = store.get_respond_evidence_multiplier_map()
    except Exception:
        mmap = {}

    feedback_influence = build_respond_feedback_influence(
        store, prompt_norm_hash, str(eff_pf or "general")
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

    conf_pre = _compute_response_confidence(
        profile,
        top_score,
        top_d[0] if top_d else None,
        agreement_boost,
        decision_family_aligned=top_family_aligned,
        path_multipliers=None,
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

    answer, reasoning = _apply_feedback_replacement_overlay(
        answer,
        reasoning,
        feedback_influence=feedback_influence,
        phrase_seed=phrase_seed,
        answer_focus=answer_focus,
        eff_pf=str(eff_pf or "general"),
    )
    answer, reasoning, used_example_overlay = _apply_example_overlay(
        answer,
        reasoning,
        example_influence=example_influence,
        answer_focus=answer_focus,
        phrase_seed=phrase_seed,
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
    answer = _phase41_style_realism_pass(answer, answer_focus=answer_focus)

    path_m = _respond_path_multipliers(evidence_path, mmap)
    conf = _compute_response_confidence(
        profile,
        top_score,
        top_d[0] if top_d else None,
        agreement_boost,
        decision_family_aligned=top_family_aligned,
        path_multipliers=path_m or None,
    )
    for cap in conf_caps:
        conf = min(conf, cap)

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
    )

