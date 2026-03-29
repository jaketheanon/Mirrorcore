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
    decision_memory_relevance_multiplier,
    gossip_or_backchannel_user_prompt,
    personal_response_decision_families_aligned,
    profile_memory_fit_score,
    respond_main_decision_passes_shape_gate,
    style_memory_passes_respond_conflict_shape,
    style_memory_relevance_multiplier,
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
    prompt_norm = normalize_input(prompt)
    for row in rows:
        s, reasons = score_style_memory_row(row, keywords, prompt_counts)
        s *= style_memory_relevance_multiplier(prompt_norm, row, s)
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
    return max(0.12, min(0.9, base))


def _shorten_sentence(text: str, aggressive: bool) -> str:
    t = (text or "").strip()
    if not aggressive or len(t) < 90:
        return t
    cut = t[:87].rsplit(" ", 1)[0]
    return cut + "…"


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
) -> Tuple[str, str, float, List[str]]:
    """Cautious likely-you line when strict conflict gating finds no decision row."""
    extra_basis: List[str] = []
    gossip = gossip_or_backchannel_user_prompt(prompt_norm)

    for row, sc, _ in s_ranked:
        if (row.get("correction_status") or "") == "not_really":
            continue
        if sc < 0.28:
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
        )
        reasoning = (
            "No same-shape decision save on file; a style calibration that fits this conflict shape still points this way. "
            "Indirect evidence only — confidence stays low."
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
        return answer, reasoning, conf_cap, extra_basis

    peace = float(tmap.get("tendency_peace_over_confrontation", 0) or 0)
    clar = float(tmap.get("tendency_clarity_priority", 0) or 0)
    if max(peace, clar) >= 0.48:
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
        reasoning = (
            "No same-shape conflict save; check-in tendencies on file nudge toward how you'd phrase this. "
            "Thin evidence — I am not treating this as certain."
        )
        return answer, reasoning, 0.37, extra_basis

    fit = profile_memory_fit_score(CONFLICT_FAMILY, prompt_norm)
    clause = _conflict_profile_tone_clause(profile_risk, communication_style)
    if clause and fit >= 0.38:
        opts = (
            f"You'd probably say it straight and keep the heat low — on file you tend to be {clause}.",
            f"My read is you'd speak up plainly; {clause} is close to how you sound when you are being direct.",
        )
        answer = opts[_stable_index(f"{phrase_seed}:cprof", len(opts))]
        reasoning = (
            "No tight gossip-or-conflict save; this leans on your overall communication pattern from past answers, "
            "not one labeled situation. Indirect only — confidence stays moderate or low."
        )
        return answer, reasoning, 0.35, extra_basis

    if gossip:
        opts = (
            "You'd probably address it directly: name what you heard, keep your voice steady, and say you want the behind-the-back talk to stop.",
            "My read is you'd go short and clear — not a long fight — but you would not pretend you did not notice people talking about you behind your back.",
        )
    else:
        opts = (
            "You'd probably say your piece plainly and look for a clean resolution, without dragging lots of people into it.",
            "My read is you'd keep it honest and direct — enough to clear the air without turning it into a pile-on.",
        )
    answer = opts[_stable_index(f"{phrase_seed}:cbase", len(opts))]
    reasoning = (
        "No same-shape conflict decision on file; this is a cautious pattern read for this kind of situation, not a quote from your saves."
    )
    return answer, reasoning, 0.34, extra_basis


def _strict_spending_pressure_evidence_fallback(
    *,
    phrase_seed: str,
    dims: Mapping[str, float],
    rel_facts: Sequence[Mapping[str, Any]],
    cross_boost: float,
    profile_risk: Optional[str],
    store: DatabaseStore,
) -> Tuple[str, str, float, List[str]]:
    """Bills-first / pause-want / trim-spend read when strict spending gate has no row."""
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

    return answer, reasoning, conf_cap, extra_basis


def _natural_likely_line(
    choice: str,
    why: str,
    *,
    primary_family: str,
    seed: str,
    cautious: bool,
    blunt: float,
    aggressive_short: bool,
) -> str:
    fam = _family_label(primary_family)
    choice = (choice or "").strip()
    why_low = _to_lower_start(why)
    cautious_openers = (
        "If I'm reading your saves right, ",
        "From what's on file, ",
        "",
    )
    pref = cautious_openers[_stable_index(f"{seed}:cp", len(cautious_openers))] if cautious else ""

    if fam == "conflict":
        if choice:
            variants = (
                f"{pref}you'd probably say something like: \"{choice}.\"",
                f"{pref}you're most likely to say: \"{choice}.\"",
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

    answer = variants[_stable_index(f"{seed}:main:{fam}", len(variants))]
    if why_low:
        if blunt >= 0.62:
            answer += f" {why_low}."
        else:
            answer += f" Mostly because {why_low}."
    answer = answer.replace("..", ".").strip()
    if not answer.endswith("."):
        answer += "."
    return _shorten_sentence(answer, aggressive_short)


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

    prompt_norm = normalize_input(text)

    ordered_pf, _ = rank_families(prompt_norm)
    primary_pf = ordered_pf[0][0] if ordered_pf else "general"
    eff_pf = effective_primary_for_cross_filter(primary_pf, prompt_norm)
    strict_shape = _respond_strict_decision_shape_prompt(prompt_norm, eff_pf)
    dims_for_prompt = score_dimensions(prompt_norm)
    money_pressure_prompt = _spending_pressure_prompt(dims_for_prompt, prompt_norm)

    d_ranked = retrieve_relevant_decision_memories(d_rows, text, top_k=8, min_score=0.38)
    s_ranked = retrieve_relevant_style_memories(s_rows, text, top_k=4)

    d_gated = [
        (r, s, rs)
        for r, s, rs in d_ranked
        if respond_main_decision_passes_shape_gate(
            prompt_norm, r, effective_primary=eff_pf
        )
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
    phrase_seed = hashlib.sha256(normalize_input(text).encode("utf-8")).hexdigest()[:24]

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
    if cross_boost > 0:
        agreement_boost = min(1.0, agreement_boost + cross_boost)

    conf = _compute_response_confidence(
        profile,
        top_score,
        top_d[0] if top_d else None,
        agreement_boost,
        decision_family_aligned=top_family_aligned,
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
        or conf < 0.52
        or (strict_shape and top_score < 2.35)
    )

    # Compose answer
    if spending_quote_is_speed_trap:
        ans, reas, ccap, extra_mb = _strict_spending_pressure_evidence_fallback(
            phrase_seed=phrase_seed,
            dims=dims_for_prompt,
            rel_facts=rel_facts,
            cross_boost=cross_boost,
            profile_risk=dr,
            store=store,
        )
        answer = ans
        reasoning = reas
        conf = min(conf, ccap)
        memory_basis.extend(extra_mb)
    elif (
        top_d
        and top_score >= strong_threshold
        and top_family_aligned
        and top_d[0].get("correction_status") != "not_really"
    ):
        row = top_d[0]
        choice = str(row.get("choice_label") or "").strip()
        why = str(row.get("reasoning_label") or "").strip()
        answer = _natural_likely_line(
            choice,
            why,
            primary_family=eff_pf,
            seed=phrase_seed,
            cautious=weak_or_cross,
            blunt=blunt,
            aggressive_short=aggressive_short,
        )
        reasoning = (
            "Closest same-family saves point this way, plus your interview and style patterns."
        )
        if cross_boost >= 0.09:
            reasoning += " Recent same-theme check-ins add a little support."
        if agreement_boost >= 0.3:
            reasoning += " A few entries repeat the same values."
        if profile.has_trait_conflict:
            reasoning += " Your past answers also pull in different directions, so this is only a rough read."
            conf = min(conf, 0.62)
        if strict_shape:
            conf = min(conf, 0.82)
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
        answer = _natural_likely_line(
            choice,
            why,
            primary_family=eff_pf,
            seed=phrase_seed,
            cautious=True,
            blunt=blunt,
            aggressive_short=aggressive_short,
        )
        reasoning = (
            "There is a related save, but the match is only medium — treat this as a soft read, not a lock."
        )
        if cross_boost >= 0.09:
            reasoning += " Check-ins on file nudge the same direction a little."
        conf = min(conf, 0.48 if strict_shape else 0.55)
        if profile.has_trait_conflict:
            conf = min(conf, 0.4)
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
        conf = min(conf, 0.36)
    elif strict_shape and not top_d and eff_pf == CONFLICT_FAMILY:
        ans, reas, ccap, extra_mb = _strict_conflict_shape_evidence_fallback(
            prompt_norm=prompt_norm,
            phrase_seed=phrase_seed,
            s_ranked=s_ranked,
            profile_risk=dr,
            communication_style=cs,
            tmap=tmap,
            blunt=blunt,
            aggressive_short=aggressive_short,
            store=store,
        )
        answer = ans
        reasoning = reas
        conf = min(conf, ccap)
        memory_basis.extend(extra_mb)
    elif strict_shape and not top_d and eff_pf == SPENDING:
        ans, reas, ccap, extra_mb = _strict_spending_pressure_evidence_fallback(
            phrase_seed=phrase_seed,
            dims=dims_for_prompt,
            rel_facts=rel_facts,
            cross_boost=cross_boost,
            profile_risk=dr,
            store=store,
        )
        answer = ans
        reasoning = reas
        conf = min(conf, ccap)
        memory_basis.extend(extra_mb)
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
        conf = min(conf, 0.32)
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
        conf = min(conf, 0.36)
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
                    "Nothing saved fits this question tightly. If I’m winging it from your old answers, "
                    f"I’d sound {profile_hint} — I’d still want a few more facts before I stuck to that."
                ),
                (
                    "I can’t hook this to one past choice. The loose read from your saves is "
                    f"{profile_hint} — I’d want more detail before I trusted it."
                ),
                (
                    "No close save for this one. Guessing from patterns, you usually come across as "
                    f"{profile_hint} — I’d slow down and fill in blanks before I called that solid."
                ),
            )
            answer = variants[_stable_index(f"{phrase_seed}:pf", len(variants))]
            reasoning = (
                "That’s from your saved style and values, not a single labeled decision."
            )
            conf = min(conf, 0.48)
            if hasattr(store, "record_memory_line_surface"):
                try:
                    store.record_memory_line_surface("respond_profile_fallback")
                except Exception:
                    pass
        else:
            answer = (
                "No tight save for this question. I’d still grab one or two more facts before deciding."
            )
            reasoning = (
                "Playing it quiet — I already surfaced a profile read recently."
            )
            conf = min(conf, 0.42)
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
        conf = min(conf, 0.4)
    else:
        answer = (
            "I don't have enough saved decisions or style picks to say what you'd probably do here. "
            "I'd sit with it, grab a fact or two, then pick once the tradeoffs are clearer."
        )
        reasoning = (
            "Almost nothing on file matches this prompt; this is a careful generic take."
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

