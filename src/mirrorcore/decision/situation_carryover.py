"""
Phase 44 — deterministic short-term situation continuation (session carryover).

Not durable tendency memory: only recent, explicit rows in ``short_term_situation_memory``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .ontology import GENERAL, slot_ids_covered_by_context
from ..router import normalize_input

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "to",
        "of",
        "in",
        "on",
        "for",
        "i",
        "im",
        "ive",
        "id",
        "me",
        "my",
        "you",
        "your",
        "we",
        "they",
        "them",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "this",
        "that",
        "with",
        "as",
        "at",
        "so",
        "do",
        "does",
        "did",
        "dont",
        "not",
        "no",
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "would",
        "could",
        "should",
        "can",
        "cant",
        "about",
        "into",
        "from",
        "just",
        "really",
        "like",
        "get",
        "got",
        "out",
        "up",
        "down",
        "all",
        "any",
        "some",
        "more",
        "most",
        "very",
        "also",
        "then",
        "than",
        "too",
        "here",
        "there",
        "again",
        "still",
        "something",
        "anything",
        "everything",
        "nothing",
        "someone",
        "anyone",
    }
)

# Timing / stance clash — shrink carryover when the new prompt pushes the opposite direction.
TIMING_WAIT = frozenset(
    {"wait", "later", "tomorrow", "slow", "sleep", "pause", "hold", "delay", "week"}
)
TIMING_NOW = frozenset(
    {"now", "today", "asap", "urgent", "immediately", "tonight", "right", "soon"}
)

# Lexical cues that the user is continuing the same situation (not starting a timing-only ask).
_REFERENCE_CONTINUATION_NEEDLES: Tuple[str, ...] = (
    " same ",
    " again ",
    " another ",
    " keeps ",
    " keep ",
    " this same ",
    "this same ",
    " same passive",
    "same passive",
    " keeps doing it",
    " keep doing it",
    " keeps doing ",
    " keep doing ",
    "same coworker",
    "same person",
    "same guy",
    "same girl",
    "same boss",
    "still pushing",
    "still asking",
    "still trying",
    "still doing",
    " still deciding",
    "still deciding",
)


def reference_continuation_cues(norm: str) -> bool:
    """True when phrasing points at continuing an ongoing thread (deterministic)."""
    padded = f" {(norm or '').strip().lower()} "
    return any(n in padded for n in _REFERENCE_CONTINUATION_NEEDLES)
CONFLICT_AVOID = frozenset(
    {"ignore", "let", "drop", "avoid", "quiet", "ride", "pass"}
)
CONFLICT_FACE = frozenset(
    {"address", "confront", "speak", "tell", "boundary", "say", "call"}
)


def _stance_snippet_avoidance_leans(snippet: str) -> bool:
    low = (snippet or "").lower()
    needles = (
        "let it go",
        "let it slide",
        "move on",
        "ignore it",
        "just ignore",
        "drop it",
        "let it ride",
        "walk away",
        "rise above",
        "not worth",
    )
    return any(n in low for n in needles)


def passive_aggressive_friction_shaped(norm: str) -> bool:
    """Prompt reads as passive-aggressive / sideways interpersonal friction (deterministic)."""
    low = (norm or "").lower()
    pad = f" {low} "
    return (
        "passive aggressive" in low
        or "passive-aggressive" in low
        or " snide " in pad
        or "snide " in low
        or "backhanded" in low
        or "underhanded" in low
        or "sideways" in low
        or " cold shoulder" in low
        or "cold shoulder" in low
        or "undermin" in low
        or " backhanded" in pad
        or " digs " in pad
        or " dig at" in low
        or "digs at" in low
        or " petty " in pad
        or " indirect " in pad
    )


def passive_aggressive_thread_overlap(cur_norm: str, row_norm: str) -> bool:
    """Both prompts read as passive-aggressive / sideways-friction shaped (deterministic)."""
    return passive_aggressive_friction_shaped(
        cur_norm
    ) and passive_aggressive_friction_shaped(row_norm)


def significant_tokens(norm: str) -> FrozenSet[str]:
    raw = (norm or "").lower()
    out: List[str] = []
    cur: List[str] = []
    for ch in raw:
        if ch.isalnum() or ch == "'":
            cur.append(ch)
        else:
            if cur:
                w = "".join(cur).strip("'")
                if len(w) >= 3 and w not in STOPWORDS:
                    out.append(w)
                cur = []
    if cur:
        w = "".join(cur).strip("'")
        if len(w) >= 3 and w not in STOPWORDS:
            out.append(w)
    return frozenset(out)


def _row_soft_interpersonal_pa(norm: str) -> bool:
    """Work-adjacent row with softer sideways-friction wording (not always 'passive aggressive')."""
    low = (norm or "").lower()
    if not any(
        x in low
        for x in (
            "work",
            "coworker",
            "colleague",
            "teammate",
            "boss",
            "office",
            "job",
        )
    ):
        return False
    return any(
        x in low
        for x in (
            "sideways",
            "snide",
            "digs",
            "dig ",
            "slight",
            "undermin",
            "petty",
            "indirect",
            "backhand",
            "cold shoulder",
            "passive",
            "aggressive",
            "remarks",
            "jabs",
        )
    )


def pa_issue_carryover_bridge(cur_norm: str, row_norm: str) -> bool:
    """
    Same-thread PA continuation when the new prompt is explicit PA + continuation cues
    but the stored row used softer wording (still work + friction).
    """
    if passive_aggressive_thread_overlap(cur_norm, row_norm):
        return False
    if not passive_aggressive_friction_shaped(cur_norm):
        return False
    if not reference_continuation_cues(cur_norm):
        return False
    inter = len(significant_tokens(cur_norm) & significant_tokens(row_norm))
    if inter < 2:
        return False
    if passive_aggressive_friction_shaped(row_norm):
        return True
    return _row_soft_interpersonal_pa(row_norm)


def pa_carryover_aligned(cur_norm: str, row_norm: str) -> bool:
    """PA-shaped same issue: strict overlap or continuation bridge (deterministic)."""
    return passive_aggressive_thread_overlap(cur_norm, row_norm) or pa_issue_carryover_bridge(
        cur_norm, row_norm
    )


def carryover_shape_key(prompt_norm: str, effective_family: str) -> str:
    """Family + covered clarification slots — same prompt shape for ask and respond."""
    fam = (effective_family or "general").strip().lower() or "general"
    covered = sorted(slot_ids_covered_by_context(prompt_norm))[:10]
    return f"{fam}|slots:{','.join(covered)}"


def carryover_slots_prefix(shape_key: str) -> str:
    """Family + slot list only (strip route/evidence tail from stored combined keys)."""
    sk = (shape_key or "").strip()
    if not sk:
        return "|slots:"
    fam = sk.split("|", 1)[0]
    if "|slots:" not in sk:
        return f"{fam}|slots:"
    i = sk.index("|slots:")
    rest = sk[i + len("|slots:") :]
    j = rest.find("|")
    slots_val = rest if j < 0 else rest[:j]
    return f"{fam}|slots:{slots_val}"


def short_term_row_recent_enough(
    row: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
    max_hours: float = 42.0,
) -> bool:
    """True when the situation row is fresh enough for user-facing continuity claims."""
    updated = str(row.get("updated_at") or row.get("created_at") or "")
    return _hours_old(updated, now or datetime.utcnow()) <= max_hours


def respond_evidence_shape_key(evidence_dict: Dict[str, Any]) -> str:
    """Route/slot fingerprint after respond (aligns with Phase 42 shape idea)."""
    routes = sorted(
        [
            str(x).strip().lower()
            for x in (evidence_dict.get("route_keys") or [])
            if str(x).strip()
        ]
    )[:2]
    slots = sorted(
        [
            str(x).strip().lower()
            for x in (evidence_dict.get("clarif_slot_keys") or [])
            if str(x).strip()
        ]
    )[:1]
    rk = "|".join(routes) if routes else "none"
    sk = "|".join(slots) if slots else "none"
    return f"r:{rk};c:{sk}"


def combined_shape_key(
    prompt_norm: str, effective_family: str, evidence_dict: Optional[Dict[str, Any]]
) -> str:
    base = carryover_shape_key(prompt_norm, effective_family)
    if evidence_dict:
        return f"{base}|{respond_evidence_shape_key(evidence_dict)}"
    return base


def _parse_ts(iso: str) -> Optional[datetime]:
    if not iso:
        return None
    try:
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def _hours_old(updated_at: str, now: datetime) -> float:
    ts = _parse_ts(updated_at)
    if not ts:
        return 999.0
    n = now
    if ts.tzinfo and n.tzinfo is None:
        n = n.replace(tzinfo=ts.tzinfo)
    elif n.tzinfo and ts.tzinfo is None:
        ts = ts.replace(tzinfo=n.tzinfo)
    return max(0.0, (n - ts).total_seconds() / 3600.0)


def _stance_text_for_timing_compare(stance_snippet: str) -> str:
    """Drop boundary idioms that contain ``hold`` but are not wait-vs-act timing."""
    s = (stance_snippet or "").lower()
    for phrase in (
        "hold the line",
        "hold your boundary",
        "holding your boundary",
        "hold your ground",
        "holding your ground",
    ):
        s = s.replace(phrase, " ")
    return s


def timing_contradiction_multiplier(cur_norm: str, stance_snippet: str) -> float:
    pn = significant_tokens(cur_norm)
    st = significant_tokens(_stance_text_for_timing_compare(stance_snippet))
    w_cur = bool(pn & TIMING_WAIT)
    n_cur = bool(pn & TIMING_NOW)
    w_st = bool(st & TIMING_WAIT)
    n_st = bool(st & TIMING_NOW)
    if (w_cur and n_st) or (n_cur and w_st):
        return 0.32
    return 1.0


def conflict_stance_multiplier(cur_norm: str, stance_snippet: str) -> float:
    pn = significant_tokens(cur_norm)
    st = significant_tokens((stance_snippet or "").lower())
    av_cur = bool(pn & CONFLICT_AVOID)
    fc_cur = bool(pn & CONFLICT_FACE)
    av_st = bool(st & CONFLICT_AVOID)
    fc_st = bool(st & CONFLICT_FACE)
    if (av_cur and fc_st) or (fc_cur and av_st):
        return 0.38
    return 1.0


def continuation_strength(
    cur_norm: str,
    cur_hash: str,
    cur_family: str,
    cur_shape: str,
    row: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> float:
    now = now or datetime.utcnow()
    st = (row.get("state") or "").strip().lower()
    if st != "unresolved":
        return 0.0
    updated = str(row.get("updated_at") or row.get("created_at") or "")
    hours = _hours_old(updated, now)
    if hours > 72.0:
        return 0.0
    rf = str(row.get("effective_family") or "").strip().lower() or "general"
    cf = str(cur_family or "").strip().lower() or "general"
    if rf != cf:
        return 0.0
    row_norm = str(row.get("prompt_norm") or "")
    row_hash = str(row.get("prompt_norm_hash") or "")
    row_shape = str(row.get("shape_key") or "")
    row_tokens = significant_tokens(row_norm)
    cur_tokens = significant_tokens(cur_norm)
    inter = 0
    ref_ok = False
    if cur_hash and row_hash == cur_hash:
        base = 1.0
        ref_ok = True
    else:
        if not cur_tokens or not row_tokens:
            return 0.0
        inter = len(cur_tokens & row_tokens)
        uni = len(cur_tokens | row_tokens)
        jacc = inter / uni if uni else 0.0
        pa_overlap_gate = passive_aggressive_thread_overlap(cur_norm, row_norm)
        pa_bridge_gate = pa_issue_carryover_bridge(cur_norm, row_norm)
        ref_ok = (
            reference_continuation_cues(cur_norm)
            and inter >= 2
            and (jacc >= 0.17 or pa_overlap_gate or pa_bridge_gate)
        )
        if inter < 3 and jacc < 0.34:
            if not ref_ok:
                return 0.0
        if jacc < 0.24:
            if not ref_ok:
                return 0.0
        base = min(1.0, 0.33 + jacc * 1.18 + min(0.38, inter * 0.045))
        if ref_ok and inter >= 2:
            base = min(1.0, base + 0.09 + min(0.12, (inter - 2) * 0.04))
    stance_early = str(row.get("stance_snippet") or "")
    pn_tokens = significant_tokens(cur_norm)
    pa_align = pa_carryover_aligned(cur_norm, row_norm)
    wants_face_script = bool(pn_tokens & CONFLICT_FACE)
    ct = cf.lower()
    if ct == "conflict" and reference_continuation_cues(cur_norm):
        if (
            pa_align
            and wants_face_script
            and _stance_snippet_avoidance_leans(stance_early)
        ):
            base *= 0.40
        elif (
            pa_align
            and wants_face_script
            and not _stance_snippet_avoidance_leans(stance_early)
        ):
            base = min(1.0, base + 0.13)
        elif ref_ok and pa_align and inter >= 2:
            base = min(1.0, base + 0.08)
    # Shape: same family is not enough; mismatch weakens unless match was very strong.
    if row_shape and cur_shape and row_shape != cur_shape:
        rs_head = row_shape.split("|", 1)[0]
        cs_head = cur_shape.split("|", 1)[0]
        if rs_head != cs_head:
            base *= 0.52 if base < 0.75 else 0.68
        elif carryover_slots_prefix(row_shape) == carryover_slots_prefix(cur_shape):
            base *= 0.96
        else:
            base *= 0.82
    rec = max(0.38, 1.0 - (hours / 90.0))
    base *= rec
    stance = str(row.get("stance_snippet") or "")
    base *= timing_contradiction_multiplier(cur_norm, stance)
    csm = conflict_stance_multiplier(cur_norm, stance)
    if (
        csm < 1.0
        and ct == "conflict"
        and reference_continuation_cues(cur_norm)
        and pa_carryover_aligned(cur_norm, row_norm)
        and bool(significant_tokens(cur_norm) & CONFLICT_FACE)
    ):
        csm = max(csm, 0.90)
    base *= csm
    return max(0.0, min(1.0, base))


def pick_best_carryover(
    cur_norm: str,
    cur_hash: str,
    cur_family: str,
    cur_shape: str,
    rows: Sequence[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> Tuple[Optional[Dict[str, Any]], float]:
    best: Optional[Dict[str, Any]] = None
    best_s = 0.0
    for row in rows:
        d = dict(row) if not isinstance(row, dict) else row
        s = continuation_strength(
            cur_norm, cur_hash, cur_family, cur_shape, d, now=now
        )
        if s > best_s:
            best_s = s
            best = d
    if best_s < 0.25:
        return None, 0.0
    return best, best_s


def pick_best_carryover_for_ask(
    cur_norm: str,
    cur_hash: str,
    rows: Sequence[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """
    Best unresolved continuation for ``mirrorcore ask``: score each row using *that*
    row's family + shape (Phase 44).  Unlike ``pick_best_carryover``, this does not
    require the ask flow's preliminary primary family to match the stored row.
    """
    best: Optional[Dict[str, Any]] = None
    best_s = 0.0
    best_fam = GENERAL
    for row in rows:
        d = dict(row) if not isinstance(row, dict) else row
        fam = str(d.get("effective_family") or "").strip().lower() or GENERAL
        sk = carryover_shape_key(cur_norm, fam)
        s = continuation_strength(cur_norm, cur_hash, fam, sk, d, now=now)
        if s > best_s:
            best_s = s
            best = d
            best_fam = fam
    if best_s < 0.25 or best is None:
        return None, 0.0, GENERAL
    return best, best_s, best_fam


def diagnose_ask_carryover_candidates(
    cur_norm: str,
    cur_hash: str,
    rows: Sequence[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    pick_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Temporary Phase 44 observability: per-row scores using the same rules as
    ``pick_best_carryover_for_ask`` (deterministic, no I/O).
    """
    now = now or datetime.utcnow()
    thr = float(pick_threshold)
    scored: List[Dict[str, Any]] = []
    best_id: str = ""
    best_s: float = -1.0
    for row in rows:
        d = dict(row) if not isinstance(row, dict) else row
        fam = str(d.get("effective_family") or "").strip().lower() or GENERAL
        sk = carryover_shape_key(cur_norm, fam)
        s = continuation_strength(cur_norm, cur_hash, fam, sk, d, now=now)
        rid = str(d.get("id") or "").strip()
        updated = str(d.get("updated_at") or d.get("created_at") or "")
        hours = _hours_old(updated, now)
        st = str(d.get("state") or "").strip().lower()
        hints: List[str] = []
        if st != "unresolved":
            hints.append("state_not_unresolved")
        if hours > 72.0:
            hints.append("row_updated_gt_72h")
        row_norm = str(d.get("prompt_norm") or "")
        row_h = str(d.get("prompt_norm_hash") or "")
        if not (cur_hash and row_h == cur_hash):
            ct = significant_tokens(cur_norm)
            rt = significant_tokens(row_norm)
            if not ct or not rt:
                hints.append("no_token_overlap_path_empty_tokens")
        if s <= 0.0 and not hints:
            hints.append("continuation_strength_zero_other")
        if s > best_s:
            best_s = s
            best_id = rid
        scored.append(
            {
                "row_id": rid,
                "effective_family": fam,
                "stored_shape_key": str(d.get("shape_key") or ""),
                "scoring_shape_key": sk,
                "updated_at": updated,
                "hours_old": round(hours, 4),
                "state": st,
                "score": round(float(s), 4),
                "prompt_norm_prefix": (row_norm or "")[:120],
                "stance_snippet_prefix": (str(d.get("stance_snippet") or ""))[:120],
                "zero_score_hints": hints,
            }
        )
    accepted = best_s >= thr and bool(best_id)
    for item in scored:
        rid = item["row_id"]
        is_best = bool(rid and rid == best_id)
        item["is_best_scoring"] = is_best
        item["accepted_as_carryover"] = bool(accepted and is_best)
        reasons: List[str] = []
        if not is_best:
            reasons.append("not_highest_scoring_candidate")
        elif not accepted:
            reasons.append(f"best_score_below_pick_threshold({thr})")
        item["reject_reasons"] = reasons
    scored.sort(key=lambda x: (-float(x["score"]), str(x["row_id"])))
    return {
        "pick_threshold": thr,
        "candidate_count": len(scored),
        "best_row_id": best_id if accepted else "",
        "best_score": round(float(best_s), 4) if accepted else 0.0,
        "any_candidate_meets_threshold": accepted,
        "candidates": scored,
    }


def truncate_prompt_norm(text: str, max_len: int = 480) -> str:
    n = normalize_input(text)
    if len(n) <= max_len:
        return n
    return n[:max_len]
