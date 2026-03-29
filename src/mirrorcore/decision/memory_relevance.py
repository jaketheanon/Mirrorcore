"""
Phase 34: shared deterministic relevance helpers for decision memory and
profile snippets. Used by routed guidance and personal response retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .ontology import (
    AXIS_BACKCHANNEL_HURT,
    AXIS_BILLS_FINANCIAL_PRESSURE,
    AXIS_CONFLICT_CONFRONTATION,
    AXIS_HELPING_FAVOR,
    AXIS_LOYALTY_VS_SELF,
    AXIS_MONEY_SPENDING,
    AXIS_NEED_VS_WANT,
    AXIS_OVERLOAD_BURNOUT,
    AXIS_TIMING_WAIT_VS_ACT,
    AXIS_UNCERTAINTY_RISK,
    AXIS_WORK_OBLIGATION,
    CONFLICT_FAMILY,
    CONVENIENCE_QUALITY,
    GENERAL,
    LOYALTY_BOUNDARY,
    OBLIGATION_OVERLOAD,
    RISK_TIMING,
    SPENDING,
    score_dimensions,
    score_ontology_axes,
)
from ..router import normalize_input

# --- Respond-like-me subshape (same family is not always same situation) ---


def gossip_or_backchannel_user_prompt(norm_text: str) -> bool:
    """Trash-talk / behind-the-back phrasing (normalized text)."""
    padded = f" {norm_text} "
    return any(
        m in padded or m.strip() in norm_text
        for m in (
            " behind ",
            " gossip",
            " trash ",
            " trash talk",
            " talk shit",
            " talking shit",
            " shit ",
            " badmouth",
            " rumor",
            " rumour",
            " two faced",
            " two-faced",
        )
    )


def _gossip_or_backchannel_prompt(norm_text: str) -> bool:
    return gossip_or_backchannel_user_prompt(norm_text)


def _row_supports_backchannel_conflict(
    row_norm: str, ar: Mapping[str, float]
) -> bool:
    if float(ar.get(AXIS_BACKCHANNEL_HURT, 0) or 0) >= 1.02:
        return True
    padded = f" {row_norm} "
    return any(
        m in padded or m.strip() in row_norm
        for m in (
            " behind ",
            " behind your back",
            " gossip",
            " trash ",
            " badmouth",
            " rumor",
            " rumour",
            " two faced",
            "two-faced",
            "talk directly",
            "address it",
            " confront",
            "say something",
            "clear the air",
        )
    )


def _spending_pressure_prompt(
    dp: Mapping[str, float], prompt_norm: str
) -> bool:
    if float(dp.get("money_pressure", 0) or 0) >= 0.72:
        return True
    return any(
        x in prompt_norm
        for x in (
            "rent",
            "bill",
            "$",
            "afford",
            "broke",
            "debt",
            "salary",
            "pay ",
            " late",
            "laptop",
        )
    )


def _row_has_money_decision_shape(
    ar: Mapping[str, float], dr: Mapping[str, float], row_norm: str
) -> bool:
    ax = (
        float(ar.get(AXIS_BILLS_FINANCIAL_PRESSURE, 0) or 0) * 1.05
        + float(ar.get(AXIS_MONEY_SPENDING, 0) or 0)
        + float(ar.get(AXIS_NEED_VS_WANT, 0) or 0) * 0.9
    )
    if ax >= 1.18:
        return True
    dims = float(dr.get("money_pressure", 0) or 0) + float(
        dr.get("need_vs_want_signal", 0) or 0
    )
    if dims >= 1.22:
        return True
    if any(
        x in row_norm
        for x in (
            "rent",
            "bill",
            "$",
            "afford",
            "pay",
            "debt",
            "broke",
            "salary",
            "save money",
            "spend",
            "need the money",
            "need vs want",
        )
    ):
        return ax >= 0.52
    return False


def _apply_respond_subshape_penalties(
    prompt_norm: str,
    row_norm: str,
    top_p: str,
    top_r: str,
    dp: Mapping[str, float],
    dr: Mapping[str, float],
    ar: Mapping[str, float],
    base: float,
) -> float:
    b = float(base)
    if top_p == CONFLICT_FAMILY and _gossip_or_backchannel_prompt(prompt_norm):
        if not _row_supports_backchannel_conflict(row_norm, ar):
            b = min(b, 0.18)
    if top_p == SPENDING and _spending_pressure_prompt(dp, prompt_norm):
        has_money = _row_has_money_decision_shape(ar, dr, row_norm)
        if top_r == RISK_TIMING and not has_money:
            b = min(b, 0.1)
        elif (
            top_r == GENERAL
            and float(ar.get(AXIS_TIMING_WAIT_VS_ACT, 0) or 0) >= 1.15
            and not has_money
        ):
            b = min(b, 0.12)
        if top_r == CONFLICT_FAMILY and float(dp.get("money_pressure", 0) or 0) >= 0.55:
            b = min(b, 0.09)
    return b


# --- Ontology axis overlap (prompt vs stored text) ---


def ontology_axes_overlap(prompt_norm: str, row_norm: str) -> float:
    """Sum of min(axis scores) on shared ontology axes (deterministic, 0..)."""
    ap = score_ontology_axes(prompt_norm)
    ar = score_ontology_axes(row_norm)
    keys = set(ap) & set(ar)
    if not keys:
        return 0.0
    return sum(min(ap[k], ar[k]) for k in keys)


def decision_memory_relevance_multiplier(
    prompt_norm: str,
    row: Mapping[str, Any],
    raw_score: float,
) -> float:
    """
    Down-rank stored decision rows when the prompt’s decision shape does not
    match the saved scenario (Phase 32 + Phase 34 tightening + Phase 36 respond).
    """
    row_norm = normalize_input(decision_row_text_blob(row))
    if not row_norm.strip():
        return 0.35

    from .ontology import rank_families_full

    ordered_p, dp, ap = rank_families_full(prompt_norm)
    ordered_r, dr, ar = rank_families_full(row_norm)
    top_p = ordered_p[0][0]
    top_r = ordered_r[0][0]

    keys = set(dp) & set(dr)
    overlap = sum(min(dp[k], dr[k]) for k in keys) if keys else 0.0

    axis_ov = ontology_axes_overlap(prompt_norm, row_norm)
    axis_factor = 1.0
    if axis_ov >= 2.1:
        axis_factor = 1.08
    elif axis_ov >= 1.15:
        axis_factor = 1.0
    elif axis_ov >= 0.55:
        axis_factor = 0.78
    else:
        axis_factor = 0.42

    prompt_conf = dp.get("interpersonal_hurt", 0) + dp.get("conflict_intensity", 0)
    prompt_obl = dp.get("obligation", 0) + dp.get("overload", 0)
    row_conf = dr.get("interpersonal_hurt", 0) + dr.get("conflict_intensity", 0)
    row_obl = dr.get("obligation", 0) + dr.get("overload", 0)
    dual_topic_prompt = prompt_obl >= 1.0 and prompt_conf >= 1.0

    conf_prompt_strong = top_p == CONFLICT_FAMILY or prompt_conf >= 1.05
    obl_prompt_strong = top_p == OBLIGATION_OVERLOAD or prompt_obl >= 1.05
    conf_row_strong = top_r == CONFLICT_FAMILY or row_conf >= 1.05
    obl_row_strong = top_r == OBLIGATION_OVERLOAD or row_obl >= 1.2
    loy_row_primary = top_r == LOYALTY_BOUNDARY

    if top_p == top_r:
        base = 1.0
    elif (
        conf_prompt_strong
        and (obl_row_strong or (loy_row_primary and row_conf < 0.95))
        and not dual_topic_prompt
    ):
        base = 0.04
    elif obl_prompt_strong and conf_row_strong and not dual_topic_prompt:
        base = 0.04
    elif (
        top_p == CONFLICT_FAMILY
        and top_r == SPENDING
        and dp.get("money_pressure", 0) < 0.85
    ):
        base = 0.07
    elif (
        top_p == SPENDING
        and top_r == CONFLICT_FAMILY
        and dr.get("money_pressure", 0) < 0.85
    ):
        base = 0.07
    elif overlap >= 1.55:
        base = 0.92
    elif overlap >= 1.05:
        base = 0.78
    elif overlap >= 0.65:
        base = 0.52
    else:
        money_row = dr.get("money_pressure", 0) >= 1.0 or any(
            x in row_norm
            for x in ("rent", "bill", "$", "afford", "pay", "loan", "salary", "broke")
        )
        money_prompt = dp.get("money_pressure", 0) >= 0.85 or any(
            x in prompt_norm
            for x in (
                "rent",
                "bill",
                "$",
                "afford",
                "pay",
                "loan",
                "salary",
                "buy",
                "spend",
                "money",
            )
        )
        if money_row and not money_prompt and dp.get("money_pressure", 0) < 0.55:
            base = 0.12
        else:
            obl_row = dr.get("obligation", 0) + dr.get("overload", 0) >= 1.25
            obl_prompt = dp.get("obligation", 0) + dp.get("overload", 0) >= 0.85
            if obl_row and not obl_prompt:
                base = 0.15
            else:
                conf_row = dr.get("interpersonal_hurt", 0) + dr.get(
                    "conflict_intensity", 0
                ) >= 1.1
                conf_prompt = dp.get("interpersonal_hurt", 0) + dp.get(
                    "conflict_intensity", 0
                ) >= 0.85
                if conf_row and not conf_prompt:
                    base = 0.14
                elif overlap >= 0.38:
                    base = max(0.18, 0.38 + 0.08 * min(1.0, overlap))
                else:
                    base = 0.14

    base = _apply_respond_subshape_penalties(
        prompt_norm, row_norm, top_p, top_r, dp, dr, ar, base
    )
    return max(0.08, min(1.15, base * axis_factor))


def personal_response_decision_families_aligned(
    prompt_norm: str,
    row: Mapping[str, Any],
    *,
    min_dim_overlap: float = 1.28,
) -> bool:
    """
    True when the saved row is safe to treat as a “closest” same-family match
    for respond-like-me (Phase 36).
    """
    row_norm = normalize_input(decision_row_text_blob(row))
    if not row_norm.strip():
        return False

    from .ontology import rank_families_full

    ordered_p, dp, _ = rank_families_full(prompt_norm)
    ordered_r, dr, _ = rank_families_full(row_norm)
    top_p = ordered_p[0][0]
    top_r = ordered_r[0][0]
    if top_p == top_r:
        return True
    keys = set(dp) & set(dr)
    overlap = sum(min(dp[k], dr[k]) for k in keys) if keys else 0.0
    if overlap >= min_dim_overlap:
        return True
    prompt_conf = dp.get("interpersonal_hurt", 0) + dp.get("conflict_intensity", 0)
    prompt_obl = dp.get("obligation", 0) + dp.get("overload", 0)
    row_conf = dr.get("interpersonal_hurt", 0) + dr.get("conflict_intensity", 0)
    row_obl = dr.get("obligation", 0) + dr.get("overload", 0)
    dual_p = prompt_obl >= 1.0 and prompt_conf >= 1.0
    dual_r = row_obl >= 1.0 and row_conf >= 1.0
    if dual_p and dual_r and overlap >= 0.95:
        return True
    return False


def respond_main_decision_passes_shape_gate(
    prompt_norm: str,
    row: Mapping[str, Any],
    *,
    effective_primary: str,
) -> bool:
    """
    Stricter than ``personal_response_decision_families_aligned``: the row must
    match the prompt’s effective decision *shape* (Phase 37.1 respond-like-me).
    """
    from .ontology import interpersonal_conflict_markers_present, rank_families_full

    row_norm = normalize_input(decision_row_text_blob(row))
    if not row_norm.strip():
        return False

    _, dp, _ = rank_families_full(prompt_norm)
    ordered_r, dr, ar = rank_families_full(row_norm)
    top_r = ordered_r[0][0]
    row_c = float(dr.get("interpersonal_hurt", 0) or 0) + float(
        dr.get("conflict_intensity", 0) or 0
    )

    if effective_primary == OBLIGATION_OVERLOAD:
        if top_r in (OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY):
            return True
        if top_r == CONFLICT_FAMILY:
            return (
                float(dr.get("obligation", 0) or 0)
                + float(dr.get("overload", 0) or 0)
                >= 0.95
            )
        if top_r == GENERAL:
            return (
                float(dr.get("obligation", 0) or 0)
                + float(dr.get("overload", 0) or 0)
                >= 1.02
            )
        return False

    if effective_primary == CONFLICT_FAMILY:
        if top_r in (OBLIGATION_OVERLOAD, SPENDING, RISK_TIMING):
            return False
        if top_r not in (CONFLICT_FAMILY, GENERAL, LOYALTY_BOUNDARY):
            return False
        if top_r != CONFLICT_FAMILY:
            if row_c < 1.02:
                return False
            ax_c = float(ar.get(AXIS_CONFLICT_CONFRONTATION, 0) or 0) + float(
                ar.get(AXIS_BACKCHANNEL_HURT, 0) or 0
            )
            if ax_c < 1.0:
                return False
        if _gossip_or_backchannel_prompt(prompt_norm):
            return _row_supports_backchannel_conflict(row_norm, ar)
        if interpersonal_conflict_markers_present(prompt_norm) and top_r == GENERAL:
            return row_c >= 0.9
        return True

    if effective_primary == SPENDING:
        money_press = _spending_pressure_prompt(dp, prompt_norm)
        has_money_row = _row_has_money_decision_shape(ar, dr, row_norm)
        if money_press:
            if top_r == RISK_TIMING and not has_money_row:
                return False
            if top_r == CONFLICT_FAMILY:
                return False
            return top_r == SPENDING or has_money_row
        if top_r == SPENDING or has_money_row:
            return personal_response_decision_families_aligned(
                prompt_norm, row, min_dim_overlap=1.05
            )
        return personal_response_decision_families_aligned(
            prompt_norm, row, min_dim_overlap=1.15
        )

    if effective_primary == RISK_TIMING:
        if top_r == RISK_TIMING:
            return True
        if top_r == GENERAL:
            return (
                float(dr.get("wait_vs_act", 0) or 0)
                + float(dr.get("uncertainty", 0) or 0)
                >= 1.02
            )
        return personal_response_decision_families_aligned(
            prompt_norm, row, min_dim_overlap=1.18
        )

    if effective_primary == LOYALTY_BOUNDARY:
        if top_r in (LOYALTY_BOUNDARY, OBLIGATION_OVERLOAD):
            return True
        return personal_response_decision_families_aligned(
            prompt_norm, row, min_dim_overlap=1.2
        )

    if effective_primary == CONVENIENCE_QUALITY:
        return personal_response_decision_families_aligned(
            prompt_norm, row, min_dim_overlap=1.12
        )

    return personal_response_decision_families_aligned(
        prompt_norm, row, min_dim_overlap=1.22
    )


def decision_row_text_blob(row: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("scenario_text") or ""),
            str(row.get("choice_label") or ""),
            str(row.get("reasoning_label") or ""),
            " ".join(str(t) for t in (row.get("value_tags") or [])),
        ]
    )


def style_row_text_blob(row: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("prompt_text") or ""),
            str(row.get("selected_label") or ""),
            " ".join(str(t) for t in (row.get("style_tags") or [])),
        ]
    )


def style_memory_passes_respond_conflict_shape(
    prompt_norm: str,
    row: Mapping[str, Any],
) -> bool:
    """
    Phase 37.2: for strict conflict / gossip prompts, style rows must show
    backchannel or confrontation shape — never generic avoidance / “let it go”.
    """
    from .ontology import interpersonal_conflict_markers_present, rank_families_full

    style_norm = normalize_input(style_row_text_blob(row))
    if not style_norm.strip():
        return False

    label_l = (str(row.get("selected_label") or "")).lower()
    tags_l = " ".join(str(t).lower() for t in (row.get("style_tags") or []))
    blob_l = style_norm.lower()
    if _gossip_or_backchannel_prompt(prompt_norm):
        avoid_phrases = (
            "let it go",
            "let it slide",
            "let it roll",
            "just ignore",
            "ignore it",
            "not worth",
            "don't engage",
            "dont engage",
            "walk away",
            "rise above",
        )
        if any(p in label_l or p in tags_l for p in avoid_phrases):
            return False
        _, _, ar = rank_families_full(style_norm)
        if not _row_supports_backchannel_conflict(style_norm, ar):
            return False
        return True

    if interpersonal_conflict_markers_present(prompt_norm):
        _, dr, ar = rank_families_full(style_norm)
        row_c = float(dr.get("interpersonal_hurt", 0) or 0) + float(
            dr.get("conflict_intensity", 0) or 0
        )
        ax_c = float(ar.get(AXIS_CONFLICT_CONFRONTATION, 0) or 0) + float(
            ar.get(AXIS_BACKCHANNEL_HURT, 0) or 0
        )
        if row_c >= 0.92 or ax_c >= 1.0:
            return True
        if any(
            w in blob_l
            for w in (
                "say something",
                "speak up",
                "address",
                "direct",
                "clear the air",
                "talk it out",
                "boundary",
            )
        ):
            return ax_c >= 0.55
        return False

    return True


def style_memory_relevance_multiplier(
    prompt_norm: str,
    row: Mapping[str, Any],
    raw_score: float,
) -> float:
    """
    Down-rank style calibration rows on clear cross-domain mismatch only.

    Style prompts are often generic (“how you explain…”), so this uses softer
    floors than ``decision_memory_relevance_multiplier`` and only hard-penalizes
    obvious pulls (money vs non-money, conflict vs non-conflict, etc.).
    """
    if raw_score >= 2.2:
        return 1.0

    row_norm = normalize_input(style_row_text_blob(row))
    if not row_norm.strip():
        return 0.55

    from .ontology import rank_families_full

    ordered_p, dp, _ = rank_families_full(prompt_norm)
    ordered_r, dr, _ = rank_families_full(row_norm)
    top_p = ordered_p[0][0]
    top_r = ordered_r[0][0]

    keys = set(dp) & set(dr)
    overlap = sum(min(dp[k], dr[k]) for k in keys) if keys else 0.0
    axis_ov = ontology_axes_overlap(prompt_norm, row_norm)
    axis_factor = 1.0
    if axis_ov >= 1.4:
        axis_factor = 1.04
    elif axis_ov >= 0.75:
        axis_factor = 1.0
    elif axis_ov >= 0.35:
        axis_factor = 0.9
    else:
        axis_factor = 0.78

    if top_p == top_r or top_p == GENERAL or top_r == GENERAL:
        base = 1.0
    elif overlap >= 1.1:
        base = 0.9
    elif overlap >= 0.65:
        base = 0.84
    elif overlap >= 0.38:
        base = 0.76
    else:
        base = 0.68

    money_row = dr.get("money_pressure", 0) >= 1.05 or any(
        x in row_norm for x in ("rent", "bill", "$", "afford", "pay", "salary", "broke")
    )
    money_prompt = dp.get("money_pressure", 0) >= 0.75 or any(
        x in prompt_norm for x in ("rent", "bill", "$", "afford", "pay", "buy", "money")
    )
    if money_row and not money_prompt and dp.get("money_pressure", 0) < 0.45:
        base = min(base, 0.32)

    obl_row = dr.get("obligation", 0) + dr.get("overload", 0) >= 1.2
    obl_prompt = dp.get("obligation", 0) + dp.get("overload", 0) >= 0.75
    if obl_row and not obl_prompt and dp.get("obligation", 0) + dp.get("overload", 0) < 0.5:
        base = min(base, 0.36)

    conf_row = dr.get("interpersonal_hurt", 0) + dr.get("conflict_intensity", 0) >= 1.1
    conf_prompt = dp.get("interpersonal_hurt", 0) + dp.get("conflict_intensity", 0) >= 0.75
    if conf_row and not conf_prompt and dp.get("interpersonal_hurt", 0) < 0.45:
        base = min(base, 0.34)

    # Strong lexical retrieval already found a fit — do not collapse it.
    if raw_score >= 0.45:
        base = max(base, 0.82)

    return max(0.38, min(1.1, base * axis_factor))


# --- Fit between current question shape and trait-style profile blurbs ---


def profile_memory_fit_score(
    primary_family: str,
    norm_text: str,
    dimensions: Optional[Dict[str, float]] = None,
    axes: Optional[Dict[str, float]] = None,
) -> float:
    """
    How strongly the current routed decision matches “saved habit” blurbs.
    Used to gate profile/tendency lines so generic traits do not attach to
    unrelated asks.
    """
    dims = dimensions if dimensions is not None else score_dimensions(norm_text)
    ax = axes if axes is not None else score_ontology_axes(norm_text)
    padded = f" {norm_text} "

    def d(name: str) -> float:
        return float(dims.get(name, 0.0) or 0.0)

    if primary_family == SPENDING:
        s = d("money_pressure") * 0.45 + d("need_vs_want_signal") * 0.35
        s += min(2.0, ax.get(AXIS_MONEY_SPENDING, 0.0)) * 0.35
        s += min(2.0, ax.get(AXIS_BILLS_FINANCIAL_PRESSURE, 0.0)) * 0.45
        if any(x in norm_text for x in ("buy", "afford", "rent", "bill", "pay", "$")):
            s += 0.35
        return min(1.0, s / 2.1)

    if primary_family == OBLIGATION_OVERLOAD:
        s = d("obligation") * 0.4 + d("overload") * 0.42 + d("boundary_strain") * 0.35
        s += min(2.0, ax.get(AXIS_WORK_OBLIGATION, 0.0)) * 0.35
        s += min(2.0, ax.get(AXIS_HELPING_FAVOR, 0.0)) * 0.32
        s += min(2.0, ax.get(AXIS_OVERLOAD_BURNOUT, 0.0)) * 0.38
        if any(
            w in padded
            for w in (
                " shift ",
                " overtime ",
                " cover ",
                " exhausted ",
                " favor ",
                " asked me ",
            )
        ):
            s += 0.4
        return min(1.0, s / 2.05)

    if primary_family == CONFLICT_FAMILY:
        s = d("interpersonal_hurt") * 0.42 + d("conflict_intensity") * 0.4
        s += min(2.0, ax.get(AXIS_CONFLICT_CONFRONTATION, 0.0)) * 0.42
        if any(
            w in norm_text
            for w in ("rude", "unfair", "upset", "boss", "coworker", "say something")
        ):
            s += 0.35
        return min(1.0, s / 1.85)

    if primary_family == RISK_TIMING:
        s = d("wait_vs_act") * 0.42 + d("uncertainty") * 0.38 + d("risk_level") * 0.35
        s += min(2.0, ax.get(AXIS_TIMING_WAIT_VS_ACT, 0.0)) * 0.45
        s += min(2.0, ax.get(AXIS_UNCERTAINTY_RISK, 0.0)) * 0.38
        return min(1.0, s / 1.9)

    if primary_family == LOYALTY_BOUNDARY:
        s = d("boundary_strain") * 0.4 + d("regret_risk") * 0.35
        s += min(2.0, ax.get(AXIS_LOYALTY_VS_SELF, 0.0)) * 0.45
        s += d("relationship_stakes") * 0.3
        return min(1.0, s / 1.75)

    if primary_family == CONVENIENCE_QUALITY:
        s = d("convenience_vs_correctness") * 0.45 + d("short_vs_long") * 0.35
        return min(1.0, s / 1.45)

    # GENERAL: only when the ask is still clearly about stakes / timing
    if primary_family == GENERAL:
        s = d("uncertainty") * 0.35 + d("wait_vs_act") * 0.35
        s += min(2.0, ax.get(AXIS_UNCERTAINTY_RISK, 0.0)) * 0.35
        if d("money_pressure") >= 0.85 or d("overload") >= 0.85:
            s += 0.25
        return min(1.0, s / 1.55)

    return 0.55


def profile_risk_snippet_allowed(
    primary_family: str,
    norm_text: str,
    dimensions: Optional[Dict[str, float]] = None,
    axes: Optional[Dict[str, float]] = None,
) -> bool:
    """Whether a cautious/risk-trait line may attach to this decision."""
    fit = profile_memory_fit_score(primary_family, norm_text, dimensions, axes)
    dims = dimensions if dimensions is not None else score_dimensions(norm_text)
    ax = axes if axes is not None else score_ontology_axes(norm_text)

    if fit >= 0.62:
        return True

    # Secondary path: strong axis even when family is broad
    if ax.get(AXIS_TIMING_WAIT_VS_ACT, 0) >= 1.2 and dims.get("wait_vs_act", 0) >= 0.95:
        return True
    if ax.get(AXIS_UNCERTAINTY_RISK, 0) >= 1.2 and dims.get("uncertainty", 0) >= 0.95:
        return True
    if primary_family == SPENDING and (
        dims.get("money_pressure", 0) >= 1.1 or ax.get(AXIS_BILLS_FINANCIAL_PRESSURE, 0) >= 1.2
    ):
        return True

    return False


def profile_decision_speed_snippet_allowed(
    primary_family: str,
    norm_text: str,
    dimensions: Optional[Dict[str, float]] = None,
) -> bool:
    """Gates the “pause before big calls” line — timing / conflict / purchase."""
    dims = dimensions if dimensions is not None else score_dimensions(norm_text)
    if primary_family in (RISK_TIMING, CONFLICT_FAMILY, SPENDING):
        return True
    if dims.get("uncertainty", 0) >= 0.85 and dims.get("wait_vs_act", 0) >= 0.55:
        return True
    if primary_family == GENERAL and (
        dims.get("wait_vs_act", 0) + dims.get("uncertainty", 0) >= 1.35
    ):
        return True
    return False


def profile_value_tag_snippet_allowed(primary_family: str) -> bool:
    """“Play it safe” value line — money / obligation / loyalty shapes only."""
    return primary_family in (SPENDING, OBLIGATION_OVERLOAD, LOYALTY_BOUNDARY, GENERAL)
