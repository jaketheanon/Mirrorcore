"""
Phase 34: shared deterministic relevance helpers for decision memory and
profile snippets. Used by routed guidance and personal response retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .ontology import (
    AXIS_BILLS_FINANCIAL_PRESSURE,
    AXIS_CONFLICT_CONFRONTATION,
    AXIS_HELPING_FAVOR,
    AXIS_LOYALTY_VS_SELF,
    AXIS_MONEY_SPENDING,
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
    match the saved scenario (Phase 32 + Phase 34 tightening).
    """
    if raw_score >= 2.35:
        return 1.0

    row_norm = normalize_input(decision_row_text_blob(row))
    if not row_norm.strip():
        return 0.35

    from .ontology import rank_families_full

    ordered_p, dp, _ = rank_families_full(prompt_norm)
    ordered_r, dr, _ = rank_families_full(row_norm)
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

    if top_p == top_r:
        base = 1.0
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

    return max(0.08, min(1.15, base * axis_factor))


def decision_row_text_blob(row: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("scenario_text") or ""),
            str(row.get("choice_label") or ""),
            str(row.get("reasoning_label") or ""),
            " ".join(str(t) for t in (row.get("value_tags") or [])),
        ]
    )


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
