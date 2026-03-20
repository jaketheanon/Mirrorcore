"""
Unified personal profile aggregation (Phase 29).

Combines decision memory, style memory, and correction metadata into a
lightweight, evidence-weighted summary. Traits are only surfaced when
enough grounded support exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from ..db.store import DatabaseStore

# Minimum cumulative weight before a numeric trait is reported
_MIN_TRAIT_WEIGHT = 1.05
# Minimum cumulative weight before a value/style tag is reported
_MIN_TAG_WEIGHT = 1.0


def _entry_quality_multiplier(correction_status: str) -> float:
    return {
        "accurate": 1.28,
        "uncorrected": 1.0,
        "partially_true": 0.62,
        "not_really": 0.07,
    }.get((correction_status or "").strip(), 0.88)


def _effective_entry_weight(
    correction_status: str,
    confidence_score: float,
    correction_metadata: Optional[Mapping[str, Any]],
) -> float:
    base = _entry_quality_multiplier(correction_status)
    conf = float(confidence_score) if confidence_score is not None else 0.75
    conf = max(0.15, min(1.0, conf))
    meta = correction_metadata or {}
    rejected = set(meta.get("rejected_traits") or [])
    if rejected and correction_status == "partially_true":
        base *= 0.85
    return base * conf


def _adjust_trait_signals_for_rejection(
    trait_signals: Mapping[str, float],
    correction_status: str,
    correction_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    out = dict(trait_signals)
    if correction_status != "partially_true":
        return out
    meta = correction_metadata or {}
    for key in meta.get("rejected_traits") or []:
        if key in out:
            out[key] = out[key] * 0.35
    return out


@dataclass
class TraitEstimate:
    name: str
    weighted_mean: float
    evidence_weight: float
    weighted_std: float
    confidence: float
    conflicted: bool


@dataclass
class PersonalProfile:
    """Aggregated, evidence-bound view of stored personal patterns."""

    trait_estimates: List[TraitEstimate] = field(default_factory=list)
    value_tag_weights: List[Tuple[str, float]] = field(default_factory=list)
    style_tag_weights: List[Tuple[str, float]] = field(default_factory=list)
    total_evidence_weight: float = 0.0
    has_trait_conflict: bool = False
    decision_entries_used: int = 0
    style_entries_used: int = 0

    def decision_risk_summary(self) -> Optional[str]:
        for t in self.trait_estimates:
            if t.name == "risk_tolerance" and t.confidence >= 0.35:
                if t.weighted_mean >= 0.62:
                    return "leans risk-tolerant"
                if t.weighted_mean <= 0.38:
                    return "leans cautious"
        return None

    def communication_style_summary(self) -> Optional[str]:
        parts = []
        verb = next((t for t in self.trait_estimates if t.name == "verbosity"), None)
        dip = next((t for t in self.trait_estimates if t.name == "diplomacy"), None)
        bln = next((t for t in self.trait_estimates if t.name == "bluntness"), None)
        if verb and verb.confidence >= 0.35:
            if verb.weighted_mean >= 0.62:
                parts.append("more detailed explanations")
            elif verb.weighted_mean <= 0.38:
                parts.append("shorter, direct explanations")
        if dip and bln and dip.confidence >= 0.3 and bln.confidence >= 0.3:
            if dip.weighted_mean >= bln.weighted_mean + 0.15:
                parts.append("diplomatic phrasing")
            elif bln.weighted_mean >= dip.weighted_mean + 0.15:
                parts.append("blunter phrasing")
        if not parts:
            top_tags = [x[0] for x in self.style_tag_weights[:3] if x[1] >= _MIN_TAG_WEIGHT]
            if top_tags:
                return "tends toward: " + ", ".join(top_tags)
            return None
        return "; ".join(parts)


def _accumulate_weighted_trait(
    buckets: MutableMapping[str, List[Tuple[float, float]]],
    trait: str,
    value: float,
    w: float,
) -> None:
    if w <= 0 or not math.isfinite(value):
        return
    buckets.setdefault(trait, []).append((float(value), w))


def _finalize_traits(
    buckets: Mapping[str, List[Tuple[float, float]]],
) -> List[TraitEstimate]:
    estimates: List[TraitEstimate] = []
    for name, pairs in sorted(buckets.items()):
        sum_w = sum(p[1] for p in pairs)
        if sum_w < _MIN_TRAIT_WEIGHT:
            continue
        mean = sum(v * w for v, w in pairs) / sum_w
        if len(pairs) >= 2:
            var = sum(w * (v - mean) ** 2 for v, w in pairs) / sum_w
            std = math.sqrt(max(0.0, var))
        else:
            std = 0.0
        conflicted = std > 0.32 and len(pairs) >= 2
        # Confidence rises with support and falls with spread / conflict
        conf = min(1.0, sum_w / 3.2) * (0.55 if conflicted else 1.0) * max(0.2, 1.0 - std * 1.1)
        conf = max(0.0, min(0.95, conf))
        estimates.append(
            TraitEstimate(
                name=name,
                weighted_mean=max(0.0, min(1.0, mean)),
                evidence_weight=sum_w,
                weighted_std=std,
                confidence=conf,
                conflicted=conflicted,
            )
        )
    return estimates


def _merge_tag_counts(
    target: MutableMapping[str, float], tags: List[str], w: float
) -> None:
    for tag in tags:
        key = (tag or "").strip().lower()
        if key:
            target[key] = target.get(key, 0.0) + w


def build_personal_profile_from_rows(
    decisions: Sequence[Mapping[str, Any]],
    styles: Sequence[Mapping[str, Any]],
) -> PersonalProfile:
    """Aggregate profile from already-loaded memory rows."""
    trait_pairs: Dict[str, List[Tuple[float, float]]] = {}
    value_weights: Dict[str, float] = {}
    style_weights: Dict[str, float] = {}
    total_w = 0.0
    has_conflict = False

    for row in decisions:
        ew = _effective_entry_weight(
            row.get("correction_status") or "uncorrected",
            row.get("confidence_score") or 0.75,
            row.get("correction_metadata"),
        )
        if ew <= 0:
            continue
        total_w += ew
        ts = _adjust_trait_signals_for_rejection(
            row.get("trait_signals") or {},
            row.get("correction_status") or "uncorrected",
            row.get("correction_metadata"),
        )
        for k, v in ts.items():
            if isinstance(v, (int, float)):
                _accumulate_weighted_trait(trait_pairs, k, float(v), ew)
        _merge_tag_counts(value_weights, row.get("value_tags") or [], ew)

    for row in styles:
        ew = _effective_entry_weight(
            row.get("correction_status") or "uncorrected",
            row.get("confidence_score") or 0.75,
            row.get("correction_metadata"),
        )
        if ew <= 0:
            continue
        total_w += ew
        for k, v in (row.get("tone_signals") or {}).items():
            if isinstance(v, (int, float)):
                _accumulate_weighted_trait(trait_pairs, k, float(v), ew)
        _merge_tag_counts(style_weights, row.get("style_tags") or [], ew)

    estimates = _finalize_traits(trait_pairs)
    has_conflict = any(t.conflicted for t in estimates) or has_conflict

    value_sorted = sorted(value_weights.items(), key=lambda x: (-x[1], x[0]))
    value_sorted = [p for p in value_sorted if p[1] >= _MIN_TAG_WEIGHT][:12]

    style_sorted = sorted(style_weights.items(), key=lambda x: (-x[1], x[0]))
    style_sorted = [p for p in style_sorted if p[1] >= _MIN_TAG_WEIGHT][:12]

    return PersonalProfile(
        trait_estimates=sorted(estimates, key=lambda t: (-t.evidence_weight, t.name)),
        value_tag_weights=value_sorted,
        style_tag_weights=style_sorted,
        total_evidence_weight=total_w,
        has_trait_conflict=has_conflict,
        decision_entries_used=len(decisions),
        style_entries_used=len(styles),
    )


def build_personal_profile(
    store: DatabaseStore,
    decision_limit: int = 800,
    style_limit: int = 800,
) -> PersonalProfile:
    """Aggregate profile from recent decision and style memory rows."""
    decisions = store.get_recent_decision_memory(limit=decision_limit)
    styles = store.get_recent_style_memory(limit=style_limit)
    return build_personal_profile_from_rows(decisions, styles)
