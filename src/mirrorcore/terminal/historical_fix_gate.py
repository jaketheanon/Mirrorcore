"""
Gating for learned historical fix guidance (ranked terminal outcomes).

Suppresses weak one-off matches (e.g. port-kill fix suggested for a traceback-heavy log)
while preserving repeated-success patterns (e.g. pip --user after multiple wins).

Strict rules enforced:
A. Multi-signal protection  – traceback + mixed families → block unless very strong
B. Weak evidence rejection  – single success requires domain + subsystem alignment
C. Subsystem alignment      – detected subsystem must match incident expectations
D. Category alignment       – tracebacks block network fixes unless network dominates
E. Confidence threshold     – ambiguous (3+ families) requires >= 0.9
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Set

_LOG_DOMAIN_BY_LEC: Dict[str, str] = {
    "traceback": "runtime",
    "exception": "runtime",
    "segmentation_fault": "runtime",
    "import_error": "runtime",
    "module_not_found": "runtime",
    "syntax_error": "config",
    "parse_error": "config",
    "connection_refused": "network",
    "timeout": "network",
    "dns": "network",
    "permission_denied": "permission",
    "command_not_found": "cmd",
    "service_failed": "service",
    "failed": "generic",
    "error": "generic",
    "unknown": "generic",
}

_INCIDENT_ALLOWED_DOMAINS: Dict[str, FrozenSet[str]] = {
    "pip_permission_denied": frozenset({"permission"}),
    "network_connectivity": frozenset({"network"}),
    "runtime_validation_error": frozenset({"runtime", "config"}),
    "config_syntax_error": frozenset({"config", "runtime"}),
    "git_auth_remote": frozenset({"permission", "git"}),
    "docker_container_restart": frozenset({"docker", "service", "generic", "runtime"}),
    "systemd_service_failed": frozenset({"service", "generic"}),
    "command_not_found": frozenset({"cmd", "generic"}),
}

_INCIDENT_PRIMARY_DOMAIN: Dict[str, str] = {
    "pip_permission_denied": "permission",
    "network_connectivity": "network",
    "runtime_validation_error": "runtime",
    "config_syntax_error": "config",
    "git_auth_remote": "git",
    "docker_container_restart": "docker",
    "systemd_service_failed": "service",
    "command_not_found": "cmd",
}

_INCIDENT_ALLOWED_SUBSYSTEMS: Dict[str, FrozenSet[str]] = {
    "pip_permission_denied": frozenset({"pip", "python"}),
    "network_connectivity": frozenset({"curl", "wget", "ssh", "scp", "http", "https"}),
    "docker_container_restart": frozenset({"docker"}),
    "systemd_service_failed": frozenset({"systemctl", "journalctl"}),
    "git_auth_remote": frozenset({"git", "ssh"}),
}

_RUNTIME_SIGNALS: Set[str] = {
    "traceback",
    "exception",
    "import_error",
    "module_not_found",
    "segmentation_fault",
    "syntax_error",
    "parse_error",
    "python_runtime",
}
_NETWORK_SIGNALS: Set[str] = {
    "connection_refused",
    "timeout",
    "dns",
    "port_listening",
    "port_not_listening",
}

_MIN_CONFIDENCE_SINGLE_SUCCESS = 0.88
_STRONG_CONFIDENCE = 0.92
_AMBIGUOUS_CONFIDENCE = 0.9
_EXTREMELY_STRONG_SUCCESS = 3
_EXTREMELY_STRONG_CONFIDENCE = 0.95


def _log_domain_from_lec(lec: str) -> str:
    return _LOG_DOMAIN_BY_LEC.get((lec or "").lower().strip(), "generic")


def _families_set(families: Optional[List[str]]) -> Set[str]:
    return {str(f).lower().strip() for f in (families or []) if f}


def _mixed_runtime_and_network(families: Set[str]) -> bool:
    return bool(families & _RUNTIME_SIGNALS) and bool(families & _NETWORK_SIGNALS)


def _derive_domain_from_user_text(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(traceback|valueerror|typeerror|attributeerror|importerror)\b", t):
        return "runtime"
    if "traceback" in t or 'file "' in t:
        return "runtime"
    if any(
        p in t
        for p in (
            "connection refused",
            "connection timed out",
            "name resolution failed",
            "address already in use",
        )
    ):
        return "network"
    if "permission denied" in t and "pip" in t:
        return "permission"
    if "git" in t and ("push" in t or "pull" in t or "clone" in t) and "denied" in t:
        return "git"
    return "generic"


def _text_has_traceback(text: str) -> bool:
    """Detect traceback / runtime-error indicators in raw text."""
    t = (text or "").lower()
    return bool(re.search(
        r"\b(traceback|valueerror|typeerror|attributeerror|importerror|runtimeerror)\b", t
    )) or "traceback" in t or 'file "' in t


def _derive_signal_families_from_text(text: str) -> Set[str]:
    """Infer signal families from raw text when explicit families are unavailable."""
    t = (text or "").lower()
    families: Set[str] = set()
    if "traceback" in t or 'file "' in t:
        families.add("traceback")
    if re.search(r"\b(valueerror|typeerror|attributeerror|importerror|runtimeerror)\b", t):
        families.add("error")
    if re.search(r"\bexception\b", t):
        families.add("exception")
    if "connection refused" in t:
        families.add("connection_refused")
    if re.search(r"\btimeout\b", t) or "timed out" in t:
        families.add("timeout")
    if re.search(r"\bdns\b", t) or "name resolution" in t:
        families.add("dns")
    if "permission denied" in t:
        families.add("permission_denied")
    if "command not found" in t:
        families.add("command_not_found")
    if re.search(r"service.*(failed|inactive)|failed to start", t):
        families.add("service_failed")
    return families


def should_surface_historical_fix_guidance(
    top_fix: Dict[str, Any],
    incident_type: str,
    incident_confidence: float,
    *,
    likely_error_category: Optional[str] = None,
    signal_families: Optional[List[str]] = None,
    detected_subsystem: Optional[str] = None,
    user_input: Optional[str] = None,
) -> bool:
    sc = int(top_fix.get("success_count") or 0)
    fc = int(top_fix.get("failed_count") or 0)

    if sc < 1:
        return False

    lec_raw = (likely_error_category or "").strip().lower()
    fams = _families_set(signal_families)

    # --- Derive log domain ---
    if lec_raw and lec_raw != "unknown":
        log_dom = _log_domain_from_lec(lec_raw)
    elif user_input:
        log_dom = _derive_domain_from_user_text(user_input)
    else:
        log_dom = "generic"

    if log_dom == "generic" and user_input:
        log_dom = _derive_domain_from_user_text(user_input)

    # Build effective signal families: explicit + text-derived fallback.
    # This ensures safety checks work even when callers omit signal metadata.
    effective_fams = set(fams)
    if user_input:
        effective_fams |= _derive_signal_families_from_text(user_input)

    has_traceback = lec_raw == "traceback" or (
        user_input is not None and _text_has_traceback(user_input)
    )

    # ------------------------------------------------------------------
    # RULE A – Multi-signal protection
    # Traceback + mixed runtime/network signals → block unless the fix
    # has extremely strong evidence (>= 3 successes AND >= 0.95 conf).
    # ------------------------------------------------------------------
    if has_traceback and _mixed_runtime_and_network(effective_fams):
        if sc < _EXTREMELY_STRONG_SUCCESS or incident_confidence < _EXTREMELY_STRONG_CONFIDENCE:
            return False

    # ------------------------------------------------------------------
    # RULE B – Weak evidence rejection
    # A single untested success is low-confidence evidence.  Require
    # exact domain alignment, adequate confidence, no signal ambiguity.
    # ------------------------------------------------------------------
    if sc == 1 and fc == 0:
        primary = _INCIDENT_PRIMARY_DOMAIN.get(incident_type, "generic")
        if primary != log_dom:
            return False

        if incident_confidence < _MIN_CONFIDENCE_SINGLE_SUCCESS:
            return False

        if _mixed_runtime_and_network(effective_fams):
            return False

        if len(effective_fams) >= 4:
            return False

    # ------------------------------------------------------------------
    # RULE C – Subsystem alignment
    # If we know the current subsystem, it must match what the incident
    # type expects.  Incident types not in the map are broad/unconstrained.
    # ------------------------------------------------------------------
    sub = (detected_subsystem or "").lower().strip()
    if sub:
        allowed_subs = _INCIDENT_ALLOWED_SUBSYSTEMS.get(incident_type)
        if allowed_subs and sub not in allowed_subs:
            return False

    # ------------------------------------------------------------------
    # RULE D – Category alignment
    # Tracebacks should not surface narrow network fixes (e.g. port kill)
    # unless network signals clearly dominate runtime signals.
    # ------------------------------------------------------------------
    if has_traceback:
        fix_primary = _INCIDENT_PRIMARY_DOMAIN.get(incident_type, "generic")
        if fix_primary == "network" and effective_fams:
            net_count = len(effective_fams & _NETWORK_SIGNALS)
            runtime_count = len(effective_fams & _RUNTIME_SIGNALS)
            if runtime_count >= net_count:
                return False

    # ------------------------------------------------------------------
    # RULE E – Confidence threshold for ambiguous cases
    # Three or more distinct signal families → require higher confidence.
    # ------------------------------------------------------------------
    if len(effective_fams) >= 3 and incident_confidence < _AMBIGUOUS_CONFIDENCE:
        return False

    # ------------------------------------------------------------------
    # Domain alignment (preserved from original logic)
    # ------------------------------------------------------------------
    allowed = _INCIDENT_ALLOWED_DOMAINS.get(incident_type)
    if allowed is None:
        primary = _INCIDENT_PRIMARY_DOMAIN.get(incident_type, "generic")
        allowed = frozenset({primary}) if primary != "generic" else frozenset({"generic"})

    if log_dom not in allowed:
        primary = _INCIDENT_PRIMARY_DOMAIN.get(incident_type, "generic")
        if incident_confidence >= _STRONG_CONFIDENCE and primary == log_dom and primary != "generic":
            pass
        else:
            return False

    return True
