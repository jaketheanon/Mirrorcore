"""
Gating for learned historical fix guidance (ranked terminal outcomes).

Suppresses weak one-off matches (e.g. port-kill fix suggested for a traceback-heavy log)
while preserving repeated-success patterns (e.g. pip --user after multiple wins).
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

# Log domains compatible with each incident type (single-success path)
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
    if sc < 1:
        return False
    if sc >= 2:
        return True

    if incident_confidence < _MIN_CONFIDENCE_SINGLE_SUCCESS:
        return False

    lec_raw = (likely_error_category or "").strip().lower()
    fams = _families_set(signal_families)

    if lec_raw and lec_raw != "unknown":
        log_dom = _log_domain_from_lec(lec_raw)
    elif user_input:
        log_dom = _derive_domain_from_user_text(user_input)
    else:
        log_dom = "generic"

    if log_dom == "generic" and user_input:
        log_dom = _derive_domain_from_user_text(user_input)

    allowed = _INCIDENT_ALLOWED_DOMAINS.get(incident_type)
    if allowed is None:
        primary = _INCIDENT_PRIMARY_DOMAIN.get(incident_type, "generic")
        allowed = frozenset({primary}) if primary != "generic" else frozenset({"generic"})

    domain_ok = log_dom in allowed
    if not domain_ok:
        primary = _INCIDENT_PRIMARY_DOMAIN.get(incident_type, "generic")
        if incident_confidence >= _STRONG_CONFIDENCE and primary == log_dom and primary != "generic":
            domain_ok = True
        else:
            return False

    if _mixed_runtime_and_network(fams):
        return False

    if len(fams) >= 4:
        return False

    if incident_type == "network_connectivity":
        if log_dom != "network":
            return False
        net_subs = frozenset({"curl", "wget", "ssh", "scp", "http", "https"})
        sub = (detected_subsystem or "").lower()
        if sub and sub not in net_subs:
            return False

    return True
