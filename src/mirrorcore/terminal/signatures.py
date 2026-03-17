"""
Terminal Incident Signatures

Lightweight deterministic incident signature detection system.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class IncidentSignature:
    """Represents a detected incident signature."""
    incident_type: str
    confidence: float
    matched_terms: List[str]
    likely_causes: List[str]
    recommended_first_checks: List[str]
    low_risk_initial_actions: List[str]
    allowed_subsystems: List[str]  # New: subsystems this signature applies to


class IncidentSignatureDetector:
    """Detects common terminal incident patterns using deterministic rules."""
    
    def __init__(self):
        self.signatures = self._create_incident_signatures()
        # Minimum confidence required to return a concrete incident match.
        # Below this threshold, callers should fall back to multi-signal diagnostic mode.
        self.min_confidence_threshold = 0.75

    def _get_incident_signal_family(self, incident_type: str) -> str:
        """Map an incident type to the signal family it primarily explains."""
        family_map = {
            # Must align with families produced by _detect_conflicting_signals().
            "pip_permission_denied": "permissions",
            "docker_container_restart": "docker_container",
            "systemd_service_failed": "systemd_service",
            "git_auth_remote": "git_auth",
            "network_connectivity": "network_connectivity",
            "config_syntax_error": "config_syntax",
            "command_not_found": "error",
        }
        return family_map.get(incident_type, "error")
    
    def _create_incident_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Create incident signature patterns."""
        return {
            # pip permission issues
            "pip_permission_denied": {
                "keywords": ["pip", "install", "permission", "denied", "error", "failed"],
                "phrases": ["permission denied", "error: could not create", "unable to create", "can't create"],
                "patterns": [
                    r"pip\s+install.*permission\s+denied",
                    r"pip\s+install.*error.*permission",
                    r"permission\s+denied.*pip"
                ],
                "confidence": 0.9,
                "allowed_subsystems": ["pip", "python"],
                "likely_causes": [
                    "System-wide pip installation without sudo",
                    "Site-packages directory permissions",
                    "Virtual environment not activated",
                    "User pip installation conflicts"
                ],
                "recommended_first_checks": [
                    "Check if virtual environment is active: `echo $VIRTUAL_ENV`",
                    "Verify pip installation location: `which pip`",
                    "Check site-packages permissions: `ls -la $(python -m site --user-site)`"
                ],
                "low_risk_initial_actions": [
                    "Try user installation: `pip install --user package_name`",
                    "Create/activate virtual environment: `python -m venv venv && source venv/bin/activate`",
                    "Check if package already installed system-wide"
                ]
            },
            
            # docker container issues
            "docker_container_restart": {
                "keywords": ["docker", "container", "restart", "dies", "stops", "crash", "exit", "keep", "keeps", "automatically", "running", "stay", "loop", "won't", "restarting", "dying", "exiting"],
                "phrases": [
                    "container restart", "container dies", "container stops", "randomly restart", "keeps restarting", "automatically restarts", "container keeps dying", "container won't stay running", "won't stay running", "keep restarting", "container keeps dying", "container won't stay running", "keep dying", "restart loop", "infinite loop", "keeps exiting", "exiting", "docker container keeps restarting", "docker container dies and restarts automatically", "docker container randomly dies", "container dies and restarts automatically", "container won't stay running", "container keeps restarting", "container keeps dying", "container keeps exiting", "container is exiting", "container is restarting", "container is dying", "container is stopping", "container stopped working", "container crashes", "container failed", "container error", "docker container failure", "docker container problem", "docker container issue", "docker container trouble", "docker container won't start", "docker container can't start", "docker container not starting", "docker container broken"
                ],
                "patterns": [
                    r"docker\s+container.*restart",
                    r"container.*dies",
                    r"container.*stops.*randomly",
                    r"container.*keeps.*restart",
                    r"container.*automatically.*restart",
                    r"container.*won't.*stay.*running",
                    r"container.*keep.*restart",
                    r"container.*keep.*dying",
                    r"container.*loop.*restart",
                    r"container.*infinite.*loop",
                    r"container.*keeps.*exiting",
                    r"container.*exiting.*repeatedly",
                    r"docker\s+container.*keeps.*restarting",
                    r"docker\s+container.*dies.*and.*restarts",
                    r"docker\s+container.*dies.*and.*restarts.*automatically",
                    r"docker\s+container.*won't.*stay.*running",
                    r"docker\s+container.*can't.*start",
                    r"docker\s+container.*is.*dying",
                    r"docker\s+container.*is.*stopping",
                    r"docker\s+container.*stopped.*working",
                    r"docker\s+container.*crashes",
                    r"docker\s+container.*failed",
                    r"docker\s+container.*error",
                    r"docker\s+container.*problem",
                    r"docker\s+container.*trouble",
                    r"docker\s+container.*issue",
                    r"docker\s+container.*won't.*start",
                    r"docker\s+container.*can't.*start",
                    r"docker\s+container.*not.*starting"
                ],
                "confidence": 0.85,
                "allowed_subsystems": ["docker"],
                "likely_causes": [
                    "Resource exhaustion (memory/CPU)",
                    "Application crash/exception",
                    "Health check failures",
                    "Configuration conflicts",
                    "Dependency issues"
                ],
                "recommended_first_checks": [
                    "Check container logs: `docker logs container_name`",
                    "Monitor resource usage: `docker stats container_name`",
                    "Inspect container state: `docker inspect container_name`",
                    "Check for infinite loops: `docker logs --tail 100 container_name`"
                ],
                "low_risk_initial_actions": [
                    "Check recent application changes or deployments",
                    "Verify external service connectivity",
                    "Review resource limits and requests",
                    "Check upstream documentation or bug reports"
                ]
            },
            
            # systemd/service failures
            "systemd_service_failed": {
                "keywords": ["systemd", "service", "failed", "start", "restart", "status", "inactive", "journalctl", "dependency", "systemctl", "fails", "start myservice", "myservice", "service won't start", "service can't start"],
                "phrases": ["failed to start", "service failed", "inactive (dead)", "cannot start", "won't start", "dependency errors", "journalctl shows", "systemctl status shows", "systemctl start fails", "service won't start", "start myservice fails", "myservice failed", "myservice won't start"],
                "patterns": [
                    r"service.*failed.*start",
                    r"systemctl.*failed",
                    r"service.*inactive.*dead",
                    r"journalctl.*dependency.*error",
                    r"systemctl.*status.*inactive",
                    r"service.*won't.*start",
                    r"systemctl.*start.*fails",
                    r"start.*myservice.*fails",
                    r"myservice.*failed",
                    r"myservice.*won't.*start"
                ],
                "confidence": 0.9,
                "likely_causes": [
                    "Configuration file errors",
                    "Port conflicts or binding issues",
                    "Missing dependencies",
                    "Service dependency order issues"
                ],
                "recommended_first_checks": [
                    "Check service status: `systemctl status service_name`",
                    "Review service logs: `journalctl -u service_name`",
                    "Check dependencies: `systemctl list-dependencies service_name`"
                ],
                "low_risk_initial_actions": [
                    "Check if required ports are available: `netstat -tlnp | grep port`",
                    "Verify configuration file permissions",
                    "Test configuration with validation tools"
                ]
            },
            
            # command not found
            "command_not_found": {
                "keywords": ["command", "not", "found", "not found", "bash", "shell"],
                "phrases": ["command not found", "not found", "no such file", "command: not found"],
                "patterns": [
                    r"command\s+not\s+found",
                    r"bash:.*not\s+found",
                    r".*:\s+command\s+not\s+found"
                ],
                "confidence": 0.95,
                "allowed_subsystems": [],  # Broad/multi-subsystem allowed
                "likely_causes": [
                    "Command not installed",
                    "PATH environment issue",
                    "Typo in command name",
                    "Wrong shell or environment"
                ],
                "recommended_first_checks": [
                    "Verify command exists: `which command_name`",
                    "Check PATH: `echo $PATH`",
                    "Search for command: `find /usr -name command_name 2>/dev/null`"
                ],
                "low_risk_initial_actions": [
                    "Check correct spelling and case",
                    "Try full path to command",
                    "Install missing package if identified"
                ]
            },
            
            # network connectivity issues
            "network_connectivity": {
                "keywords": ["network", "connection", "refused", "timeout", "dns", "resolve", "host"],
                "phrases": ["connection refused", "connection timed out", "name resolution failed", "dns"],
                "patterns": [
                    r"connection\s+refused",
                    r"connection\s+timed\s+out",
                    r"name\s+resolution\s+failed",
                    r"dns.*failed"
                ],
                "confidence": 0.85,
                "allowed_subsystems": ["curl", "wget", "ssh", "scp"],  # Cross-subsystem allowed
                "likely_causes": [
                    "Service not running on target host",
                    "Firewall blocking connection",
                    "DNS resolution problems",
                    "Network configuration issues"
                ],
                "recommended_first_checks": [
                    "Test basic connectivity: `ping host`",
                    "Check port availability: `telnet host port`",
                    "Verify DNS resolution: `nslookup host`"
                ],
                "low_risk_initial_actions": [
                    "Check local network interface: `ip addr show`",
                    "Verify firewall rules: `iptables -L`",
                    "Test with different host/port"
                ]
            },
            
            # configuration/syntax errors
            "config_syntax_error": {
                "keywords": ["config", "syntax", "error", "parse", "invalid", "malformed"],
                "phrases": ["syntax error", "parse error", "invalid syntax", "malformed"],
                "patterns": [
                    r"syntax\s+error",
                    r"parse\s+error",
                    r"invalid\s+syntax",
                    r"malformed.*config"
                ],
                "confidence": 0.9,
                "allowed_subsystems": [],  # Broad/cross-subsystem allowed
                "likely_causes": [
                    "Typo in configuration file",
                    "Wrong format or version",
                    "Missing required sections",
                    "Incorrect quoting or escaping"
                ],
                "recommended_first_checks": [
                    "Validate config syntax: `config_validator file`",
                    "Check file encoding and line endings",
                    "Review recent changes to configuration"
                ],
                "low_risk_initial_actions": [
                    "Backup current config and restore known-good version",
                    "Use configuration validation tools",
                    "Check documentation for correct format"
                ]
            },
            
            # git authentication/remote issues
            "git_auth_remote": {
                "keywords": ["git", "push", "pull", "clone", "auth", "authentication", "remote", "origin"],
                "phrases": ["authentication failed", "permission denied", "remote rejected", "git push"],
                "patterns": [
                    r"git.*push.*denied",
                    r"authentication.*failed",
                    r"remote.*rejected",
                    r"permission.*denied.*git"
                ],
                "confidence": 0.85,
                "allowed_subsystems": ["git"],
                "likely_causes": [
                    "Incorrect credentials or token",
                    "SSH key not configured",
                    "Repository permissions",
                    "Network/firewall issues"
                ],
                "recommended_first_checks": [
                    "Test git authentication: `git config --list | grep user`",
                    "Check remote URL: `git remote -v`",
                    "Test SSH connection: `ssh -T git@github.com`"
                ],
                "low_risk_initial_actions": [
                    "Verify git credentials are current",
                    "Check SSH key configuration",
                    "Test with HTTPS alternative if SSH fails"
                ]
            },
        }
    def detect_incident(self, user_input: str, command_context: Optional[Dict[str, Any]] = None) -> Optional[IncidentSignature]:
        """Detect incident signature from user input with optional command context."""
        user_input_lower = user_input.lower()
        
        # Detect signal families for conflict analysis
        signal_families = self._detect_conflicting_signals(user_input)
        signal_weights = self._calculate_signal_family_weights(user_input_lower)
        dominant_family = self._get_dominant_signal_family(signal_weights)
        
        # Calculate adjusted confidence for all incident signatures
        candidate_incidents = []
        
        for incident_type, signature_config in self.signatures.items():
            if not self._passes_negative_filters(incident_type, user_input_lower, signal_weights):
                continue

            # Check basic keyword/phrases/patterns matching
            keyword_matches = self._check_keywords(user_input_lower, signature_config.get('keywords', []))
            phrase_matches = self._check_phrases(user_input_lower, signature_config.get('phrases', []))
            pattern_matches = self._check_patterns(user_input, signature_config.get('patterns', []))
            
            # Calculate base confidence
            total_matches = len(keyword_matches) + len(phrase_matches) + len(pattern_matches)
            if total_matches == 0:
                continue
            
            base_confidence = signature_config.get('confidence', 0.5)
            
            # Apply command context boost if available
            context_boost = self._calculate_context_boost(incident_type, command_context)
            boosted_confidence = min(0.95, base_confidence + context_boost)  # Cap at 0.95
            
            # Apply subsystem scoping penalties
            scoped_confidence = self._apply_subsystem_scoping(incident_type, signature_config, command_context, boosted_confidence)
            
            # Apply signal family conflict penalties
            conflict_penalty = self._calculate_conflict_penalty_for_incident(scoped_confidence, signal_families)
            final_confidence = max(0.0, scoped_confidence - conflict_penalty)

            # Weight toward the dominant signal family (prevents permissive keyword matches from winning)
            incident_family = self._get_incident_signal_family(incident_type)
            if incident_family and dominant_family and incident_family == dominant_family:
                # Small deterministic boost; the rest of scoring remains unchanged.
                final_confidence = min(0.95, final_confidence + 0.08)
            elif dominant_family and incident_family != dominant_family and dominant_family in signal_families:
                # If we have a clear dominant family and this incident doesn't explain it, penalize lightly.
                final_confidence = max(0.0, final_confidence - 0.08)

            # Reward specificity: patterns/phrases are stronger evidence than loose keywords.
            final_confidence = min(
                0.95,
                final_confidence
                + (0.04 * len(pattern_matches))
                + (0.02 * len(phrase_matches))
                + (0.005 * len(keyword_matches)),
            )
            
            # Store candidate with all scoring information
            candidate_incidents.append({
                'incident_type': incident_type,
                'confidence': final_confidence,
                'base_confidence': base_confidence,
                'boosted_confidence': boosted_confidence,
                'scoped_confidence': scoped_confidence,
                'conflict_penalty': conflict_penalty,
                'matched_terms': keyword_matches + phrase_matches + pattern_matches,
                'signature_config': signature_config
            })
        
        # Apply ambiguity blocking rules
        if len(signal_families) >= 3:
            # High ambiguity: require stricter threshold
            min_threshold = max(self.min_confidence_threshold, 0.85)
        elif len(signal_families) >= 2:
            # Moderate ambiguity: require moderate threshold
            min_threshold = max(self.min_confidence_threshold, 0.8)
        else:
            # Low ambiguity: normal threshold
            min_threshold = self.min_confidence_threshold
        
        # Sort candidates by final adjusted confidence
        candidate_incidents.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Check if best candidate meets threshold
        if candidate_incidents and candidate_incidents[0]['confidence'] >= min_threshold:
            best_candidate = candidate_incidents[0]
            
            # --- Sanity check: candidate must explain observed signal families ---
            incident_family = self._get_incident_signal_family(best_candidate['incident_type']) 

            # Block incidents that clearly don't match observed signals
            if signal_families and incident_family not in signal_families:
                # Example: pip incident selected when signals are python + network
                return None
            
            # Additional check: if conflicting families include clearly different categories
            # and confidence is not very high, prefer no match
            if len(signal_families) >= 3 and best_candidate['confidence'] < 0.85:
                # Different categories detected (python + network + config, etc.)
                # Prefer no confident incident match
                return None
            
            return IncidentSignature(
                incident_type=best_candidate['incident_type'],
                confidence=best_candidate['confidence'],
                matched_terms=best_candidate['matched_terms'],
                likely_causes=best_candidate['signature_config'].get('likely_causes', []),
                recommended_first_checks=best_candidate['signature_config'].get('recommended_first_checks', []),
                low_risk_initial_actions=best_candidate['signature_config'].get('low_risk_initial_actions', []),
                allowed_subsystems=best_candidate['signature_config'].get('allowed_subsystems', [])
            )
        
        # No confident incident match
        return None

    def _calculate_signal_family_weights(self, user_input_lower: str) -> Dict[str, float]:
        """Calculate weighted strength per signal family from raw text.

        This is intentionally simple and deterministic: strong phrases contribute more than loose keywords.
        """
        weights: Dict[str, float] = {
            "python_runtime": 0.0,
            "network_connectivity": 0.0,
            "permissions": 0.0,
            "systemd_service": 0.0,
            "docker_container": 0.0,
            "git_auth": 0.0,
            "config_syntax": 0.0,
            "error": 0.0,
        }

        # Strong runtime evidence
        if "traceback" in user_input_lower:
            weights["python_runtime"] += 2.5
        if "exception" in user_input_lower or "valueerror" in user_input_lower or "typeerror" in user_input_lower:
            weights["python_runtime"] += 1.5

        # Strong network/service evidence
        if "connection refused" in user_input_lower:
            weights["network_connectivity"] += 2.5
        if "timed out" in user_input_lower or "timeout" in user_input_lower:
            weights["network_connectivity"] += 2.0
        if "port" in user_input_lower and ("not listening" in user_input_lower or "refused" in user_input_lower):
            weights["network_connectivity"] += 1.5

        # Service management evidence
        if "systemctl" in user_input_lower or "journalctl" in user_input_lower or "systemd" in user_input_lower:
            weights["systemd_service"] += 2.0
        if "failed to start" in user_input_lower and "service" in user_input_lower:
            weights["systemd_service"] += 1.5

        # Permissions evidence
        if "permission denied" in user_input_lower:
            weights["permissions"] += 2.5
        # Loose permission signals should not overpower stronger families
        if "permission" in user_input_lower or "denied" in user_input_lower:
            weights["permissions"] += 0.5

        # Pip-specific evidence (kept as part of permissions, but used for negative filtering)
        if "pip" in user_input_lower and "install" in user_input_lower:
            weights["permissions"] += 1.0

        return weights

    def _get_dominant_signal_family(self, signal_weights: Dict[str, float]) -> Optional[str]:
        """Return the dominant signal family if clearly dominant, else None."""
        if not signal_weights:
            return None
        # Deterministic tie-breaking by key name to keep ordering stable
        best_family, best_score = sorted(signal_weights.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if best_score <= 0.0:
            return None
        # Require dominance over runner-up to avoid flip-flopping on weak evidence
        sorted_scores = sorted(signal_weights.values(), reverse=True)
        runner_up = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        if best_score >= runner_up + 1.0:
            return best_family
        return None

    def _passes_negative_filters(self, incident_type: str, user_input_lower: str, signal_weights: Dict[str, float]) -> bool:
        """Block clearly incorrect incident matches using deterministic negative rules."""
        # If runtime evidence exists, don't classify pip permission issues.
        has_traceback = "traceback" in user_input_lower or signal_weights.get("python_runtime", 0.0) >= 2.0
        has_network = (
            "connection refused" in user_input_lower
            or "not listening" in user_input_lower
            or signal_weights.get("network_connectivity", 0.0) >= 2.0
        )

        if incident_type == "pip_permission_denied":
            # Must have explicit pip evidence; generic "permission denied" isn't enough.
            has_pip_terms = ("pip" in user_input_lower) or ("pip install" in user_input_lower)
            if not has_pip_terms:
                return False
            # If strong network or runtime evidence is present, this is almost certainly not pip permission.
            if has_traceback or has_network:
                return False

        # Prefer network/service when strong connectivity signals appear.
        if has_network and incident_type in ("pip_permission_denied", "config_syntax_error", "git_auth_remote"):
            return False

        return True
    
    def _apply_subsystem_scoping(self, incident_type: str, signature_config: Dict[str, Any], command_context: Optional[Dict[str, Any]], base_confidence: float) -> float:
        """Apply subsystem scoping and conflict penalties to confidence."""
        if not command_context:
            return base_confidence
        
        detected_subsystem = command_context.get('subsystem')
        allowed_subsystems = signature_config.get('allowed_subsystems', [])
        
        # Subsystem scoping: prefer signatures that match detected subsystem
        if detected_subsystem and allowed_subsystems:
            if detected_subsystem in allowed_subsystems:
                # Boost confidence for correct subsystem match
                return base_confidence
            else:
                # Heavy penalty for subsystem mismatch
                return max(0.0, base_confidence - 0.3)
        
        # For broad signatures (empty allowed_subsystems), no scoping applied
        return base_confidence
    
    def _detect_conflicting_signals(self, user_input: str) -> List[str]:
        """Detect conflicting signal families in input."""
        signal_families = {
            'python_runtime': ['traceback', 'runtimeerror', 'typeerror', 'attributeerror', 'importerror'],
            'network_connectivity': ['connection', 'timeout', 'refused', 'dns', 'resolve', 'host', 'network'],
            'permissions': ['permission', 'denied', 'access', 'forbidden', 'unauthorized'],
            'docker_container': ['container', 'docker', 'restart', 'dies', 'stops', 'crash'],
            'git_auth': ['git', 'push', 'pull', 'clone', 'auth', 'authentication', 'remote'],
            'systemd_service': ['systemd', 'service', 'systemctl', 'journalctl', 'system'],
            'config_syntax': ['config', 'syntax', 'parse', 'invalid', 'malformed']
        }
        
        user_input_lower = user_input.lower()
        detected_families = []
        
        for family, keywords in signal_families.items():
            if any(keyword in user_input_lower for keyword in keywords):
                detected_families.append(family)
        
        return detected_families
    
    def _calculate_confidence_penalty(self, base_confidence: float, conflicting_families: List[str]) -> float:
        """Calculate confidence penalty for conflicting signal families."""
        if len(conflicting_families) <= 1:
            return base_confidence
        
        # Penalty for multiple conflicting signal families
        conflict_penalty = min(0.4, 0.1 * len(conflicting_families))
        return max(0.0, base_confidence - conflict_penalty)
    
    def _calculate_conflict_penalty_for_incident(self, base_confidence: float, signal_families: List[str]) -> float:
        """Calculate conflict penalty for an incident based on signal families."""
        if len(signal_families) <= 1:
            return 0.0
        
        # Penalty for multiple conflicting signal families
        conflict_penalty = min(0.4, 0.1 * len(signal_families))
        return conflict_penalty
    
    def _check_keywords(self, user_input: str, keywords: List[str]) -> List[str]:
        """Check for keyword matches."""
        matches = []
        for keyword in keywords:
            if keyword.lower() in user_input.lower():
                matches.append(keyword)
        return matches
    
    def _check_phrases(self, user_input: str, phrases: List[str]) -> List[str]:
        """Check for phrase matches."""
        matches = []
        for phrase in phrases:
            if phrase in user_input:
                matches.append(phrase)
        return matches
    
    def _check_patterns(self, user_input: str, patterns: List[str]) -> List[str]:
        """Check for regex pattern matches."""
        matches = []
        for pattern in patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                matches.append(pattern)
        return matches
    
    def _calculate_context_boost(self, incident_type: str, command_context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence boost based on command context."""
        if not command_context:
            return 0.0
        
        subsystem = command_context.get('subsystem')
        if not subsystem:
            return 0.0
        
        # Define context boosts for specific incident type + subsystem combinations
        context_boosts = {
            # pip + permission_denied -> pip_permission_denied
            ('pip_permission_denied', 'pip'): 0.15,
            ('pip_permission_denied', 'python'): 0.10,
            
            # docker + container issues
            ('docker_container_restart', 'docker'): 0.15,
            ('docker_permission_denied', 'docker'): 0.15,
            ('docker_command_not_found', 'docker'): 0.10,
            
            # systemctl + service issues
            ('systemd_service_failed', 'systemctl'): 0.15,
            ('systemd_service_failed', 'journalctl'): 0.10,
            ('systemd_permission_denied', 'systemctl'): 0.15,
            
            # git + auth issues
            ('git_auth_remote', 'git'): 0.15,
            ('git_permission_denied', 'git'): 0.10,
            ('git_command_not_found', 'git'): 0.10,
            
            # npm/node + command issues
            ('npm_permission_denied', 'npm'): 0.15,
            ('npm_permission_denied', 'node'): 0.10,
            ('npm_command_not_found', 'npm'): 0.10,
            ('npm_command_not_found', 'node'): 0.10,
            
            # chmod/chown + permission issues
            ('permission_denied', 'chmod'): 0.15,
            ('permission_denied', 'chown'): 0.15,
            ('permission_denied', 'sudo'): 0.10,
            
            # curl/wget + network issues
            ('network_connectivity', 'curl'): 0.10,
            ('network_connectivity', 'wget'): 0.10,
            
            # ssh/scp + auth/network issues
            ('git_auth_remote', 'ssh'): 0.10,
            ('network_connectivity', 'ssh'): 0.10,
            ('network_connectivity', 'scp'): 0.10,
            
            # General command not found
            ('command_not_found', 'pip'): 0.10,
            ('command_not_found', 'docker'): 0.10,
            ('command_not_found', 'git'): 0.10,
            ('command_not_found', 'npm'): 0.10,
            ('command_not_found', 'node'): 0.10,
            ('command_not_found', 'python'): 0.10,
            ('command_not_found', 'systemctl'): 0.10,
            ('command_not_found', 'journalctl'): 0.10,
        }
        
        return context_boosts.get((incident_type, subsystem), 0.0)
