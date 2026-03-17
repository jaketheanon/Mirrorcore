"""
Reasoning Response Engine

Lightweight first-pass reasoning response generator for Mirrorcore.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import re


@dataclass
class MessageCategory:
    """Message classification categories."""
    TROUBLESHOOTING = "troubleshooting"
    DECISION_SUPPORT = "decision_support"
    GENERAL_REASONING = "general_reasoning"
    UNKNOWN = "unknown"


@dataclass
class RootCauseHypothesis:
    """Structured root-cause hypothesis with ranking metadata."""
    text: str
    related_families: List[str]
    related_subsystems: List[str]
    category: str
    score: float
    reason: str
    eliminated: bool = False
    elimination_reason: Optional[str] = None
    supporting_evidence: List[str] = None
    conflicting_evidence: List[str] = None


@dataclass
class ResponseContext:
    """Context for response generation."""
    user_input: str
    persona_profile: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    memory_store: Any
    incident_id: Optional[str] = None
    suggested_fixes: List[str] = None
    extracted_signals: List[str] = None


@dataclass
class ResponseResult:
    """Result of response generation with optional outcome capture."""
    response: str
    should_prompt_outcome: bool = False
    incident_id: Optional[str] = None
    suggested_fixes: List[str] = None
    escalation_guidance: Optional[str] = None


@dataclass
class ConfidenceEvaluation:
    """Root cause confidence evaluation result."""
    confidence_level: str  # high, moderate, low, unknown
    incident_confidence: float
    failed_fixes_ratio: float
    successful_fixes_count: int
    conflicting_signals: bool
    no_incident_match: bool
    escalation_triggered: bool

@dataclass
class InvestigationStallDetection:
    """Investigation stall detection result."""
    is_stalled: bool
    stall_reason: str
    failed_families: List[str]
    alternative_family: str
    progress_metrics: Dict[str, float]

class ReasoningResponseEngine:
    """Lightweight deterministic reasoning response generator."""
    
    def __init__(self):
        # Message classification keywords
        self.troubleshooting_keywords = [
            "error", "failed", "crash", "exception", "broken", "not working",
            "can't", "cannot", "unable", "refused", "denied", "timeout",
            "missing", "not found", "permission", "access", "install"
        ]
        
        # Investigation strategy families
        self.strategy_families = {
            'connectivity': ['direct_connection_test', 'port_connectivity_check', 'network_path_validation'],
            'configuration': ['config_file_validation', 'environment_variable_check', 'dependency_verification'],
            'service': ['service_status_check', 'service_log_review', 'service_dependency_check'],
            'permissions': ['file_permission_check', 'user_access_verification', 'resource_accessibility_test']
        }

        self.troubleshooting_patterns = [
            r".*error.*", r".*failed.*", r".*exception.*", r".*crash.*",
            r".*not.*work.*", r".*can't.*", r".*cannot.*", r".*unable.*"
        ]
        
        self.decision_keywords = [
            "should", "could", "would", "might", "better", "worse", "choose",
            "option", "alternative", "prefer", "recommend", "suggest"
        ]
        
        self.decison_keywords = self.decision_keywords  # Keep for compatibility
        
        # Escalation threshold constants
        self.LOW_CONFIDENCE_THRESHOLD = 0.6
        self.HIGH_FAILED_RATIO_THRESHOLD = 0.7
        self.MULTIPLE_SIGNALS_THRESHOLD = 3
        
        # Root-cause hypothesis templates
        self.hypothesis_templates = {
            # config + runtime
            ("config", "runtime"): [
                "Configuration errors may be causing the application/runtime failure.",
                "Invalid configuration values could be preventing proper startup or execution.",
                "The runtime failure may be a symptom of underlying configuration problems."
            ],
            # runtime + network
            ("runtime", "network"): [
                "The application may be failing before or during backend communication.",
                "Runtime exceptions could be preventing proper network connection establishment.",
                "The network connectivity failures may be downstream symptoms of the application failure."
            ],
            # config + network
            ("config", "network"): [
                "Misconfiguration may be preventing correct backend connection settings.",
                "Invalid configuration could be blocking network service initialization.",
                "Network connection failures may be caused by incorrect configuration parameters."
            ],
            # permissions + runtime
            ("permissions", "runtime"): [
                "Permission problems may be blocking startup or normal execution.",
                "Access denied errors could be preventing the application from running properly.",
                "The runtime failure may be caused by insufficient filesystem or system permissions."
            ],
            # docker + runtime
            ("docker", "runtime"): [
                "Container restart behavior may be driven by an internal application exception.",
                "The container may be crashing due to runtime failures within the application.",
                "Application runtime errors could be causing the container to exit and restart."
            ],
            # service + network
            ("service", "network"): [
                "Service startup failures may be preventing network service availability.",
                "The service may be failing to bind to network ports or establish connections.",
                "Network connectivity issues could be preventing the service from starting properly."
            ],
            # config + runtime + network (complex case)
            ("config", "runtime", "network"): [
                "Configuration errors may be causing both runtime failures and network connectivity problems.",
                "Invalid configuration could be preventing proper application startup and backend communication.",
                "The runtime failure may be primary, with network issues as downstream symptoms of misconfiguration."
            ],
            # permissions + runtime + network (complex case)
            ("permissions", "runtime", "network"): [
                "Permission issues may be blocking both application execution and network access.",
                "Access denied errors could prevent the application from starting and connecting to services.",
                "Both runtime and network failures may be caused by insufficient system permissions."
            ]
        }
        
        # Diagnostic command templates for hypothesis categories
        self.diagnostic_templates = {
            # config_runtime
            "config_runtime": [
                {
                    "description": "Inspect configuration file for syntax errors",
                    "command": "cat config.yaml | grep -n -A5 -B5 error",
                    "subsystem_specific": {
                        "python": "python -c \"import yaml; yaml.safe_load(open('config.yaml'))\"",
                        "node": "node -e \"require('./config.json')\"",
                        "docker": "docker exec <container> cat /app/config.yaml"
                    }
                },
                {
                    "description": "Validate environment variables",
                    "command": "env | grep -E '(CONFIG|APP|DATABASE)' | sort",
                    "subsystem_specific": {
                        "python": "python -c \"import os; print({k:v for k,v in os.environ.items() if 'CONFIG' in k or 'APP' in k or 'DB' in k})\"",
                        "node": "node -e \"console.log(Object.keys(process.env).filter(k => /CONFIG|APP|DB/i.test(k)))\"",
                        "docker": "docker exec <container> env | grep -E '(CONFIG|APP|DATABASE)'"
                    }
                },
                {
                    "description": "Run configuration validation",
                    "command": "./validate-config.sh",
                    "subsystem_specific": {
                        "python": "python validate_config.py",
                        "node": "node validate_config.js",
                        "docker": "docker exec <container> python validate_config.py"
                    }
                }
            ],
            # runtime_network
            "runtime_network": [
                {
                    "description": "Inspect the full traceback",
                    "command": "cat application.log | grep -A20 Traceback",
                    "subsystem_specific": {
                        "python": "python -m traceback -l application.log",
                        "node": "node --trace-warnings app.js",
                        "docker": "docker logs <container> | grep -A20 Traceback"
                    }
                },
                {
                    "description": "Run the application in verbose/debug mode",
                    "command": "python app.py --debug",
                    "subsystem_specific": {
                        "python": "python app.py --debug -v",
                        "node": "DEBUG=* node app.js",
                        "docker": "docker exec <container> python app.py --debug"
                    }
                },
                {
                    "description": "Check backend connectivity directly",
                    "command": "curl localhost:<backend_port>/health",
                    "subsystem_specific": {
                        "python": "python -c \"import requests; print(requests.get('http://localhost:8080/health').status_code)\"",
                        "node": "curl -f http://localhost:8080/health",
                        "docker": "docker exec <container> curl localhost:8080/health"
                    }
                }
            ],
            # config_network
            "config_network": [
                {
                    "description": "Check network configuration settings",
                    "command": "cat /etc/network/interfaces",
                    "subsystem_specific": {
                        "python": "python -c \"import socket; print(socket.gethostbyname(socket.gethostname()))\"",
                        "node": "node -e \"console.log(require('os').networkInterfaces())\"",
                        "docker": "docker exec <container> cat /etc/hosts"
                    }
                },
                {
                    "description": "Validate backend endpoint configuration",
                    "command": "grep -r 'localhost\\|127.0.0.1' config/",
                    "subsystem_specific": {
                        "python": "python -c \"import json; print(json.load(open('config.json'))['backend_url'])\"",
                        "node": "node -e \"console.log(JSON.parse(require('fs').readFileSync('config.json')).backend_url)\"",
                        "docker": "docker exec <container> grep -r 'backend_url' /app/config/"
                    }
                },
                {
                    "description": "Test DNS resolution",
                    "command": "nslookup backend.service",
                    "subsystem_specific": {
                        "python": "python -c \"import socket; print(socket.gethostbyname('backend.service'))\"",
                        "node": "node -e \"require('dns').lookup('backend.service', console.log)\"",
                        "docker": "docker exec <container> nslookup backend.service"
                    }
                }
            ],
            # permissions_runtime
            "permissions_runtime": [
                {
                    "description": "Check file ownership and permissions",
                    "command": "ls -l /app/data/",
                    "subsystem_specific": {
                        "python": "python -c \"import os; print(oct(os.stat('/app/data').st_mode)[-3:])\"",
                        "node": "node -e \"console.log(require('fs').statSync('/app/data').mode.toString(8))\"",
                        "docker": "docker exec <container> ls -l /app/data/"
                    }
                },
                {
                    "description": "Inspect permission errors in detail",
                    "command": "grep -i 'permission\\|denied\\|access' application.log",
                    "subsystem_specific": {
                        "python": "python -c \"import logging; logging.basicConfig(level=logging.DEBUG); import app\"",
                        "node": "node --trace-warnings app.js 2>&1 | grep -i permission",
                        "docker": "docker logs <container> | grep -i permission"
                    }
                },
                {
                    "description": "Verify executing user context",
                    "command": "whoami && groups",
                    "subsystem_specific": {
                        "python": "python -c \"import os; print(f'User: {os.getenv(\"USER\")}, Groups: {os.getenv(\"GROUPS\")}')\"",
                        "node": "node -e \"console.log(`User: ${require('os').userInfo().username}, Groups: ${process.getgroups()}`)\"",
                        "docker": "docker exec <container> whoami && docker exec <container> groups"
                    }
                }
            ],
            # docker_runtime
            "docker_runtime": [
                {
                    "description": "Inspect container logs for errors",
                    "command": "docker logs <container>",
                    "subsystem_specific": {
                        "python": "docker logs <container> | grep -A10 -B10 error",
                        "node": "docker logs <container> | grep -A10 -B10 ERROR",
                        "docker": "docker logs <container> --tail 50"
                    }
                },
                {
                    "description": "Check container restart status",
                    "command": "docker ps -a | grep <container>",
                    "subsystem_specific": {
                        "python": "docker inspect <container> | grep -A5 -B5 'RestartPolicy'",
                        "node": "docker inspect <container> | jq '.[0].RestartPolicy'",
                        "docker": "docker inspect <container> --format='{{.RestartPolicy}}'"
                    }
                },
                {
                    "description": "Examine container configuration",
                    "command": "docker inspect <container>",
                    "subsystem_specific": {
                        "python": "docker inspect <container> | grep -A10 -B10 'Environment'",
                        "node": "docker inspect <container> | jq '.[0].Config.Env'",
                        "docker": "docker inspect <container> --format='{{.Config.Env}}'"
                    }
                }
            ],
            # service_network
            "service_network": [
                {
                    "description": "Check systemd service status",
                    "command": "systemctl status <service>",
                    "subsystem_specific": {
                        "python": "systemctl status python-app.service",
                        "node": "systemctl status node-app.service",
                        "docker": "systemctl status docker-app.service"
                    }
                },
                {
                    "description": "Inspect service logs",
                    "command": "journalctl -u <service> -n 20",
                    "subsystem_specific": {
                        "python": "journalctl -u python-app.service -n 20 --no-pager",
                        "node": "journalctl -u node-app.service -n 20 --no-pager",
                        "docker": "journalctl -u docker-app.service -n 20 --no-pager"
                    }
                },
                {
                    "description": "Verify port listening status",
                    "command": "ss -tulnp | grep <port>",
                    "subsystem_specific": {
                        "python": "netstat -tulnp | grep :8080",
                        "node": "netstat -tulnp | grep :3000",
                        "docker": "ss -tulnp | grep :8080"
                    }
                }
            ],
            # concurrent_failures
            "concurrent_failures": [
                {
                    "description": "Check system resource usage",
                    "command": "top -n 1 | head -20",
                    "subsystem_specific": {
                        "python": "python -c \"import psutil; print(psutil.cpu_percent(), psutil.virtual_memory().percent)\"",
                        "node": "node -e \"console.log(require('os').loadavg(), require('os').totalmem() / 1024 / 1024)\"",
                        "docker": "docker stats --no-stream"
                    }
                },
                {
                    "description": "Verify network connectivity",
                    "command": "ping -c 3 8.8.8.8",
                    "subsystem_specific": {
                        "python": "python -c \"import subprocess; subprocess.run(['ping', '-c', '3', '8.8.8.8'])\"",
                        "node": "require('child_process').execSync('ping -c 3 8.8.8.8').toString()",
                        "docker": "docker exec <container> ping -c 3 8.8.8.8"
                    }
                },
                {
                    "description": "Check system logs for errors",
                    "command": "dmesg | tail -20",
                    "subsystem_specific": {
                        "python": "journalctl -k -n 20",
                        "node": "journalctl -k -n 20",
                        "docker": "dmesg | tail -20"
                    }
                }
            ]
        }
        
        # Signal family to layer mapping
        self.signal_family_to_layer = {
            "python_runtime": "runtime",
            "error": "runtime", 
            "traceback": "runtime",
            "exception": "runtime",
            "config_syntax": "config",
            "syntax_error": "config",  # Add syntax_error mapping
            "permissions": "permissions",
            "permission_denied": "permissions",
            "network_connectivity": "network",
            "connection_refused": "network",
            "timeout": "network",
            "docker_container": "docker",
            "systemd_service": "service"
        }
    
    def generate_root_cause_hypotheses(self, parsed_log, command_context=None) -> List[str]:
        """Generate deterministic root-cause hypotheses from signal families."""
        if not parsed_log.signal_families or len(parsed_log.signal_families) <= 1:
            return []
        
        # Map signal families to layers
        layers = []
        for family in parsed_log.signal_families:
            layer = self.signal_family_to_layer.get(family)
            if layer and layer not in layers:
                layers.append(layer)
        
        # Sort layers for consistent template matching
        layers.sort()
        
        # Find matching template
        hypotheses = []
        layer_tuple = tuple(layers)
        
        # Try exact match first
        if layer_tuple in self.hypothesis_templates:
            hypotheses = self.hypothesis_templates[layer_tuple][:3]  # Limit to 3
        else:
            # Try partial matches for complex cases
            if len(layers) >= 3:
                # For 3+ layers, try combinations
                for template_layers in self.hypothesis_templates:
                    if len(template_layers) == 3 and all(layer in layers for layer in template_layers):
                        hypotheses = self.hypothesis_templates[template_layers][:3]
                        break
            elif len(layers) == 2:
                # For 2 layers, try exact match or generic
                for template_layers in self.hypothesis_templates:
                    if len(template_layers) == 2 and set(template_layers) == set(layers):
                        hypotheses = self.hypothesis_templates[template_layers][:3]
                        break
                
                # Generic fallback for 2-layer combinations
                if not hypotheses:
                    layer1, layer2 = layers
                    hypotheses = [
                        f"The {layer1} issues may be causing or related to the {layer2} problems.",
                        f"Both {layer1} and {layer2} failures may be symptoms of a common underlying issue.",
                        f"Investigate whether the {layer1} failure is primary and the {layer2} issues are downstream."
                    ]
        
        # Ground hypotheses in observed content
        grounded_hypotheses = []
        summary_lower = parsed_log.compact_summary_text.lower()
        
        for hypothesis in hypotheses:
            grounded = hypothesis
            
            # Add specific observed content
            if "configuration" in hypothesis.lower() and any(word in summary_lower for word in ["config", "configuration", "invalid"]):
                grounded = grounded.replace("Configuration", "Invalid configuration")
            elif "network" in hypothesis.lower() and any(word in summary_lower for word in ["connection", "refused", "timeout"]):
                if "connection refused" in summary_lower:
                    grounded = grounded.replace("network", "backend service (connection refused)")
                elif "timeout" in summary_lower:
                    grounded = grounded.replace("network", "backend service (timeout)")
            
            grounded_hypotheses.append(grounded)
        
        return grounded_hypotheses[:3]  # Return max 3 hypotheses
    
    def generate_ranked_root_cause_hypotheses(self, parsed_log, command_context=None, memory_store=None) -> List[RootCauseHypothesis]:
        """Generate and rank root-cause hypotheses with historical support."""
        # Generate basic hypotheses first
        basic_hypotheses = self.generate_root_cause_hypotheses(parsed_log, command_context)
        if not basic_hypotheses:
            return []
        
        # Convert to structured hypotheses
        structured_hypotheses = []
        layers = []
        for family in parsed_log.signal_families:
            layer = self.signal_family_to_layer.get(family)
            if layer and layer not in layers:
                layers.append(layer)
        
        detected_subsystem = command_context.get('subsystem') if command_context else None
        
        for i, hypothesis_text in enumerate(basic_hypotheses):
            # Determine hypothesis category based on layers
            if len(layers) >= 2:
                # Use consistent ordering: runtime before network, config before runtime, etc.
                ordered_layers = sorted(layers, key=lambda x: {
                    'config': 0, 'runtime': 1, 'network': 2, 'permissions': 3, 
                    'docker': 4, 'service': 5
                }.get(x, 99))
                category = "_".join(ordered_layers[:2])
            else:
                category = layers[0] if layers else "unknown"
            
            # Create structured hypothesis
            hypothesis = RootCauseHypothesis(
                text=hypothesis_text,
                related_families=parsed_log.signal_families[:],
                related_subsystems=[detected_subsystem] if detected_subsystem else [],
                category=category,
                score=0.0,  # Will be calculated below
                reason=""   # Will be calculated below
            )
            structured_hypotheses.append(hypothesis)
        
        # Score and rank hypotheses
        ranked_hypotheses = self.rank_root_cause_hypotheses(structured_hypotheses, parsed_log, memory_store)
        
        return ranked_hypotheses
    
    def rank_root_cause_hypotheses(self, hypotheses: List[RootCauseHypothesis], parsed_log, memory_store=None) -> List[RootCauseHypothesis]:
        """Rank hypotheses using deterministic scoring formula."""
        
        # Scoring weights
        FAMILY_OVERLAP_WEIGHT = 0.4
        SUBSYSTEM_MATCH_WEIGHT = 0.2
        HISTORICAL_SUCCESS_WEIGHT = 0.3
        HISTORICAL_FAILURE_WEIGHT = 0.2
        RECENCY_BONUS = 0.1
        
        # Get historical support data
        historical_support = self._get_historical_support(parsed_log, memory_store) if memory_store else {}
        
        for hypothesis in hypotheses:
            score = 0.0
            reason_parts = []
            
            # 1. Family overlap score
            overlap_count = len(set(hypothesis.related_families) & set(parsed_log.signal_families))
            family_score = FAMILY_OVERLAP_WEIGHT * overlap_count
            score += family_score
            if overlap_count > 0:
                reason_parts.append(f"strong {hypothesis.category} overlap")
            
            # 2. Subsystem match score
            if hypothesis.related_subsystems:
                subsystem_score = SUBSYSTEM_MATCH_WEIGHT
                score += subsystem_score
                reason_parts.append(f"matches {hypothesis.related_subsystems[0]} subsystem")
            
            # 3. Historical success/failure scores
            category_key = hypothesis.category
            if category_key in historical_support:
                success_count = historical_support[category_key].get('success_count', 0)
                failure_count = historical_support[category_key].get('failure_count', 0)
                
                success_score = HISTORICAL_SUCCESS_WEIGHT * min(success_count / 5.0, 1.0)  # Cap at 1.0
                failure_penalty = HISTORICAL_FAILURE_WEIGHT * min(failure_count / 5.0, 1.0)  # Cap at 1.0
                
                score += success_score
                score -= failure_penalty
                
                if success_count > 0:
                    reason_parts.append(f"similar sessions resolved by {hypothesis.category} correction")
                if failure_count > success_count:
                    reason_parts.append(f"historical support weaker for {hypothesis.category}")
            
            # 4. Recency bonus (simplified - just add small bonus for recent activity)
            if category_key in historical_support and historical_support[category_key].get('recent_success', False):
                score += RECENCY_BONUS
                reason_parts.append("recent successful resolution")
            
            # Normalize score to 0-1 range
            hypothesis.score = max(0.0, min(1.0, score))
            
            # Generate reason text
            if reason_parts:
                hypothesis.reason = "; ".join(reason_parts[:2])  # Limit to 2 parts
            else:
                hypothesis.reason = f"based on {hypothesis.category} pattern analysis"
        
        # Sort by score (descending)
        hypotheses.sort(key=lambda h: h.score, reverse=True)
        
        return hypotheses
    
    def _get_historical_support(self, parsed_log, memory_store) -> Dict[str, Dict[str, Any]]:
        """Retrieve historical support data for hypothesis categories."""
        support_data = {}
        
        try:
            # Get signal families for context
            signal_families = parsed_log.signal_families
            
            # Map signal families to potential fix patterns
            family_fix_patterns = {
                'config': ['config', 'configuration', 'settings', 'environment'],
                'runtime': ['restart', 'reload', 'reboot', 'debug', 'fix'],
                'network': ['connection', 'network', 'backend', 'service', 'port'],
                'permissions': ['permission', 'access', 'chmod', 'chown', 'user'],
                'docker': ['docker', 'container', 'image', 'build', 'run'],
                'service': ['service', 'systemctl', 'daemon', 'start', 'stop']
            }
            
            # For each hypothesis category, find relevant historical outcomes
            for category in support_data.keys() if support_data else []:
                pass  # Placeholder - will be filled below
            
            # Try to get relevant fixes from memory store
            if hasattr(memory_store, 'get_all_terminal_incidents'):
                incidents = memory_store.get_all_terminal_incidents(limit=100)
                
                # Categorize incidents by pattern matching
                category_stats = {}
                
                for incident in incidents:
                    solution_lower = incident.get('solution', '').lower()
                    success = incident.get('success', False)
                    
                    # Determine which category this incident belongs to
                    for category, patterns in family_fix_patterns.items():
                        if any(pattern in solution_lower for pattern in patterns):
                            if category not in category_stats:
                                category_stats[category] = {'success_count': 0, 'failure_count': 0, 'recent_success': False}
                            
                            if success:
                                category_stats[category]['success_count'] += 1
                                # Mark as recent if within last 10 incidents (simplified)
                                if len(incidents) <= 10:
                                    category_stats[category]['recent_success'] = True
                            else:
                                category_stats[category]['failure_count'] += 1
                            break
                
                support_data = category_stats
                
        except Exception:
            # If memory retrieval fails, return empty support data
            pass
        
        return support_data
    
    def generate_diagnostic_commands(self, top_hypothesis: RootCauseHypothesis, parsed_log, command_context=None) -> Dict[str, Any]:
        """Generate diagnostic command suggestions based on top hypothesis."""
        if not top_hypothesis:
            return {"primary": [], "additional": []}
        
        # Get diagnostic templates for the hypothesis category
        category = top_hypothesis.category
        templates = self.diagnostic_templates.get(category, [])
        
        if not templates:
            # Fallback to generic templates
            templates = self.diagnostic_templates.get("concurrent_failures", [])
        
        detected_subsystem = command_context.get('subsystem') if command_context else None
        
        # Generate primary commands (first 3)
        primary_commands = []
        for template in templates[:3]:
            command = self._adapt_command_to_subsystem(template, detected_subsystem)
            primary_commands.append({
                "description": template["description"],
                "command": command
            })
        
        # Generate additional commands (next 2)
        additional_commands = []
        for template in templates[3:5]:
            command = self._adapt_command_to_subsystem(template, detected_subsystem)
            additional_commands.append({
                "description": template["description"],
                "command": command
            })
        
        return {
            "primary": primary_commands,
            "additional": additional_commands
        }
    
    def _adapt_command_to_subsystem(self, template: Dict[str, Any], detected_subsystem: Optional[str]) -> str:
        """Adapt command template to detected subsystem."""
        base_command = template["command"]
        
        # If no subsystem detected or no subsystem-specific commands, return base
        if not detected_subsystem or "subsystem_specific" not in template:
            return base_command
        
        # Get subsystem-specific command
        subsystem_commands = template["subsystem_specific"]
        subsystem_command = subsystem_commands.get(detected_subsystem)
        
        if subsystem_command:
            return subsystem_command
        
        # Fallback to base command if no specific match
        return base_command
    
    def update_hypotheses_with_evidence(self, original_hypotheses: List[RootCauseHypothesis], 
                                       evidence_signals: List[str], session_context: Dict[str, Any]) -> List[RootCauseHypothesis]:
        """Update hypothesis scores based on new evidence."""
        updated_hypotheses = []
        
        # Evidence-to-hypothesis effect rules
        evidence_effects = {
            'connection_confirmed': {
                'boost': ['config_network'],
                'penalize': ['runtime_network', 'service_network']
            },
            'connection_refused': {
                'boost': ['runtime_network', 'service_network'],
                'penalize': ['config_network']
            },
            'timeout': {
                'boost': ['runtime_network', 'service_network'],
                'penalize': ['config_network']
            },
            'service_active': {
                'boost': ['config_runtime', 'permissions_runtime'],
                'penalize': ['service_network']
            },
            'service_inactive': {
                'boost': ['service_network'],
                'penalize': ['config_runtime', 'permissions_runtime']
            },
            'port_listening': {
                'boost': ['config_runtime', 'permissions_runtime'],
                'penalize': ['service_network']
            },
            'port_not_listening': {
                'boost': ['service_network'],
                'penalize': ['config_runtime', 'permissions_runtime']
            },
            'config_valid': {
                'boost': ['runtime_network', 'permissions_runtime'],
                'penalize': ['config_runtime']
            },
            'config_invalid': {
                'boost': ['config_runtime'],
                'penalize': ['runtime_network', 'permissions_runtime']
            },
            'permission_granted': {
                'boost': ['config_runtime', 'runtime_network'],
                'penalize': ['permissions_runtime']
            },
            'permission_denied': {
                'boost': ['permissions_runtime'],
                'penalize': ['config_runtime', 'runtime_network']
            },
            'module_present': {
                'boost': ['config_runtime', 'runtime_network'],
                'penalize': ['permissions_runtime']
            },
            'module_missing': {
                'boost': ['permissions_runtime'],
                'penalize': ['config_runtime', 'runtime_network']
            }
        }
        
        for hypothesis in original_hypotheses:
            updated_hypothesis = RootCauseHypothesis(
                text=hypothesis.text,
                related_families=hypothesis.related_families[:],
                related_subsystems=hypothesis.related_subsystems[:],
                category=hypothesis.category,
                score=hypothesis.score,  # Start with original score
                reason=hypothesis.reason
            )
            
            # Apply evidence effects
            score_adjustment = 0.0
            evidence_reasons = []
            
            for evidence in evidence_signals:
                if evidence in evidence_effects:
                    effects = evidence_effects[evidence]
                    
                    # Boost matching categories
                    if updated_hypothesis.category in effects['boost']:
                        score_adjustment += 0.2
                        evidence_reasons.append(f"{evidence} supports {updated_hypothesis.category}")
                    
                    # Penalize conflicting categories
                    if updated_hypothesis.category in effects['penalize']:
                        score_adjustment -= 0.3
                        evidence_reasons.append(f"{evidence} weakens {updated_hypothesis.category}")
            
            # Update score and reason
            updated_hypothesis.score = max(0.0, min(1.0, updated_hypothesis.score + score_adjustment))
            
            # Track supporting and conflicting evidence
            updated_hypothesis.supporting_evidence = []
            updated_hypothesis.conflicting_evidence = []
            
            for evidence in evidence_signals:
                if evidence in evidence_effects:
                    effects = evidence_effects[evidence]
                    
                    if updated_hypothesis.category in effects['boost']:
                        updated_hypothesis.supporting_evidence.append(evidence)
                    
                    if updated_hypothesis.category in effects['penalize']:
                        updated_hypothesis.conflicting_evidence.append(evidence)
            
            if evidence_reasons:
                updated_hypothesis.reason = "; ".join(evidence_reasons[:2])  # Limit to 2 reasons
            else:
                updated_hypothesis.reason = f"no strong evidence effect on {updated_hypothesis.category}"
            
            updated_hypotheses.append(updated_hypothesis)
        
        # Apply hypothesis elimination rules
        updated_hypotheses = self.apply_elimination_rules(updated_hypotheses, evidence_signals)
        
        # Re-sort by updated score (eliminated hypotheses go to bottom)
        updated_hypotheses.sort(key=lambda h: (h.eliminated, -h.score))
        
        return updated_hypotheses
    
    def apply_elimination_rules(self, hypotheses: List[RootCauseHypothesis], evidence_signals: List[str]) -> List[RootCauseHypothesis]:
        """Apply deterministic elimination rules based on strong evidence."""
        elimination_rules = {
            'connection_confirmed': {
                'eliminate': ['runtime_network', 'service_network'],
                'reason': 'backend connectivity confirmed successful'
            },
            'config_valid': {
                'eliminate': ['config_runtime'],
                'reason': 'configuration validation passed'
            },
            'service_active': {
                'eliminate': ['service_network'],
                'reason': 'backend service is running'
            },
            'module_present': {
                'eliminate': ['permissions_runtime'],
                'reason': 'required module is available'
            }
        }
        
        for hypothesis in hypotheses:
            for evidence in evidence_signals:
                if evidence in elimination_rules:
                    rule = elimination_rules[evidence]
                    if hypothesis.category in rule['eliminate']:
                        hypothesis.eliminated = True
                        hypothesis.elimination_reason = rule['reason']
                        break
        
        return hypotheses
    
    def generate_investigation_status(self, session_id: str, evidence_signals: List[str], db_store) -> Dict[str, Any]:
        """Generate investigation status summary."""
        # Get investigation steps
        steps = db_store.get_investigation_steps(session_id)
        
        completed_steps = [s for s in steps if s['status'] == 'completed']
        pending_steps = [s for s in steps if s['status'] == 'pending']
        
        # Find next recommended step
        next_step = None
        if pending_steps:
            next_step = pending_steps[0]  # First pending step
        
        return {
            'completed_steps': completed_steps,
            'pending_steps': pending_steps,
            'next_step': next_step,
            'total_steps': len(steps),
            'completed_count': len(completed_steps),
            'pending_count': len(pending_steps)
        }
    
    def generate_deduplicated_diagnostic_commands(self, session_id: str, top_hypothesis, command_context, db_store) -> Dict[str, Any]:
        """Generate diagnostic commands excluding completed steps."""
        # Get investigation steps to avoid duplication
        steps = db_store.get_investigation_steps(session_id)
        completed_commands = {s['command'] for s in steps if s['status'] == 'completed'}
        
        # Generate new diagnostic commands
        all_commands = self.generate_diagnostic_commands(top_hypothesis, None, command_context)
        
        # Filter out completed commands
        filtered_primary = []
        for cmd in all_commands['primary']:
            if cmd['command'] not in completed_commands:
                filtered_primary.append(cmd)
        
        filtered_additional = []
        for cmd in all_commands['additional']:
            if cmd['command'] not in completed_commands:
                filtered_additional.append(cmd)
        
        # If all commands are completed, generate additional deterministic suggestions
        if not filtered_primary and not filtered_additional:
            filtered_primary = [
                {
                    'description': 'Review all collected evidence and investigation logs',
                    'command': 'mirrorcore analyze-followup'
                },
                {
                    'description': 'Consider escalating to manual investigation or expert consultation',
                    'command': 'echo "Investigation requires manual review"'
                }
            ]
        
        return {
            'primary': filtered_primary,
            'additional': filtered_additional
        }
    
    def detect_investigation_stall(self, investigation_status: Dict[str, Any], 
                             current_strategy_family: str, strategies_attempted: List[str],
                             evidence_history: List[List[str]], hypothesis_history: List[Dict[str, Any]]) -> InvestigationStallDetection:
        """Detect investigation stall using deterministic rules."""
        
        # Calculate progress metrics
        completion_rate = investigation_status['completed_count'] / investigation_status['total_steps'] if investigation_status['total_steps'] > 0 else 0.0
        
        # Stall detection rules
        is_stalled = False
        stall_reason = ""
        
        # Rule 1: Low completion rate with multiple attempts
        if completion_rate < 0.3 and len(strategies_attempted) >= 2:
            is_stalled = True
            stall_reason = "Low completion rate with multiple strategy attempts"
        
        # Rule 2: Repeated evidence patterns (no new evidence)
        elif len(evidence_history) >= 3 and len(set(tuple(e) for e in evidence_history[-3:])) <= 1:
            is_stalled = True
            stall_reason = "Repeated evidence patterns with no new information"
        
        # Rule 3: No hypothesis ranking changes
        elif len(hypothesis_history) >= 3:
            top_categories = [h[0]['category'] if h and h[0] else None for h in hypothesis_history[-3:]]
            if len(set(top_categories)) <= 1 and top_categories[0] is not None:
                is_stalled = True
                stall_reason = "No hypothesis ranking changes detected"
        
        # Find alternative strategy family
        available_families = [f for f in self.strategy_families.keys() if f not in strategies_attempted]
        alternative_family = available_families[0] if available_families else None
        
        progress_metrics = {
            'completion_rate': completion_rate,
            'strategies_attempted': len(strategies_attempted),
            'evidence_diversity': len(set(tuple(e) for e in evidence_history)) if evidence_history else 0
        }
        
        return InvestigationStallDetection(
            is_stalled=is_stalled,
            stall_reason=stall_reason,
            failed_families=strategies_attempted,
            alternative_family=alternative_family,
            progress_metrics=progress_metrics
        )
    
    def generate_alternative_strategy_commands(self, alternative_family: str, command_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate lightweight generic diagnostic commands for alternative strategy family."""
        
        if alternative_family == 'connectivity':
            return {
                'primary': [
                    {'description': 'Test basic network connectivity', 'command': 'ping -c 3 localhost'},
                    {'description': 'Check if target port is accessible', 'command': 'nc -zv localhost 8080'}
                ],
                'additional': []
            }
        elif alternative_family == 'configuration':
            return {
                'primary': [
                    {'description': 'Check configuration file syntax', 'command': 'python -c "import json; json.load(open(\'config.json\'))"'},
                    {'description': 'List environment variables', 'command': 'env | head -20'}
                ],
                'additional': []
            }
        elif alternative_family == 'service':
            return {
                'primary': [
                    {'description': 'Check system service status', 'command': 'systemctl list-units --type=service --state=running'},
                    {'description': 'Review recent system logs', 'command': 'journalctl --since "10 minutes ago" --no-pager'}
                ],
                'additional': []
            }
        elif alternative_family == 'permissions':
            return {
                'primary': [
                    {'description': 'Check current user permissions', 'command': 'id'},
                    {'description': 'Verify file access permissions', 'command': 'ls -la /tmp'}
                ],
                'additional': []
            }
        
        return {'primary': [], 'additional': []}

    def generate_evidence_summary(self, evidence_signals: List[str]) -> Dict[str, List[str]]:
        """Generate summary of evidence effects on hypotheses."""
        supports = []
        weakens = []
        
        evidence_descriptions = {
            'connection_confirmed': 'backend connectivity confirmed',
            'connection_refused': 'backend connectivity failure',
            'timeout': 'backend connectivity timeout',
            'service_active': 'backend service is running',
            'service_inactive': 'backend service is not running',
            'port_listening': 'expected port is listening',
            'port_not_listening': 'expected port is not listening',
            'config_valid': 'configuration is valid',
            'config_invalid': 'configuration is invalid',
            'permission_granted': 'permissions are sufficient',
            'permission_denied': 'permissions are insufficient',
            'module_present': 'required module is available',
            'module_missing': 'required module is missing'
        }
        
        for evidence in evidence_signals:
            description = evidence_descriptions.get(evidence, evidence)
            
            if evidence in ['connection_confirmed', 'service_active', 'port_listening', 
                          'config_valid', 'permission_granted', 'module_present']:
                supports.append(description)
            else:
                weakens.append(description)
        
        return {
            'supports': supports,
            'weakens': weakens
        }
    
    def classify_message(self, user_input: str) -> MessageCategory:
        """Classify user message into category."""
        user_input_lower = user_input.lower()
        
        # Check for troubleshooting indicators (keywords + patterns)
        if (any(keyword in user_input_lower for keyword in self.troubleshooting_keywords) or
            any(pattern in user_input_lower for pattern in self.troubleshooting_patterns)):
            return MessageCategory.TROUBLESHOOTING
        
        # Check for decision support indicators
        if any(keyword in user_input_lower for keyword in self.decision_keywords):
            return MessageCategory.DECISION_SUPPORT
        
        return MessageCategory.GENERAL_REASONING
    
    def _generate_targeted_response(self, incident, context: ResponseContext) -> ResponseResult:
        """Generate targeted response for recognized incident."""
        # Extract suggested fixes for outcome capture
        suggested_fixes = []
        if incident.recommended_first_checks:
            suggested_fixes.extend(incident.recommended_first_checks[:2])
        if incident.low_risk_initial_actions:
            suggested_fixes.extend(incident.low_risk_initial_actions[:1])
        
        # Build styled response
        styled_response = f"I recognize this as a {incident.incident_type.replace('_', ' ').title()} issue. Let me provide targeted guidance:\n\n"
        
        if incident.likely_causes:
            styled_response += "Most likely causes:\n"
            for i, cause in enumerate(incident.likely_causes[:3], 1):
                styled_response += f"{i}. {cause}\n"
            styled_response += "\n"
        
        if incident.recommended_first_checks:
            styled_response += "First checks to run:\n"
            for i, check in enumerate(incident.recommended_first_checks[:3], 1):
                styled_response += f"{i}. {check}\n"
            styled_response += "\n"
        
        if incident.low_risk_initial_actions:
            styled_response += "Low-risk initial actions:\n"
            for action in incident.low_risk_initial_actions[:3]:
                styled_response += f"• {action}\n"
            styled_response += "\n"
        
        # Add historical context if available
        if context.memory_store:
            try:
                ranked_fixes = context.memory_store.get_ranked_fixes_by_incident_type(
                    incident.incident_type, limit=3
                )
                
                if ranked_fixes:
                    successful_fixes = [f for f in ranked_fixes if f['success_count'] > 0]
                    if successful_fixes:
                        top_fix = successful_fixes[0]
                        styled_response += f"Based on previous outcomes for this issue:\n"
                        styled_response += f"Most successful approach: {top_fix['normalized_fix']}\n"
                        styled_response += f"  Reason: {top_fix['success_count']} successful, {top_fix['failed_count']} failed attempts\n\n"
            except Exception:
                pass  # Don't fail if memory retrieval fails
        
        styled_response += "These steps should help identify the root cause. Let me know what you discover!"
        
        # Evaluate confidence and determine if escalation is needed
        extracted_signals = getattr(context, 'extracted_signals', [])
        evaluation = self.evaluate_confidence(incident, context, extracted_signals)
        
        # Generate escalation guidance if triggered
        escalation_guidance = ""
        if evaluation.escalation_triggered:
            escalation_guidance = self.generate_escalation_guidance(evaluation)
        
        # Return result with outcome capture support and escalation info.
        # Escalation guidance is returned separately to ensure callers can print it once.
        return ResponseResult(
            response=styled_response,
            should_prompt_outcome=True,  # Enable outcome capture for targeted responses
            incident_id=getattr(context, 'incident_id', None),
            suggested_fixes=suggested_fixes,
            escalation_guidance=escalation_guidance if evaluation.escalation_triggered else None
        )
    
    def evaluate_confidence(self, incident, context: ResponseContext, extracted_signals: List[str]) -> ConfidenceEvaluation:
        """Evaluate confidence level and determine if escalation is needed."""
        if not incident:
            return ConfidenceEvaluation(
                confidence_level="unknown",
                incident_confidence=0.0,
                failed_fixes_ratio=0.0,
                successful_fixes_count=0,
                conflicting_signals=len(extracted_signals) > 1,
                no_incident_match=True,
                escalation_triggered=False
            )
        
        # Get historical fix data
        failed_fixes_count = 0
        successful_fixes_count = 0
        failed_fixes_ratio = 0.0
        
        if context.memory_store:
            try:
                ranked_fixes = context.memory_store.get_ranked_fixes_by_incident_type(
                    incident.incident_type, limit=10
                )
                
                if ranked_fixes:
                    successful_fixes = [f for f in ranked_fixes if f['success_count'] > 0]
                    failed_fixes = [f for f in ranked_fixes if f['failed_count'] > f['success_count']]
                    
                    successful_fixes_count = sum(f['success_count'] for f in successful_fixes)
                    failed_fixes_count = sum(f['failed_count'] for f in failed_fixes)
                    
                    total_attempts = successful_fixes_count + failed_fixes_count
                    if total_attempts > 0:
                        failed_fixes_ratio = failed_fixes_count / total_attempts
            except Exception:
                pass  # Don't fail if memory retrieval fails
        
        # Get signal families from parsed log if available
        conflicting_families = []
        if hasattr(context, 'signal_families'):
            conflicting_families = context.signal_families
        
        # Apply conflict penalties for multiple signal families
        conflict_penalty = 0.0
        if len(conflicting_families) > 1:
            conflict_penalty = min(0.4, 0.1 * len(conflicting_families))
        
        # Evaluate escalation criteria
        escalation_triggered = False
        confidence_level = "moderate"
        
        # Check 1: Low incident confidence
        if incident.confidence < self.LOW_CONFIDENCE_THRESHOLD:
            confidence_level = "low"
            escalation_triggered = True
        
        # Check 2: High failed fix ratio
        if failed_fixes_ratio > self.HIGH_FAILED_RATIO_THRESHOLD:
            confidence_level = "low"
            escalation_triggered = True
        
        # Check 3: Multiple conflicting error signals
        if len(extracted_signals) > self.MULTIPLE_SIGNALS_THRESHOLD:
            confidence_level = "low"
            escalation_triggered = True
        
        # Check 4: No incident match but signals exist
        if not extracted_signals:
            confidence_level = "unknown"
        elif len(extracted_signals) == 1 and incident.confidence < 0.8:
            confidence_level = "low"
            escalation_triggered = True
        
        # Check 5: Apply conflict penalties
        if conflict_penalty > 0.2:  # High conflict penalty
            confidence_level = "low"
            escalation_triggered = True
        elif conflict_penalty > 0.1:  # Moderate conflict penalty
            if confidence_level == "moderate":
                confidence_level = "low"
        
        # Adjust confidence level upward for strong signals
        if not escalation_triggered and incident.confidence > 0.8 and successful_fixes_count > 2:
            confidence_level = "high"
        
        # Apply conflict penalty to incident confidence for evaluation
        adjusted_incident_confidence = max(0.0, incident.confidence - conflict_penalty)
        
        return ConfidenceEvaluation(
            confidence_level=confidence_level,
            incident_confidence=adjusted_incident_confidence,
            failed_fixes_ratio=failed_fixes_ratio,
            successful_fixes_count=successful_fixes_count,
            conflicting_signals=len(conflicting_families) > 1,
            no_incident_match=incident is None,
            escalation_triggered=escalation_triggered
        )
    
    def generate_escalation_guidance(self, evaluation: ConfidenceEvaluation) -> str:
        """Generate escalation guidance based on confidence evaluation."""
        guidance = "\n⚠️ INVESTIGATION ESCALATION RECOMMENDED\n"
        guidance += "Standard fixes may be insufficient for this issue.\n\n"
        
        guidance += "Possible investigation steps:\n"
        guidance += "• Review service logs and error patterns\n"
        guidance += "• Inspect configuration files for conflicts\n"
        guidance += "• Verify dependency versions and compatibility\n"
        guidance += "• Check system environment differences\n"
        guidance += "• Review upstream documentation or bug reports\n\n"
        
        # Add specific guidance based on evaluation factors
        if evaluation.conflicting_signals:
            guidance += "⚠️ Multiple error signals detected - issue may be complex\n"
        
        if evaluation.failed_fixes_ratio > self.HIGH_FAILED_RATIO_THRESHOLD:
            guidance += f"⚠️ High failure rate ({evaluation.failed_fixes_ratio:.1%}) for similar issues\n"
        
        if evaluation.no_incident_match:
            guidance += "⚠️ No recognized incident pattern - requires general investigation\n"
        
        return guidance
    
    def _calculate_confidence_penalty(self, base_confidence: float, conflicting_families: List[str]) -> float:
        """Calculate confidence penalty for conflicting signal families."""
        if len(conflicting_families) <= 1:
            return base_confidence
        
        # Penalty for multiple conflicting signal families
        conflict_penalty = min(0.4, 0.1 * len(conflicting_families))
        return max(0.0, base_confidence - conflict_penalty)
    
    def classify_message(self, user_input: str) -> MessageCategory:
        """Classify user message into category."""
        user_input_lower = user_input.lower()
        
        # Check for troubleshooting indicators (keywords + patterns)
        if (any(keyword in user_input_lower for keyword in self.troubleshooting_keywords) or
            any(pattern in user_input_lower for pattern in self.troubleshooting_patterns)):
            return MessageCategory.TROUBLESHOOTING
        
        # Check for decision support indicators
        if any(keyword in user_input_lower for keyword in self.decision_keywords):
            return MessageCategory.DECISION_SUPPORT
        
        # Check for reasoning indicators
        if any(keyword in user_input_lower for keyword in self.reasoning_keywords):
            return MessageCategory.GENERAL_REASONING
        
        return MessageCategory.UNKNOWN
    
    def generate_troubleshooting_response(self, context: ResponseContext) -> ResponseResult:
        """Generate troubleshooting response with outcome capture support."""
        user_input = context.user_input
        
        # First try incident signature detection
        if self.incident_detector:
            incident = self.incident_detector.detect_incident(user_input)
            if incident:
                return self._generate_targeted_response(incident, context)
        
        # Fallback to general troubleshooting
        response = self._generate_general_troubleshooting_response(user_input, context)
        return ResponseResult(response=response, should_prompt_outcome=False)
    
    def _generate_targeted_response(self, incident, context: ResponseContext) -> ResponseResult:
        """Generate targeted response for detected incident."""
        incident_type = incident.incident_type.replace('_', ' ').title()
        
        response = f"I recognize this as a {incident_type} issue. Let me provide targeted guidance:\n\n"
        
        # Add likely causes
        if incident.likely_causes:
            response += "Most likely causes:\n"
            for i, cause in enumerate(incident.likely_causes[:3], 1):  # Show top 3
                response += f"{i}. {cause}\n"
            response += "\n"
        
        # Add recommended first checks
        if incident.recommended_first_checks:
            response += "First checks to run:\n"
            for i, check in enumerate(incident.recommended_first_checks[:4], 1):  # Show top 4
                response += f"{i}. {check}\n"
            response += "\n"
        
        # Add low-risk initial actions
        if incident.low_risk_initial_actions:
            response += "Low-risk initial actions:\n"
            for i, action in enumerate(incident.low_risk_initial_actions[:3], 1):  # Show top 3
                response += f"• {action}\n"
            response += "\n"
        
        # Add historical context if available
        suggested_fixes = []
        if hasattr(context, 'memory_store') and context.memory_store:
            try:
                # Get ranked fixes for this incident type
                ranked_fixes = context.memory_store.get_ranked_fixes_by_incident_type(
                    incident.incident_type, limit=3
                )
                
                if ranked_fixes:
                    response += "Based on previous outcomes for this issue:\n"
                    
                    # Show top ranked fix with explanation
                    top_fix = ranked_fixes[0]
                    response += f"Most successful approach: {top_fix['normalized_fix']}\n"
                    response += f"  Reason: {top_fix['success_count']} successful, {top_fix['failed_count']} failed attempts\n"
                    
                    # Collect suggested fixes for outcome capture
                    suggested_fixes.append(top_fix['normalized_fix'])
                    
                    # Show other successful options
                    successful_fixes = [f for f in ranked_fixes if f['success_count'] > 0]
                    if len(successful_fixes) > 1:
                        response += "Other successful approaches:\n"
                        for i, fix in enumerate(successful_fixes[1:3], 2):  # Show next 2
                            response += f"  {i}. {fix['normalized_fix']} ({fix['success_count']} success)\n"
                            suggested_fixes.append(fix['normalized_fix'])
                    
                    # Warn about commonly failed approaches
                    failed_fixes = [f for f in ranked_fixes if f['failed_count'] > f['success_count']]
                    if failed_fixes:
                        response += "Approaches that typically fail:\n"
                        for fix in failed_fixes[:2]:  # Show top 2 failed
                            response += f"  • {fix['normalized_fix']} ({fix['failed_count']} failed)\n"
                    
                    response += "\n"
                    
            except Exception:
                # Don't fail if memory retrieval fails
                pass
        
        # Add recommended fixes to suggested fixes if available
        if incident.recommended_first_checks:
            for check in incident.recommended_first_checks[:2]:  # Add top 2 checks
                if check not in suggested_fixes:
                    suggested_fixes.append(check)
        
        response += "These steps should help identify the root cause. Let me know what you discover!"
        
        # Apply persona style
        styled_response = self._apply_persona_style(response, context, "troubleshooting")
        
        # Evaluate confidence and determine if escalation is needed
        extracted_signals = getattr(context, 'extracted_signals', [])
        evaluation = self.evaluate_confidence(incident, context, extracted_signals)
        
        # Generate escalation guidance if triggered
        escalation_guidance = ""
        if evaluation.escalation_triggered:
            escalation_guidance = self.generate_escalation_guidance(evaluation)
        
        # Return result with outcome capture support and escalation info.
        # Escalation guidance is returned separately to ensure callers can print it once.
        return ResponseResult(
            response=styled_response,
            should_prompt_outcome=True,  # Enable outcome capture for targeted responses
            incident_id=getattr(context, 'incident_id', None),
            suggested_fixes=suggested_fixes,
            escalation_guidance=escalation_guidance if evaluation.escalation_triggered else None
        )
    
    def _generate_general_troubleshooting_response(self, user_input: str, context: ResponseContext) -> str:
        """Generate general troubleshooting response when no signature matches."""
        user_input_lower = user_input.lower()
        
        # Identify likely issue type
        if any(word in user_input_lower for word in ["error", "failed", "crash", "broken"]):
            response = "I can help troubleshoot that. Let's approach this systematically:\n\n"
            response += "1. What exactly happened or what error message did you see?\n"
            response += "2. What were you trying to do when it occurred?\n"
            response += "3. Have you tried any basic fixes (restart, check permissions, verify syntax)?\n\n"
            response += "Once you provide these details, I can help identify the root cause and suggest specific steps."
            
        elif any(word in user_input_lower for word in ["command", "bash", "terminal"]):
            response = "For command-line issues, let's break this down:\n\n"
            response += "1. What command are you trying to run?\n"
            response += "2. What's the exact error or unexpected output?\n"
            response += "3. Can you show me the full command and any relevant context?\n\n"
            response += "I'll help you debug the command structure or identify the issue."
            
        elif any(word in user_input for word in ["docker", "container", "service", "daemon", "server"]):
            response = "For container/service issues, let's investigate systematically:\n\n"
            response += "1. What specific behavior are you seeing (crashes, restarts, hangs)?\n"
            response += "2. What do the logs show when this happens?\n"
            response += "3. Are there resource constraints (CPU, memory, disk space)?\n"
            response += "4. Has anything changed recently (updates, config changes)?\n\n"
            response += "This will help identify whether it's a resource issue, configuration problem, or external dependency."
            
        elif any(word in user_input for word in ["cpu", "memory", "performance", "load"]):
            response = "For performance/resource issues, let's analyze this methodically:\n\n"
            response += "1. What are the current resource metrics (CPU %, memory usage, load average)?\n"
            response += "2. When did the performance issue start and what triggers it?\n"
            response += "3. Are there specific processes or operations consuming resources?\n"
            response += "4. Have you checked for memory leaks, infinite loops, or resource exhaustion?\n\n"
            response += "This will help distinguish between normal load and actual performance problems."
            
        elif any(word in user_input for word in ["logs", "debug", "investigate", "diagnose"]):
            response = "For debugging and investigation, let's establish a systematic approach:\n\n"
            response += "1. What specific problem are you trying to diagnose?\n"
            response += "2. What logging or monitoring tools are available?\n"
            response += "3. What have you already checked and what were the findings?\n"
            response += "4. Can you reproduce the issue or is it intermittent?\n\n"
            response += "I'll help you build a structured investigation plan to identify the root cause."
            
        else:
            response = "I understand you're dealing with a technical issue. "
            response += "To help you effectively:\n\n"
            response += "• Describe the specific problem or error\n"
            response += "• Share any relevant error messages\n"
            response += "• Let me know what you've already tried\n\n"
            response += "This will help me provide targeted assistance rather than general suggestions."
        
        return self._apply_persona_style(response, context, "troubleshooting")
    
    def prompt_assisted_outcome_capture(self, incident_id: str, suggested_fixes: List[str]) -> str:
        """Generate prompt for assisted outcome capture."""
        prompt = "\n" + "=" * 60 + "\n"
        prompt += "🎯 OUTCOME CAPTURE\n"
        prompt += "=" * 60 + "\n"
        prompt += "Did one of these suggested fixes solve the issue? [y/N]\n"
        
        if suggested_fixes:
            prompt += "\nSuggested fixes from this session:\n"
            for i, fix in enumerate(suggested_fixes[:5], 1):  # Show top 5
                prompt += f"  {i}. {fix}\n"
        
        prompt += "\nIf you'd like to record the outcome now, type 'y' and I'll collect:\n"
        prompt += "  • Which fix you attempted\n"
        prompt += "  • Result (success/failed/partial/unknown)\n"
        prompt += "  • Confirmed root cause (optional)\n"
        prompt += "  • Additional notes (optional)\n"
        prompt += "\nYou can also use 'mirrorcore debug-outcome' anytime to record outcomes.\n"
        
        return prompt
    
    def generate_decision_support_response(self, context: ResponseContext) -> str:
        """Generate decision support response."""
        response = "I can help you think through this decision. "
        
        # Extract key elements
        user_input = context.user_input.lower()
        
        # Look for decision elements
        if "or" in user_input and any(word in user_input for word in ["choose", "select", "pick"]):
            response += "I see you're choosing between options. "
            response += "Let me help you evaluate the tradeoffs.\n\n"
            response += "What are the key factors you're considering? (cost, time, risk, quality, etc.)\n"
            response += "What are the main pros and cons of each option?\n"
            response += "Is there a time constraint or deadline?\n\n"
            response += "Once I understand these details, I can help you structure this decision."
            
        elif any(word in user_input for word in ["better", "prefer", "recommend"]):
            response += "I can help analyze that recommendation. "
            response += "Let's consider:\n\n"
            response += "• What evidence supports this recommendation?\n"
            response += "• What are the potential downsides or risks?\n"
            response += "• Does this align with your past experiences or goals?\n\n"
            response += "This will help us evaluate whether it's the right choice for you."
            
        else:
            response += "Let me help you work through this decision systematically. "
            response += "I'll ask some clarifying questions to understand the full context:\n\n"
            response += "1. What's the main objective you're trying to achieve?\n"
            response += "2. What are the available options or approaches?\n"
            response += "3. What constraints or considerations are most important?\n\n"
            response += "This framework will help us make a more informed choice."
        
        return self._apply_persona_style(response, context, "decision_support")
    
    def generate_general_reasoning_response(self, context: ResponseContext) -> str:
        """Generate general reasoning response."""
        user_input = context.user_input.lower()
        
        # Look for reasoning patterns
        if any(word in user_input for word in ["why", "how"]):
            response = "That's a good question. "
            
            if "why" in user_input:
                response += "Understanding the 'why' helps us address root causes rather than symptoms. "
                response += "What specific situation or outcome are you trying to understand?"
            else:  # "how"
                response += "Knowing 'how' helps with implementation and process. "
                response += "What process or approach are you trying to figure out?"
            
            response += "\n\nBreaking this down systematically will lead to better insights."
            
        elif any(word in user_input for word in ["think", "believe", "consider"]):
            response = "I appreciate you sharing your thinking process. "
            response += "Exploring different angles helps strengthen our reasoning. "
            response += "What alternative perspectives or counter-arguments should we consider?"
            response += "What evidence would strengthen or challenge this view?"
            
        elif any(word in user_input for word in ["understand", "explain"]):
            response = "Let me help clarify this concept. "
            response += "What specific aspect is most important to understand first?"
            response += "Are there any particular examples or contexts that would make this clearer?"
            
        else:
            response = "That's an interesting perspective. "
            response += "Let me explore this with you:\n\n"
            response += "• What's the core question or issue here?\n"
            response += "• What assumptions are we working with?\n"
            response += "• What would be the most useful way to move forward?\n\n"
            response += "This will help us develop a more complete understanding."
        
        return self._apply_persona_style(response, context, "general_reasoning")
    
    def generate_unknown_response(self, context: ResponseContext) -> str:
        """Generate response for unknown message types."""
        response = "I'm here to help and learn from our conversation. "
        response += "Could you tell me more about what you're working on or what you'd like assistance with?\n\n"
        response += "I can help with:\n"
        response += "• Technical troubleshooting and debugging\n"
        response += "• Decision analysis and tradeoff evaluation\n"
        response += "• General reasoning and problem-solving\n"
        response += "• Learning from patterns and preferences\n\n"
        response += "What would be most helpful for you right now?"
        
        return self._apply_persona_style(response, context, "unknown")
    
    def _apply_persona_style(self, response: str, context: ResponseContext, response_type: str) -> str:
        """Apply persona-aware styling to responses."""
        if not context.persona_profile:
            return response
        
        # Extract relevant persona traits
        traits = context.persona_profile
        
        # Helper function to safely get numeric trait values
        def get_trait_value(trait_name: str, default: float = 0.5) -> float:
            trait_data = traits.get(trait_name, {})
            if isinstance(trait_data, dict):
                value = trait_data.get('value', default)
                # Convert to float if it's a string or int
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            elif isinstance(trait_data, (int, float)):
                return float(trait_data)
            else:
                return default
        
        # Apply evidence_threshold styling
        evidence_threshold = get_trait_value('evidence_threshold', 0.5)
        if evidence_threshold > 0.7:  # High evidence preference
            if response_type in ["troubleshooting", "general_reasoning"]:
                response += "\n\nLet me suggest a systematic approach to verify this."
            elif response_type == "decision_support":
                response += "\n\nI recommend gathering data before committing to a decision."
        
        # Apply ambiguity_tolerance styling
        ambiguity_tolerance = get_trait_value('ambiguity_tolerance', 0.5)
        if ambiguity_tolerance < 0.3:  # Low ambiguity tolerance
            response = response.replace("uncertain", "unclear").replace("might be", "could be")
        elif ambiguity_tolerance > 0.7:  # High ambiguity tolerance
            if "options" in response:
                response += "\n\nThere are multiple valid approaches here."
        
        # Apply action_bias styling
        action_bias = get_trait_value('action_bias', 0.5)
        if action_bias > 0.7:  # High action bias
            if response_type == "troubleshooting":
                response += "\n\nLet's try a quick fix and iterate if needed."
            elif response_type == "decision_support":
                response += "\n\nWhat's a small step we could take to move forward?"
        elif action_bias < 0.3:  # Low action bias
            if response_type == "troubleshooting":
                response += "\n\nLet's analyze this thoroughly before taking action."
            elif response_type == "decision_support":
                response += "\n\nLet's make sure we understand all aspects first."
        
        # Apply self_reliance styling
        self_reliance = get_trait_value('self_reliance_level', 0.5)
        if self_reliance > 0.7:  # High self-reliance
            response += "\n\nI'll provide guidance, but you should trust your judgment too."
        elif self_reliance < 0.3:  # Low self-reliance
            response += "\n\nLet me work through this together step by step."
        
        return response
    
    def generate_response(self, user_input: str, persona_profile: Dict[str, Any], 
                      conversation_history: List[Dict[str, Any]], 
                      session_id: str = None, db_store=None) -> str:
        """Generate appropriate response based on user input and persona."""
        context = ResponseContext(
            user_input=user_input,
            persona_profile=persona_profile,
            conversation_history=conversation_history,
            memory_store=db_store  # Add memory store for outcome tracking
        )
        
        # Classify the message
        category = self.classify_message(user_input)
        
        # Generate response based on category
        if category == MessageCategory.TROUBLESHOOTING:
            # Check for incident signature
            if self.incident_detector and db_store:
                incident = self.incident_detector.detect_incident(user_input)
                if incident:
                    # Store the incident signature
                    try:
                        incident_id = db_store.save_incident_signature(user_input, incident, session_id)
                        context.incident_id = incident_id  # Add incident ID to context
                    except Exception:
                        # Don't fail if storage fails
                        pass
                    
                    # Generate targeted response
                    response_result = self._generate_targeted_response(incident, context)
                    return response_result
            
            # Fallback to general troubleshooting
            response = self._generate_general_troubleshooting_response(user_input, context)
            return ResponseResult(response=response, should_prompt_outcome=False)
        elif category == MessageCategory.DECISION_SUPPORT:
            response = self.generate_decision_support_response(context)
            return ResponseResult(response=response, should_prompt_outcome=False)
        elif category == MessageCategory.GENERAL_REASONING:
            response = self.generate_general_reasoning_response(context)
            return ResponseResult(response=response, should_prompt_outcome=False)
        else:
            response = self.generate_unknown_response(context)
            return ResponseResult(response=response, should_prompt_outcome=False)
