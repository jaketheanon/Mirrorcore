"""
Memory Extractor

Extracts patterns and insights from interactions and episodes.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re
import uuid


@dataclass
class Pattern:
    """Represents a discovered pattern."""
    id: str
    name: str
    category: str
    description: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0


@dataclass
class Insight:
    """Represents an extracted insight."""
    id: str
    type: str
    description: str
    evidence: List[str]
    confidence: float
    timestamp: str


@dataclass
class LearningSignal:
    """Represents a conversation learning signal."""
    signal_id: str
    session_id: str
    timestamp: str
    signal_type: str
    inferred_direction: str  # "positive", "negative", "neutral"
    value: float  # -1.0 to 1.0
    confidence: float
    source_module: str
    evidence_excerpt: str
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal_id": self.signal_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "signal_type": self.signal_type,
            "inferred_direction": self.inferred_direction,
            "value": self.value,
            "confidence": self.confidence,
            "source_module": self.source_module,
            "evidence_excerpt": self.evidence_excerpt,
            "tags": self.tags
        }


class ConversationExtractor:
    """Extracts persona-relevant signals from interactive conversations."""
    
    def __init__(self):
        # Define keyword patterns for different signal types
        self.signal_patterns = {
            "directness_preference": {
                "positive": ["direct", "straightforward", "clear", "concise", "get to the point"],
                "negative": ["subtle", "hint", "suggest", "maybe", "perhaps", "consider"]
            },
            "evidence_preference": {
                "positive": ["proof", "evidence", "data", "verify", "check", "confirm", "test"],
                "negative": ["trust", "believe", "assume", "guess", "intuition", "gut feeling"]
            },
            "ambiguity_tolerance": {
                "positive": ["uncertain", "maybe", "could be", "not sure", "ambiguous", "unclear"],
                "negative": ["certain", "definitely", "always", "never", "clear", "precise"]
            },
            "action_bias": {
                "positive": ["let's try", "implement", "do it", "take action", "move forward"],
                "negative": ["wait", "hold on", "delay", "postpone", "think more"]
            },
            "trust_verification_style": {
                "positive": ["official docs", "documentation", "source", "verify", "check source"],
                "negative": ["trust me", "believe me", "experience", "expert says"]
            },
            "self_reliance_level": {
                "positive": ["figure it out", "myself", "own", "independent", "without help"],
                "negative": ["help", "team", "collaborate", "ask someone", "get advice"]
            },
            # Troubleshooting-specific patterns
            "evidence_first_troubleshooting": {
                "positive": ["check logs", "verify", "test", "diagnose", "investigate", "gather evidence"],
                "negative": ["guess", "assume", "try random", "shoot in the dark"]
            },
            "quick_fix_preference": {
                "positive": ["quick fix", "try something", "simple solution", "easy fix"],
                "negative": ["thorough", "systematic", "investigate", "root cause"]
            },
            "root_cause_investigation": {
                "positive": ["root cause", "why", "investigate", "understand", "systematic"],
                "negative": ["just fix it", "workaround", "patch", "temporary"]
            },
            "communication_preference": {
                "positive": ["technical", "details", "specific", "precise", "exact"],
                "negative": ["emotional", "feelings", "supportive", "encouraging", "gentle"]
            },
            "uncertainty_response": {
                "positive": ["admit", "don't know", "uncertain", "need more info", "clarify"],
                "negative": ["confident", "sure", "definitely", "always", "certain"]
            }
        }
    
    def extract_signals(self, session_id: str, conversation: List[Dict[str, Any]]) -> List[LearningSignal]:
        """Extract persona-relevant signals from conversation."""
        signals = []
        
        for message in conversation:
            text = message.get("content", "").lower()
            role = message.get("role", "user")
            
            # Only extract from user messages (avoid over-inference from assistant responses)
            if role != "user":
                continue
            
            # Extract signals for each pattern type
            for signal_type, patterns in self.signal_patterns.items():
                signal = self._extract_signal_type(text, signal_type, patterns, session_id, message)
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _extract_signal_type(self, text: str, signal_type: str, patterns: Dict[str, List[str]], 
                         session_id: str, message: Dict[str, Any]) -> Optional[LearningSignal]:
        """Extract a specific type of signal from text."""
        positive_matches = sum(1 for keyword in patterns["positive"] if keyword in text)
        negative_matches = sum(1 for keyword in patterns["negative"] if keyword in text)
        
        # Only extract if there's a clear signal
        if positive_matches == 0 and negative_matches == 0:
            return None
        
        # Determine direction and value
        if positive_matches > negative_matches:
            direction = "positive"
            value = min(positive_matches / (positive_matches + negative_matches), 1.0)
        elif negative_matches > positive_matches:
            direction = "negative"
            value = -min(negative_matches / (positive_matches + negative_matches), 1.0)
        else:
            # Equal matches - treat as neutral/uncertain
            return None
        
        # Calculate confidence based on match strength and context
        confidence = min((positive_matches + negative_matches) / 3.0, 0.8)  # Max 0.8 for single message
        
        # Create excerpt (first 100 chars with keyword highlighted)
        excerpt = text[:100] + "..." if len(text) > 100 else text
        
        return LearningSignal(
            signal_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            signal_type=signal_type,
            inferred_direction=direction,
            value=value,
            confidence=confidence,
            source_module="conversation",
            evidence_excerpt=excerpt,
            tags=[signal_type, direction, "interactive"]
        )


class MemoryExtractor:
    """Extracts patterns and insights from memory data."""
    
    def __init__(self):
        self.pattern_extractors = {
            "troubleshooting": self._extract_troubleshooting_patterns,
            "decision": self._extract_decision_patterns,
            "communication": self._extract_communication_patterns
        }
        self.conversation_extractor = ConversationExtractor()
    
    def extract_patterns(self, episodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Extract patterns from a collection of episodes."""
        all_patterns = []
        
        # Group episodes by domain
        episodes_by_domain = self._group_by_domain(episodes)
        
        for domain, domain_episodes in episodes_by_domain.items():
            extractor = self.pattern_extractors.get(domain)
            if extractor:
                patterns = extractor(domain_episodes)
                all_patterns.extend(patterns)
        
        return all_patterns
    
    def extract_conversation_signals(self, session_id: str, conversation: List[Dict[str, Any]]) -> List[LearningSignal]:
        """Extract learning signals from interactive conversation."""
        return self.conversation_extractor.extract_signals(session_id, conversation)
    
    def extract_insights(self, interactions: List[Dict[str, Any]]) -> List[Insight]:
        """Extract insights from interaction data."""
        insights = []
        
        # Extract success patterns
        success_insights = self._extract_success_insights(interactions)
        insights.extend(success_insights)
        
        # Extract learning patterns
        learning_insights = self._extract_learning_insights(interactions)
        insights.extend(learning_insights)
        
        # Extract preference patterns
        preference_insights = self._extract_preference_insights(interactions)
        insights.extend(preference_insights)
        
        return insights
    
    def _group_by_domain(self, episodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group episodes by domain."""
        grouped = {}
        
        for episode in episodes:
            domain = episode.get("context", {}).get("domain", "general")
            if domain not in grouped:
                grouped[domain] = []
            grouped[domain].append(episode)
        
        return grouped
    
    def _extract_troubleshooting_patterns(self, episodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Extract troubleshooting patterns."""
        patterns = []
        
        # Look for common error-solution pairs
        error_solution_pairs = self._find_error_solution_pairs(episodes)
        
        for error_type, solutions in error_solution_pairs.items():
            if len(solutions) >= 2:  # Pattern appears multiple times
                # Calculate success rate for each solution
                successful_solutions = []
                for solution in solutions:
                    success_rate = self._calculate_solution_success_rate(solution, episodes)
                    if success_rate > 0.5:
                        successful_solutions.append((solution, success_rate))
                
                if successful_solutions:
                    # Sort by success rate
                    successful_solutions.sort(key=lambda x: x[1], reverse=True)
                    
                    pattern = Pattern(
                        id=f"troubleshooting_{error_type}",
                        name=f"Fix {error_type} issues",
                        category="troubleshooting",
                        description=f"Common solutions for {error_type} problems",
                        conditions=[{"field": "error_type", "operator": "equals", "value": error_type}],
                        actions=[
                            {
                                "step": i + 1,
                                "action": "solution",
                                "content": solution,
                                "success_rate": success_rate
                            }
                            for i, (solution, success_rate) in enumerate(successful_solutions[:3])
                        ],
                        confidence=min(len(solutions) / 5.0, 1.0)  # More occurrences = higher confidence
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_error_solution_pairs(self, episodes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Find recurring error-solution patterns."""
        pairs = {}
        
        for episode in episodes:
            content = episode.get("content", {})
            input_text = content.get("input", "").lower()
            output_text = content.get("output", "").lower()
            
            # Extract error type
            error_type = self._extract_error_type(input_text)
            if error_type:
                # Extract solution
                solution = self._extract_solution(output_text)
                if solution:
                    if error_type not in pairs:
                        pairs[error_type] = []
                    pairs[error_type].append(solution)
        
        return pairs
    
    def _extract_error_type(self, text: str) -> Optional[str]:
        """Extract error type from text."""
        error_patterns = {
            "permission": r"permission denied|access denied|sudo|admin",
            "network": r"connection refused|timeout|network|unreachable",
            "dependency": r"module not found|import error|dependency|package",
            "syntax": r"syntax error|invalid syntax|parse error",
            "git": r"git|merge conflict|push failed|pull"
        }
        
        for error_type, pattern in error_patterns.items():
            if re.search(pattern, text):
                return error_type
        
        return None
    
    def _extract_solution(self, text: str) -> Optional[str]:
        """Extract solution from text."""
        # Look for command patterns
        command_pattern = r"`([^`]+)`|(\$ )(.+)|(# )(.+)"
        matches = re.findall(command_pattern, text)
        
        if matches:
            # Return the first command found
            for match in matches:
                for part in match:
                    if part.strip():
                        return part.strip()
        
        # Look for explicit solution keywords
        solution_keywords = ["solution:", "fix:", "try:", "use:", "run:"]
        lines = text.split('\n')
        
        for line in lines:
            for keyword in solution_keywords:
                if keyword in line.lower():
                    return line.strip()
        
        return None
    
    def _calculate_solution_success_rate(self, solution: str, episodes: List[Dict[str, Any]]) -> float:
        """Calculate success rate for a solution."""
        successes = 0
        total = 0
        
        for episode in episodes:
            content = episode.get("content", {})
            output_text = content.get("output", "").lower()
            
            if solution.lower() in output_text:
                total += 1
                if content.get("outcome") == "success":
                    successes += 1
        
        return successes / total if total > 0 else 0.0
    
    def _extract_decision_patterns(self, episodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Extract decision-making patterns."""
        # Placeholder for decision pattern extraction
        return []
    
    def _extract_communication_patterns(self, episodes: List[Dict[str, Any]]) -> List[Pattern]:
        """Extract communication patterns."""
        # Placeholder for communication pattern extraction
        return []
    
    def _extract_success_insights(self, interactions: List[Dict[str, Any]]) -> List[Insight]:
        """Extract insights about successful interactions."""
        insights = []
        
        successful_interactions = [i for i in interactions if i.get("outcome") == "success"]
        
        if len(successful_interactions) > 5:
            # Find common factors in successful interactions
            common_factors = self._find_common_factors(successful_interactions)
            
            for factor in common_factors:
                insight = Insight(
                    id=f"success_factor_{factor['type']}",
                    type="success_factor",
                    description=f"Successful interactions often involve {factor['description']}",
                    evidence=factor.get("evidence", []),
                    confidence=factor.get("confidence", 0.7),
                    timestamp=datetime.utcnow().isoformat()
                )
                insights.append(insight)
        
        return insights
    
    def _extract_learning_insights(self, interactions: List[Dict[str, Any]]) -> List[Insight]:
        """Extract learning-related insights."""
        insights = []
        
        # Look for improvement over time
        improvement_patterns = self._detect_improvement_patterns(interactions)
        
        for pattern in improvement_patterns:
            insight = Insight(
                id=f"learning_{pattern['area']}",
                type="learning_pattern",
                description=pattern["description"],
                evidence=pattern.get("evidence", []),
                confidence=pattern.get("confidence", 0.6),
                timestamp=datetime.utcnow().isoformat()
            )
            insights.append(insight)
        
        return insights
    
    def _extract_preference_insights(self, interactions: List[Dict[str, Any]]) -> List[Insight]:
        """Extract user preference insights."""
        insights = []
        
        # Analyze response patterns
        response_patterns = self._analyze_response_patterns(interactions)
        
        for pattern in response_patterns:
            insight = Insight(
                id=f"preference_{pattern['type']}",
                type="preference",
                description=pattern["description"],
                evidence=pattern.get("evidence", []),
                confidence=pattern.get("confidence", 0.7),
                timestamp=datetime.utcnow().isoformat()
            )
            insights.append(insight)
        
        return insights
    
    def _find_common_factors(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common factors in successful interactions."""
        # Placeholder implementation
        return [
            {
                "type": "clear问题描述",
                "description": "clear problem descriptions",
                "confidence": 0.8,
                "evidence": ["multiple successful interactions had detailed problem descriptions"]
            }
        ]
    
    def _detect_improvement_patterns(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect improvement patterns over time."""
        # Placeholder implementation
        return []
    
    def _analyze_response_patterns(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze user response patterns."""
        # Placeholder implementation
        return []
