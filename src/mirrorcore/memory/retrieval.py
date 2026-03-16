"""
Memory Retrieval

Retrieves relevant memory entries for context and learning.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from ..persona.drift import DriftEvaluation
from ..reasoning.response_engine import ConfidenceEvaluation, RootCauseHypothesis


@dataclass
class RetrievalQuery:
    """Represents a memory retrieval query."""
    text: str
    context: Dict[str, Any]
    limit: int = 10
    min_relevance: float = 0.3
    time_window_days: Optional[int] = None


@dataclass
class RetrievedMemory:
    """Represents a retrieved memory with relevance score."""
    memory: Dict[str, Any]
    relevance_score: float
    retrieval_reason: str


@dataclass
class InvestigationMemoryMatch:
    """Represents a retrieved past investigation session."""
    session_id: str
    timestamp: str
    subsystem: Optional[str]
    root_cause_category: Optional[str]
    strategy_family: Optional[str]
    resolution_summary: Optional[str]
    similarity_score: float


class MemoryRetrieval:
    """Retrieves relevant memories based on context and queries."""
    
    def __init__(self):
        self.retrieval_strategies = {
            "text_similarity": self._text_similarity_search,
            "context_matching": self._context_matching_search,
            "temporal_proximity": self._temporal_proximity_search,
            "pattern_matching": self._pattern_matching_search
        }
    
    def search_memories(self, query: RetrievalQuery, memory_store) -> List[RetrievedMemory]:
        """Search memories using multiple strategies."""
        all_results = []
        
        # Apply different retrieval strategies
        for strategy_name, strategy_func in self.retrieval_strategies.items():
            results = strategy_func(query, memory_store)
            all_results.extend(results)
        
        # Deduplicate and rank results
        deduplicated_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(deduplicated_results, query)
        
        # Filter by minimum relevance and limit
        filtered_results = [
            r for r in ranked_results 
            if r.relevance_score >= query.min_relevance
        ][:query.limit]
        
        return filtered_results
    
    def fetch_learning_signals(self, memory_store, signal_type: Optional[str] = None, 
                           session_id: Optional[str] = None,
                           days_back: int = 30) -> List[Dict[str, Any]]:
        """Fetch conversation learning signals for persona update evaluation."""
        # Create query for learning signals
        query = RetrievalQuery(
            text="",
            context={"memory_type": "learning_signal"},
            limit=100,
            time_window_days=days_back,
            min_relevance=0.1
        )
        
        # Get all learning signals
        all_signals = []
        memories = memory_store.search_memories("", limit=100)
        
        for memory in memories:
            content = memory.get("content", "")
            
            # Check if this is a learning signal
            if ("signal_type" in content and "session_id" in content and 
                memory.get("memory_type") == "learning_signal"):
                
                # Parse the content (should be JSON string)
                try:
                    import json
                    signal_data = json.loads(content) if isinstance(content, str) else content
                    
                    # Filter by signal type if specified
                    if signal_type and signal_data.get("signal_type") != signal_type:
                        continue
                    
                    # Filter by session ID if specified
                    if session_id and signal_data.get("session_id") != session_id:
                        continue
                    
                    all_signals.append(signal_data)
                except:
                    continue
        
        return all_signals
    
    def fetch_signals_by_trait(self, trait_name: str, memory_store, days_back: int = 30) -> List[Dict[str, Any]]:
        """Fetch learning signals for a specific trait."""
        # Map traits to signal types
        trait_to_signals = {
            "communication_preference": ["directness_preference", "communication_preference"],
            "evidence_threshold": ["evidence_preference"],
            "ambiguity_tolerance": ["ambiguity_tolerance"],
            "action_bias": ["action_bias"],
            "trust_verification_style": ["trust_verification_style"],
            "self_reliance_level": ["self_reliance_level"],
            "confidence_calibration": ["uncertainty_response"]
        }
        
        relevant_signal_types = trait_to_signals.get(trait_name, [])
        all_signals = []
        
        for signal_type in relevant_signal_types:
            signals = self.fetch_learning_signals(
                signal_type=signal_type,
                days_back=days_back,
                memory_store=memory_store
            )
            all_signals.extend(signals)
        
        return all_signals
    
    def summarize_signal_directions(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize signal directions across multiple signals."""
        if not signals:
            return {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
        
        positive_count = sum(1 for s in signals if s.get("inferred_direction") == "positive")
        negative_count = sum(1 for s in signals if s.get("inferred_direction") == "negative")
        total_count = len(signals)
        
        return {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": total_count - positive_count - negative_count,
            "total": total_count,
            "positive_ratio": positive_count / total_count if total_count > 0 else 0,
            "negative_ratio": negative_count / total_count if total_count > 0 else 0
        }
    
    def get_similar_situations(self, current_situation: Dict[str, Any], memory_store, limit: int = 5) -> List[RetrievedMemory]:
        """Get similar situations to current context."""
        query = RetrievalQuery(
            text=current_situation.get("description", ""),
            context=current_situation.get("context", {}),
            limit=limit
        )
        
        return self.search_memories(query, memory_store)
    
    def get_successful_solutions(self, problem_type: str, memory_store, limit: int = 10) -> List[RetrievedMemory]:
        """Get successful solutions for a specific problem type."""
        # Create query for successful solutions
        query = RetrievalQuery(
            text=problem_type,
            context={"outcome": "success", "problem_type": problem_type},
            limit=limit,
            min_relevance=0.5
        )
        
        return self.search_memories(query, memory_store)
    
    def _text_similarity_search(self, query: RetrievalQuery, memory_store) -> List[RetrievedMemory]:
        """Search based on text similarity."""
        results = []
        
        # Get all memories from store
        all_memories = memory_store.search_memories(query.text, limit=100)
        
        for memory in all_memories:
            # Calculate text similarity
            similarity = self._calculate_text_similarity(query.text, memory)
            
            if similarity > 0.3:  # Minimum similarity threshold
                result = RetrievedMemory(
                    memory=memory,
                    relevance_score=similarity * 0.8,  # Weight for text similarity
                    retrieval_reason="text_similarity"
                )
                results.append(result)
        
        return results
    
    def _context_matching_search(self, query: RetrievalQuery, memory_store) -> List[RetrievedMemory]:
        """Search based on context matching."""
        results = []
        
        # For now, use simple context matching
        # In a full implementation, this would query the database directly
        all_memories = memory_store.search_memories("", limit=100)  # Get all memories
        
        for memory in all_memories:
            context_score = self._calculate_context_similarity(query.context, memory.get("context", {}))
            
            if context_score > 0.3:
                result = RetrievedMemory(
                    memory=memory,
                    relevance_score=context_score * 0.7,  # Weight for context matching
                    retrieval_reason="context_matching"
                )
                results.append(result)
        
        return results
    
    def _temporal_proximity_search(self, query: RetrievalQuery, memory_store) -> List[RetrievedMemory]:
        """Search based on temporal proximity."""
        results = []
        
        if query.time_window_days:
            # Calculate time window
            cutoff_date = datetime.utcnow() - timedelta(days=query.time_window_days)
            
            # Get recent memories
            all_memories = memory_store.search_memories("", limit=100)
            
            for memory in all_memories:
                memory_time = datetime.fromisoformat(memory.get("timestamp", "").replace('Z', '+00:00'))
                
                if memory_time >= cutoff_date:
                    # Calculate recency score
                    days_old = (datetime.utcnow() - memory_time).days
                    recency_score = max(0, 1 - (days_old / query.time_window_days))
                    
                    result = RetrievedMemory(
                        memory=memory,
                        relevance_score=recency_score * 0.3,  # Weight for recency
                        retrieval_reason="temporal_proximity"
                    )
                    results.append(result)
        
        return results
    
    def _pattern_matching_search(self, query: RetrievalQuery, memory_store) -> List[RetrievedMemory]:
        """Search based on pattern matching."""
        results = []
        
        # Look for patterns in the query
        patterns = self._extract_patterns(query.text)
        
        if patterns:
            all_memories = memory_store.search_memories("", limit=100)
            
            for memory in all_memories:
                pattern_score = self._calculate_pattern_match(patterns, memory)
                
                if pattern_score > 0.4:
                    result = RetrievedMemory(
                        memory=memory,
                        relevance_score=pattern_score * 0.6,  # Weight for pattern matching
                        retrieval_reason="pattern_matching"
                    )
                    results.append(result)
        
        return results
    
    def _calculate_text_similarity(self, text1: str, memory: Dict[str, Any]) -> float:
        """Calculate text similarity between query and memory."""
        # Simple word overlap similarity
        # In a full implementation, this would use embeddings or more sophisticated NLP
        
        content_text = memory.get("content", {}).get("input", "") + " " + memory.get("content", {}).get("output", "")
        
        words1 = set(text1.lower().split())
        words2 = set(content_text.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate context similarity."""
        if not context1 or not context2:
            return 0.0
        
        # Simple field matching
        matching_fields = 0
        total_fields = 0
        
        for key in context1:
            if key in context2:
                total_fields += 1
                if context1[key] == context2[key]:
                    matching_fields += 1
        
        if total_fields == 0:
            return 0.0
        
        return matching_fields / total_fields
    
    def _extract_patterns(self, text: str) -> List[str]:
        """Extract patterns from query text."""
        patterns = []
        
        # Common error patterns
        error_patterns = ["error", "failed", "denied", "timeout", "refused"]
        for pattern in error_patterns:
            if pattern in text.lower():
                patterns.append(f"error_{pattern}")
        
        # Command patterns
        if any(cmd in text.lower() for cmd in ["git", "docker", "npm", "pip"]):
            patterns.append("command_related")
        
        return patterns
    
    def _calculate_pattern_match(self, patterns: List[str], memory: Dict[str, Any]) -> float:
        """Calculate pattern match score."""
        memory_text = (memory.get("content", {}).get("input", "") + " " + 
                      memory.get("content", {}).get("output", "")).lower()
        
        matches = 0
        for pattern in patterns:
            if pattern.replace("_", " ") in memory_text:
                matches += 1
        
        return matches / len(patterns) if patterns else 0.0
    
    def _deduplicate_results(self, results: List[RetrievedMemory]) -> List[RetrievedMemory]:
        """Remove duplicate results."""
        seen_ids = set()
        deduplicated = []
        
        for result in results:
            memory_id = result.memory.get("id")
            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)
                deduplicated.append(result)
        
        return deduplicated
    
    def fetch_drift_evaluations(self, trait_name: Optional[str] = None, 
                                days_back: int = 90, memory_store=None) -> List[DriftEvaluation]:
        """Fetch drift evaluations for analysis."""
        if memory_store is None:
            return []
        
        # Search for drift evaluation memory entries
        query = RetrievalQuery(
            text=f"persona_drift_evaluation {trait_name or ''}",
            context={"evaluation_type": "persona_drift"},
            limit=50,
            min_relevance=0.5
        )
        
        retrieved = self.search_memories(query, memory_store)
        drift_evaluations = []
        
        for memory_entry in retrieved:
            try:
                # Parse drift evaluation from stored content
                evaluation_data = json.loads(memory_entry.content)
                
                # Filter by trait if specified
                if trait_name and evaluation_data.get("trait_name") != trait_name:
                    continue
                
                # Filter by time window
                if days_back:
                    entry_date = datetime.fromisoformat(memory_entry.timestamp)
                    if (datetime.utcnow() - entry_date).days > days_back:
                        continue
                
                # Create DriftEvaluation object (simplified version)
                from ..persona.drift import DriftPressure, DriftAction
                drift_pressure = DriftPressure(
                    trait_name=evaluation_data.get("trait_name", ""),
                    baseline_value=evaluation_data.get("baseline_value", 0.0),
                    baseline_confidence=evaluation_data.get("baseline_confidence", 0.5),
                    reinforcing_count=0,  # Not stored in evaluation
                    contradictory_count=0,
                    neutral_count=0,
                    recent_direction="stable"
                )
                
                drift_eval = DriftEvaluation(
                    trait_name=evaluation_data.get("trait_name", ""),
                    baseline_value=evaluation_data.get("baseline_value", 0.0),
                    baseline_confidence=evaluation_data.get("baseline_confidence", 0.5),
                    drift_pressure=drift_pressure,
                    evaluation_timestamp=evaluation_data.get("evaluation_timestamp", ""),
                    recommended_action=DriftAction(evaluation_data.get("recommended_action", "retain")),
                    action_rationale=evaluation_data.get("action_rationale", ""),
                    confidence_adjustment=evaluation_data.get("confidence_adjustment", 0.0),
                    revision_suggestion=evaluation_data.get("revision_suggestion")
                )
                
                drift_evaluations.append(drift_eval)
                
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        
        return drift_evaluations
    
    def summarize_drift_patterns(self, days_back: int = 30, memory_store=None) -> Dict[str, Any]:
        """Summarize drift patterns across all traits."""
        drift_evaluations = self.fetch_drift_evaluations(days_back=days_back, memory_store=memory_store)
        
        if not drift_evaluations:
            return {"message": "No drift evaluations found"}
        
        # Analyze patterns
        traits_under_pressure = []
        revision_suggestions = []
        confidence_trends = {}
        
        for evaluation in drift_evaluations:
            trait_name = evaluation.trait_name
            
            # Check for active drift pressure
            if abs(evaluation.drift_pressure.drift_pressure) > 0.3:
                traits_under_pressure.append(trait_name)
            
            # Track confidence trends
            if trait_name not in confidence_trends:
                confidence_trends[trait_name] = []
            confidence_trends[trait_name].append(evaluation.confidence_adjustment)
            
            # Collect revision suggestions
            if evaluation.revision_suggestion:
                revision_suggestions.append({
                    "trait": trait_name,
                    "suggestion": evaluation.revision_suggestion,
                    "timestamp": evaluation.evaluation_timestamp
                })
        
        # Calculate trend statistics
        trend_summary = {}
        for trait, adjustments in confidence_trends.items():
            if adjustments:
                trend_summary[trait] = {
                    "total_adjustments": len(adjustments),
                    "avg_adjustment": sum(adjustments) / len(adjustments),
                    "trend": "improving" if sum(adjustments) > 0 else "declining"
                }
        
        return {
            "evaluation_period_days": days_back,
            "total_evaluations": len(drift_evaluations),
            "traits_under_pressure": traits_under_pressure,
            "revision_suggestions": revision_suggestions,
            "confidence_trends": trend_summary,
            "summary": f"Found {len(traits_under_pressure)} traits under drift pressure"
        }
    
    def retrieve_ranked_fixes(self, memory_store, incident_type: str, limit: int = 5) -> List[RetrievedMemory]:
        """Retrieve ranked fixes for a specific incident type."""
        query = RetrievalQuery(
            text=f"ranked fixes for {incident_type}",
            context={"memory_type": "terminal_fix_outcome", "ranked": True},
            limit=limit
        )
        
        # Get ranked fix outcomes from memory store
        ranked_fixes = memory_store.get_ranked_fixes_by_incident_type(incident_type, limit)
        
        # Convert to memory entries
        memories = []
        for fix in ranked_fixes:
            memory_content = {
                "incident_type": incident_type,
                "normalized_fix": fix.get('normalized_fix'),
                "score": fix.get('score'),
                "success_count": fix.get('success_count'),
                "partial_count": fix.get('partial_count'),
                "failed_count": fix.get('failed_count'),
                "total_attempts": fix.get('total_attempts'),
                "latest_timestamp": fix.get('latest_timestamp'),
                "latest_result": fix.get('latest_result'),
                "ranking_reason": f"Score: {fix.get('score')} (success: {fix.get('success_count')}, partial: {fix.get('partial_count')}, failed: {fix.get('failed_count')})"
            }
            
            memories.append(RetrievedMemory(
                memory=memory_content,
                relevance_score=1.0,  # Ranked fixes are highly relevant
                retrieval_reason=f"Ranked fix for {incident_type}"
            ))
        
        return self._rank_results(memories, query)
    
    def retrieve_successful_fixes(self, memory_store, incident_type: str, limit: int = 5) -> List[RetrievedMemory]:
        """Retrieve successful fixes for a specific incident type."""
        query = RetrievalQuery(
            text=f"successful fixes for {incident_type}",
            context={"memory_type": "terminal_fix_outcome", "result_status": "success"},
            limit=limit
        )
        
        # Get fix outcomes from memory store
        outcomes = memory_store.get_terminal_fix_outcomes(incident_type=incident_type, result_status="success", limit=limit)
        
        # Convert to memory entries
        memories = []
        for outcome in outcomes:
            memory_content = {
                "incident_type": outcome.get("incident_type"),
                "suggested_fix": outcome.get("suggested_fix"),
                "attempted_fix": outcome.get("attempted_fix"),
                "result_status": outcome.get("result_status"),
                "confirmed_root_cause": outcome.get("confirmed_root_cause"),
                "notes": outcome.get("notes"),
                "timestamp": outcome.get("timestamp")
            }
            
            memories.append(RetrievedMemory(
                id=outcome["id"],
                memory_type="terminal_fix_outcome",
                content=json.dumps(memory_content),
                relevance_score=1.0,  # Direct match is highly relevant
                timestamp=datetime.fromisoformat(outcome["timestamp"]),
                source_module="terminal_fix_outcomes"
            ))
        
        return self._rank_results(memories, query)
    
    def retrieve_common_failed_fixes(self, memory_store, incident_type: str, limit: int = 3) -> List[RetrievedMemory]:
        """Retrieve commonly failed fixes for an incident type."""
        query = RetrievalQuery(
            text=f"failed fixes for {incident_type}",
            context={"memory_type": "terminal_fix_outcome", "result_status": "failed"},
            limit=limit
        )
        
        # Get failed fix outcomes from memory store
        outcomes = memory_store.get_terminal_fix_outcomes(incident_type=incident_type, result_status="failed", limit=limit)
        
        # Convert to memory entries
        memories = []
        for outcome in outcomes:
            memory_content = {
                "incident_type": outcome.get("incident_type"),
                "suggested_fix": outcome.get("suggested_fix"),
                "attempted_fix": outcome.get("attempted_fix"),
                "result_status": outcome.get("result_status"),
                "confirmed_root_cause": outcome.get("confirmed_root_cause"),
                "notes": outcome.get("notes"),
                "timestamp": outcome.get("timestamp")
            }
            
            memories.append(RetrievedMemory(
                id=outcome["id"],
                memory_type="terminal_fix_outcome",
                content=json.dumps(memory_content),
                relevance_score=0.8,  # Failed fixes are moderately relevant
                timestamp=datetime.fromisoformat(outcome["timestamp"]),
                source_module="terminal_fix_outcomes"
            ))
        
        return self._rank_results(memories, query)
    
    def retrieve_recent_fixes(self, memory_store, incident_type: str, days_back: int = 30, limit: int = 10) -> List[RetrievedMemory]:
        """Retrieve recent fixes for an incident type within time window."""
        query = RetrievalQuery(
            text=f"recent fixes for {incident_type}",
            context={"memory_type": "terminal_fix_outcome"},
            limit=limit
        )
        
        # Get fix outcomes from memory store
        outcomes = memory_store.get_terminal_fix_outcomes(incident_type=incident_type, limit=limit)
        
        # Filter by time window
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Convert to memory entries
        memories = []
        for outcome in outcomes:
            outcome_date = datetime.fromisoformat(outcome["timestamp"])
            if outcome_date >= cutoff_date:
                memory_content = {
                    "incident_type": outcome.get("incident_type"),
                    "suggested_fix": outcome.get("suggested_fix"),
                    "attempted_fix": outcome.get("attempted_fix"),
                    "result_status": outcome.get("result_status"),
                    "confirmed_root_cause": outcome.get("confirmed_root_cause"),
                    "notes": outcome.get("notes"),
                    "timestamp": outcome.get("timestamp")
                }
                
                memories.append(RetrievedMemory(
                    id=outcome["id"],
                    memory_type="terminal_fix_outcome",
                    content=json.dumps(memory_content),
                    relevance_score=0.9,  # Recent fixes are highly relevant
                    timestamp=outcome_date,
                    source_module="terminal_fix_outcomes"
                ))
        
        return self._rank_results(memories, query)
    
    def retrieve_similar_investigations(
        self,
        parsed_log: Any,
        ranked_hypotheses: List[RootCauseHypothesis],
        memory_store: Any,
        limit: int = 3,
    ) -> List[InvestigationMemoryMatch]:
        """Retrieve similar past investigation sessions using deterministic scoring."""
        # Derive current investigation context
        current_subsystem = getattr(parsed_log, "detected_subsystem", None)
        current_signal_families = getattr(parsed_log, "signal_families", None) or []
        
        top_hypothesis_category: Optional[str] = None
        if ranked_hypotheses:
            top_hypothesis_category = ranked_hypotheses[0].category
        
        # Derive current strategy family deterministically from top hypothesis category
        if top_hypothesis_category and "network" in top_hypothesis_category:
            current_strategy_family = "connectivity"
        else:
            current_strategy_family = "configuration"
        
        # Fetch resolved sessions from the underlying store
        if not hasattr(memory_store, "get_resolved_analysis_sessions"):
            return []
        
        try:
            past_sessions = memory_store.get_resolved_analysis_sessions(limit=100)
        except Exception:
            return []
        
        matches: List[InvestigationMemoryMatch] = []
        
        for session in past_sessions:
            past_subsystem = session.get("detected_subsystem")
            past_families = session.get("signal_families") or []
            past_category = session.get("top_hypothesis_category")
            past_strategy_family = session.get("current_strategy_family")
            past_strategies_attempted = session.get("strategies_attempted") or []
            
            if not past_strategy_family and past_strategies_attempted:
                past_strategy_family = past_strategies_attempted[0]
            
            score = 0.0
            
            # Subsystem match (binary)
            if current_subsystem and past_subsystem and current_subsystem == past_subsystem:
                score += 0.4
            
            # Signal family overlap (Jaccard)
            current_set = set(current_signal_families)
            past_set = set(past_families)
            if current_set and past_set:
                intersection = len(current_set & past_set)
                union = len(current_set | past_set) or 1
                signal_score = 0.3 * (intersection / union)
                score += signal_score
            
            # Root-cause hypothesis category match
            if top_hypothesis_category and past_category and top_hypothesis_category == past_category:
                score += 0.2
            
            # Strategy family match
            if current_strategy_family and past_strategy_family and current_strategy_family == past_strategy_family:
                score += 0.1
            
            if score <= 0.0:
                continue
            
            matches.append(
                InvestigationMemoryMatch(
                    session_id=session["id"],
                    timestamp=session["timestamp"],
                    subsystem=past_subsystem,
                    root_cause_category=past_category,
                    strategy_family=past_strategy_family,
                    resolution_summary=session.get("analysis_summary"),
                    similarity_score=round(score, 3),
                )
            )
        
        # Deterministic ranking: score desc, then timestamp asc, then session_id asc
        matches.sort(key=lambda m: (-m.similarity_score, m.timestamp or "", m.session_id))
        
        return matches[:limit]

    def _rank_results(self, results: List[RetrievedMemory], query: RetrievalQuery) -> List[RetrievedMemory]:
        """Rank results by relevance score."""
        # Sort by relevance score (descending)
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)
    
    def evaluate_confidence_for_incident(self, incident_type: str, memory_store) -> ConfidenceEvaluation:
        """Evaluate confidence level for a specific incident type."""
        try:
            # Get historical fix data for this incident type
            ranked_fixes = memory_store.get_ranked_fixes_by_incident_type(incident_type, limit=10)
            
            failed_fixes_count = 0
            successful_fixes_count = 0
            failed_fixes_ratio = 0.0
            
            if ranked_fixes:
                successful_fixes = [f for f in ranked_fixes if f['success_count'] > 0]
                failed_fixes = [f for f in ranked_fixes if f['failed_count'] > f['success_count']]
                
                successful_fixes_count = sum(f['success_count'] for f in successful_fixes)
                failed_fixes_count = sum(f['failed_count'] for f in failed_fixes)
                
                total_attempts = successful_fixes_count + failed_fixes_count
                if total_attempts > 0:
                    failed_fixes_ratio = failed_fixes_count / total_attempts
            
            # Determine confidence level
            from ..reasoning.response_engine import ReasoningResponseEngine
            engine = ReasoningResponseEngine()
            
            if failed_fixes_ratio > engine.HIGH_FAILED_RATIO_THRESHOLD:
                confidence_level = "low"
            elif successful_fixes_count > 5:
                confidence_level = "high"
            elif successful_fixes_count > 2:
                confidence_level = "moderate"
            else:
                confidence_level = "low"
            
            return ConfidenceEvaluation(
                confidence_level=confidence_level,
                incident_confidence=0.8,  # Default for historical evaluation
                failed_fixes_ratio=failed_fixes_ratio,
                successful_fixes_count=successful_fixes_count,
                conflicting_signals=False,
                no_incident_match=False,
                escalation_triggered=confidence_level == "low"
            )
            
        except Exception:
            # Return unknown if evaluation fails
            return ConfidenceEvaluation(
                confidence_level="unknown",
                incident_confidence=0.0,
                failed_fixes_ratio=0.0,
                successful_fixes_count=0,
                conflicting_signals=False,
                no_incident_match=False,
                escalation_triggered=False
            )
