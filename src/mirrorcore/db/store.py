"""
Database Store

Handles all database operations for Mirrorcore.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import sqlite3
import json
from datetime import datetime
from uuid import uuid4
from ..memory.extractor import LearningSignal
from ..persona.drift import DriftEvaluation
from .models import (
    get_db_connection, 
    initialize_database,
    insert_persona_trait,
    insert_memory_entry,
    insert_terminal_incident,
    insert_decision_record,
    insert_session_log
)


class DatabaseStore:
    """Lightweight database store interface for Mirrorcore."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path
        self._connection = None
    
    def get_db_connection(self):
        """Get a database connection, creating it if needed."""
        if self._connection is None:
            self._connection = get_db_connection(self.db_path)
        return self._connection
    
    def initialize_database(self):
        """Initialize the database with all required tables."""
        conn = self.get_db_connection()
        initialize_database(conn)
        return True
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    # Persona profile operations
    def add_persona_trait(self, trait_name: str, trait_value: str, confidence: float, source: str) -> str:
        """Add a persona trait to the database."""
        conn = self.get_db_connection()
        return insert_persona_trait(conn, trait_name, trait_value, confidence, source)
    
    def get_persona_traits(self, trait_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get persona traits, optionally filtered by trait name."""
        conn = self.get_db_connection()
        
        if trait_name:
            query = "SELECT * FROM persona_profile WHERE trait_name = ? ORDER BY updated_at DESC"
            rows = conn.execute(query, (trait_name,)).fetchall()
        else:
            query = "SELECT * FROM persona_profile ORDER BY updated_at DESC"
            rows = conn.execute(query).fetchall()
        
        return [dict(row) for row in rows]
    
    def persona_exists(self) -> bool:
        """Check if any persona profile exists in the database."""
        conn = self.get_db_connection()
        
        query = "SELECT COUNT(*) as count FROM persona_profile"
        result = conn.execute(query).fetchone()
        
        return result['count'] > 0
    
    def load_persona_profile(self) -> Dict[str, Dict[str, Any]]:
        """Load all persona traits in structured dictionary format."""
        conn = self.get_db_connection()
        
        query = "SELECT trait_name, trait_value, confidence, source FROM persona_profile ORDER BY updated_at DESC"
        rows = conn.execute(query).fetchall()
        
        persona_profile = {}
        
        for row in rows:
            trait_name = row['trait_name']
            
            # If trait already exists, keep the most recent one (first in ordered results)
            if trait_name not in persona_profile:
                persona_profile[trait_name] = {
                    "value": row['trait_value'],
                    "confidence": row['confidence'],
                    "source": row['source']
                }
        
        return persona_profile
    
    def save_assessment_responses(self, responses: List[Dict[str, Any]]) -> List[str]:
        """Save raw assessment responses as memory entries with session grouping."""
        saved_ids = []
        
        for response in responses:
            # Create a structured content for the response
            content = {
                "assessment_session_id": response.get("assessment_session_id"),
                "question_id": response.get("question_id"),
                "question_type": response.get("question_type"),
                "selected_option_id": response.get("selected_option_id"),
                "custom_response": response.get("custom_response"),
                "explanation": response.get("explanation"),
                "trigger_reason": response.get("trigger_reason"),
                "order_index": response.get("order_index")
            }
            
            # Include session ID in tags for easy retrieval
            session_id = response.get("assessment_session_id", "unknown")
            question_type = response.get("question_type", "unknown")
            
            # Save as memory entry
            memory_id = self.add_memory_entry(
                memory_type="initial_assessment_response",
                content=json.dumps(content),  # Serialize as JSON for storage
                source_module="intake",
                confidence=0.9,  # High confidence in recorded responses
                tags=f"assessment,response,{question_type},session_{session_id}"
            )
            saved_ids.append(memory_id)
        
        return saved_ids
    
    def save_persona_traits(self, persona_traits: Dict[str, Any]) -> List[str]:
        """Save inferred persona traits to database."""
        saved_ids = []
        
        for trait_name, trait_data in persona_traits.items():
            if isinstance(trait_data, dict):
                trait_value = trait_data.get("value", 0.5)
                confidence = trait_data.get("confidence", 0.0)
            else:
                # Handle simple numeric values
                trait_value = float(trait_data)
                confidence = 0.5  # Default confidence for simple values
            
            trait_id = self.add_persona_trait(
                trait_name=trait_name,
                trait_value=str(trait_value),
                confidence=confidence,
                source="inferred"
            )
            saved_ids.append(trait_id)
        
        return saved_ids
    
    def save_persona_traits(self, persona_traits: Dict[str, Any]) -> List[str]:
        """Save inferred persona traits to database."""
        saved_ids = []
        
        for trait_name, trait_data in persona_traits.items():
            # Handle both PersonaTrait objects and simple values
            if hasattr(trait_data, 'value'):
                # PersonaTrait object
                trait_value = trait_data.value
                confidence = trait_data.confidence
                source = "assessment"
            else:
                # Simple value
                trait_value = trait_data.get("value", trait_data)
                confidence = trait_data.get("confidence", 0.7)
                source = trait_data.get("source", "assessment")
            
            trait_id = self.add_persona_trait(
                trait_name=trait_name,
                trait_value=str(trait_value),
                confidence=confidence,
                source=source
            )
            saved_ids.append(trait_id)
        
        return saved_ids
    
    def save_learning_signals(self, signals: List[Dict[str, Any]]) -> List[str]:
        """Save conversation learning signals as memory entries."""
        saved_ids = []
        
        for signal in signals:
            # Convert to dict if it's a LearningSignal object
            if hasattr(signal, 'to_dict'):
                signal_data = signal.to_dict()
            else:
                signal_data = signal
            
            # Save learning signal as memory entry
            memory_id = self.add_memory_entry(
                memory_type="learning_signal",
                content=json.dumps(signal_data),  # Serialize as JSON
                source_module=signal_data.get("source_module", "conversation"),
                confidence=signal_data.get("confidence", 0.5),
                tags=f"learning,signal,{signal_data.get('signal_type', 'unknown')}"
            )
            saved_ids.append(memory_id)
        
        return saved_ids
    
    def fetch_learning_signals(self, signal_type: Optional[str] = None, 
                           session_id: Optional[str] = None,
                           days_back: int = 30) -> List[Dict[str, Any]]:
        """Fetch learning signals with optional filtering."""
        # Get all learning signal memories
        memories = self.search_memories("", limit=100)
        
        filtered_signals = []
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        for memory in memories:
            if memory.get("memory_type") != "learning_signal":
                continue
            
            # Parse signal data
            try:
                signal_data = json.loads(memory.get("content", "{}"))
            except:
                continue
            
            # Filter by signal type
            if signal_type and signal_data.get("signal_type") != signal_type:
                continue
            
            # Filter by session ID
            if session_id and signal_data.get("session_id") != session_id:
                continue
            
            # Filter by recency
            try:
                signal_time = datetime.fromisoformat(signal_data.get("timestamp", ""))
                if signal_time < cutoff_date:
                    continue
            except:
                continue
            
            filtered_signals.append(signal_data)
        
        return filtered_signals
    
    def get_persona_refinement_suggestions(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get stored persona refinement suggestions."""
        # Get refinement memories
        memories = self.search_memories("", limit=100)
        
        refinements = []
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        for memory in memories:
            if memory.get("memory_type") != "persona_refinement":
                continue
            
            # Parse refinement data
            try:
                refinement_data = json.loads(memory.get("content", "{}"))
                
                # Filter by recency
                try:
                    refinement_time = datetime.fromisoformat(refinement_data.get("timestamp", ""))
                    if refinement_time >= cutoff_date:
                        refinements.append(refinement_data)
                except:
                    continue
            except:
                continue
        
        return refinements
    
    # Memory operations
    def add_memory_entry(self, memory_type: str, content: str, source_module: str, 
                        confidence: float, tags: str = None) -> str:
        """Add a memory entry to the database."""
        conn = self.get_db_connection()
        return insert_memory_entry(conn, memory_type, content, source_module, confidence, tags)
    
    def search_memories(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory entries for matching content."""
        conn = self.get_db_connection()
        
        search_pattern = f"%{query_text}%"
        query = """
            SELECT * FROM memory_entries 
            WHERE content LIKE ? OR tags LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        rows = conn.execute(query, (search_pattern, search_pattern, limit)).fetchall()
        return [dict(row) for row in rows]
    
    def get_memories_by_type(self, memory_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get memory entries by type."""
        conn = self.get_db_connection()
        
        query = "SELECT * FROM memory_entries WHERE memory_type = ? ORDER BY timestamp DESC LIMIT ?"
        rows = conn.execute(query, (memory_type, limit)).fetchall()
        
        return [dict(row) for row in rows]
    
    # Terminal incident operations
    def add_terminal_incident(self, command: str, error_message: str, root_cause: str = None,
                             solution: str = None, success: bool = False, notes: str = None) -> str:
        """Add a terminal incident to the database."""
        conn = self.get_db_connection()
        return insert_terminal_incident(conn, command, error_message, root_cause, solution, success, notes)
    
    def save_terminal_incident(self, command: str, error_message: str, root_cause: str, 
                           solution: str, success: bool = False, notes: str = None) -> str:
        """Save a terminal incident to database."""
        conn = self.get_db_connection()
        return insert_terminal_incident(conn, command, error_message, root_cause, solution, success, notes)
    
    def save_incident_signature(self, user_input: str, incident_signature, session_id: str = None) -> str:
        """Save an incident signature detection to database."""
        conn = self.get_db_connection()
        
        # Format the incident data for storage
        incident_type = incident_signature.incident_type
        likely_causes = "; ".join(incident_signature.likely_causes[:3])  # Store top 3
        recommended_checks = "; ".join(incident_signature.recommended_first_checks[:4])  # Store top 4
        initial_actions = "; ".join(incident_signature.low_risk_initial_actions[:3])  # Store top 3
        
        # Create notes with incident details
        notes = f"Incident Type: {incident_type}\n"
        notes += f"Confidence: {incident_signature.confidence:.2f}\n"
        notes += f"Matched Terms: {', '.join(incident_signature.matched_terms)}\n"
        notes += f"Session ID: {session_id or 'unknown'}"
        
        # Use the existing terminal incidents table
        return self.save_terminal_incident(
            command=user_input,
            error_message=f"Detected: {incident_type}",
            root_cause=likely_causes,
            solution=recommended_checks,
            success=False,  # Mark as unresolved initially
            notes=notes
        )
    
    def save_terminal_fix_outcome(self, incident_id: str, suggested_fix: str, attempted_fix: str, 
                               result_status: str, confirmed_root_cause: str = None, 
                               notes: str = None) -> str:
        """Save terminal fix outcome to database."""
        conn = self.get_db_connection()
        
        outcome_id = str(uuid4())
        now = datetime.utcnow().isoformat()
        
        query = """
            INSERT INTO terminal_fix_outcomes 
            (id, incident_id, suggested_fix, attempted_fix, result_status, confirmed_root_cause, notes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        conn.execute(query, (outcome_id, incident_id, suggested_fix, attempted_fix, result_status, confirmed_root_cause, notes, now))
        conn.commit()
        
        return outcome_id
    
    def get_terminal_fix_outcomes(self, incident_type: str = None, result_status: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get terminal fix outcomes, optionally filtered by incident type or result status."""
        conn = self.get_db_connection()
        
        if incident_type and result_status:
            query = """
                SELECT tfo.*, ti.error_message as incident_type 
                FROM terminal_fix_outcomes tfo
                JOIN terminal_incidents ti ON tfo.incident_id = ti.id
                WHERE ti.error_message LIKE ? AND tfo.result_status = ?
                ORDER BY tfo.timestamp DESC LIMIT ?
            """
            search_pattern = f"%{incident_type}%"
            rows = conn.execute(query, (search_pattern, result_status, limit)).fetchall()
        elif incident_type:
            query = """
                SELECT tfo.*, ti.error_message as incident_type 
                FROM terminal_fix_outcomes tfo
                JOIN terminal_incidents ti ON tfo.incident_id = ti.id
                WHERE ti.error_message LIKE ?
                ORDER BY tfo.timestamp DESC LIMIT ?
            """
            search_pattern = f"%{incident_type}%"
            rows = conn.execute(query, (search_pattern, limit)).fetchall()
        elif result_status:
            query = """
                SELECT tfo.*, ti.error_message as incident_type 
                FROM terminal_fix_outcomes tfo
                JOIN terminal_incidents ti ON tfo.incident_id = ti.id
                WHERE tfo.result_status = ?
                ORDER BY tfo.timestamp DESC LIMIT ?
            """
            rows = conn.execute(query, (result_status, limit)).fetchall()
        else:
            query = """
                SELECT tfo.*, ti.error_message as incident_type 
                FROM terminal_fix_outcomes tfo
                JOIN terminal_incidents ti ON tfo.incident_id = ti.id
                ORDER BY tfo.timestamp DESC LIMIT ?
            """
            rows = conn.execute(query, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    def get_ranked_fixes_by_incident_type(self, incident_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get ranked fixes for a specific incident type with deterministic ranking."""
        conn = self.get_db_connection()
        
        query = """
            SELECT tfo.*, ti.error_message as incident_type 
            FROM terminal_fix_outcomes tfo
            JOIN terminal_incidents ti ON tfo.incident_id = ti.id
            WHERE ti.error_message LIKE ?
            ORDER BY tfo.timestamp DESC
            LIMIT ?
        """
        search_pattern = f"%{incident_type}%"
        rows = conn.execute(query, (search_pattern, limit * 3)).fetchall()  # Get more for ranking
        
        # Group and rank fixes
        return self._rank_fix_attempts([dict(row) for row in rows], limit)
    
    def _rank_fix_attempts(self, fix_attempts: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Rank fix attempts using deterministic scoring."""
        # Group similar fix attempts
        grouped_fixes = {}
        
        for attempt in fix_attempts:
            attempted_fix = attempt.get('attempted_fix', '').strip().lower()
            if not attempted_fix:
                continue
                
            # Normalize fix text
            normalized_fix = self._normalize_fix_text(attempted_fix)
            
            if normalized_fix not in grouped_fixes:
                grouped_fixes[normalized_fix] = {
                    'original_fixes': [],
                    'success_count': 0,
                    'partial_count': 0,
                    'failed_count': 0,
                    'latest_timestamp': None,
                    'latest_result': None
                }
            
            group = grouped_fixes[normalized_fix]
            group['original_fixes'].append(attempt)
            
            # Count results
            result_status = attempt.get('result_status', '').lower()
            if result_status == 'success':
                group['success_count'] += 1
            elif result_status == 'partial':
                group['partial_count'] += 1
            elif result_status == 'failed':
                group['failed_count'] += 1
            
            # Track latest attempt
            attempt_time = attempt.get('timestamp')
            if attempt_time and (group['latest_timestamp'] is None or attempt_time > group['latest_timestamp']):
                group['latest_timestamp'] = attempt_time
                group['latest_result'] = result_status
        
        # Calculate scores and rank
        ranked_fixes = []
        for normalized_fix, group in grouped_fixes.items():
            score = self._calculate_fix_score(group)
            ranked_fixes.append({
                'normalized_fix': normalized_fix,
                'original_fixes': group['original_fixes'],
                'score': score,
                'success_count': group['success_count'],
                'partial_count': group['partial_count'],
                'failed_count': group['failed_count'],
                'latest_timestamp': group['latest_timestamp'],
                'latest_result': group['latest_result'],
                'total_attempts': len(group['original_fixes'])
            })
        
        # Sort by score (descending), then by recency for ties
        ranked_fixes.sort(key=lambda x: (-x['score'], x['latest_timestamp'] or ''))
        
        return ranked_fixes[:limit]
    
    def _normalize_fix_text(self, fix_text: str) -> str:
        """Normalize fix text for grouping."""
        import re
        
        # Convert to lowercase
        normalized = fix_text.lower()
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        # Collapse repeated whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common command prefixes/suffixes for better grouping
        prefixes_to_remove = ['sudo ', 'run ', 'execute ', 'try ']
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        # Remove common suffixes
        suffixes_to_remove = [' command', ' and see', ' and check', ' and test']
        for suffix in suffixes_to_remove:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        
        return normalized.strip()
    
    def _calculate_fix_score(self, group: Dict[str, Any]) -> float:
        """Calculate deterministic score for a fix group."""
        # Base scoring weights
        success_weight = 10.0
        partial_weight = 5.0
        failure_weight = -2.0
        recency_bonus = 0.1  # Small bonus for recent attempts
        
        # Calculate base score
        base_score = (
            group['success_count'] * success_weight +
            group['partial_count'] * partial_weight +
            group['failed_count'] * failure_weight
        )
        
        # Add recency bonus if recent (within last 30 days)
        if group['latest_timestamp']:
            from datetime import datetime, timedelta
            try:
                latest_time = datetime.fromisoformat(group['latest_timestamp'])
                if datetime.utcnow() - latest_time < timedelta(days=30):
                    base_score += recency_bonus
            except (ValueError, TypeError):
                pass  # Ignore timestamp parsing errors
        
        # Penalize if no successful attempts
        if group['success_count'] == 0:
            base_score -= 5.0
        
        return round(base_score, 2)

    def get_successful_fixes_by_incident_type(self, incident_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get successful fixes for a specific incident type."""
        return self.get_terminal_fix_outcomes(incident_type=incident_type, result_status="success", limit=limit)
    
    def get_terminal_incidents(self, success: Optional[bool] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get terminal incidents, optionally filtered by success status."""
        conn = self.get_db_connection()
        
        if success is not None:
            query = "SELECT * FROM terminal_incidents WHERE success = ? ORDER BY timestamp DESC LIMIT ?"
            rows = conn.execute(query, (success, limit)).fetchall()
        else:
            query = "SELECT * FROM terminal_incidents ORDER BY timestamp DESC LIMIT ?"
            rows = conn.execute(query, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    def search_terminal_incidents(self, search_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search terminal incidents for matching commands or errors."""
        conn = self.get_db_connection()
        
        search_pattern = f"%{search_text}%"
        query = """
            SELECT * FROM terminal_incidents 
            WHERE command LIKE ? OR error_message LIKE ? OR solution LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        rows = conn.execute(query, (search_pattern, search_pattern, search_pattern, limit)).fetchall()
        return [dict(row) for row in rows]
    
    # Decision history operations
    def add_decision_record(self, problem_description: str, options: str, recommended_option: str,
                          reasoning: str, outcome: str = None, confidence: float = 0.0) -> str:
        """Add a decision record to the database."""
        conn = self.get_db_connection()
        return insert_decision_record(conn, problem_description, options, recommended_option, reasoning, outcome, confidence)
    
    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get decision history."""
        conn = self.get_db_connection()
        
        query = "SELECT * FROM decision_history ORDER BY timestamp DESC LIMIT ?"
        rows = conn.execute(query, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    def search_decisions(self, search_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search decision history for matching problems or solutions."""
        conn = self.get_db_connection()
        
        search_pattern = f"%{search_text}%"
        query = """
            SELECT * FROM decision_history 
            WHERE problem_description LIKE ? OR recommended_option LIKE ? OR reasoning LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        
        rows = conn.execute(query, (search_pattern, search_pattern, search_pattern, limit)).fetchall()
        return [dict(row) for row in rows]
    
    # Session log operations
    def start_session(self, notes: str = None) -> str:
        """Start a new session and return the session ID."""
        from datetime import datetime
        
        session_start = datetime.utcnow().isoformat()
        conn = self.get_db_connection()
        
        return insert_session_log(conn, session_start, interaction_count=0, notes=notes)
    
    def end_session(self, session_id: str, interaction_count: int = 0, notes: str = None):
        """End a session with final interaction count and notes."""
        from datetime import datetime
        
        session_end = datetime.utcnow().isoformat()
        conn = self.get_db_connection()
        
        query = """
            UPDATE session_logs 
            SET session_end = ?, interaction_count = ?, notes = ?
            WHERE id = ?
        """
        
        conn.execute(query, (session_end, interaction_count, notes, session_id))
        conn.commit()
    
    def get_recent_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent session logs."""
        conn = self.get_db_connection()
        
        query = "SELECT * FROM session_logs ORDER BY session_start DESC LIMIT ?"
        rows = conn.execute(query, (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    # Statistics and analytics
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        conn = self.get_db_connection()
        
        stats = {}
        
        # Persona traits count
        stats['persona_traits'] = conn.execute("SELECT COUNT(*) as count FROM persona_profile").fetchone()['count']
        
        # Memory entries count
        stats['memory_entries'] = conn.execute("SELECT COUNT(*) as count FROM memory_entries").fetchone()['count']
        
        # Terminal incidents count
        stats['terminal_incidents'] = conn.execute("SELECT COUNT(*) as count FROM terminal_incidents").fetchone()['count']
        stats['successful_incidents'] = conn.execute("SELECT COUNT(*) as count FROM terminal_incidents WHERE success = TRUE").fetchone()['count']
        
        # Decision records count
        stats['decision_records'] = conn.execute("SELECT COUNT(*) as count FROM decision_history").fetchone()['count']
        
        # Session count
        stats['total_sessions'] = conn.execute("SELECT COUNT(*) as count FROM session_logs").fetchone()['count']
        
        # Average session duration (for completed sessions)
        avg_duration_result = conn.execute("""
            SELECT AVG(CAST(strftime('%s', session_end) - strftime('%s', session_start) AS INTEGER)) as avg_duration
            FROM session_logs 
            WHERE session_end IS NOT NULL
        """).fetchone()
        
        stats['average_session_duration_seconds'] = avg_duration_result['avg_duration'] or 0
        
        return stats
    
    def save_drift_evaluation(self, evaluation: DriftEvaluation) -> str:
        """Save a drift evaluation to memory."""
        evaluation_data = {
            "trait_name": evaluation.trait_name,
            "baseline_value": evaluation.baseline_value,
            "baseline_confidence": evaluation.baseline_confidence,
            "drift_pressure": evaluation.drift_pressure.drift_pressure,
            "evidence_strength": evaluation.drift_pressure.evidence_strength,
            "recommended_action": evaluation.recommended_action.value,
            "action_rationale": evaluation.action_rationale,
            "confidence_adjustment": evaluation.confidence_adjustment,
            "evaluation_timestamp": evaluation.evaluation_timestamp,
            "revision_suggestion": evaluation.revision_suggestion
        }
        
        memory_id = self.add_memory_entry(
            memory_type="persona_drift_evaluation",
            content=json.dumps(evaluation_data),
            source_module="drift_controller",
            confidence=0.8,  # High confidence in evaluation process
            tags=f"persona,drift,{evaluation.trait_name}"
        )
        
        return memory_id
    
    def fetch_drift_evaluations(self, trait_name: Optional[str] = None, 
                                days_back: int = 90) -> List[Dict[str, Any]]:
        """Fetch drift evaluations for analysis."""
        # Search for drift evaluation memory entries
        memories = self.search_memories("", limit=100)
        
        filtered_evaluations = []
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        for memory in memories:
            if memory.get("memory_type") != "persona_drift_evaluation":
                continue
            
            # Filter by trait if specified
            if trait_name:
                try:
                    content = json.loads(memory["content"])
                    if content.get("trait_name") != trait_name:
                        continue
                except json.JSONDecodeError:
                    continue
            
            # Filter by time window
            memory_date = datetime.fromisoformat(memory["timestamp"])
            if memory_date < cutoff_date:
                continue
            
            filtered_evaluations.append(memory)
        
        return filtered_evaluations
    
    def save_session_log(self, session_id: str, session_start: str, 
                       interaction_count: int = 0, session_end: str = None, 
                       notes: str = None) -> str:
        """Save a session log to database."""
        conn = self.get_db_connection()
        
        # Always create a new session log with the provided session_id
        query = """
            INSERT INTO session_logs 
            (id, session_start, session_end, interaction_count, notes)
            VALUES (?, ?, ?, ?, ?)
        """
        
        conn.execute(query, (session_id, session_start, session_end, interaction_count, notes))
        conn.commit()
        
        return session_id
    
    def get_all_terminal_incidents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all terminal incidents for historical analysis."""
        conn = self.get_db_connection()
        query = """
        SELECT id, command, error_message, root_cause, solution, success, 
               timestamp, notes
        FROM terminal_incidents 
        ORDER BY timestamp DESC
        LIMIT ?
        """
        result = conn.execute(query, (limit,)).fetchall()
        
        incidents = []
        for row in result:
            incident = {
                'id': row['id'],
                'command': row['command'],
                'error_message': row['error_message'],
                'root_cause': row['root_cause'],
                'solution': row['solution'],
                'success': bool(row['success']),
                'timestamp': row['timestamp'],
                'notes': row['notes']
            }
            incidents.append(incident)
        
        return incidents
    
    def store_analysis_session(self, session_data: Dict[str, Any]) -> str:
        """Store multi-signal analysis session for follow-up reanalysis."""
        from uuid import uuid4
        from datetime import datetime
        import json
        
        session_id = str(uuid4())
        
        conn = self.get_db_connection()
        query = """
        INSERT INTO analysis_sessions 
        (id, timestamp, detected_subsystem, signal_families, original_hypotheses, 
         top_hypothesis_category, suggested_commands, analysis_summary, session_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        conn.execute(query, (
            session_id,
            datetime.utcnow().isoformat(),
            session_data.get('detected_subsystem'),
            json.dumps(session_data.get('signal_families', [])),
            json.dumps(session_data.get('original_hypotheses', [])),
            session_data.get('top_hypothesis_category'),
            json.dumps(session_data.get('suggested_commands', [])),
            session_data.get('analysis_summary'),
            session_data.get('session_status', 'active')
        ))
        conn.commit()
        
        return session_id
    
    def get_latest_analysis_session(self, subsystem: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis session for follow-up."""
        import json
        
        conn = self.get_db_connection()
        
        if subsystem:
            query = """
            SELECT * FROM analysis_sessions 
            WHERE detected_subsystem = ? AND session_status = 'active'
            ORDER BY timestamp DESC LIMIT 1
            """
            result = conn.execute(query, (subsystem,)).fetchone()
        else:
            query = """
            SELECT * FROM analysis_sessions 
            WHERE session_status = 'active'
            ORDER BY timestamp DESC LIMIT 1
            """
            result = conn.execute(query).fetchone()
        
        if result:
            session = {
                'id': result['id'],
                'timestamp': result['timestamp'],
                'detected_subsystem': result['detected_subsystem'],
                'signal_families': json.loads(result['signal_families'] or '[]'),
                'original_hypotheses': json.loads(result['original_hypotheses'] or '[]'),
                'top_hypothesis_category': result['top_hypothesis_category'],
                'suggested_commands': json.loads(result['suggested_commands'] or '[]'),
                'analysis_summary': result['analysis_summary'],
                'session_status': result['session_status']
            }
            return session
        
        return None
    
    def store_investigation_steps(self, session_id: str, diagnostic_commands: Dict[str, Any]) -> List[str]:
        """Store investigation steps for a session."""
        from uuid import uuid4
        from datetime import datetime
        import json
        
        step_ids = []
        
        conn = self.get_db_connection()
        
        # Store primary steps
        for i, cmd in enumerate(diagnostic_commands.get('primary', []), 1):
            step_id = str(uuid4())
            query = """
            INSERT INTO investigation_steps 
            (id, session_id, step_id, description, command, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            conn.execute(query, (
                step_id,
                session_id,
                i,
                cmd['description'],
                cmd['command'],
                'pending'
            ))
            step_ids.append(step_id)
        
        # Store additional steps with higher step IDs
        for i, cmd in enumerate(diagnostic_commands.get('additional', []), len(diagnostic_commands.get('primary', [])) + 1):
            step_id = str(uuid4())
            query = """
            INSERT INTO investigation_steps 
            (id, session_id, step_id, description, command, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """
            
            conn.execute(query, (
                step_id,
                session_id,
                i,
                cmd['description'],
                cmd['command'],
                'pending'
            ))
            step_ids.append(step_id)
        
        conn.commit()
        return step_ids
    
    def get_investigation_steps(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all investigation steps for a session."""
        import json
        
        conn = self.get_db_connection()
        query = """
        SELECT id, step_id, description, command, status, evidence_generated, timestamp_completed
        FROM investigation_steps 
        WHERE session_id = ?
        ORDER BY step_id
        """
        
        results = conn.execute(query, (session_id,)).fetchall()
        
        steps = []
        for result in results:
            step = {
                'id': result['id'],
                'step_id': result['step_id'],
                'description': result['description'],
                'command': result['command'],
                'status': result['status'],
                'evidence_generated': json.loads(result['evidence_generated'] or '[]'),
                'timestamp_completed': result['timestamp_completed']
            }
            steps.append(step)
        
        return steps
    
    def complete_investigation_step(self, step_id: str, evidence_signals: List[str]):
        """Mark an investigation step as completed with evidence."""
        from datetime import datetime
        import json
        
        conn = self.get_db_connection()
        query = """
        UPDATE investigation_steps 
        SET status = 'completed', 
            evidence_generated = ?, 
            timestamp_completed = ?
        WHERE id = ?
        """
        
        conn.execute(query, (
            json.dumps(evidence_signals),
            datetime.utcnow().isoformat(),
            step_id
        ))
        conn.commit()
    
    def find_matching_step(self, session_id: str, command_text: str) -> Optional[str]:
        """Find the most relevant pending step for a command using deterministic matching."""
        conn = self.get_db_connection()
        
        # Get all pending steps ordered by step_id
        query = """
        SELECT id, step_id, description, command
        FROM investigation_steps 
        WHERE session_id = ? AND status = 'pending'
        ORDER BY step_id
        """
        
        results = conn.execute(query, (session_id,)).fetchall()
        
        if not results:
            return None
        
        # Simple deterministic matching: exact command match first
        for result in results:
            if command_text.strip() == result['command'].strip():
                return result['id']
        
        # Then check for command similarity (contains matching)
        for result in results:
            if (result['command'].lower() in command_text.lower() or 
                command_text.lower() in result['command'].lower()):
                return result['id']
        
        # Return the first pending step as fallback
        return results[0]['id']
    
    def update_analysis_session_status(self, session_id: str, status: str):
        """Update session status (e.g., mark as completed)."""
        conn = self.get_db_connection()
        query = "UPDATE analysis_sessions SET session_status = ? WHERE id = ?"
        conn.execute(query, (status, session_id))
        conn.commit()
    
    def get_session_count(self) -> int:
        """Get total number of sessions."""
        conn = self.get_db_connection()
        result = conn.execute("SELECT COUNT(*) as count FROM session_logs").fetchone()
        return result['count'] if result else 0
