"""
Database Store

Handles all database operations for Mirrorcore.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import sqlite3
import json
from datetime import datetime, timedelta
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
        """Get a database connection, creating it if needed.

        On first connection the schema migrations are applied
        automatically so that callers never need to remember to
        call ``initialize_database()`` explicitly.
        """
        if self._connection is None:
            self._connection = get_db_connection(self.db_path)
            self._run_migrations()
        return self._connection

    def _run_migrations(self):
        """Apply all idempotent schema migrations on the current connection."""
        self.ensure_step21_columns()
        self.ensure_phase27_columns()
        self.ensure_phase28_style_memory()
        self.ensure_phase30_1_interview_rotation_state()
        self.ensure_phase31_2_decision_router_memory()
        self.ensure_phase34_memory_line_surface()

    def initialize_database(self):
        """Initialize the database with all required tables.

        Safe to call multiple times — the underlying operations are
        idempotent.  Normally not needed because ``get_db_connection``
        handles this automatically, but kept for explicit use in
        ``main()`` and tests.
        """
        conn = self.get_db_connection()
        initialize_database(conn)
        self._run_migrations()
        return True

    def ensure_step21_columns(self):
        """Ensure Step 21 investigation tracking columns exist on existing databases."""
        conn = self.get_db_connection()
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'analysis_sessions' LIMIT 1"
        ).fetchone()
        if not table_exists:
            return

        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(analysis_sessions)").fetchall()
        }

        required_columns = {
            "investigation_state": "TEXT DEFAULT 'active'",
            "current_strategy_family": "TEXT",
            "strategies_attempted": "TEXT",
            "progress_metrics": "TEXT",
            "evidence_history": "TEXT",
            "hypothesis_history": "TEXT",
        }

        for column, definition in required_columns.items():
            if column not in existing_columns:
                conn.execute(
                    f"ALTER TABLE analysis_sessions ADD COLUMN {column} {definition}"
                )

        conn.commit()

    def ensure_phase27_columns(self):
        """Ensure Phase 27 correction_metadata column exists on existing databases."""
        conn = self.get_db_connection()
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'decision_memory' LIMIT 1"
        ).fetchone()
        if not table_exists:
            return

        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(decision_memory)").fetchall()
        }
        if "correction_metadata" not in existing_columns:
            conn.execute(
                "ALTER TABLE decision_memory ADD COLUMN correction_metadata TEXT"
            )
            conn.commit()

    def ensure_phase28_style_memory(self):
        """Ensure Phase 28 style memory table and indexes exist."""
        conn = self.get_db_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS style_memory (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                prompt_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                selected_label TEXT NOT NULL,
                selected_value TEXT NOT NULL,
                optional_notes TEXT,
                style_tags_json TEXT,
                tone_signals_json TEXT,
                confidence_score REAL NOT NULL DEFAULT 0.0,
                correction_status TEXT NOT NULL DEFAULT 'uncorrected',
                correction_metadata TEXT,
                source TEXT NOT NULL DEFAULT 'style_calibration'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_style_memory_timestamp ON style_memory(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_style_memory_prompt_id ON style_memory(prompt_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_style_memory_source ON style_memory(source)"
        )
        conn.commit()

    def ensure_phase30_1_interview_rotation_state(self):
        """Ensure Phase 30.1 interview rotation state table exists."""
        conn = self.get_db_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_rotation_state (
                id TEXT PRIMARY KEY,
                counter INTEGER NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.commit()

    def ensure_phase31_2_decision_router_memory(self):
        """Ensure routed decision clarification tables exist (Phase 31.2)."""
        conn = self.get_db_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_router_situation_facts (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                slot_key TEXT NOT NULL,
                slot_value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_router_tendencies (
                slot_key TEXT PRIMARY KEY,
                strength REAL NOT NULL DEFAULT 0.0,
                sample_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decision_router_situation_ts "
            "ON decision_router_situation_facts(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_decision_router_situation_key "
            "ON decision_router_situation_facts(slot_key)"
        )
        conn.commit()

    def ensure_phase34_memory_line_surface(self):
        """Phase 34: log surfaced memory-line keys for repetition control."""
        conn = self.get_db_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_line_surface_events (
                id TEXT PRIMARY KEY,
                line_key TEXT NOT NULL,
                shown_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_line_surface_time "
            "ON memory_line_surface_events(shown_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_line_surface_key "
            "ON memory_line_surface_events(line_key)"
        )
        conn.commit()

    def count_recent_memory_line_surfaces(self, line_key: str, window: int = 24) -> int:
        """How often ``line_key`` appears among the last ``window`` surfaces (all keys)."""
        conn = self.get_db_connection()
        rows = conn.execute(
            """
            SELECT line_key FROM memory_line_surface_events
            ORDER BY shown_at DESC LIMIT ?
            """,
            (window,),
        ).fetchall()
        return sum(1 for r in rows if r["line_key"] == line_key)

    def should_surface_memory_line(
        self, line_key: str, *, max_in_window: int = 1, window: int = 40
    ) -> bool:
        """True if this line has not been overused in the recent global window."""
        return self.count_recent_memory_line_surfaces(line_key, window) < max_in_window

    def record_memory_line_surface(self, line_key: str) -> None:
        """Append one surface event and prune old rows."""
        conn = self.get_db_connection()
        conn.execute(
            """
            INSERT INTO memory_line_surface_events (id, line_key, shown_at)
            VALUES (?, ?, ?)
            """,
            (str(uuid4()), line_key, datetime.utcnow().isoformat()),
        )
        self._prune_memory_line_surface_events(conn)
        conn.commit()

    def _prune_memory_line_surface_events(self, conn: sqlite3.Connection, keep_last: int = 2000) -> None:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM memory_line_surface_events"
        ).fetchone()
        if not row or int(row["n"]) <= keep_last + 400:
            return
        excess = int(row["n"]) - keep_last
        conn.execute(
            """
            DELETE FROM memory_line_surface_events WHERE id IN (
                SELECT id FROM memory_line_surface_events
                ORDER BY shown_at ASC LIMIT ?
            )
            """,
            (excess,),
        )

    def get_next_decision_interview_rotation_index(self) -> int:
        """Return the next persisted rotation index for interview scenarios.

        This index increments once per completed decision interview session, so
        scenario rotation is stable across process runs.
        """
        conn = self.get_db_connection()
        rotation_id = "decision_interview_rotation"
        now = datetime.utcnow().isoformat()

        row = conn.execute(
            "SELECT counter FROM interview_rotation_state WHERE id = ?",
            (rotation_id,),
        ).fetchone()

        if not row:
            conn.execute(
                """
                INSERT INTO interview_rotation_state (id, counter, updated_at)
                VALUES (?, ?, ?)
                """,
                # Store the *next* index value so the second call advances.
                (rotation_id, 1, now),
            )
            conn.commit()
            return 0

        current = int(row["counter"])
        next_value = current + 1
        conn.execute(
            """
            UPDATE interview_rotation_state
            SET counter = ?, updated_at = ?
            WHERE id = ?
            """,
            (next_value, now, rotation_id),
        )
        conn.commit()
        return current

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
        
        self._update_user_fix_preference(attempted_fix, result_status)
        
        return outcome_id
    
    def _update_user_fix_preference(self, attempted_fix: str, result_status: str):
        """Upsert cross-incident-type fix preference after each outcome."""
        normalized = self._normalize_fix_text(attempted_fix.strip().lower())
        if not normalized:
            return
        
        status = result_status.strip().lower()
        if status not in ("success", "partial", "failed"):
            return

        # One-hot delta for this outcome (used for insert and as excluded.* on conflict).
        s = 1 if status == "success" else 0
        f = 1 if status == "failed" else 0
        p = 1 if status == "partial" else 0

        conn = self.get_db_connection()
        now = datetime.utcnow().isoformat()

        conn.execute(
            """
            INSERT INTO user_fix_preferences (normalized_fix, success_count, failure_count, partial_count, last_used)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(normalized_fix) DO UPDATE SET
                success_count = user_fix_preferences.success_count + excluded.success_count,
                failure_count = user_fix_preferences.failure_count + excluded.failure_count,
                partial_count = user_fix_preferences.partial_count + excluded.partial_count,
                last_used = excluded.last_used
            """,
            (normalized, s, f, p, now),
        )
        conn.commit()
    
    def _get_user_fix_preferences(self, normalized_fixes: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch-fetch user fix preferences for a set of normalized fix texts."""
        if not normalized_fixes:
            return {}
        conn = self.get_db_connection()
        placeholders = ",".join("?" for _ in normalized_fixes)
        rows = conn.execute(
            f"SELECT * FROM user_fix_preferences WHERE normalized_fix IN ({placeholders})",
            tuple(normalized_fixes),
        ).fetchall()
        return {row["normalized_fix"]: dict(row) for row in rows}
    
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
        
        # Group and rank fixes (with user preference blending)
        return self._rank_fix_attempts([dict(row) for row in rows], limit)
    
    def _rank_fix_attempts(self, fix_attempts: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Rank fix attempts using deterministic scoring with user preference blending."""
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
        
        # Batch-fetch user preferences for all grouped fixes
        user_prefs = self._get_user_fix_preferences(list(grouped_fixes.keys()))
        
        # Calculate scores and rank
        ranked_fixes = []
        for normalized_fix, group in grouped_fixes.items():
            user_pref = user_prefs.get(normalized_fix)
            score = self._calculate_fix_score(group, user_pref=user_pref)
            ranked_fixes.append({
                'normalized_fix': normalized_fix,
                'original_fixes': group['original_fixes'],
                'score': score,
                'success_count': group['success_count'],
                'partial_count': group['partial_count'],
                'failed_count': group['failed_count'],
                'latest_timestamp': group['latest_timestamp'],
                'latest_result': group['latest_result'],
                'total_attempts': len(group['original_fixes']),
                'user_preference_applied': user_pref is not None
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
    
    def _calculate_fix_score(self, group: Dict[str, Any], *, user_pref: Dict[str, Any] = None) -> float:
        """Calculate deterministic score for a fix group.

        Blends per-incident-type counts (global) with cross-incident-type
        user preference data when available.

        Global score (unchanged):
            10*success + 5*partial - 2*failed  (+ recency, penalties)

        User preference bonus (guardrailed):
            raw_bonus  = 3*user_success + 1*user_partial - 2*user_failure
            Positive bonuses are gated by:
              - evidence threshold  (< 2 successes → 25% strength)
              - recency decay       (>30d → 50%, >90d → 25%)
              - bad-habit cap       (failures >= successes → no boost)
              - absolute cap        (max +5.0 points)
              - global safety       (negative global → no rescue)
            Negative bonuses (penalties) always apply at full scale.
        """
        # --- Global scoring (unchanged) ---
        SUCCESS_W = 10.0
        PARTIAL_W = 5.0
        FAILURE_W = -2.0
        RECENCY_BONUS = 0.1

        base_score = (
            group['success_count'] * SUCCESS_W +
            group['partial_count'] * PARTIAL_W +
            group['failed_count'] * FAILURE_W
        )

        if group['latest_timestamp']:
            try:
                latest_time = datetime.fromisoformat(group['latest_timestamp'])
                if datetime.utcnow() - latest_time < timedelta(days=30):
                    base_score += RECENCY_BONUS
            except (ValueError, TypeError):
                pass

        if group['success_count'] == 0:
            base_score -= 5.0

        # --- User preference bonus (guardrailed) ---
        USER_PREF_SCALE = 0.5
        USER_PREF_CAP = 5.0

        if user_pref:
            u_sc = int(user_pref.get('success_count', 0))
            u_pc = int(user_pref.get('partial_count', 0))
            u_fc = int(user_pref.get('failure_count', 0))
            raw_bonus = u_sc * 3.0 + u_pc * 1.0 - u_fc * 2.0

            if raw_bonus > 0:
                # 1) Evidence threshold: require 2+ successes for full strength
                evidence_factor = 1.0 if u_sc >= 2 else 0.25

                # 2) Recency decay based on last_used
                recency_factor = 1.0
                last_used = user_pref.get('last_used')
                if last_used:
                    try:
                        age = datetime.utcnow() - datetime.fromisoformat(last_used)
                        if age > timedelta(days=90):
                            recency_factor = 0.25
                        elif age > timedelta(days=30):
                            recency_factor = 0.5
                    except (ValueError, TypeError):
                        pass

                effective = raw_bonus * USER_PREF_SCALE * evidence_factor * recency_factor

                # 3) Bad-habit prevention: failures >= successes → no positive boost
                if u_fc >= u_sc and u_fc > 0:
                    effective = 0.0

                # 4) Cap to prevent user preference from dominating
                effective = min(effective, USER_PREF_CAP)

                # 5) Global safety: don't rescue a globally-poor fix
                if base_score < 0:
                    effective = 0.0

                base_score += effective
            else:
                # Negative user bonus (penalties) always apply at full scale
                base_score += raw_bonus * USER_PREF_SCALE

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
        
        # Open sessions: `session_status` is only set to 'active' on insert and
        # 'completed' (or rarely 'resolved') when closed — never 'updated'/'stalled'.
        # Stall/progress lives in `investigation_state` (see update_investigation_tracking).
        open_status_sql = (
            "(session_status IS NULL OR session_status NOT IN ('completed', 'resolved'))"
        )
        if subsystem:
            query = f"""
            SELECT * FROM analysis_sessions
            WHERE detected_subsystem = ? AND {open_status_sql}
            ORDER BY timestamp DESC LIMIT 1
            """
            result = conn.execute(query, (subsystem,)).fetchone()
        else:
            query = f"""
            SELECT * FROM analysis_sessions
            WHERE {open_status_sql}
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
                'session_status': result['session_status'],
                # Exposed for follow-up / UI; stall lives here, not in session_status.
                'investigation_state': result['investigation_state'] or 'active',
            }
            return session
        
        return None
    
    def get_resolved_analysis_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently resolved analysis sessions for similarity retrieval."""
        import json
        
        conn = self.get_db_connection()
        query = """
        SELECT id, timestamp, detected_subsystem, signal_families,
               top_hypothesis_category, analysis_summary,
               investigation_state, current_strategy_family, strategies_attempted
        FROM analysis_sessions
        WHERE session_status IN ('completed','resolved')
        ORDER BY timestamp DESC
        LIMIT ?
        """
        
        rows = conn.execute(query, (limit,)).fetchall()
        
        sessions: List[Dict[str, Any]] = []
        for row in rows:
            sessions.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "detected_subsystem": row["detected_subsystem"],
                    "signal_families": json.loads(row["signal_families"] or "[]"),
                    "top_hypothesis_category": row["top_hypothesis_category"],
                    "analysis_summary": row["analysis_summary"],
                    "investigation_state": row["investigation_state"],
                    "current_strategy_family": row["current_strategy_family"],
                    "strategies_attempted": json.loads(row["strategies_attempted"] or "[]"),
                }
            )
        
        return sessions
    
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

    def update_investigation_tracking(self, session_id: str, current_strategy_family: str, 
                                strategies_attempted: List[str], progress_metrics: Dict[str, Any],
                                investigation_state: str = 'active', evidence_history: List[List[str]] = None,
                                hypothesis_history: List[List[Dict[str, Any]]] = None):
        """Update investigation tracking fields."""
        import json
        
        conn = self.get_db_connection()
        query = """
        UPDATE analysis_sessions 
        SET investigation_state = ?, current_strategy_family = ?, 
            strategies_attempted = ?, progress_metrics = ?, 
            evidence_history = ?, hypothesis_history = ?
        WHERE id = ?
        """
        
        conn.execute(query, (
            investigation_state,
            current_strategy_family,
            json.dumps(strategies_attempted),
            json.dumps(progress_metrics),
            json.dumps(evidence_history or []),
            json.dumps(hypothesis_history or []),
            session_id
        ))
        conn.commit()
    
    def get_investigation_tracking(self, session_id: str) -> Dict[str, Any]:
        """Get investigation tracking fields."""
        import json
        
        conn = self.get_db_connection()
        query = """
        SELECT investigation_state, current_strategy_family, strategies_attempted, 
            progress_metrics, evidence_history, hypothesis_history
        FROM analysis_sessions 
        WHERE id = ?
        """
        
        result = conn.execute(query, (session_id,)).fetchone()
        
        if result:
            return {
                'investigation_state': result['investigation_state'] or 'active',
                'current_strategy_family': result['current_strategy_family'],
                'strategies_attempted': json.loads(result['strategies_attempted'] or '[]'),
                'progress_metrics': json.loads(result['progress_metrics'] or '{}'),
                'evidence_history': json.loads(result['evidence_history'] or '[]'),
                'hypothesis_history': json.loads(result['hypothesis_history'] or '[]')
            }
        
        return {
            'investigation_state': 'active',
            'current_strategy_family': None,
            'strategies_attempted': [],
            'progress_metrics': {},
            'evidence_history': [],
            'hypothesis_history': []
        }
    
    def get_session_count(self) -> int:
        """Get total number of sessions."""
        conn = self.get_db_connection()
        result = conn.execute("SELECT COUNT(*) as count FROM session_logs").fetchone()
        return result['count'] if result else 0

    # Decision memory operations (Phase 26)
    def record_decision_memory(self, scenario_id: str, scenario_text: str,
                               choice_label: str, choice_value: str,
                               reasoning_label: str, reasoning_value: str,
                               value_tags: List[str], trait_signals: Dict[str, Any],
                               confidence_score: float = 0.8,
                               optional_notes: str = None,
                               correction_status: str = "uncorrected",
                               source: str = "interview") -> str:
        """Record a structured decision memory entry."""
        conn = self.get_db_connection()

        entry_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        query = """
            INSERT INTO decision_memory
            (id, timestamp, scenario_id, scenario_text,
             choice_label, choice_value, reasoning_label, reasoning_value,
             optional_notes, value_tags_json, trait_signals_json,
             confidence_score, correction_status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        conn.execute(query, (
            entry_id, now, scenario_id, scenario_text,
            choice_label, choice_value, reasoning_label, reasoning_value,
            optional_notes, json.dumps(value_tags), json.dumps(trait_signals),
            confidence_score, correction_status, source
        ))
        conn.commit()

        return entry_id

    def get_recent_decision_memory(self, limit: int = 20,
                                   scenario_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent decision memory entries."""
        conn = self.get_db_connection()

        if scenario_id:
            query = """
                SELECT * FROM decision_memory
                WHERE scenario_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """
            rows = conn.execute(query, (scenario_id, limit)).fetchall()
        else:
            query = """
                SELECT * FROM decision_memory
                ORDER BY timestamp DESC LIMIT ?
            """
            rows = conn.execute(query, (limit,)).fetchall()

        results = []
        for row in rows:
            entry = dict(row)
            try:
                entry['value_tags'] = json.loads(entry.pop('value_tags_json', '[]'))
            except (json.JSONDecodeError, TypeError):
                entry['value_tags'] = []
            try:
                entry['trait_signals'] = json.loads(entry.pop('trait_signals_json', '{}'))
            except (json.JSONDecodeError, TypeError):
                entry['trait_signals'] = {}
            raw_meta = entry.get("correction_metadata")
            if isinstance(raw_meta, str):
                try:
                    entry["correction_metadata"] = json.loads(raw_meta or "{}")
                except (json.JSONDecodeError, TypeError):
                    entry["correction_metadata"] = {}
            elif raw_meta is None:
                entry["correction_metadata"] = {}
            results.append(entry)

        return results

    def update_decision_memory_correction(self, entry_id: str,
                                          correction_status: str,
                                          correction_metadata: Optional[Dict[str, Any]] = None):
        """Update correction status and metadata for a decision memory entry."""
        conn = self.get_db_connection()

        if correction_metadata is not None:
            conn.execute(
                "UPDATE decision_memory SET correction_status = ?, correction_metadata = ? WHERE id = ?",
                (correction_status, json.dumps(correction_metadata), entry_id),
            )
        else:
            conn.execute(
                "UPDATE decision_memory SET correction_status = ? WHERE id = ?",
                (correction_status, entry_id),
            )
        conn.commit()

    # Style memory operations (Phase 28)
    def record_style_memory(self, prompt_id: str, prompt_text: str,
                            selected_label: str, selected_value: str,
                            style_tags: List[str], tone_signals: Dict[str, Any],
                            confidence_score: float = 0.8,
                            optional_notes: str = None,
                            correction_status: str = "uncorrected",
                            correction_metadata: Optional[Dict[str, Any]] = None,
                            source: str = "style_calibration") -> str:
        """Record a structured style memory entry."""
        conn = self.get_db_connection()

        entry_id = str(uuid4())
        now = datetime.utcnow().isoformat()

        query = """
            INSERT INTO style_memory
            (id, timestamp, prompt_id, prompt_text,
             selected_label, selected_value, optional_notes,
             style_tags_json, tone_signals_json, confidence_score,
             correction_status, correction_metadata, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        conn.execute(query, (
            entry_id, now, prompt_id, prompt_text,
            selected_label, selected_value, optional_notes,
            json.dumps(style_tags), json.dumps(tone_signals), confidence_score,
            correction_status,
            json.dumps(correction_metadata) if correction_metadata is not None else None,
            source
        ))
        conn.commit()
        return entry_id

    def get_recent_style_memory(self, limit: int = 20,
                                prompt_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent style memory entries."""
        conn = self.get_db_connection()

        if prompt_id:
            query = """
                SELECT * FROM style_memory
                WHERE prompt_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """
            rows = conn.execute(query, (prompt_id, limit)).fetchall()
        else:
            query = """
                SELECT * FROM style_memory
                ORDER BY timestamp DESC LIMIT ?
            """
            rows = conn.execute(query, (limit,)).fetchall()

        results = []
        for row in rows:
            entry = dict(row)
            try:
                entry["style_tags"] = json.loads(entry.pop("style_tags_json", "[]"))
            except (json.JSONDecodeError, TypeError):
                entry["style_tags"] = []
            try:
                entry["tone_signals"] = json.loads(entry.pop("tone_signals_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                entry["tone_signals"] = {}
            try:
                entry["correction_metadata"] = json.loads(entry.get("correction_metadata") or "{}")
            except (json.JSONDecodeError, TypeError):
                entry["correction_metadata"] = {}
            results.append(entry)

        return results

    def count_decision_memory_rows(self) -> int:
        """Return total rows in decision_memory (for onboarding / progress)."""
        conn = self.get_db_connection()
        row = conn.execute("SELECT COUNT(*) AS count FROM decision_memory").fetchone()
        return int(row["count"]) if row else 0

    def count_style_memory_rows(self) -> int:
        """Return total rows in style_memory (for onboarding / progress)."""
        conn = self.get_db_connection()
        row = conn.execute("SELECT COUNT(*) AS count FROM style_memory").fetchone()
        return int(row["count"]) if row else 0

    def record_router_situation_fact(self, slot_key: str, slot_value: str) -> str:
        """Append a one-off situation fact from routed clarification (not a trait)."""
        conn = self.get_db_connection()
        eid = str(uuid4())
        conn.execute(
            """
            INSERT INTO decision_router_situation_facts (id, timestamp, slot_key, slot_value)
            VALUES (?, ?, ?, ?)
            """,
            (eid, datetime.utcnow().isoformat(), slot_key, slot_value),
        )
        conn.commit()
        return eid

    def merge_router_tendency(self, slot_key: str, signal: float) -> None:
        """Update a long-run tendency from clarification (0..1 signal, capped, decayed)."""
        conn = self.get_db_connection()
        signal = max(0.0, min(1.0, float(signal)))
        now = datetime.utcnow().isoformat()
        row = conn.execute(
            "SELECT strength, sample_count FROM decision_router_tendencies WHERE slot_key = ?",
            (slot_key,),
        ).fetchone()
        if row:
            old = float(row["strength"] or 0.0)
            n = int(row["sample_count"] or 0) + 1
            # Conflicting low-signal updates pull strength down instead of climbing (Phase 33).
            if old >= 0.48 and signal < 0.28:
                merged = max(0.0, old * 0.86 - (0.28 - signal) * 0.5)
            else:
                merged = min(1.0, old * 0.92 + signal * 0.22)
            conn.execute(
                """
                UPDATE decision_router_tendencies
                SET strength = ?, sample_count = ?, updated_at = ?
                WHERE slot_key = ?
                """,
                (merged, n, now, slot_key),
            )
        else:
            conn.execute(
                """
                INSERT INTO decision_router_tendencies (slot_key, strength, sample_count, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (slot_key, signal, 1, now),
            )
        conn.commit()

    def get_router_tendency_map(self, min_strength: float = 0.22) -> Dict[str, float]:
        """Return tendencies that have enough accumulated strength (honest threshold)."""
        conn = self.get_db_connection()
        rows = conn.execute(
            """
            SELECT slot_key, strength, sample_count FROM decision_router_tendencies
            WHERE strength >= ? AND sample_count >= 1
            """,
            (min_strength,),
        ).fetchall()
        return {str(r["slot_key"]): float(r["strength"]) for r in rows}

    def get_recent_router_situation_facts(self, limit: int = 40) -> List[Dict[str, Any]]:
        """Recent situation facts only (newest first)."""
        conn = self.get_db_connection()
        rows = conn.execute(
            """
            SELECT slot_key, slot_value, timestamp FROM decision_router_situation_facts
            ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
