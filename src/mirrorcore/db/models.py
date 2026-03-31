"""
Mirrorcore Database Models

This module defines the database models and ORM layer for Mirrorcore.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


class DatabaseManager:
    """Main database manager for Mirrorcore."""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            from ..config import get_database_path
            db_path = get_database_path()
        
        self.db_path = db_path
        self._connection = None
    
    def connect(self) -> sqlite3.Connection:
        """Connect to the database."""
        if self._connection is None:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")
            
            # Initialize tables if needed
            initialize_database(self._connection)
        
        return self._connection
    
    def close(self):
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connection is not None
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a database query."""
        conn = self.connect()
        return conn.execute(query, params)
    
    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute query and fetch one result."""
        cursor = self.execute(query, params)
        return cursor.fetchone()
    
    def fetchall(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute query and fetch all results."""
        cursor = self.execute(query, params)
        return cursor.fetchall()
    
    def commit(self):
        """Commit the current transaction."""
        if self._connection:
            self._connection.commit()
    
    def rollback(self):
        """Rollback the current transaction."""
        if self._connection:
            self._connection.rollback()


def initialize_database(conn: sqlite3.Connection):
    """Initialize database tables with the required schema."""
    
    # 1) persona_profile table - Stores stable traits about the user
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persona_profile (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            trait_name TEXT NOT NULL,
            trait_value TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            source TEXT NOT NULL CHECK (source IN ('stated', 'inferred', 'observed', 'assessment', 'learning'))
        )
    """)
    
    # 2) memory_entries table - Stores episodic memories from conversations and actions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_entries (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source_module TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            tags TEXT
        )
    """)
    
    # 3) terminal_incidents table - Stores troubleshooting experiences
    conn.execute("""
        CREATE TABLE IF NOT EXISTS terminal_incidents (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            command TEXT NOT NULL,
            error_message TEXT NOT NULL,
            root_cause TEXT,
            solution TEXT,
            success BOOLEAN NOT NULL DEFAULT FALSE,
            notes TEXT
        )
    """)
    
    # 5) terminal_fix_outcomes table - Stores fix attempt results
    conn.execute("""
        CREATE TABLE IF NOT EXISTS terminal_fix_outcomes (
            id TEXT PRIMARY KEY,
            incident_id TEXT NOT NULL,
            suggested_fix TEXT NOT NULL,
            attempted_fix TEXT NOT NULL,
            result_status TEXT NOT NULL,
            confirmed_root_cause TEXT,
            notes TEXT,
            timestamp TIMESTAMP NOT NULL
        )
    """)
    
    # 6) analysis_sessions table - Stores multi-signal analysis context for follow-up
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_sessions (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            detected_subsystem TEXT,
            signal_families TEXT,
            original_hypotheses TEXT,
            top_hypothesis_category TEXT,
            suggested_commands TEXT,
            analysis_summary TEXT,
            session_status TEXT DEFAULT 'active',
            investigation_state TEXT DEFAULT 'active',
            current_strategy_family TEXT,
            strategies_attempted TEXT,
            progress_metrics TEXT,
            evidence_history TEXT,
            hypothesis_history TEXT
        )
    """)
    
    # 7) investigation_steps table - Stores diagnostic steps for analysis sessions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS investigation_steps (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            step_id INTEGER NOT NULL,
            description TEXT NOT NULL,
            command TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            evidence_generated TEXT,
            timestamp_completed TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
        )
    """)
    
    # 4) decision_history table - Stores analyzed decisions and outcomes
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_history (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            problem_description TEXT NOT NULL,
            options TEXT NOT NULL,  -- JSON array of options
            recommended_option TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            outcome TEXT,
            confidence REAL NOT NULL DEFAULT 0.0
        )
    """)
    
    # 5) session_logs table - Tracks individual assistant sessions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_logs (
            id TEXT PRIMARY KEY,
            session_start TIMESTAMP NOT NULL,
            session_end TIMESTAMP,
            interaction_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT
        )
    """)

    # 6) interview_rotation_state table - Persist deterministic interview rotations
    #    across process runs (Phase 30.1).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interview_rotation_state (
            id TEXT PRIMARY KEY,
            counter INTEGER NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)
    
    # 9) user_fix_preferences table - Cross-incident-type fix affinity tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_fix_preferences (
            normalized_fix TEXT PRIMARY KEY,
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0,
            partial_count INTEGER NOT NULL DEFAULT 0,
            last_used TIMESTAMP
        )
    """)
    
    # 9) decision_memory table - Structured user decision memory (Phase 26)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_memory (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            scenario_id TEXT NOT NULL,
            scenario_text TEXT NOT NULL,
            choice_label TEXT NOT NULL,
            choice_value TEXT NOT NULL,
            reasoning_label TEXT NOT NULL,
            reasoning_value TEXT NOT NULL,
            optional_notes TEXT,
            value_tags_json TEXT,
            trait_signals_json TEXT,
            confidence_score REAL NOT NULL DEFAULT 0.0,
            correction_status TEXT NOT NULL DEFAULT 'uncorrected',
            correction_metadata TEXT,
            source TEXT NOT NULL DEFAULT 'interview'
        )
    """)

    # 11) Routed decision clarification memory (Phase 31.2) — situation facts + tendencies
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_router_situation_facts (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            slot_key TEXT NOT NULL,
            slot_value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_router_tendencies (
            slot_key TEXT PRIMARY KEY,
            strength REAL NOT NULL DEFAULT 0.0,
            sample_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
    """)

    # 10) style_memory table - Structured style/persona calibration memory (Phase 28)
    conn.execute("""
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
    """)

    # Phase 34 — repetition control for memory / profile lines in guidance + respond
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_line_surface_events (
            id TEXT PRIMARY KEY,
            line_key TEXT NOT NULL,
            shown_at TEXT NOT NULL
        )
    """)

    # Phase 38 — respond-like-me active learning (feedback + per-evidence weights)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS personal_response_feedback (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            scenario_snippet TEXT NOT NULL,
            prompt_norm_hash TEXT NOT NULL,
            rating TEXT NOT NULL,
            partial_aspect TEXT,
            replacement_text TEXT,
            confidence_shown REAL NOT NULL,
            effective_family TEXT NOT NULL,
            evidence_path_json TEXT NOT NULL,
            likely_answer_snippet TEXT NOT NULL,
            feedback_target TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS respond_evidence_weights (
            evidence_key TEXT PRIMARY KEY,
            balance REAL NOT NULL DEFAULT 0.0,
            wrong_count INTEGER NOT NULL DEFAULT 0,
            right_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
    """)

    # Phase 42 — promoted reusable examples from repeated respond corrections
    conn.execute("""
        CREATE TABLE IF NOT EXISTS personal_response_examples (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            prompt_norm_hash TEXT NOT NULL,
            effective_family TEXT NOT NULL,
            shape_key TEXT NOT NULL,
            example_type TEXT NOT NULL,
            example_text TEXT NOT NULL,
            support_count INTEGER NOT NULL DEFAULT 0,
            contradict_count INTEGER NOT NULL DEFAULT 0,
            strength REAL NOT NULL DEFAULT 0.0,
            last_feedback_at TEXT NOT NULL
        )
    """)

    # Create indexes for performance
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_persona_profile_trait_name ON persona_profile(trait_name)",
        "CREATE INDEX IF NOT EXISTS idx_persona_profile_updated_at ON persona_profile(updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_memory_entries_timestamp ON memory_entries(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_memory_entries_type ON memory_entries(memory_type)",
        "CREATE INDEX IF NOT EXISTS idx_memory_entries_source_module ON memory_entries(source_module)",
        "CREATE INDEX IF NOT EXISTS idx_terminal_incidents_timestamp ON terminal_incidents(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_terminal_incidents_success ON terminal_incidents(success)",
        "CREATE INDEX IF NOT EXISTS idx_terminal_fix_outcomes_timestamp ON terminal_fix_outcomes(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_terminal_fix_outcomes_incident_id ON terminal_fix_outcomes(incident_id)",
        "CREATE INDEX IF NOT EXISTS idx_terminal_fix_outcomes_result_status ON terminal_fix_outcomes(result_status)",
        "CREATE INDEX IF NOT EXISTS idx_decision_history_timestamp ON decision_history(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_decision_history_confidence ON decision_history(confidence)",
        "CREATE INDEX IF NOT EXISTS idx_session_logs_session_start ON session_logs(session_start)",
        "CREATE INDEX IF NOT EXISTS idx_user_fix_preferences_last_used ON user_fix_preferences(last_used)",
        "CREATE INDEX IF NOT EXISTS idx_decision_memory_timestamp ON decision_memory(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_decision_memory_scenario_id ON decision_memory(scenario_id)",
        "CREATE INDEX IF NOT EXISTS idx_decision_memory_source ON decision_memory(source)",
        "CREATE INDEX IF NOT EXISTS idx_style_memory_timestamp ON style_memory(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_style_memory_prompt_id ON style_memory(prompt_id)",
        "CREATE INDEX IF NOT EXISTS idx_style_memory_source ON style_memory(source)",
        "CREATE INDEX IF NOT EXISTS idx_decision_router_situation_ts ON decision_router_situation_facts(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_decision_router_situation_key ON decision_router_situation_facts(slot_key)",
        "CREATE INDEX IF NOT EXISTS idx_memory_line_surface_time ON memory_line_surface_events(shown_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_memory_line_surface_key ON memory_line_surface_events(line_key)",
        "CREATE INDEX IF NOT EXISTS idx_personal_response_feedback_ts ON personal_response_feedback(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_personal_response_feedback_hash ON personal_response_feedback(prompt_norm_hash)",
        "CREATE INDEX IF NOT EXISTS idx_personal_response_examples_prompt_hash ON personal_response_examples(prompt_norm_hash)",
        "CREATE INDEX IF NOT EXISTS idx_personal_response_examples_family_shape ON personal_response_examples(effective_family, shape_key)",
        "CREATE INDEX IF NOT EXISTS idx_personal_response_examples_type_strength ON personal_response_examples(example_type, strength)",
    ]
    
    for index_sql in indexes:
        conn.execute(index_sql)


# Simple helper functions for database operations
def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection."""
    if db_path is None:
        from ..config import get_database_path
        db_path = get_database_path()
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Initialize tables if needed
    initialize_database(conn)
    
    return conn


# Helper functions for inserting data into each table
def insert_persona_trait(conn: sqlite3.Connection, trait_name: str, trait_value: str, 
                       confidence: float, source: str) -> str:
    """Insert a persona trait."""
    import uuid
    from datetime import datetime
    
    trait_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    query = """
        INSERT INTO persona_profile 
        (id, created_at, updated_at, trait_name, trait_value, confidence, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(query, (trait_id, now, now, trait_name, trait_value, confidence, source))
    conn.commit()
    
    return trait_id


def insert_memory_entry(conn: sqlite3.Connection, memory_type: str, content: str, 
                        source_module: str, confidence: float, tags: str = None) -> str:
    """Insert a memory entry."""
    import uuid
    from datetime import datetime
    
    memory_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    query = """
        INSERT INTO memory_entries 
        (id, timestamp, memory_type, content, source_module, confidence, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(query, (memory_id, now, memory_type, content, source_module, confidence, tags))
    conn.commit()
    
    return memory_id


def insert_terminal_incident(conn: sqlite3.Connection, command: str, error_message: str,
                           root_cause: str = None, solution: str = None, 
                           success: bool = False, notes: str = None) -> str:
    """Insert a terminal incident."""
    import uuid
    from datetime import datetime
    
    incident_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    query = """
        INSERT INTO terminal_incidents 
        (id, timestamp, command, error_message, root_cause, solution, success, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(query, (incident_id, now, command, error_message, root_cause, solution, success, notes))
    conn.commit()
    
    return incident_id


def insert_decision_record(conn: sqlite3.Connection, problem_description: str, options: str,
                         recommended_option: str, reasoning: str, outcome: str = None,
                         confidence: float = 0.0) -> str:
    """Insert a decision record."""
    import uuid
    from datetime import datetime
    
    decision_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    query = """
        INSERT INTO decision_history 
        (id, timestamp, problem_description, options, recommended_option, reasoning, outcome, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(query, (decision_id, now, problem_description, options, recommended_option, reasoning, outcome, confidence))
    conn.commit()
    
    return decision_id


def insert_session_log(conn: sqlite3.Connection, session_start: str, interaction_count: int = 0,
                      session_end: str = None, notes: str = None) -> str:
    """Insert a session log."""
    import uuid
    
    session_id = str(uuid.uuid4())
    
    query = """
        INSERT INTO session_logs 
        (id, session_start, session_end, interaction_count, notes)
        VALUES (?, ?, ?, ?, ?)
    """
    
    conn.execute(query, (session_id, session_start, session_end, interaction_count, notes))
    conn.commit()
    
    return session_id


# Convenience function to get database manager
def get_database_manager() -> DatabaseManager:
    """Get a database manager instance."""
    return DatabaseManager()
