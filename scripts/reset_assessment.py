#!/usr/bin/env python3
"""
Mirrorcore Assessment Reset Script

This script resets the user assessment and persona profile data,
allowing users to retake their initial assessment or start fresh.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore


def reset_persona_assessment():
    """Reset only the persona assessment data.
    
    Safety: Only deletes persona_profile table data.
    Preserves: memory_entries, terminal_incidents, decision_history, session_logs.
    """
    print("Mirrorcore Assessment Reset")
    print("=" * 30)
    
    # Confirm with user
    response = input("This will reset your persona profile data. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Reset cancelled.")
        return
    
    try:
        # Connect to database (use same path as main app)
        from pathlib import Path
        db_path = Path(__file__).parent.parent / "data" / "mirrorcore.db"
        db_store = DatabaseStore(db_path)
        conn = db_store.get_db_connection()
        
        # Check if persona data exists
        if not db_store.persona_exists():
            print("No persona profile data found to reset.")
            return
        
        # Get count before deletion
        cursor = conn.execute("SELECT COUNT(*) as count FROM persona_profile")
        count = cursor.fetchone()['count']
        
        # Delete persona profile data only
        conn.execute("DELETE FROM persona_profile")
        conn.commit()
        conn.close()
        
        print(f"✓ Reset complete. Deleted {count} persona profile entries.")
        print("✓ Memory entries, terminal incidents, decisions, and sessions preserved.")
        
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    reset_persona_assessment()
