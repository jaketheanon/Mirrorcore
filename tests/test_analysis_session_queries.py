"""Tests for analysis_sessions query semantics (open vs closed)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore


class TestGetLatestAnalysisSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def test_finds_session_when_stalled_is_only_on_investigation_state(self):
        """Stall is stored on investigation_state; session_status stays 'active'."""
        store = DatabaseStore(self.db_path)
        try:
            store.initialize_database()
            sid = store.store_analysis_session(
                {
                    "detected_subsystem": "test_sub",
                    "signal_families": [],
                    "original_hypotheses": [],
                    "top_hypothesis_category": "cat",
                    "suggested_commands": [],
                    "analysis_summary": "summary",
                    "session_status": "active",
                }
            )
            conn = store.get_db_connection()
            conn.execute(
                "UPDATE analysis_sessions SET investigation_state = ? WHERE id = ?",
                ("stalled", sid),
            )
            conn.commit()

            latest = store.get_latest_analysis_session()
            self.assertIsNotNone(latest)
            self.assertEqual(latest["id"], sid)
            self.assertEqual(latest["investigation_state"], "stalled")
            self.assertEqual(latest["session_status"], "active")

            sub = store.get_latest_analysis_session("test_sub")
            self.assertIsNotNone(sub)
            self.assertEqual(sub["id"], sid)
        finally:
            store.close()

    def test_excludes_completed_sessions(self):
        store = DatabaseStore(self.db_path)
        try:
            store.initialize_database()
            sid = store.store_analysis_session(
                {
                    "detected_subsystem": None,
                    "signal_families": [],
                    "original_hypotheses": [],
                    "top_hypothesis_category": None,
                    "suggested_commands": [],
                    "analysis_summary": "x",
                    "session_status": "active",
                }
            )
            store.update_analysis_session_status(sid, "completed")
            self.assertIsNone(store.get_latest_analysis_session())
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
