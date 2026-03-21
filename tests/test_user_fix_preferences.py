"""Tests for cross-incident user_fix_preferences upsert behavior."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore


class TestUserFixPreferencesUpsert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass
        self.db_path.unlink(missing_ok=True)

    def test_repeated_same_fix_increments_counts(self):
        self.store = DatabaseStore(self.db_path)
        self.store.initialize_database()

        fix = "apt update && apt install -y pkg"
        for _ in range(3):
            self.store._update_user_fix_preference(fix, "success")

        norm = self.store._normalize_fix_text(fix.strip().lower())
        row = self.store.get_db_connection().execute(
            "SELECT success_count, failure_count, partial_count FROM user_fix_preferences WHERE normalized_fix = ?",
            (norm,),
        ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["success_count"], 3)
        self.assertEqual(row["failure_count"], 0)
        self.assertEqual(row["partial_count"], 0)

    def test_same_fix_mixed_outcomes_accumulate(self):
        self.store = DatabaseStore(self.db_path)
        self.store.initialize_database()

        fix = "systemctl restart nginx"
        self.store._update_user_fix_preference(fix, "success")
        self.store._update_user_fix_preference(fix, "failed")
        self.store._update_user_fix_preference(fix, "partial")
        self.store._update_user_fix_preference(fix, "success")

        norm = self.store._normalize_fix_text(fix.strip().lower())
        row = self.store.get_db_connection().execute(
            "SELECT success_count, failure_count, partial_count FROM user_fix_preferences WHERE normalized_fix = ?",
            (norm,),
        ).fetchone()

        self.assertEqual(row["success_count"], 2)
        self.assertEqual(row["failure_count"], 1)
        self.assertEqual(row["partial_count"], 1)


if __name__ == "__main__":
    unittest.main()
