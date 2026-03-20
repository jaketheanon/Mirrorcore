"""
Regression tests for migration guards on missing tables.
"""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore


class TestMigrationTableExistenceGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def _store_with_raw_connection(self) -> DatabaseStore:
        # Create a raw connection without running initialize_database().
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        store = DatabaseStore(self.db_path)
        store._connection = conn
        return store

    def test_ensure_step21_columns_noops_when_analysis_sessions_missing(self):
        store = self._store_with_raw_connection()
        try:
            store.ensure_step21_columns()
        finally:
            store.close()

    def test_ensure_phase27_columns_noops_when_decision_memory_missing(self):
        store = self._store_with_raw_connection()
        try:
            store.ensure_phase27_columns()
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()

