"""
Tests for Memory Agent

Tests memory extraction, retrieval, and update components.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.memory.extractor import MemoryExtractor
from mirrorcore.memory.retrieval import MemoryRetrieval, RetrievalQuery
from mirrorcore.memory.updater import MemoryUpdater
from mirrorcore.reasoning.response_engine import RootCauseHypothesis
from mirrorcore.terminal.historical_fix_gate import should_surface_historical_fix_guidance


class TestMemoryExtractor(unittest.TestCase):
    """Test the memory extractor."""
    
    def setUp(self):
        self.extractor = MemoryExtractor()
    
    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor.pattern_extractors)
        self.assertIn("troubleshooting", self.extractor.pattern_extractors)
    
    def test_extract_patterns(self):
        """Test pattern extraction."""
        # Test with empty episodes
        patterns = self.extractor.extract_patterns([])
        self.assertIsInstance(patterns, list)
    
    def test_extract_insights(self):
        """Test insight extraction."""
        # Test with empty interactions
        insights = self.extractor.extract_insights([])
        self.assertIsInstance(insights, list)
    
    def test_extract_error_type(self):
        """Test error type extraction."""
        error_text = "permission denied: cannot access file"
        error_type = self.extractor._extract_error_type(error_text)
        self.assertEqual(error_type, "permission")
    
    def test_extract_solution(self):
        """Test solution extraction."""
        solution_text = "Try running: `sudo chmod +x script.sh`"
        solution = self.extractor._extract_solution(solution_text)
        self.assertIsNotNone(solution)
        self.assertIn("chmod", solution)


class TestMemoryRetrieval(unittest.TestCase):
    """Test the memory retrieval system."""
    
    def setUp(self):
        self.retrieval = MemoryRetrieval()
    
    def test_initialization(self):
        """Test retrieval initialization."""
        self.assertIsNotNone(self.retrieval.retrieval_strategies)
        self.assertIn("text_similarity", self.retrieval.retrieval_strategies)
    
    def test_create_retrieval_query(self):
        """Test creating retrieval queries."""
        query = RetrievalQuery(
            text="git push failed",
            context={"domain": "terminal"},
            limit=5
        )
        self.assertEqual(query.text, "git push failed")
        self.assertEqual(query.limit, 5)
    
    def test_text_similarity(self):
        """Test text similarity calculation."""
        memory = {"content": {"input": "git push failed with error", "output": "permission denied"}}
        similarity = self.retrieval._calculate_text_similarity("git error", memory)
        self.assertGreater(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_context_similarity(self):
        """Test context similarity calculation."""
        context1 = {"domain": "terminal", "environment": "development"}
        context2 = {"domain": "terminal", "environment": "production"}
        similarity = self.retrieval._calculate_context_similarity(context1, context2)
        self.assertGreater(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

    def test_learned_fix_gates_mixed_runtime_network_one_off(self):
        """Traceback + connectivity signals must not promote a one-off port fix."""
        store_sessions = [
            {
                "detected_subsystem": "python",
                "top_hypothesis_category": "network",
                "signal_families": ["connection_refused", "timeout"],
                "strategies_attempted": ["killed process using port 8080"],
                "investigation_state": "resolved",
            }
        ]

        class _Store:
            def get_resolved_analysis_sessions(self, limit=200):
                return store_sessions

        class _Log:
            detected_subsystem = "python"
            signal_families = ["python_runtime", "error", "connection_refused", "timeout"]
            likely_error_category = "traceback"

        hyp = RootCauseHypothesis(
            text="runtime issue",
            related_families=_Log.signal_families,
            related_subsystems=["python"],
            category="runtime_network",
            score=1.0,
            reason="test",
        )
        r = self.retrieval.match_best_fix_pattern_for_issue(
            _Log(), [hyp], _Store()
        )
        self.assertIsNone(r)

    def test_learned_fix_shows_repeated_permission_strategy(self):
        """Repeated successful pip --user still surfaces after two successes."""
        key = {
            "detected_subsystem": "python",
            "top_hypothesis_category": "permissions",
            "signal_families": ["permission_denied"],
            "strategies_attempted": ["used pip install --user requests"],
            "investigation_state": "resolved",
        }

        class _Store:
            def get_resolved_analysis_sessions(self, limit=200):
                return [dict(key), dict(key)]

        class _Log:
            detected_subsystem = "python"
            signal_families = ["permission_denied"]
            likely_error_category = "permission_denied"

        hyp = RootCauseHypothesis(
            text="perm",
            related_families=_Log.signal_families,
            related_subsystems=["python"],
            category="permissions",
            score=1.0,
            reason="test",
        )
        r = self.retrieval.match_best_fix_pattern_for_issue(_Log(), [hyp], _Store())
        self.assertIsNotNone(r)
        _pattern, strategies = r
        self.assertGreaterEqual(strategies[0].successful_attempts, 2)

    def test_historical_gate_blocks_traceback_network_one_off(self):
        """Runtime/traceback + mixed signals: do not show network one-off port fix."""
        fix = {"success_count": 1, "failed_count": 0, "normalized_fix": "killed process using port 8080"}
        fams = ["python_runtime", "error", "connection_refused", "timeout"]
        ok = should_surface_historical_fix_guidance(
            fix,
            "network_connectivity",
            0.9,
            likely_error_category="traceback",
            signal_families=fams,
            detected_subsystem="python",
            user_input="ValueError: invalid port",
        )
        self.assertFalse(ok)

    def test_historical_gate_allows_pip_after_two_successes(self):
        fix = {"success_count": 2, "failed_count": 0, "normalized_fix": "used pip install --user requests"}
        ok = should_surface_historical_fix_guidance(
            fix,
            "pip_permission_denied",
            0.9,
            likely_error_category="permission_denied",
            signal_families=["permission_denied"],
            detected_subsystem="pip",
        )
        self.assertTrue(ok)

    def test_historical_gate_allows_network_fix_single_success_curl(self):
        fix = {"success_count": 1, "failed_count": 0, "normalized_fix": "freed port 8080"}
        ok = should_surface_historical_fix_guidance(
            fix,
            "network_connectivity",
            0.9,
            likely_error_category="connection_refused",
            signal_families=["connection_refused"],
            detected_subsystem="curl",
            user_input="curl: connection refused",
        )
        self.assertTrue(ok)


class TestMemoryUpdater(unittest.TestCase):
    """Test the memory updater."""
    
    def setUp(self):
        self.updater = MemoryUpdater()
    
    def test_initialization(self):
        """Test updater initialization."""
        self.assertIsNotNone(self.updater.update_strategies)
        self.assertIn("success_rate_update", self.updater.update_strategies)
    
    def test_create_usage_update(self):
        """Test creating usage updates."""
        update = self.updater._create_usage_update("test_memory_id")
        self.assertEqual(update.memory_id, "test_memory_id")
        self.assertEqual(update.update_type, "usage_increment")
    
    def test_create_success_rate_update(self):
        """Test creating success rate updates."""
        interaction = {"outcome": "success"}
        update = self.updater._create_success_rate_update("test_memory_id", interaction)
        self.assertEqual(update.memory_id, "test_memory_id")
        self.assertEqual(update.update_type, "success_rate_update")


if __name__ == "__main__":
    unittest.main()
