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


class TestUserPreferenceGuardrails(unittest.TestCase):
    """Test guardrails on user-specific fix preference weighting (Phase 25.5)."""

    def setUp(self):
        import tempfile, os
        self._tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self._tmp.close()
        from mirrorcore.db.store import DatabaseStore
        self.store = DatabaseStore(Path(self._tmp.name))
        self._tmp_path = self._tmp.name

    def tearDown(self):
        import os
        os.unlink(self._tmp_path)

    def _score(self, group, user_pref=None):
        return self.store._calculate_fix_score(group, user_pref=user_pref)

    def _group(self, success=0, partial=0, failed=0, ts=None):
        return {
            'success_count': success,
            'partial_count': partial,
            'failed_count': failed,
            'latest_timestamp': ts,
            'latest_result': 'success' if success > 0 else 'failed',
        }

    def _pref(self, success=0, failure=0, partial=0, last_used=None):
        return {
            'success_count': success,
            'failure_count': failure,
            'partial_count': partial,
            'last_used': last_used,
        }

    def test_one_off_user_success_does_not_dominate(self):
        """A single user success should add < 1.0 points (evidence threshold)."""
        group = self._group(success=1)
        score_base = self._score(group)
        score_one = self._score(group, self._pref(success=1))
        diff = score_one - score_base
        self.assertLess(diff, 1.0, f"One-off user success added {diff} points — too much")
        self.assertGreaterEqual(diff, 0.0, "One-off success should not penalize")

    def test_repeated_user_success_boosts_ranking(self):
        """2+ user successes should provide a meaningful boost (>= 2.0 points)."""
        group = self._group(success=1)
        score_base = self._score(group)
        score_repeated = self._score(group, self._pref(success=3))
        diff = score_repeated - score_base
        self.assertGreaterEqual(diff, 2.0, f"3 user successes only added {diff} — too weak")

    def test_user_failures_reduce_preference_weight(self):
        """User failures should reduce the effective user bonus."""
        group = self._group(success=1)
        score_good = self._score(group, self._pref(success=3))
        score_bad = self._score(group, self._pref(success=3, failure=4))
        self.assertLess(score_bad, score_good,
                        "User failures did not reduce preference weight")

    def test_user_failures_dominating_prevents_positive_boost(self):
        """When failures >= successes, user preference must not boost the score."""
        group = self._group(success=1)
        score_base = self._score(group)
        score_with_bad_pref = self._score(group, self._pref(success=2, failure=2))
        self.assertLessEqual(score_with_bad_pref, score_base,
                             "Bad-habit fix still received a positive boost")

    def test_stale_preference_weaker_than_recent(self):
        """Preference last used >90 days ago should have less weight than recent."""
        from datetime import datetime, timedelta
        group = self._group(success=1)
        recent = (datetime.utcnow() - timedelta(days=5)).isoformat()
        stale = (datetime.utcnow() - timedelta(days=120)).isoformat()
        score_recent = self._score(group, self._pref(success=3, last_used=recent))
        score_stale = self._score(group, self._pref(success=3, last_used=stale))
        self.assertGreater(score_recent, score_stale,
                           "Stale preference was not weaker than recent")

    def test_mid_age_preference_between_recent_and_stale(self):
        """Preference 30-90 days old should score between recent and stale."""
        from datetime import datetime, timedelta
        group = self._group(success=1)
        recent = (datetime.utcnow() - timedelta(days=5)).isoformat()
        mid = (datetime.utcnow() - timedelta(days=60)).isoformat()
        stale = (datetime.utcnow() - timedelta(days=120)).isoformat()
        s_recent = self._score(group, self._pref(success=3, last_used=recent))
        s_mid = self._score(group, self._pref(success=3, last_used=mid))
        s_stale = self._score(group, self._pref(success=3, last_used=stale))
        self.assertGreater(s_recent, s_mid)
        self.assertGreater(s_mid, s_stale)

    def test_user_pref_does_not_rescue_bad_global(self):
        """User preference must not rescue a fix with negative global evidence."""
        bad_global = self._group(success=0, failed=5)
        score_no_pref = self._score(bad_global)
        score_with_pref = self._score(bad_global, self._pref(success=5))
        self.assertLess(score_no_pref, 0, "Precondition: global score should be negative")
        self.assertEqual(score_no_pref, score_with_pref,
                         "User preference rescued a globally-bad fix")

    def test_user_pref_cap_prevents_dominance(self):
        """Even with extreme user success, bonus should be capped."""
        group = self._group(success=1)
        score_base = self._score(group)
        score_extreme = self._score(group, self._pref(success=50))
        diff = score_extreme - score_base
        self.assertLessEqual(diff, 5.0, f"User preference bonus {diff} exceeded cap")

    def test_negative_user_bonus_always_applies(self):
        """Penalty from user failures should always apply (not gated by evidence)."""
        group = self._group(success=1)
        score_base = self._score(group)
        score_penalized = self._score(group, self._pref(success=0, failure=3))
        self.assertLess(score_penalized, score_base,
                        "Negative user bonus was not applied")

    def test_global_behavior_unchanged_without_user_pref(self):
        """When no user preference exists, scoring must match original formula."""
        group = self._group(success=2, partial=1, failed=1)
        score = self._score(group)
        expected = 2 * 10.0 + 1 * 5.0 + 1 * (-2.0)
        self.assertEqual(score, expected, "Global-only score does not match expected")


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
