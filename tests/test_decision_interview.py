"""
Tests for Phase 26: Decision Interview and Decision Memory

Covers:
- Decision memory DB write and read
- Extraction mapping correctness (deterministic signal merging)
- Interview flow storage behaviour
"""

import unittest
import sys
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore
from mirrorcore.decision.interview import (
    InterviewOption,
    InterviewScenario,
    InterviewQuestion,
    InterviewResult,
    SCENARIOS,
    get_scenario,
    extract_signals,
    compute_confidence,
    run_interview,
)


class TestDecisionMemoryDB(unittest.TestCase):
    """Verify record_decision_memory and get_recent_decision_memory."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_record_and_retrieve(self):
        entry_id = self.db.record_decision_memory(
            scenario_id="test_scenario",
            scenario_text="A test scenario",
            choice_label="Option A",
            choice_value="opt_a",
            reasoning_label="Because quality",
            reasoning_value="reason_quality",
            value_tags=["quality", "caution"],
            trait_signals={"risk_tolerance": 0.3, "thoroughness": 0.8},
            confidence_score=0.75,
        )

        self.assertIsNotNone(entry_id)

        rows = self.db.get_recent_decision_memory(limit=10)
        self.assertEqual(len(rows), 1)

        row = rows[0]
        self.assertEqual(row["id"], entry_id)
        self.assertEqual(row["scenario_id"], "test_scenario")
        self.assertEqual(row["choice_value"], "opt_a")
        self.assertEqual(row["reasoning_value"], "reason_quality")
        self.assertEqual(sorted(row["value_tags"]), ["caution", "quality"])
        self.assertAlmostEqual(row["trait_signals"]["risk_tolerance"], 0.3)
        self.assertAlmostEqual(row["confidence_score"], 0.75)
        self.assertEqual(row["correction_status"], "accepted")
        self.assertEqual(row["source"], "interview")

    def test_filter_by_scenario_id(self):
        self.db.record_decision_memory(
            scenario_id="s1", scenario_text="S1",
            choice_label="A", choice_value="a",
            reasoning_label="R", reasoning_value="r",
            value_tags=[], trait_signals={},
        )
        self.db.record_decision_memory(
            scenario_id="s2", scenario_text="S2",
            choice_label="B", choice_value="b",
            reasoning_label="R", reasoning_value="r",
            value_tags=[], trait_signals={},
        )

        s1_rows = self.db.get_recent_decision_memory(scenario_id="s1")
        self.assertEqual(len(s1_rows), 1)
        self.assertEqual(s1_rows[0]["scenario_id"], "s1")

    def test_multiple_entries_ordered_by_timestamp(self):
        for i in range(5):
            self.db.record_decision_memory(
                scenario_id=f"s{i}", scenario_text=f"Scenario {i}",
                choice_label=f"C{i}", choice_value=f"c{i}",
                reasoning_label="R", reasoning_value="r",
                value_tags=[], trait_signals={},
            )

        rows = self.db.get_recent_decision_memory(limit=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["scenario_id"], "s4")

    def test_optional_notes(self):
        entry_id = self.db.record_decision_memory(
            scenario_id="s", scenario_text="S",
            choice_label="C", choice_value="c",
            reasoning_label="R", reasoning_value="r",
            value_tags=[], trait_signals={},
            optional_notes="User added a note",
        )
        row = self.db.get_recent_decision_memory()[0]
        self.assertEqual(row["optional_notes"], "User added a note")


class TestExtractionMapping(unittest.TestCase):
    """Verify deterministic signal extraction and confidence computation."""

    def test_extract_signals_merges_tags(self):
        main = InterviewOption(
            id="a", label="A",
            value_tags=["speed", "pragmatism"],
            trait_signals={"risk_tolerance": 0.7},
        )
        followup = InterviewOption(
            id="b", label="B",
            value_tags=["pragmatism", "efficiency"],
            trait_signals={"self_direction": 0.6},
        )

        tags, signals = extract_signals(main, followup)

        self.assertEqual(tags, ["efficiency", "pragmatism", "speed"])
        self.assertAlmostEqual(signals["risk_tolerance"], 0.7)
        self.assertAlmostEqual(signals["self_direction"], 0.6)

    def test_extract_signals_averages_shared_traits(self):
        main = InterviewOption(
            id="a", label="A",
            value_tags=[],
            trait_signals={"risk_tolerance": 0.8, "thoroughness": 0.4},
        )
        followup = InterviewOption(
            id="b", label="B",
            value_tags=[],
            trait_signals={"risk_tolerance": 0.2, "thoroughness": 0.6},
        )

        _, signals = extract_signals(main, followup)

        self.assertAlmostEqual(signals["risk_tolerance"], 0.5)
        self.assertAlmostEqual(signals["thoroughness"], 0.5)

    def test_compute_confidence_full_overlap(self):
        a = InterviewOption(id="a", label="A", value_tags=["quality", "caution"])
        b = InterviewOption(id="b", label="B", value_tags=["quality", "caution"])
        self.assertAlmostEqual(compute_confidence(a, b), 1.0)

    def test_compute_confidence_no_overlap(self):
        a = InterviewOption(id="a", label="A", value_tags=["speed"])
        b = InterviewOption(id="b", label="B", value_tags=["quality"])
        self.assertAlmostEqual(compute_confidence(a, b), 0.6)

    def test_compute_confidence_partial_overlap(self):
        a = InterviewOption(id="a", label="A", value_tags=["speed", "pragmatism"])
        b = InterviewOption(id="b", label="B", value_tags=["pragmatism", "efficiency"])
        conf = compute_confidence(a, b)
        self.assertGreater(conf, 0.6)
        self.assertLess(conf, 1.0)

    def test_compute_confidence_empty_tags(self):
        a = InterviewOption(id="a", label="A", value_tags=[])
        b = InterviewOption(id="b", label="B", value_tags=[])
        self.assertAlmostEqual(compute_confidence(a, b), 0.5)


class TestScenarioDefinitions(unittest.TestCase):
    """Sanity-check scenario data integrity."""

    def test_at_least_one_scenario_exists(self):
        self.assertGreater(len(SCENARIOS), 0)

    def test_get_scenario_by_id(self):
        s = get_scenario(SCENARIOS[0].id)
        self.assertEqual(s.id, SCENARIOS[0].id)

    def test_get_scenario_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_scenario("nonexistent_scenario")

    def test_scenario_has_required_structure(self):
        for scenario in SCENARIOS:
            self.assertTrue(scenario.id)
            self.assertTrue(scenario.text)
            self.assertGreaterEqual(len(scenario.main_question.options), 2)
            self.assertGreaterEqual(len(scenario.followup_question.options), 2)

            for opt in scenario.main_question.options:
                self.assertTrue(opt.id)
                self.assertTrue(opt.label)
                self.assertIsInstance(opt.value_tags, list)
                self.assertIsInstance(opt.trait_signals, dict)


class TestInterviewFlowStorage(unittest.TestCase):
    """Simulate a full interview flow and verify storage."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    @patch("builtins.input", side_effect=["1", "2"])
    def test_run_interview_stores_result(self, mock_input):
        result = run_interview()
        self.assertIsInstance(result, InterviewResult)

        entry_id = self.db.record_decision_memory(
            scenario_id=result.scenario_id,
            scenario_text=result.scenario_text,
            choice_label=result.choice_label,
            choice_value=result.choice_value,
            reasoning_label=result.reasoning_label,
            reasoning_value=result.reasoning_value,
            value_tags=result.value_tags,
            trait_signals=result.trait_signals,
            confidence_score=result.confidence_score,
        )

        rows = self.db.get_recent_decision_memory()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scenario_id"], result.scenario_id)
        self.assertEqual(rows[0]["choice_value"], result.choice_value)
        self.assertEqual(rows[0]["reasoning_value"], result.reasoning_value)

    @patch("builtins.input", side_effect=["3", "3"])
    def test_all_option_combos_produce_valid_signals(self, mock_input):
        result = run_interview()
        self.assertIsInstance(result.value_tags, list)
        self.assertIsInstance(result.trait_signals, dict)
        self.assertGreater(len(result.trait_signals), 0)
        self.assertGreater(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)


if __name__ == "__main__":
    unittest.main()
