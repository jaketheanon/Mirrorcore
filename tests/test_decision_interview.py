"""
Tests for Phase 26 + Phase 27: Decision Interview, Decision Memory,
Multi-Scenario Sessions, Reflection, Correction Loop, and Response Variation.
"""

import unittest
import sys
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Dict

from mirrorcore.db.store import DatabaseStore
from mirrorcore.decision.interview import (
    InterviewOption,
    InterviewScenario,
    InterviewQuestion,
    InterviewResult,
    ReflectionLine,
    CorrectionResult,
    SessionResult,
    SCENARIOS,
    get_scenario,
    get_scenarios_for_session,
    get_followup_question_for_main,
    extract_signals,
    compute_confidence,
    generate_reflections,
    run_interview,
    run_interview_session,
    _rotate_phrase,
    _CONFIRM_PHRASES,
    _TRANSITION_PHRASES,
    _ACCURATE_PHRASES,
    _PARTIAL_PHRASES,
    _FULL_CORRECTION_PHRASES,
)


# ===================================================================
# Phase 26 tests (preserved)
# ===================================================================

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
        self.assertEqual(row["correction_status"], "uncorrected")
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
        # Follow-up-only traits are softened toward a neutral baseline.
        # MAIN_WEIGHT=0.7, NEUTRAL=0.5 → 0.7*0.5 + 0.3*0.6 = 0.53
        self.assertAlmostEqual(signals["self_direction"], 0.53)

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

        # MAIN_WEIGHT=0.7, FOLLOW_WEIGHT=0.3
        self.assertAlmostEqual(signals["risk_tolerance"], 0.62)
        self.assertAlmostEqual(signals["thoroughness"], 0.46)

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

    def test_multiple_everyday_scenarios_exist(self):
        self.assertGreaterEqual(len(SCENARIOS), 5)

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
            main_ids = {o.id for o in scenario.main_question.options}
            self.assertEqual(
                set(scenario.followup_by_main_choice.keys()),
                main_ids,
                f"{scenario.id}: follow-up map must cover every main option",
            )
            for mid, fq in scenario.followup_by_main_choice.items():
                self.assertGreaterEqual(len(fq.options), 2, f"{scenario.id}/{mid}")

            for opt in scenario.main_question.options:
                self.assertTrue(opt.id)
                self.assertTrue(opt.label)
                self.assertIsInstance(opt.value_tags, list)
                self.assertIsInstance(opt.trait_signals, dict)

    def test_scenario_ids_are_unique(self):
        ids = [s.id for s in SCENARIOS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_followup_option_ids_are_unique_across_scenarios(self):
        ids = []
        for scenario in SCENARIOS:
            for _mid, fq in scenario.followup_by_main_choice.items():
                for opt in fq.options:
                    ids.append(opt.id)
        # Follow-up option IDs should not be reused across scenarios.
        # This is a lightweight guardrail against copying a generic reason pool.
        self.assertEqual(len(ids), len(set(ids)))

    def test_speed_vs_safety_followup_is_contextual(self):
        s = get_scenario("speed_vs_safety_v1")
        short = get_followup_question_for_main(s, "take_shortcut")
        labels = [o.label.lower() for o in short.options]
        self.assertTrue(any("move fast" in l for l in labels))
        right = get_followup_question_for_main(s, "do_it_right")
        labels_r = [o.label.lower() for o in right.options]
        self.assertTrue(any("proper" in l for l in labels_r))
        self.assertTrue(any("mess" in l for l in labels_r))
        self.assertFalse(any("save money" in l for l in labels + labels_r))

    def test_help_overwhelmed_say_no_reasons_match_boundary(self):
        s = get_scenario("help_when_overwhelmed_v1")
        fq = get_followup_question_for_main(s, "say_no")
        labels = " ".join(o.label.lower() for o in fq.options)
        self.assertIn("protect", labels)
        self.assertIn("overload", labels)
        self.assertIn("regret", labels)
        self.assertNotIn("small way", labels)


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


# ===================================================================
# Phase 27 tests
# ===================================================================

class TestGetScenariosForSession(unittest.TestCase):
    """Verify multi-scenario selection logic."""

    def test_returns_requested_count(self):
        scenarios = get_scenarios_for_session(4)
        self.assertEqual(len(scenarios), 4)

    def test_clamps_minimum_to_3(self):
        scenarios = get_scenarios_for_session(1)
        self.assertEqual(len(scenarios), 3)

    def test_clamps_maximum_to_5(self):
        scenarios = get_scenarios_for_session(100)
        self.assertEqual(len(scenarios), 5)

    def test_returns_interview_scenario_objects(self):
        for s in get_scenarios_for_session(3):
            self.assertIsInstance(s, InterviewScenario)

    def test_different_session_indices_produce_different_scenarios(self):
        a = get_scenarios_for_session(4, session_index=0)
        b = get_scenarios_for_session(4, session_index=1)
        a_ids = [s.id for s in a]
        b_ids = [s.id for s in b]
        self.assertNotEqual(a_ids, b_ids)

    def test_rotation_wraps_around(self):
        n = len(SCENARIOS)
        a = get_scenarios_for_session(4, session_index=0)
        # After enough rotations the cycle should restart
        wrap = get_scenarios_for_session(4, session_index=n)
        self.assertEqual([s.id for s in a], [s.id for s in wrap])

    def test_all_scenarios_covered_across_sessions(self):
        seen_ids: set = set()
        for idx in range(len(SCENARIOS)):
            for s in get_scenarios_for_session(3, session_index=idx):
                seen_ids.add(s.id)
        all_ids = {s.id for s in SCENARIOS}
        self.assertEqual(seen_ids, all_ids)


class TestSimpleChoiceExtraction(unittest.TestCase):
    """Verify that simple human-friendly choices map to deterministic signals."""

    def test_felt_right_maps_to_intuitive(self):
        opt = InterviewOption(
            id="felt_right", label="Felt right",
            value_tags=["intuition", "gut_feeling"],
            trait_signals={"analytical_thinking": 0.3, "intuitive_leaning": 0.8},
        )
        self.assertIn("intuition", opt.value_tags)
        self.assertGreater(opt.trait_signals["intuitive_leaning"], 0.5)

    def test_thought_it_through_maps_to_analytical(self):
        opt = InterviewOption(
            id="thought_it_through", label="Thought it through",
            value_tags=["deliberation", "planning"],
            trait_signals={"analytical_thinking": 0.8, "self_direction": 0.7},
        )
        self.assertIn("deliberation", opt.value_tags)
        self.assertGreater(opt.trait_signals["analytical_thinking"], 0.5)

    def test_every_scenario_option_has_signals(self):
        for scenario in SCENARIOS:
            for opt in scenario.main_question.options:
                self.assertTrue(len(opt.value_tags) > 0, f"{scenario.id} main option {opt.id} has no value_tags")
                self.assertTrue(len(opt.trait_signals) > 0, f"{scenario.id} main option {opt.id} has no trait_signals")
            for mid, fq in scenario.followup_by_main_choice.items():
                for opt in fq.options:
                    self.assertTrue(
                        len(opt.value_tags) > 0,
                        f"{scenario.id}/{mid} followup option {opt.id} has no value_tags",
                    )
                    self.assertTrue(
                        len(opt.trait_signals) > 0,
                        f"{scenario.id}/{mid} followup option {opt.id} has no trait_signals",
                    )


class TestReflectionGeneration(unittest.TestCase):
    """Verify mid-interview reflection generation logic."""

    def _make_result(self, trait_signals: Dict) -> InterviewResult:
        return InterviewResult(
            scenario_id="test", scenario_text="test",
            choice_label="A", choice_value="a",
            reasoning_label="B", reasoning_value="b",
            value_tags=[], trait_signals=trait_signals,
            confidence_score=0.8,
        )

    def test_generates_reflections_for_strong_traits(self):
        results = [
            self._make_result({"patience": 0.9, "risk_tolerance": 0.1}),
            self._make_result({"patience": 0.8, "risk_tolerance": 0.2}),
            self._make_result({"patience": 0.85}),
        ]
        reflections = generate_reflections(results)
        self.assertGreater(len(reflections), 0)
        traits = [r.trait for r in reflections]
        self.assertIn("patience", traits)

    def test_no_reflections_for_insufficient_evidence(self):
        results = [
            self._make_result({"patience": 0.9}),
        ]
        reflections = generate_reflections(results)
        patience_refs = [r for r in reflections if r.trait == "patience"]
        self.assertEqual(len(patience_refs), 0)

    def test_no_reflections_for_middling_traits(self):
        results = [
            self._make_result({"patience": 0.5}),
            self._make_result({"patience": 0.5}),
            self._make_result({"patience": 0.5}),
        ]
        reflections = generate_reflections(results)
        patience_refs = [r for r in reflections if r.trait == "patience"]
        self.assertEqual(len(patience_refs), 0)

    def test_low_trait_generates_low_reflection(self):
        results = [
            self._make_result({"patience": 0.2}),
            self._make_result({"patience": 0.1}),
            self._make_result({"patience": 0.15}),
        ]
        reflections = generate_reflections(results)
        patience_refs = [r for r in reflections if r.trait == "patience"]
        self.assertEqual(len(patience_refs), 1)
        self.assertIn("quickly", patience_refs[0].text.lower())

    def test_reflections_capped_at_5(self):
        signals = {f"trait_{i}": 0.9 for i in range(10)}
        results = [self._make_result(signals) for _ in range(3)]
        reflections = generate_reflections(results)
        self.assertLessEqual(len(reflections), 5)

    def test_reflection_line_has_trait_and_strength(self):
        results = [
            self._make_result({"patience": 0.9}),
            self._make_result({"patience": 0.85}),
        ]
        reflections = generate_reflections(results)
        for ref in reflections:
            self.assertIsInstance(ref.trait, str)
            self.assertIsInstance(ref.strength, float)
            self.assertTrue(ref.text)

    def test_low_reflection_for_risk_tolerance_is_specific(self):
        # Two results → enough evidence to surface a reflection.
        results = [
            self._make_result({"risk_tolerance": 0.1}),
            self._make_result({"risk_tolerance": 0.2}),
        ]
        reflections = generate_reflections(results)
        risk_refs = [r for r in reflections if r.trait == "risk_tolerance"]
        self.assertEqual(len(risk_refs), 1)
        self.assertIn("avoid unnecessary risk", risk_refs[0].text.lower())

    def test_intuitive_reflection_does_not_overpower_cautious_main(self):
        # Main choice is cautious (low risk); follow-up is "gut-ish".
        # Extract logic should soften follow-up-only gut signals.
        main = InterviewOption(
            id="do_it_right",
            label="Do it the proper way",
            value_tags=[],
            trait_signals={"risk_tolerance": 0.2, "thoroughness": 0.9},
        )
        followup = InterviewOption(
            id="wanted_it_now",
            label="Wanted it now",
            value_tags=[],
            trait_signals={"intuitive_leaning": 0.8, "analytical_thinking": 0.2},
        )
        _, trait_signals = extract_signals(main, followup)

        results = [
            self._make_result(trait_signals),
            self._make_result(trait_signals),
        ]
        reflections = generate_reflections(results)
        traits = [r.trait for r in reflections]
        self.assertNotIn("intuitive_leaning", traits)


class TestCorrectionLoop(unittest.TestCase):
    """Verify correction loop behavior."""

    def test_accurate_correction_accepts_all(self):
        reflections = [
            ReflectionLine(text="You think before acting", trait="patience", strength=0.85),
            ReflectionLine(text="You avoid risk", trait="risk_tolerance", strength=0.2),
        ]
        result = CorrectionResult(
            status="accurate",
            accepted_traits=["patience", "risk_tolerance"],
        )
        self.assertEqual(result.status, "accurate")
        self.assertEqual(len(result.accepted_traits), 2)
        self.assertEqual(len(result.rejected_traits), 0)

    def test_partial_correction_splits_traits(self):
        result = CorrectionResult(
            status="partially_true",
            accepted_traits=["patience"],
            rejected_traits=["risk_tolerance"],
        )
        self.assertEqual(result.status, "partially_true")
        self.assertIn("patience", result.accepted_traits)
        self.assertIn("risk_tolerance", result.rejected_traits)

    def test_full_correction_stores_replacements(self):
        result = CorrectionResult(
            status="not_really",
            rejected_traits=["patience", "risk_tolerance"],
            replacement_choices={"patience": "opposite", "risk_tolerance": "contextual"},
        )
        self.assertEqual(result.status, "not_really")
        self.assertEqual(result.replacement_choices["patience"], "opposite")
        self.assertEqual(result.replacement_choices["risk_tolerance"], "contextual")


class TestCorrectionMetadataDB(unittest.TestCase):
    """Verify correction metadata is stored and retrieved correctly."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_update_correction_status(self):
        entry_id = self.db.record_decision_memory(
            scenario_id="s", scenario_text="S",
            choice_label="C", choice_value="c",
            reasoning_label="R", reasoning_value="r",
            value_tags=[], trait_signals={},
        )

        self.db.update_decision_memory_correction(entry_id, "accurate")
        row = self.db.get_recent_decision_memory()[0]
        self.assertEqual(row["correction_status"], "accurate")

    def test_update_correction_metadata(self):
        entry_id = self.db.record_decision_memory(
            scenario_id="s", scenario_text="S",
            choice_label="C", choice_value="c",
            reasoning_label="R", reasoning_value="r",
            value_tags=[], trait_signals={},
        )

        metadata = {
            "status": "partially_true",
            "accepted_traits": ["patience"],
            "rejected_traits": ["risk_tolerance"],
        }
        self.db.update_decision_memory_correction(entry_id, "partially_true", metadata)
        row = self.db.get_recent_decision_memory()[0]
        self.assertEqual(row["correction_status"], "partially_true")
        stored_meta = row.get("correction_metadata") or {}
        if isinstance(stored_meta, str):
            stored_meta = json.loads(stored_meta or "{}")
        self.assertEqual(stored_meta["accepted_traits"], ["patience"])
        self.assertEqual(stored_meta["rejected_traits"], ["risk_tolerance"])


class TestResponseVariation(unittest.TestCase):
    """Verify deterministic response rotation."""

    def test_rotate_phrase_cycles(self):
        phrases = ["a", "b", "c"]
        self.assertEqual(_rotate_phrase(phrases, 0), "a")
        self.assertEqual(_rotate_phrase(phrases, 1), "b")
        self.assertEqual(_rotate_phrase(phrases, 2), "c")
        self.assertEqual(_rotate_phrase(phrases, 3), "a")

    def test_all_phrase_lists_non_empty(self):
        for phrases in [_CONFIRM_PHRASES, _TRANSITION_PHRASES,
                        _ACCURATE_PHRASES, _PARTIAL_PHRASES,
                        _FULL_CORRECTION_PHRASES]:
            self.assertGreater(len(phrases), 1)

    def test_no_duplicate_phrases_within_list(self):
        for phrases in [_CONFIRM_PHRASES, _TRANSITION_PHRASES,
                        _ACCURATE_PHRASES, _PARTIAL_PHRASES,
                        _FULL_CORRECTION_PHRASES]:
            self.assertEqual(len(phrases), len(set(phrases)))


class TestMultiScenarioInterviewSession(unittest.TestCase):
    """Simulate multi-scenario session with mocked input."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    @patch("builtins.input")
    def test_session_collects_multiple_results_then_accurate(self, mock_input):
        # 4 scenarios x 2 choices + correction choice (1 = accurate)
        inputs = ["1", "1"] * 4 + ["1"]
        mock_input.side_effect = inputs

        session = run_interview_session(4)
        self.assertIsInstance(session, SessionResult)
        self.assertEqual(len(session.results), 4)

        for r in session.results:
            self.assertIsInstance(r, InterviewResult)
            self.assertTrue(r.scenario_id)
            self.assertTrue(r.choice_label)
            self.assertTrue(r.reasoning_label)

    @patch("builtins.input")
    def test_session_stores_all_results_in_db(self, mock_input):
        inputs = ["1", "2"] * 3 + ["1"]
        mock_input.side_effect = inputs

        session = run_interview_session(3)

        for result in session.results:
            self.db.record_decision_memory(
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

        rows = self.db.get_recent_decision_memory(limit=10)
        self.assertEqual(len(rows), 3)

    @patch("builtins.input")
    def test_session_partial_correction(self, mock_input):
        # 4 scenarios x 2 choices + correction (2 = partially true) + right ones
        # Provide a valid "right ones" input even if fewer reflections appear.
        inputs = ["1", "1"] * 4 + ["2", "1"]
        mock_input.side_effect = inputs

        session = run_interview_session(4)
        self.assertIsNotNone(session.correction)
        self.assertEqual(session.correction.status, "partially_true")
        self.assertGreater(len(session.correction.accepted_traits), 0)

    @patch("builtins.input")
    def test_session_full_correction(self, mock_input):
        # 4 scenarios x 2 choices + correction (3 = not really) + replacement for each reflection
        inputs = ["1", "1"] * 4 + ["3"]
        reflections_expected = 5  # max reflections
        for _ in range(reflections_expected):
            inputs.append("a")
        mock_input.side_effect = inputs

        session = run_interview_session(4)
        self.assertIsNotNone(session.correction)
        self.assertEqual(session.correction.status, "not_really")


class TestPhase27SchemaCompatibility(unittest.TestCase):
    """Verify Phase 27 schema migration is backwards-compatible."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_correction_metadata_column_exists(self):
        conn = self.db.get_db_connection()
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(decision_memory)").fetchall()
        }
        self.assertIn("correction_metadata", columns)

    def test_old_entries_without_metadata_still_readable(self):
        entry_id = self.db.record_decision_memory(
            scenario_id="old_scenario",
            scenario_text="Old style",
            choice_label="A", choice_value="a",
            reasoning_label="R", reasoning_value="r",
            value_tags=["x"], trait_signals={"y": 0.5},
        )
        rows = self.db.get_recent_decision_memory()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["correction_status"], "uncorrected")


class TestDecisionInterviewRotationPersistence(unittest.TestCase):
    """Verify scenario rotation persists across process runs."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_rotation_counter_persists_and_changes_scenario_order(self):
        idx1 = self.db.get_next_decision_interview_rotation_index()
        idx2 = self.db.get_next_decision_interview_rotation_index()
        self.assertEqual(idx1, 0)
        self.assertEqual(idx2, 1)

        first = [s.id for s in get_scenarios_for_session(4, session_index=idx1)]
        second = [s.id for s in get_scenarios_for_session(4, session_index=idx2)]
        self.assertNotEqual(first, second)

        # Simulate a new process run by reopening the same DB file.
        self.db.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        idx3 = self.db.get_next_decision_interview_rotation_index()
        self.assertEqual(idx3, 2)


if __name__ == "__main__":
    unittest.main()
