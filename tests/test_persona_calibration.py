"""
Tests for Phase 28: Style / Persona Calibration Foundation.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore
from mirrorcore.persona.calibration import (
    STYLE_PROMPTS,
    StyleOption,
    StyleCalibrationResult,
    StyleReflectionLine,
    get_prompts_for_session,
    compute_style_confidence,
    generate_style_reflections,
    run_style_calibration_session,
    _rotate_phrase,
    _TRANSITIONS,
    _CONFIRMS,
)


class TestStyleMemoryDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_record_and_read_style_memory(self):
        entry_id = self.db.record_style_memory(
            prompt_id="blunt_vs_gentle_v1",
            prompt_text="Which reply sounds more like you?",
            selected_label="Straight to the point",
            selected_value="straight_callout",
            style_tags=["blunt", "direct"],
            tone_signals={"bluntness": 0.9, "diplomacy": 0.2},
            confidence_score=0.81,
            correction_status="accurate",
            correction_metadata={"status": "accurate", "accepted_traits": ["bluntness"]},
        )
        rows = self.db.get_recent_style_memory(limit=5)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["id"], entry_id)
        self.assertEqual(row["prompt_id"], "blunt_vs_gentle_v1")
        self.assertEqual(row["selected_value"], "straight_callout")
        self.assertEqual(sorted(row["style_tags"]), ["blunt", "direct"])
        self.assertAlmostEqual(row["tone_signals"]["bluntness"], 0.9)
        self.assertEqual(row["correction_status"], "accurate")
        self.assertEqual(row["correction_metadata"]["status"], "accurate")

    def test_filter_by_prompt_id(self):
        self.db.record_style_memory(
            prompt_id="a",
            prompt_text="A",
            selected_label="A",
            selected_value="a",
            style_tags=[],
            tone_signals={},
        )
        self.db.record_style_memory(
            prompt_id="b",
            prompt_text="B",
            selected_label="B",
            selected_value="b",
            style_tags=[],
            tone_signals={},
        )
        rows = self.db.get_recent_style_memory(prompt_id="a")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt_id"], "a")


class TestDeterministicMapping(unittest.TestCase):
    def test_prompt_options_include_style_tags_and_tone_signals(self):
        for prompt in STYLE_PROMPTS:
            self.assertTrue(prompt.id)
            self.assertTrue(prompt.text)
            self.assertGreaterEqual(len(prompt.options), 2)
            for opt in prompt.options:
                self.assertTrue(opt.label)
                self.assertTrue(len(opt.style_tags) > 0)
                self.assertTrue(len(opt.tone_signals) > 0)

    def test_compute_style_confidence_deterministic(self):
        option = StyleOption(
            id="x",
            label="X",
            style_tags=["direct"],
            tone_signals={"bluntness": 0.8, "diplomacy": 0.2},
        )
        a = compute_style_confidence(option)
        b = compute_style_confidence(option)
        self.assertEqual(a, b)
        self.assertGreaterEqual(a, 0.55)

    def test_get_prompts_for_session_rotates(self):
        a = [p.id for p in get_prompts_for_session(prompt_count=4, session_index=0)]
        b = [p.id for p in get_prompts_for_session(prompt_count=4, session_index=1)]
        self.assertNotEqual(a, b)


class TestReflectionGeneration(unittest.TestCase):
    def _result(self, signals):
        return StyleCalibrationResult(
            prompt_id="p",
            prompt_text="t",
            selected_label="l",
            selected_value="v",
            style_tags=[],
            tone_signals=signals,
            confidence_score=0.8,
        )

    def test_reflection_requires_evidence(self):
        reflections = generate_style_reflections([self._result({"bluntness": 0.9})])
        self.assertEqual(reflections, [])

    def test_reflection_generates_for_strong_trait(self):
        reflections = generate_style_reflections(
            [
                self._result({"bluntness": 0.9}),
                self._result({"bluntness": 0.8}),
                self._result({"bluntness": 0.85}),
            ]
        )
        self.assertTrue(any(r.trait == "bluntness" for r in reflections))

    def test_reflection_cap(self):
        signals = {
            "bluntness": 0.9,
            "verbosity": 0.1,
            "diplomacy": 0.9,
            "sarcasm": 0.9,
            "warmth": 0.9,
            "seriousness": 0.9,
            "casualness": 0.9,
        }
        reflections = generate_style_reflections([self._result(signals), self._result(signals)])
        self.assertLessEqual(len(reflections), 5)


class TestStyleCalibrationFlow(unittest.TestCase):
    @patch("builtins.input")
    def test_session_runs_and_returns_results(self, mock_input):
        mock_input.side_effect = ["1"] * 4 + ["1"]
        session = run_style_calibration_session(prompt_count=4, session_index=0)
        self.assertEqual(len(session.results), 4)
        self.assertIsNotNone(session.correction)
        self.assertEqual(session.correction.status, "accurate")

    @patch("builtins.input")
    def test_partial_correction_flow(self, mock_input):
        # 4 prompts + correction(2) + right ones + replacements for wrong traits
        mock_input.side_effect = ["1"] * 4 + ["2", "1", "b", "b", "b", "b"]
        session = run_style_calibration_session(prompt_count=4, session_index=0)
        self.assertIsNotNone(session.correction)
        self.assertEqual(session.correction.status, "partially_true")
        self.assertGreaterEqual(len(session.correction.replacement_choices), 1)

    @patch("builtins.input")
    def test_not_really_correction_flow(self, mock_input):
        # 4 prompts + correction(3) + 5 replacement picks (max reflections)
        mock_input.side_effect = ["1"] * 4 + ["3", "a", "a", "a", "a", "a"]
        session = run_style_calibration_session(prompt_count=4, session_index=0)
        self.assertIsNotNone(session.correction)
        self.assertEqual(session.correction.status, "not_really")
        self.assertGreaterEqual(len(session.correction.replacement_choices), 1)


class TestVariationHelpers(unittest.TestCase):
    def test_rotate_phrase(self):
        self.assertEqual(_rotate_phrase(["a", "b"], 0), "a")
        self.assertEqual(_rotate_phrase(["a", "b"], 1), "b")
        self.assertEqual(_rotate_phrase(["a", "b"], 2), "a")

    def test_phrase_lists_non_empty(self):
        self.assertGreater(len(_TRANSITIONS), 1)
        self.assertGreater(len(_CONFIRMS), 1)


if __name__ == "__main__":
    unittest.main()

