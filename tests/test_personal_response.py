"""
Tests for Phase 29: unified personal profile aggregation, memory retrieval,
grounded response generation, and confidence behavior.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore
from mirrorcore.main import create_parser, handle_respond_like_me
from types import SimpleNamespace
from mirrorcore.persona.profile import build_personal_profile_from_rows
from mirrorcore.persona.respond import (
    generate_personal_response,
    retrieve_relevant_decision_memories,
    retrieve_relevant_style_memories,
    tokenize_prompt,
)


class TestPersonalProfileAggregation(unittest.TestCase):
    def test_trait_surfaces_with_enough_evidence_and_downweights_not_really(self):
        strong = {
            "id": "a",
            "timestamp": "2025-01-01",
            "scenario_id": "s1",
            "scenario_text": "x",
            "choice_label": "A",
            "choice_value": "a",
            "reasoning_label": "R",
            "reasoning_value": "r",
            "value_tags": ["caution"],
            "trait_signals": {"risk_tolerance": 0.2},
            "confidence_score": 0.9,
            "correction_status": "accurate",
            "correction_metadata": {},
            "source": "interview",
        }
        weak = dict(strong)
        weak["id"] = "b"
        weak["trait_signals"] = {"risk_tolerance": 0.95}
        weak["correction_status"] = "not_really"
        weak["value_tags"] = ["impulsiveness"]

        profile = build_personal_profile_from_rows([strong, weak], [])
        names = [t.name for t in profile.trait_estimates if t.name == "risk_tolerance"]
        self.assertTrue(names)
        risk = next(t for t in profile.trait_estimates if t.name == "risk_tolerance")
        self.assertLess(risk.weighted_mean, 0.5)

    def test_partially_true_rejected_trait_dampens_signal(self):
        base = {
            "timestamp": "2025-01-01",
            "scenario_id": "s1",
            "scenario_text": "x",
            "choice_label": "A",
            "choice_value": "a",
            "reasoning_label": "R",
            "reasoning_value": "r",
            "value_tags": [],
            "trait_signals": {"patience": 0.9},
            "confidence_score": 0.85,
            "correction_status": "partially_true",
            "correction_metadata": {"rejected_traits": ["patience"]},
            "source": "interview",
        }
        dampened_rows = [
            dict(base, id="a"),
            dict(base, id="b"),
            dict(base, id="e"),
        ]
        profile_damp = build_personal_profile_from_rows(dampened_rows, [])
        baseline_rows = [
            dict(base, id="c", correction_status="accurate", correction_metadata={}),
            dict(base, id="d", correction_status="accurate", correction_metadata={}),
        ]
        for r in baseline_rows:
            r["trait_signals"] = {"patience": 0.9}
        profile_base = build_personal_profile_from_rows(baseline_rows, [])
        pd = next(t for t in profile_damp.trait_estimates if t.name == "patience")
        pb = next(t for t in profile_base.trait_estimates if t.name == "patience")
        self.assertLess(pd.weighted_mean, pb.weighted_mean)


class TestRetrieval(unittest.TestCase):
    def test_deterministic_ranking_tiebreak(self):
        rows = [
            {
                "id": "z",
                "timestamp": "2025-01-02",
                "scenario_id": "expensive_purchase_v1",
                "scenario_text": "buying something expensive you want",
                "choice_label": "Wait",
                "choice_value": "w",
                "reasoning_label": "Think",
                "reasoning_value": "t",
                "value_tags": ["caution"],
                "trait_signals": {"risk_tolerance": 0.2},
                "correction_status": "uncorrected",
                "confidence_score": 0.8,
            },
            {
                "id": "a",
                "timestamp": "2025-01-02",
                "scenario_id": "expensive_purchase_v1",
                "scenario_text": "buying something expensive you want",
                "choice_label": "Buy",
                "choice_value": "b",
                "reasoning_label": "Feel",
                "reasoning_value": "f",
                "value_tags": ["caution"],
                "trait_signals": {"risk_tolerance": 0.8},
                "correction_status": "uncorrected",
                "confidence_score": 0.8,
            },
        ]
        a = retrieve_relevant_decision_memories(rows, "expensive purchase want", top_k=2)
        b = retrieve_relevant_decision_memories(rows, "expensive purchase want", top_k=2)
        self.assertEqual([x[0]["id"] for x in a], [x[0]["id"] for x in b])

    def test_accurate_boosts_rank(self):
        unc = {
            "id": "u",
            "timestamp": "2025-01-03",
            "scenario_id": "s",
            "scenario_text": "coffee shop line is long",
            "choice_label": "Stay",
            "choice_value": "s",
            "reasoning_label": "Patience",
            "reasoning_value": "p",
            "value_tags": ["patience"],
            "trait_signals": {},
            "correction_status": "uncorrected",
            "confidence_score": 0.8,
        }
        acc = dict(unc)
        acc["id"] = "c"
        acc["correction_status"] = "accurate"
        ranked = retrieve_relevant_decision_memories([unc, acc], "long line coffee shop", top_k=1)
        self.assertEqual(ranked[0][0]["id"], "c")


class TestRespondGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_low_evidence_fallback(self):
        pr = generate_personal_response("should I quit my job tomorrow", self.db)
        low = pr.likely_answer.lower()
        self.assertTrue(
            "don’t have enough" in low
            or "don't have enough" in low
            or "little or no relevant" in pr.reasoning_brief.lower()
        )
        self.assertLessEqual(pr.confidence, 0.35)

    def test_uses_decision_memory_when_keywords_align(self):
        self.db.record_decision_memory(
            scenario_id="side_project_v1",
            scenario_text="You have a side project but limited free time.",
            choice_label="Cut scope",
            choice_value="cut",
            reasoning_label="Ship something small",
            reasoning_value="ship_small",
            value_tags=["pragmatism"],
            trait_signals={"risk_tolerance": 0.35},
            confidence_score=0.82,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "side project limited free time what would I do", self.db
        )
        self.assertIn("choose", pr.likely_answer.lower())
        self.assertIn("Cut scope", pr.likely_answer)
        self.assertGreater(pr.confidence, 0.35)
        self.assertTrue(pr.memory_basis)

    def test_not_really_blocks_high_confidence_match(self):
        eid = self.db.record_decision_memory(
            scenario_id="match_me",
            scenario_text="ice cream flavor-picking scenario",
            choice_label="Pick chocolate",
            choice_value="choc",
            reasoning_label="Mood",
            reasoning_value="m",
            value_tags=[],
            trait_signals={},
            confidence_score=0.9,
            correction_status="not_really",
        )
        self.assertIsNotNone(eid)
        pr = generate_personal_response("ice cream flavor picking", self.db)
        self.assertNotIn("chocolate", pr.likely_answer.lower())


class TestStyleRetrieval(unittest.TestCase):
    def test_style_keyword_overlap(self):
        rows = [
            {
                "id": "1",
                "timestamp": "t",
                "prompt_id": "short_vs_detailed_v1",
                "prompt_text": "When you explain something",
                "selected_label": "Short",
                "selected_value": "short",
                "style_tags": ["concise"],
                "tone_signals": {"verbosity": 0.2},
                "correction_status": "accurate",
                "confidence_score": 0.9,
            }
        ]
        r = retrieve_relevant_style_memories(
            rows, "I need a short explanation", top_k=1
        )
        self.assertGreater(r[0][1], 0.5)


class TestTokenize(unittest.TestCase):
    def test_tokenize_dedupes(self):
        self.assertEqual(
            tokenize_prompt("buy buy expensive purchase"),
            ["buy", "expensive", "purchase"],
        )


class TestRespondLikeMeCLI(unittest.TestCase):
    def test_subcommand_parses(self):
        p = create_parser()
        args = p.parse_args(["respond-like-me", "should I upgrade today"])
        self.assertEqual(args.command, "respond-like-me")
        self.assertEqual(args.scenario, "should I upgrade today")

    @patch("mirrorcore.persona.respond.generate_personal_response")
    @patch("mirrorcore.db.store.DatabaseStore")
    def test_handler_invokes_generator_with_scenario(self, mock_db_class, mock_gen):
        mock_gen.return_value = SimpleNamespace(
            likely_answer="Out",
            reasoning_brief="Because",
            confidence=0.41,
            confidence_label="fair",
            memory_basis=["Decision memory (x)"],
            profile_hint=None,
        )
        args = SimpleNamespace(scenario="  pick a paint color  ")
        handle_respond_like_me(args)
        mock_gen.assert_called_once()
        called_store = mock_gen.call_args[0][1]
        self.assertIs(called_store, mock_db_class.return_value)
        self.assertEqual(mock_gen.call_args[0][0], "pick a paint color")


if __name__ == "__main__":
    unittest.main()
