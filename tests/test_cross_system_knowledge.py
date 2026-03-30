"""Phase 36: cross-system knowledge helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.decision.cross_system_knowledge import (
    build_ask_interview_memory_line_candidates,
    clarification_cross_evidence_boost,
    effective_primary_for_cross_filter,
    filter_relevant_situation_facts,
    situation_fact_family_match,
)
from mirrorcore.decision.ontology import CONFLICT_FAMILY, GENERAL, SPENDING
from mirrorcore.router import normalize_input


class TestSituationFactFilter(unittest.TestCase):
    def test_housing_fact_does_not_attach_to_conflict_primary(self):
        facts = [{"slot_key": "housing_bill_pressure", "slot_value": "open"}]
        out = filter_relevant_situation_facts(
            facts,
            primary_family=CONFLICT_FAMILY,
            prompt_norm=normalize_input("my coworker was rude"),
        )
        self.assertEqual(out, [])

    def test_housing_fact_kept_for_spending_prompt(self):
        facts = [{"slot_key": "housing_bill_pressure", "slot_value": "open"}]
        out = filter_relevant_situation_facts(
            facts,
            primary_family=SPENDING,
            prompt_norm=normalize_input("should i buy a laptop rent is late"),
        )
        self.assertEqual(len(out), 1)

    def test_situation_key_family_map(self):
        self.assertTrue(situation_fact_family_match("housing_bill_pressure", SPENDING))
        self.assertFalse(situation_fact_family_match("housing_bill_pressure", CONFLICT_FAMILY))

    def test_effective_primary_upgrades_general_when_money_strong(self):
        self.assertEqual(
            effective_primary_for_cross_filter(GENERAL, normalize_input("rent late broke buy")),
            SPENDING,
        )

    def test_phase40_conflict_stance_fact_kept_for_boundary_prompt(self):
        facts = [{"slot_key": "conflict_stance", "slot_value": "boundary"}]
        out = filter_relevant_situation_facts(
            facts,
            primary_family=CONFLICT_FAMILY,
            prompt_norm=normalize_input("coworker keeps pushing my boundaries what do i do"),
        )
        self.assertEqual(len(out), 1)

    def test_phase40_conflict_stance_dropped_without_conflict_cues(self):
        facts = [{"slot_key": "conflict_stance", "slot_value": "speak"}]
        out = filter_relevant_situation_facts(
            facts,
            primary_family=CONFLICT_FAMILY,
            prompt_norm=normalize_input("should i learn python this weekend"),
        )
        self.assertEqual(out, [])


class TestClarificationCrossBoost(unittest.TestCase):
    def test_boost_from_aligned_facts_and_tendency(self):
        facts = [{"slot_key": "energy_available", "slot_value": "low"}]
        tmap = {"tendency_guilt_about_no": 0.55}
        b = clarification_cross_evidence_boost(
            normalize_input("exhausted cover another shift guilt"),
            "obligation_overload",
            facts,
            tmap,
        )
        self.assertGreater(b, 0.05)

    def test_phase40_conflict_tendency_boosts(self):
        tmap = {
            "tendency_speak_up_conflict": 0.5,
            "tendency_hard_boundary": 0.48,
            "tendency_pull_back_contact": 0.47,
        }
        b = clarification_cross_evidence_boost(
            normalize_input("rude disrespect say something"),
            CONFLICT_FAMILY,
            [],
            tmap,
        )
        self.assertGreater(b, 0.08)


class TestAskInterviewCandidates(unittest.TestCase):
    def test_strong_interview_row_produces_candidate(self):
        rows = [
            {
                "id": "e1",
                "scenario_id": "expensive_purchase_v1",
                "scenario_text": "thinking about buying something expensive you want but dont need",
                "choice_label": "Wait a few days",
                "choice_value": "w",
                "reasoning_label": "Avoid impulse",
                "reasoning_value": "x",
                "value_tags": ["caution", "deliberation"],
                "trait_signals": {"patience": 0.9, "risk_tolerance": 0.3},
                "correction_status": "accurate",
                "confidence_score": 0.9,
            }
        ]
        merged = normalize_input(
            "should i buy an expensive gadget i want but dont need right now"
        )
        initial = normalize_input("should i buy an expensive gadget")
        cands = build_ask_interview_memory_line_candidates(
            rows,
            merged_norm=merged,
            initial_norm=initial,
            primary_family=SPENDING,
            seed="seedseedseed",
        )
        self.assertTrue(cands)
        self.assertIn("Wait", cands[0][2])

    def test_not_really_row_skipped(self):
        rows = [
            {
                "id": "bad",
                "scenario_id": "x",
                "scenario_text": "ice cream",
                "choice_label": "Chocolate",
                "reasoning_label": "Mood",
                "reasoning_value": "m",
                "value_tags": [],
                "trait_signals": {},
                "correction_status": "not_really",
                "confidence_score": 0.9,
            }
        ]
        cands = build_ask_interview_memory_line_candidates(
            rows,
            merged_norm=normalize_input("ice cream pick"),
            initial_norm=normalize_input("ice cream"),
            primary_family=GENERAL,
            seed="s2",
        )
        self.assertEqual(cands, [])


if __name__ == "__main__":
    unittest.main()
