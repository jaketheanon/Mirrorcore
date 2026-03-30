"""
Phase 33: decision ontology axes, adaptive slot coverage, and regression corpus.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.decision.ontology import (
    AXIS_BACKCHANNEL_HURT,
    AXIS_BILLS_FINANCIAL_PRESSURE,
    AXIS_CONFLICT_CONFRONTATION,
    AXIS_OVERLOAD_BURNOUT,
    AXIS_TIMING_WAIT_VS_ACT,
    AXIS_WORK_OBLIGATION,
    rank_families_full,
    score_ontology_axes,
    slot_ids_covered_by_context,
)
from mirrorcore.decision.routed_clarification import (
    CONFLICT_FAMILY,
    OBLIGATION_OVERLOAD,
    RISK_TIMING,
    SPENDING,
    pick_next_question,
    rank_families,
)
from mirrorcore.router import normalize_input


class TestOntologyAxes(unittest.TestCase):
    def test_multi_label_coworker_burnt_out(self):
        t = normalize_input(
            "a coworker needs me to cover a shift but im burnt out what do i do?"
        )
        ax = score_ontology_axes(t)
        self.assertGreater(ax.get(AXIS_WORK_OBLIGATION, 0), 0)
        self.assertGreater(ax.get(AXIS_OVERLOAD_BURNOUT, 0), 0)

    def test_rent_late_bills_axis(self):
        t = normalize_input("Should I buy an $800 laptop if my rent is late?")
        ax = score_ontology_axes(t)
        self.assertGreater(ax.get(AXIS_BILLS_FINANCIAL_PRESSURE, 0), 0)

    def test_gossip_backchannel_axis(self):
        t = normalize_input("someone keeps talking shit behind my back. what do i do?")
        ax = score_ontology_axes(t)
        self.assertGreater(ax.get(AXIS_BACKCHANNEL_HURT, 0), 0)

    def test_phase40_passive_aggressive_hits_conflict_axis(self):
        t = normalize_input("my teammate is passive aggressive and snide in meetings")
        ax = score_ontology_axes(t)
        self.assertGreater(ax.get(AXIS_CONFLICT_CONFRONTATION, 0), 0)

    def test_wait_or_act_axis(self):
        t = normalize_input("I don't know whether to wait or act now.")
        ax = score_ontology_axes(t)
        self.assertGreater(ax.get(AXIS_TIMING_WAIT_VS_ACT, 0), 0)


class TestRegressionCorpus(unittest.TestCase):
    """Minimum acceptance prompts from Phase 33 spec."""

    def _first_slot(self, text: str):
        norm = normalize_input(text)
        ordered, dims = rank_families(norm)
        fam_order = [f for f, sc in ordered if sc >= 0.4][:8]
        if "general" not in fam_order:
            fam_order.append("general")
        q = pick_next_question(
            context_parts=[text],
            asked_ids=[],
            domain_order=fam_order,
            dimensions=dims,
        )
        return q, ordered[0][0], fam_order

    def test_laptop_rent_late(self):
        q, primary, _ = self._first_slot("Should I buy an $800 laptop if my rent is late?")
        self.assertEqual(primary, SPENDING)
        self.assertIsNotNone(q)
        self.assertNotEqual(q.slot_id, "peace_vs_clarity")

    def test_coworker_cover_burnt_out(self):
        q, primary, _ = self._first_slot(
            "a coworker needs me to cover a shift but im burnt out what do i do?"
        )
        self.assertEqual(primary, OBLIGATION_OVERLOAD)
        self.assertIsNotNone(q)
        self.assertNotEqual(q.slot_id, "peace_vs_clarity")

    def test_talking_shit_behind_back(self):
        q, primary, _ = self._first_slot(
            "someone keeps talking shit behind my back. what do i do?"
        )
        self.assertEqual(primary, CONFLICT_FAMILY)
        self.assertIsNotNone(q)
        self.assertIn(q.slot_id, ("peace_vs_clarity", "pattern_vs_once", "stakes_real", "timing_conflict"))

    def test_wait_or_act_now(self):
        q, primary, _ = self._first_slot("I don't know whether to wait or act now.")
        self.assertEqual(primary, RISK_TIMING)
        self.assertIsNotNone(q)
        self.assertIn(q.slot_id, ("real_deadline", "reversibility", "worst_hurt"))

    def test_mom_favor_drained(self):
        q, primary, _ = self._first_slot(
            "my mom wants a favor but im already drained. what should i do?"
        )
        self.assertIn(primary, (OBLIGATION_OVERLOAD, SPENDING))
        self.assertIsNotNone(q)
        self.assertNotEqual(q.slot_id, "peace_vs_clarity")

    def test_need_vs_want_plain(self):
        q, primary, _ = self._first_slot("i want this but i don't really need it")
        self.assertIsNotNone(q)
        self.assertNotEqual(q.slot_id, "peace_vs_clarity")

    def test_let_go_or_say_something(self):
        ordered, _, _ = rank_families_full(
            normalize_input("should i let this go or say something")
        )
        self.assertEqual(ordered[0][0], CONFLICT_FAMILY)

    def test_risky_waiting_cost(self):
        ordered, _, _ = rank_families_full(
            normalize_input("this feels risky but waiting might cost me too")
        )
        self.assertEqual(ordered[0][0], RISK_TIMING)

    def test_unrelated_memory_suppressed_not_obvious_misroute(self):
        """Gossip conflict should not rank obligation first."""
        ordered, _, _ = rank_families_full(
            normalize_input("people keep gossiping behind my back at work")
        )
        self.assertEqual(ordered[0][0], CONFLICT_FAMILY)


class TestSlotCoverage(unittest.TestCase):
    def test_rent_late_skips_bills_question(self):
        covered = slot_ids_covered_by_context(
            normalize_input("Should I buy this if rent is late?")
        )
        self.assertIn("bills_basics", covered)


if __name__ == "__main__":
    unittest.main()
