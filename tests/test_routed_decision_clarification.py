"""
Phase 31.1: routed decision clarification (deterministic, one question at a time).
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore
from mirrorcore.decision.routed_clarification import (
    CONFLICT_FAMILY,
    LEGACY_GENERIC_PHRASES,
    MONEY,
    OBLIGATION_OVERLOAD,
    RISK_TIMING,
    SPENDING,
    build_routed_decision_guidance,
    extract_clarification_evidence,
    pick_next_question,
    rank_families,
    run_routed_decision_guidance,
    sanitize_domain_order_for_obligation,
    score_decision_domains,
    score_dimensions,
)
from mirrorcore.router import classify_intent, normalize_input


class TestOntology(unittest.TestCase):
    def test_bothered_social_conflict_wins(self):
        ordered, _ = rank_families(
            normalize_input(
                "Someone said something that bothered me. What should I do?"
            )
        )
        self.assertEqual(ordered[0][0], CONFLICT_FAMILY)

    def test_wait_vs_act_risk_timing_top(self):
        ordered, _ = rank_families(
            normalize_input("I don't know whether to wait or act now.")
        )
        self.assertEqual(ordered[0][0], RISK_TIMING)

    def test_money_pressure_dimension(self):
        d = score_dimensions(normalize_input("rent is late and I'm broke"))
        self.assertGreater(d.get("money_pressure", 0), 0)

    def test_overload_family_top(self):
        ordered, _ = rank_families(
            normalize_input("Should I pick up another shift? I'm exhausted.")
        )
        self.assertEqual(ordered[0][0], OBLIGATION_OVERLOAD)

    def test_conflict_family_top(self):
        ordered, _ = rank_families(
            normalize_input("Should I confront my boss about unfair treatment?")
        )
        self.assertEqual(ordered[0][0], CONFLICT_FAMILY)

    def test_coworker_cover_shift_burnt_out_is_obligation_not_conflict(self):
        """Phase 32: peer + shift cover + fatigue → obligation/overload, not conflict slots."""
        text = normalize_input(
            "a coworker needs me to cover a shift but im burnt out. what do i do?"
        )
        ordered, dims = rank_families(text)
        self.assertEqual(ordered[0][0], OBLIGATION_OVERLOAD)
        self.assertGreater(dims.get("overload", 0), 0)
        self.assertGreater(dims.get("obligation", 0), 0)
        fam_order = [f for f, sc in ordered if sc >= 0.4][:6]
        if "general" not in fam_order:
            fam_order.append("general")
        q = pick_next_question(
            context_parts=[
                "a coworker needs me to cover a shift but im burnt out. what do i do?"
            ],
            asked_ids=[],
            domain_order=fam_order,
            dimensions=dims,
        )
        self.assertIsNotNone(q)
        self.assertNotEqual(q.slot_id, "peace_vs_clarity")
        self.assertNotEqual(q.slot_id, "pattern_vs_once")
        self.assertIn(q.slot_id, ("energy_capacity", "guilt_axis", "consequence_no", "pressure_source"))

    def test_cover_shift_burnt_out_ranks_obligation(self):
        ordered, _ = rank_families(
            normalize_input("can i say no to covering a shift im burnt out")
        )
        self.assertEqual(ordered[0][0], OBLIGATION_OVERLOAD)

    def test_extra_hours_while_overloaded_is_obligation(self):
        ordered, _ = rank_families(
            normalize_input(
                "they keep asking for extra hours helping out but im overloaded already"
            )
        )
        self.assertEqual(ordered[0][0], OBLIGATION_OVERLOAD)

    def test_mom_favor_drained_is_obligation_top(self):
        ordered, _ = rank_families(
            normalize_input(
                "my mom wants a favor but im already drained what should i do"
            )
        )
        self.assertEqual(ordered[0][0], OBLIGATION_OVERLOAD)

    def test_malicious_domain_order_defers_conflict_for_shift_cover(self):
        """Even if conflict was first, sanitize shifts it after obligation (Phase 34+)."""
        text = normalize_input(
            "a coworker needs me to cover a shift but im burnt out. what do i do?"
        )
        ordered, dims = rank_families(text)
        bad_order = [CONFLICT_FAMILY, OBLIGATION_OVERLOAD, "general"]
        fixed = sanitize_domain_order_for_obligation(text, dims, bad_order)
        self.assertEqual(fixed[0], OBLIGATION_OVERLOAD)
        self.assertEqual(fixed[-1], CONFLICT_FAMILY)
        q = pick_next_question(
            context_parts=[
                "a coworker needs me to cover a shift but im burnt out. what do i do?"
            ],
            asked_ids=[],
            domain_order=fixed,
            dimensions=dims,
        )
        self.assertIsNotNone(q)
        self.assertNotEqual(q.slot_id, "peace_vs_clarity")


class TestEvidenceExtraction(unittest.TestCase):
    def test_bills_situation_fact(self):
        sit, tend = extract_clarification_evidence("bills_basics", "Rent is still not paid")
        self.assertTrue(any(k == "housing_bill_pressure" for k, _ in sit))
        self.assertEqual(tend, [])

    def test_energy_marks_both(self):
        sit, tend = extract_clarification_evidence(
            "energy_capacity", "I'm completely exhausted, no energy left"
        )
        self.assertTrue(any("energy" in k for k, _ in sit))
        self.assertTrue(any(t[0].startswith("tendency_") for t in tend))


class TestRouterMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def test_situation_and_tendency_persist(self):
        store = DatabaseStore(self.db_path)
        try:
            store.initialize_database()
            store.record_router_situation_fact("housing_bill_pressure", "open")
            store.merge_router_tendency("tendency_guilt_about_no", 0.5)
            tmap = store.get_router_tendency_map(min_strength=0.2)
            self.assertGreaterEqual(tmap.get("tendency_guilt_about_no", 0), 0.2)
            recent = store.get_recent_router_situation_facts(limit=5)
            self.assertTrue(any(r["slot_key"] == "housing_bill_pressure" for r in recent))
        finally:
            store.close()

    def test_tendency_conflict_dampens_high_strength(self):
        """Phase 33: a very low follow-up signal pulls strength down vs blind accumulation."""
        store = DatabaseStore(self.db_path)
        try:
            store.initialize_database()
            store.merge_router_tendency("tendency_test_conflict", 0.72)
            store.merge_router_tendency("tendency_test_conflict", 0.12)
            row = store.get_db_connection().execute(
                "SELECT strength FROM decision_router_tendencies WHERE slot_key = ?",
                ("tendency_test_conflict",),
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertLess(float(row["strength"]), 0.72)
        finally:
            store.close()

    def test_profile_trait_surface_bucket_allows_once_then_blocks(self):
        """Phase 34+: trait blurbs share one bucket — one show per window."""
        store = DatabaseStore(self.db_path)
        try:
            store.initialize_database()
            self.assertTrue(store.should_surface_memory_line("surface_profile_trait_snippet"))
            store.record_memory_line_surface("surface_profile_trait_snippet")
            self.assertFalse(store.should_surface_memory_line("surface_profile_trait_snippet"))
        finally:
            store.close()


class TestDomainScoring(unittest.TestCase):
    def test_laptop_rent_money_primary(self):
        t = normalize_input("Should I buy an $800 laptop if my rent is late?")
        ranked = score_decision_domains(t)
        self.assertEqual(ranked[0][0], MONEY)

    def test_ask_still_decision_intent(self):
        c = classify_intent("Should I buy an $800 laptop if my rent is late?")
        self.assertEqual(c.ordered[0][0], "decision_help")


class TestQuestionPicking(unittest.TestCase):
    def test_bothered_opens_conflict_not_energy(self):
        ordered, _ = rank_families(
            normalize_input(
                "Someone said something that bothered me. What should I do?"
            )
        )
        fam_order = [f for f, sc in ordered if sc >= 0.4][:5]
        if "general" not in fam_order:
            fam_order.append("general")
        q = pick_next_question(
            context_parts=[
                "Someone said something that bothered me. What should I do?"
            ],
            asked_ids=[],
            domain_order=fam_order,
        )
        self.assertIsNotNone(q)
        self.assertEqual(q.slot_id, "peace_vs_clarity")

    def test_first_question_money_is_rent_not_dump(self):
        q = pick_next_question(
            context_parts=["Should I buy an $800 laptop if my rent is late?"],
            asked_ids=[],
            domain_order=[MONEY, "general"],
        )
        self.assertIsNotNone(q)
        self.assertLessEqual(q.text.count("?"), 1)
        # Rent already stated → skip redundant bills check; next slot should still fit spending.
        self.assertIn(
            q.slot_id,
            ("bills_basics", "need_vs_want", "purpose_purchase", "can_wait_purchase", "cheaper_ok"),
        )

    def test_second_question_after_answer(self):
        ctx = [
            "Should I buy an $800 laptop if my rent is late?",
            "Rent is still not paid.",
        ]
        q = pick_next_question(
            context_parts=ctx,
            asked_ids=["bills_basics"],
            domain_order=[MONEY, "general"],
        )
        self.assertIsNotNone(q)
        self.assertNotEqual(q.qid, "m_rent")


class TestSlotWordingVariation(unittest.TestCase):
    def test_same_seed_same_wording(self):
        q = pick_next_question(
            context_parts=["Should I buy an $800 laptop if my rent is late?"],
            asked_ids=[],
            domain_order=[MONEY, "general"],
        )
        self.assertIsNotNone(q)
        seed = "deadbeefcafe"
        a = q.wording(seed)
        b = q.wording(seed)
        self.assertEqual(a, b)
        low = a.lower()
        self.assertTrue(
            "rent" in low or "bill" in low or "need" in low or "want" in low,
            msg=a,
        )

    def test_variants_share_slot_id_and_markers(self):
        q = pick_next_question(
            context_parts=["Should I buy an $800 laptop if my rent is late?"],
            asked_ids=[],
            domain_order=[MONEY, "general"],
        )
        self.assertIsNotNone(q)
        self.assertEqual(q.slot_id, "need_vs_want")
        seeds = [f"s{i}" for i in range(24)]
        texts = {q.wording(s) for s in seeds}
        self.assertGreaterEqual(len(texts), 2, msg="expected multiple surface wordings")


class TestGuidanceWording(unittest.TestCase):
    def test_phase34_no_generic_cautious_phrase(self):
        """Overused cautious fallback line must not appear (Phase 34)."""
        from unittest.mock import MagicMock

        prof = MagicMock()
        prof.total_evidence_weight = 1.4
        prof.trait_estimates = [
            MagicMock(name="risk_tolerance", confidence=0.55, weighted_mean=0.25),
        ]
        prof.decision_risk_summary.return_value = "leans cautious"
        prof.value_tag_weights = []
        g = build_routed_decision_guidance(
            original_question="Should I buy this if rent is late?",
            qa_pairs=[],
            domain_order=[SPENDING],
            profile=prof,
            tendency_map={},
            situation_repeat_counts={},
        )
        self.assertNotIn("call feels heavy", g.lower())

    def test_phase35_merged_memory_tail_single_block(self):
        """Tendency + profile candidates merge to one strongest line (Phase 35)."""
        prof = MagicMock()
        prof.total_evidence_weight = 1.4
        prof.trait_estimates = [
            MagicMock(name="risk_tolerance", confidence=0.55, weighted_mean=0.25),
        ]
        prof.decision_risk_summary.return_value = "leans cautious"
        prof.value_tag_weights = []
        g = build_routed_decision_guidance(
            original_question=(
                "Should I pick up another shift? I'm exhausted and saying no feels awful."
            ),
            qa_pairs=[],
            domain_order=[OBLIGATION_OVERLOAD, "general"],
            profile=prof,
            tendency_map={"tendency_guilt_about_no": 0.95},
            situation_repeat_counts={},
        )
        paras = [p.strip() for p in g.split("\n\n") if p.strip()]
        habit_hits = sum(
            1
            for p in paras
            if "from older saves" in p.lower()
            or "past answers:" in p.lower()
            or "patterns on file:" in p.lower()
            or "saying no has nagged" in p.lower()
            or "stressed over turning people down" in p.lower()
        )
        self.assertLessEqual(
            habit_hits,
            1,
            msg="expected at most one habit/profile memory paragraph",
        )

    def test_phase34_surface_suppression_respects_store(self):
        store = MagicMock()
        store.should_surface_memory_line.return_value = False
        prof = MagicMock()
        prof.total_evidence_weight = 1.4
        prof.trait_estimates = [
            MagicMock(name="risk_tolerance", confidence=0.55, weighted_mean=0.25),
        ]
        prof.decision_risk_summary.return_value = "leans cautious"
        prof.value_tag_weights = []
        g = build_routed_decision_guidance(
            original_question="Should I buy this if rent is late?",
            qa_pairs=[],
            domain_order=[SPENDING],
            profile=prof,
            tendency_map={},
            situation_repeat_counts={},
            surface_store=store,
        )
        self.assertNotIn("from older saves", g.lower())
        store.record_memory_line_surface.assert_not_called()

    def test_housing_pattern_not_surfaced_without_money_context(self):
        g = build_routed_decision_guidance(
            original_question="Someone was rude to me at work",
            qa_pairs=[],
            domain_order=[CONFLICT_FAMILY],
            profile=None,
            tendency_map={},
            situation_repeat_counts={"housing_bill_pressure=open": 5},
        )
        self.assertNotIn("housing or bill pressure", g.lower())

    def test_housing_pattern_requires_spending_and_high_count(self):
        g = build_routed_decision_guidance(
            original_question="Should I buy a laptop if rent is late?",
            qa_pairs=[],
            domain_order=[SPENDING],
            profile=None,
            tendency_map={},
            situation_repeat_counts={"housing_bill_pressure=open": 2},
        )
        self.assertNotIn("past clarifications", g.lower())
        g2 = build_routed_decision_guidance(
            original_question="Should I buy a laptop if rent is late?",
            qa_pairs=[],
            domain_order=[SPENDING],
            profile=None,
            tendency_map={},
            situation_repeat_counts={"housing_bill_pressure=open": 3},
        )
        self.assertIn("past clarifications", g2.lower())

    def test_no_legacy_generic_phrases(self):
        g = build_routed_decision_guidance(
            original_question="Should I buy an $800 laptop if my rent is late?",
            qa_pairs=[
                ("q1", "Rent is still not paid."),
                ("q2", "It's mainly for school."),
            ],
            domain_order=[MONEY],
            profile=None,
        )
        low = g.lower()
        for phrase in LEGACY_GENERIC_PHRASES:
            self.assertNotIn(phrase, low)

    def test_unpaid_rent_concrete(self):
        g = build_routed_decision_guidance(
            original_question="Laptop purchase?",
            qa_pairs=[("q", "Rent is not paid yet")],
            domain_order=[MONEY],
            profile=None,
        )
        self.assertIn("rent", g.lower())
        self.assertNotIn("available options", g.lower())


class TestPhase40ConflictDepth(unittest.TestCase):
    def test_passive_slight_guidance_vocab(self):
        g = build_routed_decision_guidance(
            original_question="my colleague is snide and underhanded at meetings",
            qa_pairs=[],
            domain_order=[CONFLICT_FAMILY, "general"],
            profile=None,
        )
        low = g.lower()
        self.assertTrue(
            any(x in low for x in ("snide", "dig", "passive", "plain", "indirect")),
            msg=g,
        )

    def test_conflict_aim_extracts_boundary_stance(self):
        sit, tend = extract_clarification_evidence(
            "conflict_aim", "I need to draw a boundary and stop absorbing it"
        )
        self.assertTrue(any(k == "conflict_stance" for k, _ in sit))
        self.assertTrue(any(t[0] == "tendency_hard_boundary" for t in tend))


class TestInteractiveFlow(unittest.TestCase):
    @patch("mirrorcore.decision.routed_clarification.build_personal_profile")
    def test_two_rounds_incorporate_answers(self, mock_prof):
        mock_prof.return_value = MagicMock(total_evidence_weight=0.0)

        store = MagicMock()
        store.get_recent_decision_memory.return_value = []
        store.get_recent_style_memory.return_value = []
        store.get_router_tendency_map.return_value = {}
        store.get_recent_router_situation_facts.return_value = []

        replies = iter(
            [
                "Still not paid.",
                "Need it for school, old one died.",
            ]
        )

        def read_line(_p: str) -> str:
            return next(replies)

        guidance = run_routed_decision_guidance(
            initial_text="Should I buy an $800 laptop if my rent is late?",
            read_line=read_line,
            db_store=store,
        )
        low = guidance.lower()
        for phrase in LEGACY_GENERIC_PHRASES:
            self.assertNotIn(phrase, low)
        self.assertTrue(
            "school" in low or "need" in low or "rent" in low,
            msg=guidance,
        )


if __name__ == "__main__":
    unittest.main()
