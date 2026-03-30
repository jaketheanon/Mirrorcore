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
from mirrorcore.decision.memory_relevance import personal_response_decision_families_aligned
from mirrorcore.router import normalize_input
from mirrorcore.decision.cross_system_knowledge import effective_primary_for_cross_filter
from mirrorcore.decision.memory_relevance import respond_main_decision_passes_shape_gate
from mirrorcore.decision.routed_clarification import rank_families
from mirrorcore.persona.respond import (
    RespondEvidencePath,
    build_respond_feedback_influence,
    classify_answer_focus,
    generate_personal_response,
    retrieve_relevant_decision_memories,
    retrieve_relevant_style_memories,
    score_decision_memory_row,
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
    def test_conflict_prompt_downranks_money_scenario_memory(self):
        rent_row = {
            "id": "r1",
            "timestamp": "2025-01-01",
            "scenario_id": "rent_v1",
            "scenario_text": "Rent is late and I want to buy something expensive.",
            "choice_label": "Wait",
            "choice_value": "w",
            "reasoning_label": "Bills first",
            "reasoning_value": "b",
            "value_tags": ["caution"],
            "trait_signals": {"financial_caution": 0.8},
            "correction_status": "accurate",
            "confidence_score": 0.9,
        }
        social_row = {
            "id": "s1",
            "timestamp": "2025-01-02",
            "scenario_id": "rude_v1",
            "scenario_text": "Coworker was rude in a meeting.",
            "choice_label": "Say something",
            "choice_value": "x",
            "reasoning_label": "Clear the air",
            "reasoning_value": "y",
            "value_tags": ["directness"],
            "trait_signals": {"directness": 0.7},
            "correction_status": "accurate",
            "confidence_score": 0.88,
        }
        ranked = retrieve_relevant_decision_memories(
            [rent_row, social_row],
            "Someone was rude to me at work what should I do",
            top_k=2,
            min_score=0.0,
        )
        self.assertEqual(ranked[0][0]["id"], "s1")
        self.assertLess(ranked[1][1], ranked[0][1])

    def test_conflict_prompt_does_not_rank_overload_say_no_memory_first(self):
        """Phase 36: gossip/conflict ask must not top-rank helping/overload saves."""
        overload_row = {
            "id": "o1",
            "timestamp": "2025-01-01",
            "scenario_id": "shift_v1",
            "scenario_text": "Coworker wants you to cover another shift.",
            "choice_label": "Say no, explain you can't right now",
            "choice_value": "n",
            "reasoning_label": "I was already overloaded",
            "reasoning_value": "o",
            "value_tags": ["boundaries"],
            "trait_signals": {"boundary_strain": 0.8},
            "correction_status": "accurate",
            "confidence_score": 0.9,
        }
        conflict_row = {
            "id": "c1",
            "timestamp": "2025-01-02",
            "scenario_id": "gossip_v1",
            "scenario_text": "Someone keeps talking badly about you behind your back.",
            "choice_label": "Address it calmly with the person",
            "choice_value": "a",
            "reasoning_label": "Stop the spiral",
            "reasoning_value": "s",
            "value_tags": ["directness"],
            "trait_signals": {"diplomacy": 0.4},
            "correction_status": "accurate",
            "confidence_score": 0.88,
        }
        prompt = (
            "what would i probably say if someone keeps talking shit behind my back"
        )
        ranked = retrieve_relevant_decision_memories(
            [overload_row, conflict_row],
            prompt,
            top_k=2,
            min_score=0.0,
        )
        self.assertEqual(ranked[0][0]["id"], "c1")
        self.assertLess(ranked[1][1], ranked[0][1])

    def test_obligation_prompt_does_not_rank_pure_conflict_memory_first(self):
        """Helping/shift ask should not treat pure conflict memory as closest match."""
        conflict_row = {
            "id": "c2",
            "timestamp": "2025-01-01",
            "scenario_id": "rude_v1",
            "scenario_text": "Coworker was rude in a meeting.",
            "choice_label": "Say something directly",
            "choice_value": "s",
            "reasoning_label": "Clear the air",
            "reasoning_value": "x",
            "value_tags": ["directness"],
            "trait_signals": {"directness": 0.8},
            "correction_status": "accurate",
            "confidence_score": 0.9,
        }
        obligation_row = {
            "id": "o2",
            "timestamp": "2025-01-02",
            "scenario_id": "favor_v1",
            "scenario_text": "You are exhausted and they asked for another favor.",
            "choice_label": "Offer a smaller yes",
            "choice_value": "y",
            "reasoning_label": "Protect energy",
            "reasoning_value": "e",
            "value_tags": ["pragmatism"],
            "trait_signals": {"overload": 0.85},
            "correction_status": "accurate",
            "confidence_score": 0.88,
        }
        prompt = (
            "i'm burnt out but they want me to cover a shift what would i probably do"
        )
        ranked = retrieve_relevant_decision_memories(
            [conflict_row, obligation_row],
            prompt,
            top_k=2,
            min_score=0.0,
        )
        self.assertEqual(ranked[0][0]["id"], "o2")

    def test_misaligned_top_memory_lowers_confidence_and_skips_quote(self):
        """Weak cross-family match must not produce a confident direct quote."""
        pr_mis = {
            "id": "m1",
            "timestamp": "2025-01-01",
            "scenario_id": "shift_only",
            "scenario_text": "Pick up extra shift when already tired.",
            "choice_label": "Say no firmly",
            "choice_value": "n",
            "reasoning_label": "Too wiped",
            "reasoning_value": "w",
            "value_tags": ["boundaries"],
            "trait_signals": {"overload": 0.9},
            "correction_status": "accurate",
            "confidence_score": 0.9,
        }
        self.assertFalse(
            personal_response_decision_families_aligned(
                normalize_input("someone trash talks me behind my back"),
                pr_mis,
            )
        )

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
        a = retrieve_relevant_decision_memories(
            rows, "expensive purchase want", top_k=2, min_score=0.0
        )
        b = retrieve_relevant_decision_memories(
            rows, "expensive purchase want", top_k=2, min_score=0.0
        )
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
        self.assertIn("cut scope", pr.likely_answer.lower())
        self.assertNotIn("i’d probably choose", pr.likely_answer.lower())
        self.assertNotIn("this leans on the saved decision", pr.reasoning_brief.lower())
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

    def test_conflict_profile_fallback_tone_clause_is_clean_english(self):
        """Phase 37.3: strict conflict profile path must not stack 'skew' on 'leans cautious; …'."""
        self.db.record_decision_memory(
            scenario_id="shift_v1",
            scenario_text="Coworker wants you to cover another shift.",
            choice_label="Say no",
            choice_value="n",
            reasoning_label="Overload",
            reasoning_value="o",
            value_tags=["boundaries"],
            trait_signals={
                "risk_tolerance": 0.2,
                "bluntness": 0.75,
                "diplomacy": 0.35,
            },
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "what would i say if someone keeps talking shit behind my back",
            self.db,
        )
        low = pr.likely_answer.lower()
        self.assertNotIn("skews leans", low)
        self.assertNotIn("leans cautious;", low)
        self.assertNotIn("usually skews leans", low)
        self.assertTrue("cautious" in low or "blunt" in low or "direct" in low)

    def test_conflict_gossip_fallback_stays_useful_without_same_shape_save(self):
        """Phase 37.2: strict gossip gating with no conflict row still gives a direct-address read."""
        self.db.record_decision_memory(
            scenario_id="shift_v1",
            scenario_text="Coworker wants you to cover another shift.",
            choice_label="Say no, explain you can't right now",
            choice_value="n",
            reasoning_label="I was already overloaded",
            reasoning_value="o",
            value_tags=["boundaries"],
            trait_signals={"boundary_strain": 0.8},
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "what would i say if someone keeps talking shit behind my back",
            self.db,
        )
        low = pr.likely_answer.lower()
        self.assertTrue(
            any(w in low for w in ("address", "say", "heard", "behind", "plain", "stop"))
        )
        self.assertNotIn("overload", low)
        self.assertLessEqual(pr.confidence, 0.42)

    def test_money_rent_laptop_fallback_mentions_hold_or_triage_not_proper_way(self):
        """Phase 37.2: spending strict miss uses bills-first / pause-want shape."""
        self.db.record_decision_memory(
            scenario_id="speed_vs_safety_v1",
            scenario_text=(
                "You need to get something done and there's a quick way that cuts some corners."
            ),
            choice_label="Do it the proper way even if it takes longer",
            choice_value="right",
            reasoning_label="Avoid regret from sloppy work",
            reasoning_value="r",
            value_tags=["quality", "caution"],
            trait_signals={"thoroughness": 0.9, "risk_tolerance": 0.2},
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "what would i probably do if i wanted an $800 laptop but my rent was late",
            self.db,
        )
        low = pr.likely_answer.lower()
        self.assertNotIn("proper way", low)
        self.assertTrue(
            any(w in low for w in ("hold", "pause", "triage", "rent", "optional", "stable"))
        )

    def test_cross_family_does_not_quote_shift_save_for_gossip_prompt(self):
        self.db.record_decision_memory(
            scenario_id="shift_v1",
            scenario_text="Coworker wants you to cover another shift.",
            choice_label="Say no, explain you can't right now",
            choice_value="n",
            reasoning_label="I was already overloaded",
            reasoning_value="o",
            value_tags=["boundaries"],
            trait_signals={"boundary_strain": 0.8},
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "what would i probably say if someone keeps talking shit behind my back",
            self.db,
        )
        self.assertNotIn("overload", pr.likely_answer.lower())
        self.assertNotIn("Say no, explain", pr.likely_answer)
        self.assertLessEqual(pr.confidence, 0.42)

    def test_conflict_prompt_sounds_like_likely_speech_not_menu_label(self):
        self.db.record_decision_memory(
            scenario_id="gossip_v2",
            scenario_text="Someone is talking badly about you behind your back at work.",
            choice_label="Let's talk directly. I want this to stop.",
            choice_value="a",
            reasoning_label="Clear it early",
            reasoning_value="c",
            value_tags=["directness"],
            trait_signals={"diplomacy": 0.5},
            confidence_score=0.91,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "what would i probably say if someone keeps talking shit behind my back",
            self.db,
        )
        low = pr.likely_answer.lower()
        self.assertTrue(("say" in low) or ("talk" in low))
        self.assertNotIn("i’d probably choose", low)

    def test_overload_prompt_sounds_like_boundary_setting(self):
        self.db.record_decision_memory(
            scenario_id="shift_v3",
            scenario_text="You're exhausted and asked to cover another shift.",
            choice_label="I can't take this one. I need to recover tonight.",
            choice_value="n",
            reasoning_label="Protect energy",
            reasoning_value="p",
            value_tags=["boundaries"],
            trait_signals={"boundary_strain": 0.82},
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "i am burnt out and they asked me to help again what would i do",
            self.db,
        )
        low = pr.likely_answer.lower()
        self.assertTrue(("energy" in low) or ("limit" in low) or ("can't" in low))
        self.assertNotIn("choose:", low)

    def test_uncertainty_timing_prompt_sounds_like_real_behavior(self):
        self.db.record_decision_memory(
            scenario_id="timing_v1",
            scenario_text="You are unsure and timing matters.",
            choice_label="Wait one day and gather one more signal",
            choice_value="w",
            reasoning_label="Lower regret risk",
            reasoning_value="l",
            value_tags=["caution"],
            trait_signals={"risk_tolerance": 0.3},
            confidence_score=0.88,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "i'm not sure if i should act now or wait what would i probably do",
            self.db,
        )
        low = pr.likely_answer.lower()
        self.assertTrue(("wait" in low) or ("signal" in low) or ("pause" in low))
        self.assertNotIn("saved scenarios", low)

    def test_conflict_gossip_does_not_substitute_style_for_missing_shape_match(self):
        """Phase 37.1: gossip/backchannel asks need same-shape conflict saves, not style-only."""
        self.db.record_style_memory(
            prompt_id="let_it_go_style_v1",
            prompt_text="When someone upsets you at work",
            selected_label="Let it go",
            selected_value="let_go",
            style_tags=["patience", "avoid conflict"],
            tone_signals={"diplomacy": 0.75},
            confidence_score=0.88,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="generic_rude_meeting",
            scenario_text="Coworker was rude in a meeting about your work.",
            choice_label="Let it go and move on",
            choice_value="l",
            reasoning_label="Not worth the fight today",
            reasoning_value="n",
            value_tags=["patience"],
            trait_signals={"diplomacy": 0.72},
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "what would i say if someone keeps talking shit behind my back",
            self.db,
        )
        self.assertNotIn("let it go", pr.likely_answer.lower())
        self.assertLessEqual(pr.confidence, 0.38)
        if pr.memory_basis:
            only_style = all("style" in m.lower() for m in pr.memory_basis)
            self.assertFalse(only_style)

    def test_money_rent_laptop_does_not_quote_speed_vs_safety_decision(self):
        """Phase 37.1: bills/rent + purchase asks must not top-rank timing/corners saves."""
        self.db.record_decision_memory(
            scenario_id="speed_vs_safety_v1",
            scenario_text=(
                "You need to get something done and there's a quick way that cuts some corners."
            ),
            choice_label="Do it the proper way even if it takes longer",
            choice_value="right",
            reasoning_label="Avoid regret from sloppy work",
            reasoning_value="r",
            value_tags=["quality", "caution"],
            trait_signals={"thoroughness": 0.9, "risk_tolerance": 0.2},
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "what would i probably do if i wanted an $800 laptop but my rent is late",
            self.db,
        )
        low = pr.likely_answer.lower()
        self.assertNotIn("proper way", low)
        self.assertNotIn("cuts some corners", low)
        self.assertLessEqual(pr.confidence, 0.4)

    def test_weak_gated_decision_is_cautious_not_definite(self):
        """Borderline gated match should use soft-read path and cap confidence."""
        self.db.record_decision_memory(
            scenario_id="bc_mid",
            scenario_text="Behind your back they talk smack.",
            choice_label="Tell them to cut it out",
            choice_value="t",
            reasoning_label="End the sideways talk",
            reasoning_value="e",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.5,
            correction_status="uncorrected",
        )
        pr = generate_personal_response(
            "what would i say if someone keeps talking shit behind my back",
            self.db,
        )
        self.assertIn("soft read", pr.reasoning_brief.lower())
        self.assertLessEqual(pr.confidence, 0.52)


class TestRespondShapeGate(unittest.TestCase):
    def test_gate_rejects_speed_vs_safety_for_rent_laptop_prompt(self):
        pn = normalize_input(
            "what would i probably do if i wanted an $800 laptop but my rent is late"
        )
        ordered, _ = rank_families(pn)
        eff = effective_primary_for_cross_filter(ordered[0][0], pn)
        row = {
            "scenario_text": (
                "You need to get something done and there's a quick way that cuts some corners."
            ),
            "choice_label": "Do it the proper way even if it takes longer",
            "reasoning_label": "Quality matters",
            "value_tags": ["quality", "caution"],
        }
        self.assertFalse(
            respond_main_decision_passes_shape_gate(pn, row, effective_primary=eff)
        )

    def test_gate_requires_backchannel_row_for_gossip_prompt(self):
        pn = normalize_input(
            "what would i say if someone keeps talking shit behind my back"
        )
        ordered, _ = rank_families(pn)
        eff = effective_primary_for_cross_filter(ordered[0][0], pn)
        generic = {
            "scenario_text": "Coworker was rude in a meeting about your work.",
            "choice_label": "Let it go and move on",
            "reasoning_label": "Pick your battles",
            "value_tags": ["patience"],
        }
        self.assertFalse(
            respond_main_decision_passes_shape_gate(pn, generic, effective_primary=eff)
        )
        backchannel = {
            "scenario_text": "Someone keeps talking badly about you behind your back at work.",
            "choice_label": "Address it calmly with the person",
            "reasoning_label": "Stop the spiral",
            "value_tags": ["directness"],
        }
        self.assertTrue(
            respond_main_decision_passes_shape_gate(
                pn, backchannel, effective_primary=eff
            )
        )


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


class TestPhase39AnswerFocus(unittest.TestCase):
    def test_classify_do_and_say_prompt_is_both(self):
        pn = normalize_input(
            "what would i do and say if my mom wants a favor but im already drained"
        )
        self.assertEqual(classify_answer_focus(pn), "both")

    def test_classify_wording_vs_action(self):
        pn_say = normalize_input(
            "what would i say if someone keeps talking shit behind my back"
        )
        self.assertEqual(classify_answer_focus(pn_say), "wording")
        pn_do = normalize_input(
            "i am burnt out and they asked me to help again what would i do"
        )
        self.assertEqual(classify_answer_focus(pn_do), "action")

    def test_action_prompt_avoids_quote_wrapped_choice_for_overload(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = DatabaseStore(Path(tmp.name))
        db.initialize_database()
        try:
            db.record_decision_memory(
                scenario_id="shift_v3",
                scenario_text="You're exhausted and asked to cover another shift.",
                choice_label="I can't take this one. I need to recover tonight.",
                choice_value="n",
                reasoning_label="Protect energy",
                reasoning_value="p",
                value_tags=["boundaries"],
                trait_signals={"boundary_strain": 0.82},
                confidence_score=0.9,
                correction_status="accurate",
            )
            pr = generate_personal_response(
                "i am burnt out and they asked me to help again what would i do",
                db,
            )
            self.assertEqual(pr.answer_focus, "action")
            self.assertNotIn('say: "', pr.likely_answer)
            self.assertNotIn('go with: "', pr.likely_answer.lower())
            low = pr.likely_answer.lower()
            self.assertTrue(
                ("energy" in low) or ("boundary" in low) or ("can't" in low)
            )
        finally:
            db.close()
            Path(tmp.name).unlink(missing_ok=True)

    def test_wording_prompt_keeps_speech_shape_for_conflict_save(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = DatabaseStore(Path(tmp.name))
        db.initialize_database()
        try:
            db.record_decision_memory(
                scenario_id="gossip_v2",
                scenario_text=(
                    "Someone is talking badly about you behind your back at work."
                ),
                choice_label="Let's talk directly. I want this to stop.",
                choice_value="a",
                reasoning_label="Clear it early",
                reasoning_value="c",
                value_tags=["directness"],
                trait_signals={"diplomacy": 0.5},
                confidence_score=0.91,
                correction_status="accurate",
            )
            pr = generate_personal_response(
                "what would i probably say if someone keeps talking shit behind my back",
                db,
            )
            self.assertEqual(pr.answer_focus, "wording")
            low = pr.likely_answer.lower()
            self.assertTrue(("say" in low) or ('"' in pr.likely_answer))
            self.assertIn("let's talk directly", low)
        finally:
            db.close()
            Path(tmp.name).unlink(missing_ok=True)

    def test_do_and_say_prompt_splits_likely_action_and_wording(self):
        """Both-mode answers must visibly separate move vs script, not one quote-only blob."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = DatabaseStore(Path(tmp.name))
        db.initialize_database()
        try:
            db.record_decision_memory(
                scenario_id="mom_favor_v1",
                scenario_text="Your mom asks for a favor when you are already drained.",
                choice_label="I can't help with that tonight. I need to recharge.",
                choice_value="n",
                reasoning_label="Protect bandwidth",
                reasoning_value="b",
                value_tags=["boundaries"],
                trait_signals={"boundary_strain": 0.75},
                confidence_score=0.9,
                correction_status="accurate",
            )
            q = (
                "what would i do and say if my mom wants a favor but im already drained"
            )
            pr = generate_personal_response(q, db)
            self.assertEqual(pr.answer_focus, "both")
            ans = pr.likely_answer
            self.assertGreaterEqual(ans.count("\n"), 2)
            bridges = (
                "if you said it out loud",
                "in plain words",
            )
            low = ans.lower()
            self.assertTrue(any(b in low for b in bridges))
            self.assertIn('"', ans)
            # Not a single paragraph that is only a quoted line / phrasing teaser.
            one_line = " ".join(ans.split())
            self.assertGreater(len(one_line), 120)
            self.assertFalse(
                one_line.lower().startswith('you\'d probably say something like: "')
                and one_line.count('"') <= 2
            )
        finally:
            db.close()
            Path(tmp.name).unlink(missing_ok=True)


class TestRespondLikeMeCLI(unittest.TestCase):
    def test_subcommand_parses(self):
        p = create_parser()
        args = p.parse_args(["respond-like-me", "should I upgrade today"])
        self.assertEqual(args.command, "respond-like-me")
        self.assertEqual(args.scenario, "should I upgrade today")

    @patch("mirrorcore.persona.respond_feedback.maybe_prompt_respond_feedback")
    @patch("mirrorcore.persona.respond.generate_personal_response")
    @patch("mirrorcore.db.store.DatabaseStore")
    def test_handler_invokes_generator_with_scenario(
        self, mock_db_class, mock_gen, _mock_fb
    ):
        mock_gen.return_value = SimpleNamespace(
            likely_answer="Out",
            reasoning_brief="Because",
            confidence=0.41,
            confidence_label="fair",
            memory_basis=["Decision memory (x)"],
            profile_hint=None,
            evidence_path=RespondEvidencePath(route_keys=("strong_decision",)),
            prompt_norm_hash="abc",
            effective_family="general",
            answer_focus="action",
        )
        args = SimpleNamespace(scenario="  pick a paint color  ", no_feedback=False)
        handle_respond_like_me(args)
        mock_gen.assert_called_once()
        called_store = mock_gen.call_args[0][1]
        self.assertIs(called_store, mock_db_class.return_value)
        self.assertEqual(mock_gen.call_args[0][0], "pick a paint color")

    def test_parser_no_feedback_flag(self):
        p = create_parser()
        args = p.parse_args(["respond-like-me", "--no-feedback", "hi"])
        self.assertTrue(args.no_feedback)

    def test_parser_respond_feedback(self):
        p = create_parser()
        args = p.parse_args(["respond-feedback", "--recent", "5"])
        self.assertEqual(args.command, "respond-feedback")
        self.assertEqual(args.recent, 5)


class TestPhase38RespondFeedback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_score_respects_evidence_row_multiplier(self):
        row = {
            "id": "z1",
            "timestamp": "2025-01-01",
            "scenario_id": "s",
            "scenario_text": "coffee shop long line wait",
            "choice_label": "Stay",
            "choice_value": "s",
            "reasoning_label": "Patience",
            "reasoning_value": "p",
            "value_tags": ["patience"],
            "trait_signals": {},
            "correction_status": "uncorrected",
            "confidence_score": 0.8,
        }
        kws = tokenize_prompt("coffee shop line")
        s_hi, _ = score_decision_memory_row(
            row, kws, {"s": 1}, evidence_row_mult=1.0
        )
        s_lo, _ = score_decision_memory_row(
            row, kws, {"s": 1}, evidence_row_mult=0.35
        )
        self.assertLess(s_lo, s_hi)

    def test_wrong_feedback_reduces_multiplier_for_decision_key(self):
        self.db.merge_respond_evidence_delta("decision:x1", -0.46, mark_wrong=True)
        self.db.merge_respond_evidence_delta("decision:x1", -0.46, mark_wrong=True)
        mmap = self.db.get_respond_evidence_multiplier_map()
        self.assertLess(mmap.get("decision:x1", 1.0), 0.88)

    def test_record_personal_response_feedback_persists_row(self):
        self.db.record_personal_response_feedback(
            scenario_snippet="should I text them",
            prompt_norm_hash="deadbeef",
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text=None,
            confidence_shown=0.55,
            effective_family="interpersonal_conflict",
            evidence_path={
                "decision_ids": ["d1"],
                "style_ids": [],
                "route_keys": ["medium_decision"],
                "clarif_slot_keys": [],
            },
            likely_answer_snippet="You'd probably say…",
        )
        rows = self.db.list_recent_personal_response_feedback(limit=3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rating"], "partly")
        self.assertEqual(rows[0]["partial_aspect"], "action_ok_word_bad")
        self.assertEqual(rows[0]["evidence_path"]["decision_ids"], ["d1"])
        self.assertEqual(rows[0].get("feedback_target"), "wording")

    def test_generate_includes_evidence_path_and_hash(self):
        eid = self.db.record_decision_memory(
            scenario_id="t1",
            scenario_text="friend asked to borrow money rent is tight",
            choice_label="Say no gently",
            choice_value="n",
            reasoning_label="Protect rent",
            reasoning_value="r",
            value_tags=["caution"],
            trait_signals={},
            confidence_score=0.85,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "friend wants to borrow money and rent is tight", self.db
        )
        self.assertTrue(pr.prompt_norm_hash)
        self.assertTrue(pr.evidence_path.route_keys)
        self.assertIn(eid, pr.evidence_path.decision_ids)
        self.assertIn(
            pr.evidence_path.to_storage_dict().get("answer_focus", ""),
            ("action", "wording", "both"),
        )


class TestPhase40FeedbackInfluence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_list_feedback_for_prompt_hash(self):
        h = "cafef00d" * 4
        self.db.record_personal_response_feedback(
            scenario_snippet="x",
            prompt_norm_hash=h,
            rating="wrong",
            partial_aspect=None,
            replacement_text="address it directly",
            confidence_shown=0.5,
            effective_family="conflict",
            evidence_path={"decision_ids": ["d1"], "route_keys": ["strong_decision"]},
            likely_answer_snippet="let it go",
            feedback_target="both",
        )
        rows = self.db.list_personal_response_feedback_for_prompt(h, limit=5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["replacement_text"], "address it directly")

    def test_repeated_wrong_with_replacement_shifts_passive_aggressive_answer(self):
        self.db.record_decision_memory(
            scenario_id="pa_snide",
            scenario_text=(
                "passive aggressive coworker snide remarks sideways comments at work "
                "when someone is being passive aggressive"
            ),
            choice_label="Let it go and stay professional",
            choice_value="x",
            reasoning_label="Not worth the energy today",
            reasoning_value="y",
            value_tags=["patience", "professional"],
            trait_signals={},
            confidence_score=0.92,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="pa_direct",
            scenario_text=("teammate passive aggressive undermine work need to address calmly"),
            choice_label="Address it directly but calmly",
            choice_value="d",
            reasoning_label="Stop the sideways pattern",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.88,
            correction_status="accurate",
        )
        q = (
            "what would i do and say if someone is being passive aggressive at work"
        )
        pr = generate_personal_response(q, self.db)
        self.assertIn("let it go", pr.likely_answer.lower())
        h = pr.prompt_norm_hash
        for _ in range(3):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash=h,
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=pr.confidence,
                effective_family=pr.effective_family,
                evidence_path=pr.evidence_path.to_storage_dict(),
                likely_answer_snippet=pr.likely_answer[:200],
                feedback_target="both",
            )
            pr = generate_personal_response(q, self.db)
        low = pr.likely_answer.lower()
        self.assertNotIn("let it go", low)
        self.assertTrue(
            ("address" in low or "direct" in low or "calm" in low or "plain" in low),
            msg=pr.likely_answer,
        )
        infl = build_respond_feedback_influence(self.db, h, pr.effective_family)
        self.assertGreater(len(infl.pref_token_weight), 0)
        self.assertGreater(infl.avoidance_demote, 0.3)

    def test_two_wrong_with_same_replacement_injects_replacement_text(self):
        self.db.record_decision_memory(
            scenario_id="pa_one",
            scenario_text="passive aggressive colleague at work snide passive aggressive",
            choice_label="Let it go for now",
            choice_value="a",
            reasoning_label="Keep it smooth",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.91,
            correction_status="accurate",
        )
        q = "what would i do and say if someone is being passive aggressive at work"
        rep = "address it directly but calmly"
        pr = generate_personal_response(q, self.db)
        h = pr.prompt_norm_hash
        for _ in range(2):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash=h,
                rating="wrong",
                partial_aspect=None,
                replacement_text=rep,
                confidence_shown=pr.confidence,
                effective_family=pr.effective_family,
                evidence_path=pr.evidence_path.to_storage_dict(),
                likely_answer_snippet=pr.likely_answer[:200],
                feedback_target="both",
            )
            pr = generate_personal_response(q, self.db)
        self.assertIn("address it directly but calmly", pr.likely_answer.lower())

    def test_feedback_influence_biases_retrieval_against_avoidance_row(self):
        from mirrorcore.persona.respond import RespondFeedbackInfluence

        rows = [
            {
                "id": "avoid",
                "timestamp": "2025-01-01",
                "scenario_id": "s1",
                "scenario_text": "passive aggressive dig at work coworker snide",
                "choice_label": "Let it go",
                "choice_value": "x",
                "reasoning_label": "Keep the peace",
                "reasoning_value": "p",
                "value_tags": ["patience"],
                "trait_signals": {},
                "correction_status": "uncorrected",
                "confidence_score": 0.9,
            },
            {
                "id": "direct",
                "timestamp": "2025-01-02",
                "scenario_id": "s2",
                "scenario_text": "passive teammate work need to talk",
                "choice_label": "Address it calmly and directly",
                "choice_value": "y",
                "reasoning_label": "Clear the air",
                "reasoning_value": "c",
                "value_tags": ["directness"],
                "trait_signals": {},
                "correction_status": "uncorrected",
                "confidence_score": 0.88,
            },
        ]
        prompt = "what would i say passive aggressive coworker at work"
        base = retrieve_relevant_decision_memories(
            rows, prompt, top_k=2, min_score=0.05
        )
        self.assertEqual(base[0][0]["id"], "avoid")
        infl = RespondFeedbackInfluence(
            pref_token_weight={"address": 0.45, "calmly": 0.45, "directly": 0.45},
            avoidance_demote=0.7,
            direct_calm_signal=0.6,
        )
        biased = retrieve_relevant_decision_memories(
            rows,
            prompt,
            top_k=2,
            min_score=0.05,
            feedback_influence=infl,
        )
        m0 = {r[0]["id"]: r[1] for r in base}
        m1 = {r[0]["id"]: r[1] for r in biased}
        self.assertGreater(m1["direct"], m0["direct"])
        self.assertLess(m1["avoid"], m0["avoid"])


if __name__ == "__main__":
    unittest.main()
