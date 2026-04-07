"""
Tests for Phase 29: unified personal profile aggregation, memory retrieval,
grounded response generation, and confidence behavior.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
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
from mirrorcore.decision.routed_clarification import OBLIGATION_OVERLOAD, rank_families
from mirrorcore.decision.situation_carryover import (
    carryover_shape_key,
    combined_shape_key,
    pa_issue_carryover_bridge,
)
from mirrorcore.persona.respond import (
    RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN,
    RespondEvidencePath,
    _merge_action_wording_paragraphs,
    _phase41_style_realism_pass,
    _phase48_final_answer_polish,
    _respond_carryover_reasoning_line_audit,
    _evaluate_phase45_conflict_escalation,
    _respond_carryover_suppress_avoidance,
    _respond_repeated_passive_aggressive_escalation_active,
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
        rb = pr.reasoning_brief.lower()
        self.assertTrue(
            "cautious" in rb or "blunt" in rb or "profile tendency" in rb,
            msg=pr.reasoning_brief,
        )
        self.assertNotRegex(low, r"cautious on risk.*wording")
        self.assertNotIn("say it straight", low)
        self.assertNotIn("heat low", low)

    def test_phase41_no_mixed_pronouns_on_overload_action(self):
        """First-person likely-you lines must not pair I'd with your (self-energy/bandwidth)."""
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
        self.assertNotRegex(low, r"i'd[^\n]{0,120}your energy")
        self.assertNotRegex(low, r"i'd[^\n]{0,120}your bandwidth")

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

    def test_phase48_spending_both_mode_no_stacked_out_loud_wrapper(self):
        """Phase 48.1: strict spending fallback stays one practical paragraph (no talk-track echo)."""
        self.db.record_decision_memory(
            scenario_id="speed_vs_safety_phase48",
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
            "i want to buy something fun but rent is due tomorrow and i am short on money",
            self.db,
        )
        self.assertEqual(pr.effective_family, "spending")
        low = pr.likely_answer.lower()
        self.assertNotIn("\n", (pr.likely_answer or "").strip(), msg=pr.likely_answer)
        self.assertNotIn("if i said it out loud", low, msg=pr.likely_answer)
        self.assertNotIn("might sound like this", low, msg=pr.likely_answer)
        self.assertNotIn("line i'd use with myself", low, msg=pr.likely_answer)
        self.assertTrue(
            any(
                w in low
                for w in (
                    "rent",
                    "roof",
                    "want",
                    "need",
                    "pause",
                    "triage",
                    "bill",
                    "optional",
                    "cheaper",
                )
            ),
            msg=pr.likely_answer,
        )

    def test_phase48_final_polish_fixes_mid_sentence_i_cant(self):
        self.assertEqual(
            _phase48_final_answer_polish("I already said no. i cant take that on right now"),
            "I already said no. I can't take that on right now",
        )

    def test_phase48_final_polish_dedupes_legacy_double_out_loud_header(self):
        raw = (
            "I'd triage bills first.\n\n"
            "If I said it out loud, it might sound like this:\n"
            "If I said it out loud, it would sound like triage — roof first."
        )
        out = _phase48_final_answer_polish(raw)
        self.assertEqual(out.lower().count("if i said it out loud"), 1)

    def test_phase48_merge_drops_repeated_why_after_quoted_line(self):
        act = "I'd hold the line. Mostly because I was already overloaded."
        wrd = (
            'you\'d probably go with: "Say no, explain you can\'t right now." '
            "i was already overloaded."
        )
        out = _merge_action_wording_paragraphs(act, wrd, seed="phase48_why_dedupe")
        self.assertEqual(
            out.lower().count("already overloaded"),
            1,
            msg=out,
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

    def test_phase41_reduces_meta_narration_in_main_answer(self):
        self.db.record_decision_memory(
            scenario_id="gossip_v3",
            scenario_text="Coworker keeps talking behind your back.",
            choice_label="Let's talk directly. I want it handled face-to-face.",
            choice_value="a",
            reasoning_label="Clear it early and keep it calm",
            reasoning_value="c",
            value_tags=["directness"],
            trait_signals={"diplomacy": 0.45},
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr = generate_personal_response(
            "what would i say if someone keeps talking shit behind my back",
            self.db,
        )
        low = pr.likely_answer.lower()
        self.assertNotIn("my read is", low)
        self.assertNotIn("on file you tend to", low)
        self.assertNotIn("from what's on file", low)
        self.assertTrue(
            ("i'd" in low) or ("i " in low) or ("that's about how" in low),
            msg=pr.likely_answer,
        )
        self.assertNotIn("you'd probably say", low)

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
            self.assertIn("\n", ans)
            lines = [ln.strip() for ln in ans.split("\n") if ln.strip()]
            self.assertGreaterEqual(len(lines), 2, msg=ans)
            low = ans.lower()
            self.assertIn("if i said it out loud", low, msg=ans)
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


class TestPhase41OutputRealism(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_main_answer_avoids_tone_meta_phrases(self):
        """Tone is shown through phrasing, not summarized as 'cautious on risk, … wording'."""
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
        self.assertNotRegex(low, r"on risk, with .+ wording")
        self.assertNotIn("pretty blunt wording", low)
        self.assertNotIn("fairly soft wording", low)

    def test_main_answer_avoids_internal_save_jargon(self):
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
        self.assertNotIn("save describes", low)
        self.assertNotIn("the way that save", low)

    def test_both_mode_action_paragraph_is_not_script_hybrid(self):
        """Move vs script: first block should not paste the full choice like the quote block."""
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
            correction_status="uncorrected",
        )
        q = "what would i do and say if someone is being passive aggressive at work"
        pr = generate_personal_response(q, self.db)
        self.assertEqual(pr.answer_focus, "both")
        lines = [ln.strip() for ln in pr.likely_answer.split("\n") if ln.strip()]
        self.assertGreaterEqual(len(lines), 2, msg=pr.likely_answer)
        self.assertNotEqual(lines[0].lower(), lines[-1].lower())
        low = pr.likely_answer.lower()
        self.assertIn("if i said it out loud", low)
        self.assertNotRegex(
            lines[0].lower(),
            r"handle this head-on: let it go",
            msg=lines[0],
        )

    def test_action_mode_avoids_and_say_script_hook(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = DatabaseStore(Path(tmp.name))
        db.initialize_database()
        try:
            db.record_decision_memory(
                scenario_id="push_v1",
                scenario_text="Coworker keeps asking after you already said no.",
                choice_label="I already said no — I'm not reopening it tonight.",
                choice_value="n",
                reasoning_label="Hold the line",
                reasoning_value="h",
                value_tags=["boundaries"],
                trait_signals={},
                confidence_score=0.9,
                correction_status="accurate",
            )
            pr = generate_personal_response(
                "what would i do if a coworker keeps pushing after i already said no",
                db,
            )
            self.assertEqual(pr.answer_focus, "action")
            self.assertNotRegex(pr.likely_answer.lower(), r"\band say:")
        finally:
            db.close()
            Path(tmp.name).unlink(missing_ok=True)

    def test_public_disrespect_blocks_avoidance_style_row(self):
        """High-scoring 'let it go' style must not anchor public disrespect."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = DatabaseStore(Path(tmp.name))
        db.initialize_database()
        try:
            db.record_style_memory(
                prompt_id="avoid_style_v1",
                prompt_text="When someone is rude in front of others",
                selected_label="Let it go and keep your composure",
                selected_value="x",
                style_tags=["patience", "let it go"],
                tone_signals={"diplomacy": 0.82, "directness": 0.2},
                confidence_score=0.9,
                correction_status="accurate",
            )
            pr = generate_personal_response(
                "what would i say if someone disrespects me in front of other people",
                db,
            )
            self.assertNotIn("let it go", pr.likely_answer.lower())
        finally:
            db.close()
            Path(tmp.name).unlink(missing_ok=True)

    def test_public_disrespect_does_not_quote_let_it_go(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = DatabaseStore(Path(tmp.name))
        db.initialize_database()
        try:
            db.record_decision_memory(
                scenario_id="pub_rude_avoid",
                scenario_text="Someone threw shade at you among friends; stay smooth.",
                choice_label="Let it go and move on",
                choice_value="l",
                reasoning_label="Not worth the drama tonight",
                reasoning_value="n",
                value_tags=["patience"],
                trait_signals={},
                confidence_score=0.92,
                correction_status="accurate",
            )
            db.record_decision_memory(
                scenario_id="pub_rude_direct",
                scenario_text="Coworker was rude to you in front of the team.",
                choice_label="Pull them aside and say that was not okay",
                choice_value="p",
                reasoning_label="Keep it clean but direct",
                reasoning_value="k",
                value_tags=["directness"],
                trait_signals={},
                confidence_score=0.9,
                correction_status="accurate",
            )
            pr = generate_personal_response(
                "what would i say if someone disrespects me in front of other people",
                db,
            )
            low = pr.likely_answer.lower()
            self.assertNotIn("let it go", low)
            self.assertTrue(
                any(w in low for w in ("address", "direct", "aside", "okay", "say")),
                msg=pr.likely_answer,
            )
        finally:
            db.close()
            Path(tmp.name).unlink(missing_ok=True)

    def test_public_disrespect_with_only_avoidance_decision_stays_direct(self):
        """Even with only avoidance memory, public disrespect should not default to passivity."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = DatabaseStore(Path(tmp.name))
        db.initialize_database()
        try:
            db.record_decision_memory(
                scenario_id="pub_only_avoid",
                scenario_text="They were rude to me in front of everyone.",
                choice_label="Let it go and move on",
                choice_value="l",
                reasoning_label="Keep the peace",
                reasoning_value="k",
                value_tags=["patience"],
                trait_signals={},
                confidence_score=0.95,
                correction_status="accurate",
            )
            pr = generate_personal_response(
                "what would i say if someone disrespects me in front of people",
                db,
            )
            low = pr.likely_answer.lower()
            self.assertNotIn("let it go", low)
            self.assertTrue(
                any(w in low for w in ("not okay", "direct", "boundary", "calm", "clear")),
                msg=pr.likely_answer,
            )
        finally:
            db.close()
            Path(tmp.name).unlink(missing_ok=True)

    def test_public_disrespect_prefers_calm_direct_not_aggressive(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = DatabaseStore(Path(tmp.name))
        db.initialize_database()
        try:
            db.record_decision_memory(
                scenario_id="pub_mix",
                scenario_text="Disrespect in front of others at work.",
                choice_label="Let it go and move on",
                choice_value="l",
                reasoning_label="Keep peace",
                reasoning_value="k",
                value_tags=["patience"],
                trait_signals={},
                confidence_score=0.9,
                correction_status="accurate",
            )
            pr = generate_personal_response(
                "what would i say if someone disrespects me in public",
                db,
            )
            low = pr.likely_answer.lower()
            self.assertTrue(
                any(w in low for w in ("calm", "direct", "not okay", "clear")),
                msg=pr.likely_answer,
            )
            self.assertFalse(
                any(w in low for w in ("destroy", "humiliate", "make a scene")),
                msg=pr.likely_answer,
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


class TestPhase42CorrectionPromotionExamples(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_one_off_replacement_not_reusable_yet(self):
        self.db.record_personal_response_feedback(
            scenario_snippet="x",
            prompt_norm_hash="h1",
            rating="wrong",
            partial_aspect=None,
            replacement_text="address it directly but calmly",
            confidence_shown=0.5,
            effective_family="interpersonal_conflict",
            evidence_path={"route_keys": ["strong_decision"], "answer_focus": "wording"},
            likely_answer_snippet="let it go",
            feedback_target="wording",
        )
        rows = self.db.list_reusable_response_examples(
            effective_family="interpersonal_conflict",
            route_keys=["strong_decision"],
            answer_focus="wording",
        )
        self.assertEqual(rows, [])

    def test_repeated_corrections_promote_reusable_example(self):
        for _ in range(3):
            self.db.record_personal_response_feedback(
                scenario_snippet="x",
                prompt_norm_hash="h2",
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=0.5,
                effective_family="interpersonal_conflict",
                evidence_path={"route_keys": ["strong_decision"], "answer_focus": "wording"},
                likely_answer_snippet="let it go",
                feedback_target="wording",
            )
        rows = self.db.list_reusable_response_examples(
            effective_family="interpersonal_conflict",
            route_keys=["strong_decision"],
            answer_focus="wording",
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("address it directly", rows[0]["example_text"])
        self.assertGreaterEqual(float(rows[0]["strength"]), 0.55)
        self.assertGreaterEqual(int(rows[0]["support_count"]), 2)

    def test_promoted_wording_example_can_shift_future_output(self):
        self.db.record_decision_memory(
            scenario_id="pa_one",
            scenario_text="passive aggressive coworker at work",
            choice_label="Let it go for now",
            choice_value="a",
            reasoning_label="Keep it smooth",
            reasoning_value="k",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.9,
            correction_status="accurate",
        )
        q = "what would i say if someone is being passive aggressive at work"
        first = generate_personal_response(q, self.db)
        self.assertIn("let it go", first.likely_answer.lower())
        for _ in range(3):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash=first.prompt_norm_hash,
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=first.confidence,
                effective_family=first.effective_family,
                evidence_path=first.evidence_path.to_storage_dict(),
                likely_answer_snippet=first.likely_answer[:200],
                feedback_target="wording",
            )
            first = generate_personal_response(q, self.db)
        low = first.likely_answer.lower()
        self.assertTrue("address" in low or "directly" in low, msg=first.likely_answer)
        examples = self.db.list_reusable_response_examples(
            effective_family=first.effective_family,
            answer_focus="wording",
        )
        self.assertGreaterEqual(len(examples), 1)

    def test_contradicted_examples_are_tracked_and_damped(self):
        for _ in range(3):
            self.db.record_personal_response_feedback(
                scenario_snippet="x",
                prompt_norm_hash="h3",
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=0.5,
                effective_family="interpersonal_conflict",
                evidence_path={"route_keys": ["strong_decision"], "answer_focus": "wording"},
                likely_answer_snippet="let it go",
                feedback_target="wording",
            )
        for _ in range(2):
            self.db.record_personal_response_feedback(
                scenario_snippet="x",
                prompt_norm_hash="h3",
                rating="wrong",
                partial_aspect=None,
                replacement_text="let it go and move on",
                confidence_shown=0.5,
                effective_family="interpersonal_conflict",
                evidence_path={"route_keys": ["strong_decision"], "answer_focus": "wording"},
                likely_answer_snippet="address it",
                feedback_target="wording",
            )
        rows = self.db.list_reusable_response_examples(
            effective_family="interpersonal_conflict",
            route_keys=["strong_decision"],
            answer_focus="wording",
            limit=6,
            min_strength=0.2,
        )
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(any(int(r.get("contradict_count") or 0) > 0 for r in rows))


class TestPhase43ContradictionConfidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_mixed_examples_lower_confidence_and_reason_mentions_mixed(self):
        self.db.record_decision_memory(
            scenario_id="pa_one",
            scenario_text="passive aggressive coworker at work",
            choice_label="Let it go for now",
            choice_value="a",
            reasoning_label="Keep it smooth",
            reasoning_value="k",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.9,
            correction_status="accurate",
        )
        q = "what would i say if someone is being passive aggressive at work"
        for _ in range(3):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash="mix_hash_1",
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=0.55,
                effective_family="interpersonal_conflict",
                evidence_path={"route_keys": ["strong_decision"], "answer_focus": "wording"},
                likely_answer_snippet="let it go",
                feedback_target="wording",
            )
        for _ in range(3):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash="mix_hash_1",
                rating="wrong",
                partial_aspect=None,
                replacement_text="let it go and move on",
                confidence_shown=0.55,
                effective_family="interpersonal_conflict",
                evidence_path={"route_keys": ["strong_decision"], "answer_focus": "wording"},
                likely_answer_snippet="address it",
                feedback_target="wording",
            )
        pr_mixed = generate_personal_response(q, self.db)
        self.assertLessEqual(pr_mixed.confidence, 0.58)
        for _ in range(3):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash="mix_hash_1",
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=0.55,
                effective_family="interpersonal_conflict",
                evidence_path={"route_keys": ["strong_decision"], "answer_focus": "wording"},
                likely_answer_snippet="let it go",
                feedback_target="wording",
            )
        pr_resolved = generate_personal_response(q, self.db)
        self.assertLessEqual(pr_resolved.confidence, 0.86)

    def test_repeated_agreeing_examples_can_raise_confidence_carefully(self):
        q = "what would i say if someone is being passive aggressive at work"
        self.db.record_decision_memory(
            scenario_id="pa_two",
            scenario_text="passive aggressive coworker at work",
            choice_label="Address it directly but calmly",
            choice_value="d",
            reasoning_label="Stop the sideways pattern",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.9,
            correction_status="accurate",
        )
        for _ in range(6):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash="agree_hash_1",
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=0.55,
                effective_family="interpersonal_conflict",
                evidence_path={"route_keys": ["strong_decision"], "answer_focus": "wording"},
                likely_answer_snippet="let it go",
                feedback_target="wording",
            )
        pr = generate_personal_response(q, self.db)
        self.assertGreaterEqual(pr.confidence, 0.45)
        self.assertLessEqual(pr.confidence, 0.86)

    def test_one_contradictory_correction_does_not_flip_dominant_direction(self):
        """Phase 43.1: one new replacement must not override stronger repeat support."""
        self.db.record_decision_memory(
            scenario_id="pa_one",
            scenario_text="passive aggressive coworker at work",
            choice_label="Let it go for now",
            choice_value="a",
            reasoning_label="Keep it smooth",
            reasoning_value="k",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.9,
            correction_status="accurate",
        )
        q = "what would i say if someone is being passive aggressive at work"
        pr0 = generate_personal_response(q, self.db)
        h = pr0.prompt_norm_hash
        for _ in range(5):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash=h,
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=pr0.confidence,
                effective_family=pr0.effective_family,
                evidence_path=pr0.evidence_path.to_storage_dict(),
                likely_answer_snippet=pr0.likely_answer[:200],
                feedback_target="wording",
            )
        pr_stable = generate_personal_response(q, self.db)
        self.db.record_personal_response_feedback(
            scenario_snippet=q,
            prompt_norm_hash=h,
            rating="wrong",
            partial_aspect=None,
            replacement_text="just ignore it and move on",
            confidence_shown=pr_stable.confidence,
            effective_family=pr_stable.effective_family,
            evidence_path=pr_stable.evidence_path.to_storage_dict(),
            likely_answer_snippet=pr_stable.likely_answer[:200],
            feedback_target="wording",
        )
        pr_after = generate_personal_response(q, self.db)
        low = pr_after.likely_answer.lower()
        self.assertTrue(
            "address" in low or "direct" in low or "calm" in low,
            msg=pr_after.likely_answer,
        )
        self.assertFalse(
            "just ignore it and move on" in low,
            msg=pr_after.likely_answer,
        )
        self.assertLess(pr_after.confidence, pr_stable.confidence)

    def test_one_contra_lowers_confidence_and_accumulated_contra_can_shift(self):
        q = "what would i say if someone is being passive aggressive at work"
        self.db.record_decision_memory(
            scenario_id="pa_one",
            scenario_text="passive aggressive coworker at work",
            choice_label="Let it go for now",
            choice_value="a",
            reasoning_label="Keep it smooth",
            reasoning_value="k",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.9,
            correction_status="accurate",
        )
        pr0 = generate_personal_response(q, self.db)
        h = pr0.prompt_norm_hash
        for _ in range(4):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash=h,
                rating="wrong",
                partial_aspect=None,
                replacement_text="address it directly but calmly",
                confidence_shown=pr0.confidence,
                effective_family=pr0.effective_family,
                evidence_path=pr0.evidence_path.to_storage_dict(),
                likely_answer_snippet=pr0.likely_answer[:200],
                feedback_target="wording",
            )
        pr_after_support = generate_personal_response(q, self.db)
        self.db.record_personal_response_feedback(
            scenario_snippet=q,
            prompt_norm_hash=h,
            rating="wrong",
            partial_aspect=None,
            replacement_text="just ignore it and move on",
            confidence_shown=pr_after_support.confidence,
            effective_family=pr_after_support.effective_family,
            evidence_path=pr_after_support.evidence_path.to_storage_dict(),
            likely_answer_snippet=pr_after_support.likely_answer[:200],
            feedback_target="wording",
        )
        pr_one_contra = generate_personal_response(q, self.db)
        self.assertLess(pr_one_contra.confidence, pr_after_support.confidence)
        for _ in range(5):
            self.db.record_personal_response_feedback(
                scenario_snippet=q,
                prompt_norm_hash=h,
                rating="wrong",
                partial_aspect=None,
                replacement_text="just ignore it and move on",
                confidence_shown=pr_one_contra.confidence,
                effective_family=pr_one_contra.effective_family,
                evidence_path=pr_one_contra.evidence_path.to_storage_dict(),
                likely_answer_snippet=pr_one_contra.likely_answer[:200],
                feedback_target="wording",
            )
        pr_flip = generate_personal_response(q, self.db)
        low = pr_flip.likely_answer.lower()
        self.assertTrue(
            "ignore" in low or "move on" in low,
            msg=pr_flip.likely_answer,
        )


class TestPhase44RespondContinuation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseStore(Path(self.tmp.name))
        self.db.initialize_database()

    def tearDown(self):
        self.db.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_passive_aggressive_same_person_continuation_demotes_avoidance_memory(self):
        self.db.record_decision_memory(
            scenario_id="pa_avoid",
            scenario_text=(
                "passive aggressive snide colleague remarks at work sideways digs"
            ),
            choice_label="Let it go and stay professional",
            choice_value="a",
            reasoning_label="Not worth the fight today",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.96,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="pa_direct",
            scenario_text=(
                "passive aggressive teammate need to address calmly at work"
            ),
            choice_label="Address it directly but calmly",
            choice_value="d",
            reasoning_label="Stop the sideways pattern",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.72,
            correction_status="accurate",
        )
        prev = normalize_input(
            "what would i say if someone is being passive aggressive at work"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="conflict",
            shape_key=carryover_shape_key(prev, "conflict"),
            stance_snippet="address it directly calmly name what you noticed",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        q2 = "what would i say if this same passive aggressive person keeps doing it"
        pr = generate_personal_response(q2, self.db)
        low = pr.likely_answer.lower()
        self.assertNotIn("let it go", low)
        self.assertTrue(
            any(
                x in low
                for x in (
                    "direct",
                    "address",
                    "plain",
                    "name",
                    "calm",
                    "noticed",
                    "sideways",
                )
            ),
            msg=pr.likely_answer,
        )

    def test_passive_aggressive_continuation_uses_carryover_when_row_is_softer_wording(
        self,
    ):
        """Stored row may say 'jabs' + colleague/work without literal 'passive aggressive'."""
        prev = normalize_input(
            "what would i say if a colleague keeps subtle jabs at me at work"
        )
        self.assertTrue(
            pa_issue_carryover_bridge(
                normalize_input(
                    "what would i say if this same passive aggressive person keeps doing it"
                ),
                prev,
            )
        )
        self.db.record_decision_memory(
            scenario_id="pa_avoid2",
            scenario_text="colleague tension let it go professional",
            choice_label="Let it go and move on",
            choice_value="a",
            reasoning_label="Not worth engaging",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.96,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="pa_direct2",
            scenario_text="address colleague jabs calmly at work",
            choice_label="Address it calmly and directly",
            choice_value="d",
            reasoning_label="Name the pattern",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.72,
            correction_status="accurate",
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="conflict",
            shape_key=carryover_shape_key(prev, "conflict"),
            stance_snippet="address it calmly name what you noticed one clear line",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        q2 = "what would i say if this same passive aggressive person keeps doing it"
        pr = generate_personal_response(q2, self.db)
        low = pr.likely_answer.lower()
        self.assertNotIn("let it go", low)
        self.assertTrue(
            any(x in low for x in ("direct", "address", "calm", "noticed", "plain", "say")),
            msg=pr.likely_answer,
        )

    def test_spending_prompt_does_not_show_conflict_thread_continuity_line(self):
        prev = normalize_input(
            "what would i say if someone is passive aggressive at work"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="conflict",
            shape_key=carryover_shape_key(prev, "conflict"),
            stance_snippet="address it directly calmly",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        rent_q = (
            "i need to pay rent but i am broke what should i cut first "
            "groceries or going out"
        )
        pr = generate_personal_response(rent_q, self.db)
        self.assertEqual(pr.effective_family, "spending")
        rb = (pr.reasoning_brief or "").lower()
        for needle in (
            "same issue continuing",
            "recent unresolved situation",
            "recent similar thread",
            "just working through",
        ):
            self.assertNotIn(needle, rb, msg=pr.reasoning_brief)

    def test_spending_followup_different_prompt_no_continuity_line_with_prior_spending_row(
        self,
    ):
        prev = normalize_input("should i buy a new laptop when my rent is late")
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="spending",
            shape_key=carryover_shape_key(prev, "spending"),
            stance_snippet="cover rent basics before wants",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        q2 = (
            "i still need to pay rent tomorrow and i only have enough for food "
            "or fun which comes first"
        )
        pr = generate_personal_response(q2, self.db)
        self.assertEqual(pr.effective_family, "spending")
        rb = (pr.reasoning_brief or "").lower()
        for needle in (
            "same issue continuing",
            "recent unresolved situation",
            "recent similar thread",
            "just working through",
        ):
            self.assertNotIn(needle, rb, msg=pr.reasoning_brief)

    def test_repeated_pa_escalation_gate_requires_pa_carryover_align(self):
        """Unrelated carryover text must not activate narrow repeat+PA escalation."""
        prev = normalize_input("should i buy a laptop when rent is late")
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        payload = {
            "strength": 0.55,
            "carry_family": "spending",
            "match": {
                "id": "x",
                "state": "unresolved",
                "updated_at": datetime.utcnow().isoformat(),
                "prompt_norm": prev,
                "prompt_norm_hash": ph,
                "shape_key": "spending|slots:",
                "stance_snippet": "cover rent first",
            },
        }
        q = normalize_input(
            "what would i say if this same passive aggressive person keeps doing it"
        )
        qh = hashlib.sha256(q.encode("utf-8")).hexdigest()
        self.assertFalse(
            _respond_repeated_passive_aggressive_escalation_active(
                eff_pf="conflict",
                prompt_norm=q,
                prompt_norm_hash=qh,
                situation_carryover=payload,
                carry_strength=0.55,
            ),
            "spending-shaped row_norm must not PA-align with conflict continuation",
        )

    def test_phase44_repeated_pa_only_avoidant_saves_avoidant_carryover_stance(self):
        """
        Regression: only avoidance-shaped decision rows + avoidant stance_snippet
        on the active carryover row — still no \"let it go\" echo; carryover stays on.
        """
        self.db.record_decision_memory(
            scenario_id="pa_avoid_only",
            scenario_text=(
                "passive aggressive snide colleague remarks at work sideways digs"
            ),
            choice_label="Let it go and move on",
            choice_value="a",
            reasoning_label="Not worth the fight today",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.96,
            correction_status="accurate",
        )
        prev = normalize_input(
            "what would i say if someone is being passive aggressive at work"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="conflict",
            shape_key=carryover_shape_key(prev, "conflict"),
            stance_snippet="Let it go and move on.",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        exact_q = (
            "what would i say if this same passive aggressive person keeps doing it"
        )
        pr = generate_personal_response(
            exact_q, self.db, debug_phase44_carryover=True
        )
        dbg = pr.phase44_carryover_debug
        self.assertTrue(dbg["pick_best_for_ask"]["any_candidate_meets_threshold"])
        self.assertTrue(dbg["suppress_avoidance_demotion"])
        self.assertTrue(
            dbg["repeated_passive_aggressive_escalation"],
            msg=f"expected escalation gate: {dbg!r}",
        )
        self.assertTrue(dbg["strip_avoidance_decision_rows"])
        self.assertTrue(dbg["reasoning_line_audit"]["allowed"])
        low = pr.likely_answer.lower()
        self.assertNotIn("let it go", low, msg=pr.likely_answer)
        self.assertNotIn("move on", low, msg=pr.likely_answer)
        self.assertTrue(
            any(
                w in low
                for w in (
                    "pattern",
                    "direct",
                    "plain",
                    "noticed",
                    "name",
                    "calm",
                    "sideways",
                    "repeat",
                    "keeps",
                )
            ),
            msg=pr.likely_answer,
        )

    def test_phase44_exact_passive_aggressive_continuation_end_to_end_with_debug(self):
        """
        Regression: exact manual prompt must keep thread (not avoidance) when a
        matching conflict short-term row exists — debug shows the carryover path.
        """
        self.db.record_decision_memory(
            scenario_id="pa_avoid_dbg",
            scenario_text=(
                "passive aggressive snide colleague remarks at work sideways digs"
            ),
            choice_label="Let it go and stay professional",
            choice_value="a",
            reasoning_label="Not worth the fight today",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.96,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="pa_direct_dbg",
            scenario_text=(
                "passive aggressive teammate need to address calmly at work"
            ),
            choice_label="Address it directly but calmly",
            choice_value="d",
            reasoning_label="Stop the sideways pattern",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.72,
            correction_status="accurate",
        )
        prev = normalize_input(
            "what would i say if someone is being passive aggressive at work"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="conflict",
            shape_key=carryover_shape_key(prev, "conflict"),
            stance_snippet="address it directly calmly name what you noticed",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        exact_q = (
            "what would i say if this same passive aggressive person keeps doing it"
        )
        pr = generate_personal_response(
            exact_q, self.db, debug_phase44_carryover=True
        )
        dbg = pr.phase44_carryover_debug
        self.assertEqual(
            dbg["pick_best_for_ask"]["best_row_id"],
            dbg["active_match_row_id"],
            msg=f"debug best_row_id should match active carryover row: {dbg!r}",
        )
        self.assertIsNotNone(dbg)
        pick = dbg["pick_best_for_ask"]
        self.assertTrue(
            pick["any_candidate_meets_threshold"],
            msg=f"expected ask carryover pick; got {pick!r}",
        )
        self.assertFalse(dbg["family_alignment"]["dropped_by_family_align"])
        self.assertTrue(dbg["suppress_avoidance_demotion"])
        self.assertTrue(dbg["repeated_passive_aggressive_escalation"])
        self.assertTrue(dbg["phase45_escalation_evaluated"])
        self.assertTrue(dbg["phase45_escalation_active"])
        self.assertEqual(
            dbg["phase45_escalation_subtype"],
            "passive_aggressive_repeat",
        )
        self.assertEqual(dbg["phase45_escalation_reject_reasons"], [])
        self.assertTrue(dbg["phase45_demote_avoidance_due_to_escalation"])
        self.assertTrue(dbg["phase45_boost_direct_boundary_retrieval"])
        low = pr.likely_answer.lower()
        self.assertNotIn("let it go", low, msg=pr.likely_answer)

    def test_phase45_boundary_push_obligation_routes_escalation(self):
        """Regression B: same-coworker boundary persistence routes obligation; escalation fires."""
        self.db.record_decision_memory(
            scenario_id="ob_push_avoid",
            scenario_text="coworker asking to cover shifts after you said no let it slide",
            choice_label="Just let it go",
            choice_value="a",
            reasoning_label="Not worth the drama",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.96,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="ob_push_direct",
            scenario_text="coworker keeps asking after no hold boundary firm",
            choice_label="Hold the line clearly",
            choice_value="d",
            reasoning_label="Restate your no calmly",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.85,
            correction_status="accurate",
        )
        prev = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet="say you cannot cover and keep it short",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        q2 = "same coworker is still pushing after i already said no what would i say"
        pr = generate_personal_response(q2, self.db, debug_phase44_carryover=True)
        dbg = pr.phase44_carryover_debug
        self.assertEqual(pr.effective_family, OBLIGATION_OVERLOAD)
        self.assertTrue(dbg["phase45_escalation_evaluated"])
        self.assertTrue(dbg["phase45_escalation_active"], msg=f"dbg={dbg!r}")
        self.assertEqual(dbg["phase45_escalation_subtype"], "boundary_push_repeat")
        low = pr.likely_answer.lower()
        self.assertNotIn("let it go", low, msg=pr.likely_answer)
        self.assertTrue(
            any(
                x in low
                for x in (
                    "no",
                    "boundary",
                    "line",
                    "direct",
                    "clear",
                    "say",
                    "hold",
                    "stop",
                )
            ),
            msg=pr.likely_answer,
        )

    def test_phase45_boundary_push_short_prompt_single_coherent_paragraph(self):
        """
        Acceptance-style prompt (answer_focus=both): no duplicate action + out-loud layers.
        """
        self.db.record_decision_memory(
            scenario_id="ob_push_avoid3",
            scenario_text="coworker asking to cover shifts after you said no let it slide",
            choice_label="Just let it go",
            choice_value="a",
            reasoning_label="Not worth the drama",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.96,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="ob_push_direct3",
            scenario_text="coworker keeps asking after no hold boundary firm",
            choice_label="Say no, explain you can't right now",
            choice_value="d",
            reasoning_label="I was already overloaded",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.85,
            correction_status="accurate",
        )
        prev = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet="say you cannot cover and keep it short",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        q_short = "same coworker is still pushing after i already said no"
        self.assertEqual(classify_answer_focus(normalize_input(q_short)), "both")
        pr = generate_personal_response(q_short, self.db, debug_phase44_carryover=True)
        dbg = pr.phase44_carryover_debug
        self.assertTrue(dbg["phase45_escalation_active"], msg=f"dbg={dbg!r}")
        self.assertEqual(dbg["phase45_escalation_subtype"], "boundary_push_repeat")
        low = pr.likely_answer.lower()
        self.assertLessEqual(
            low.count("mostly because"),
            1,
            msg=pr.likely_answer,
        )
        self.assertNotIn("if i said it out loud", low, msg=pr.likely_answer)
        self.assertNotIn("\n", (pr.likely_answer or "").strip(), msg=pr.likely_answer)
        self.assertTrue(
            any(
                x in low
                for x in (
                    "no",
                    "boundary",
                    "line",
                    "hold",
                    "say",
                )
            ),
            msg=pr.likely_answer,
        )
        ta = (pr.likely_answer or "").strip()
        if '"' in ta:
            lastq = ta.rindex('"')
            rest = ta[lastq + 1 :].strip()
            if rest.startswith("."):
                rest = rest[1:].strip()
            self.assertEqual(
                rest,
                "",
                msg=f"unexpected text after spoken quote (rationale leak): {ta!r}",
            )
        self.assertIn("already overloaded", low, msg=pr.likely_answer)

    def test_phase45_insufficient_evidence_carryover_row_does_not_dominate(self):
        """
        Regression: a prior respond-like-me insufficient-evidence row (same prompt hash)
        must not beat a real thread carryover row or force generic answer text.
        """
        self.db.record_decision_memory(
            scenario_id="ob_push_avoid2",
            scenario_text="coworker asking to cover shifts after you said no let it slide",
            choice_label="Just let it go",
            choice_value="a",
            reasoning_label="Not worth the drama",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.96,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="ob_push_direct2",
            scenario_text="coworker keeps asking after no hold boundary firm",
            choice_label="Hold the line clearly",
            choice_value="d",
            reasoning_label="Restate your no calmly",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.85,
            correction_status="accurate",
        )
        prev = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        ph_prev = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph_prev,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet="say you cannot cover and keep it short",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        q2 = "same coworker is still pushing after i already said no what would i say"
        q2_norm = normalize_input(q2)
        ph_q2 = hashlib.sha256(q2_norm.encode("utf-8")).hexdigest()
        sk_bad = combined_shape_key(
            q2_norm,
            OBLIGATION_OVERLOAD,
            {"route_keys": ["insufficient_evidence"], "clarif_slot_keys": []},
        )
        self.db.record_short_term_situation(
            prompt_norm=q2_norm,
            prompt_norm_hash=ph_q2,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=sk_bad,
            stance_snippet=(
                "I don't have enough saved decisions or style picks to say what you'd probably do here."
            ),
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        pr = generate_personal_response(q2, self.db, debug_phase44_carryover=True)
        dbg = pr.phase44_carryover_debug
        low = pr.likely_answer.lower()
        self.assertNotIn(
            "don't have enough saved decisions",
            low,
            msg=pr.likely_answer,
        )
        self.assertTrue(dbg["phase45_escalation_active"], msg=f"dbg={dbg!r}")
        pick = dbg["pick_best_for_ask"]
        bad_hints = [
            c
            for c in pick["candidates"]
            if "insufficient_evidence" in (c.get("stored_shape_key") or "")
        ]
        self.assertTrue(
            any(
                "excluded_low_information_route:insufficient_evidence"
                in c.get("zero_score_hints", [])
                for c in bad_hints
            ),
            msg=f"expected exclusion hint on fallback row, got {bad_hints!r}",
        )
        self.assertNotEqual(
            pick.get("best_row_id"),
            "",
            msg="expected a non-fallback carryover pick",
        )

    def test_phase45_same_tone_different_issue_no_escalation(self):
        """Regression C: new interpersonal annoyance is not the same escalated thread."""
        prev = normalize_input(
            "what would i say if someone is being passive aggressive at work"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="conflict",
            shape_key=carryover_shape_key(prev, "conflict"),
            stance_snippet="address it directly calmly",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        q2 = (
            "what would i say if my roommate keeps leaving dishes everywhere "
            "and it is getting under my skin"
        )
        pr = generate_personal_response(q2, self.db, debug_phase44_carryover=True)
        dbg = pr.phase44_carryover_debug
        self.assertFalse(dbg["phase45_escalation_active"], msg=f"dbg={dbg!r}")
        reasons = dbg.get("phase45_escalation_reject_reasons") or []
        self.assertTrue(
            ("no_situation_carryover_match" in reasons)
            or ("escalation_thread_alignment_failed" in reasons)
            or ("not_escalation_eligible_family" in reasons),
            msg=f"expected explicit reject path, got {reasons!r}",
        )

    def test_phase44_respond_unit_soft_band_suppress_and_audit(self):
        """Regression: ~0.34 strength + PA bridge must pass respond gates (not 0.52/0.50 walls)."""
        prev = normalize_input(
            "at work the person keeps making vague remarks that feel targeted"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        q = normalize_input(
            "what would i say if this same passive aggressive person keeps doing it"
        )
        qh = hashlib.sha256(q.encode("utf-8")).hexdigest()
        payload = {
            "strength": 0.345,
            "carry_family": "conflict",
            "match": {
                "id": "row-unit",
                "state": "unresolved",
                "updated_at": datetime.utcnow().isoformat(),
                "prompt_norm": prev,
                "prompt_norm_hash": ph,
                "shape_key": "obligation_overload|slots:",
                "stance_snippet": "probably try not to read into every remark",
            },
        }
        self.assertTrue(
            _respond_carryover_suppress_avoidance(payload, q),
            "PA-aligned carryover should demote avoidance at moderate strength",
        )
        audit = _respond_carryover_reasoning_line_audit(payload, q, 0.345, qh, "conflict")
        self.assertTrue(audit["allowed"], msg=audit)
        self.assertTrue(audit["conflict_pa_soft_continuity_path"])
        self.assertGreater(0.345, float(RESPOND_CARRYOVER_INFLUENCE_SOFT_MIN))

    def test_phase44_soft_carry_strength_end_to_end_cli_like_row(self):
        """
        Realistic row: pick passes with strength below old 0.50 respond walls (shape age),
        neutral stance text — still suppresses avoidance memory and emits continuity.
        """
        self.db.record_decision_memory(
            scenario_id="pa_avoid_soft",
            scenario_text=(
                "passive aggressive snide colleague remarks at work sideways digs"
            ),
            choice_label="Let it go and stay professional",
            choice_value="a",
            reasoning_label="Not worth the fight today",
            reasoning_value="b",
            value_tags=["patience"],
            trait_signals={},
            confidence_score=0.96,
            correction_status="accurate",
        )
        self.db.record_decision_memory(
            scenario_id="pa_direct_soft",
            scenario_text=(
                "passive aggressive teammate need to address calmly at work"
            ),
            choice_label="Address it directly but calmly",
            choice_value="d",
            reasoning_label="Stop the sideways pattern",
            reasoning_value="s",
            value_tags=["directness"],
            trait_signals={},
            confidence_score=0.72,
            correction_status="accurate",
        )
        prev = normalize_input(
            "at work the person keeps making vague remarks that feel targeted"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="conflict",
            shape_key=carryover_shape_key(prev, "conflict"),
            stance_snippet="probably try not to read into every remark",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        conn = self.db.get_db_connection()
        aged = (datetime.utcnow() - timedelta(hours=34)).isoformat()
        conn.execute(
            """
            UPDATE short_term_situation_memory
            SET updated_at = ?, shape_key = ?
            WHERE prompt_norm_hash = ?
            """,
            (aged, "obligation_overload|slots:", ph),
        )
        conn.commit()

        exact_q = (
            "what would i say if this same passive aggressive person keeps doing it"
        )
        pr = generate_personal_response(
            exact_q, self.db, debug_phase44_carryover=True
        )
        dbg = pr.phase44_carryover_debug
        self.assertIsNotNone(dbg)
        self.assertTrue(dbg["pick_best_for_ask"]["any_candidate_meets_threshold"])
        self.assertLess(
            float(dbg["strength_after_family_align"]),
            0.50,
            msg="test expects sub-0.50 carry strength (old respond wall)",
        )
        self.assertTrue(dbg["suppress_avoidance_demotion"])
        self.assertTrue(dbg["continuity_note_emitted"])
        self.assertEqual(dbg["continuity_language_tier"], "very_soft_template")
        self.assertTrue(dbg["reasoning_line_audit"]["conflict_pa_soft_continuity_path"])
        low = pr.likely_answer.lower()
        self.assertNotIn("let it go", low, msg=pr.likely_answer)

    def test_phase44_money_pressure_no_continuity_without_spending_carryover_debug(self):
        """Money-pressure respond must not claim session continuity from a conflict-only row."""
        prev = normalize_input(
            "what would i say if someone is passive aggressive at work"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="conflict",
            shape_key=carryover_shape_key(prev, "conflict"),
            stance_snippet="address it directly calmly",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        money_q = (
            "i need to pay rent tomorrow but i am broke should i skip groceries "
            "or going out first"
        )
        pr = generate_personal_response(
            money_q, self.db, debug_phase44_carryover=True
        )
        self.assertEqual(pr.effective_family, "spending")
        rb = (pr.reasoning_brief or "").lower()
        for needle in (
            "same issue continuing",
            "recent unresolved situation",
            "recent similar thread",
            "just working through",
        ):
            self.assertNotIn(needle, rb, msg=pr.reasoning_brief)
        dbg = pr.phase44_carryover_debug
        self.assertIsNotNone(dbg)
        self.assertFalse(dbg["continuity_note_emitted"])
        self.assertEqual(dbg["continuity_language_tier"], "none")
        self.assertEqual(
            dbg["reasoning_line_audit"]["blocked_by"],
            "no_situation_carryover_match",
        )
        self.assertFalse(dbg["phase45_escalation_evaluated"])
        self.assertFalse(dbg["phase45_escalation_active"])

    def test_obligation_coworker_respond_like_me_continuation_keeps_carryover_line(self):
        prev = normalize_input(
            "my coworker keeps pushing me to cover shifts when i am burnt out"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet="name your limit without over explaining",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        q2 = "same coworker is still pushing today what would i say to shut it down"
        pr = generate_personal_response(q2, self.db)
        self.assertEqual(pr.effective_family, OBLIGATION_OVERLOAD)
        rb = (pr.reasoning_brief or "").lower()
        self.assertTrue(
            any(
                x in rb
                for x in (
                    "same issue continuing",
                    "recent unresolved situation",
                    "similar thread",
                    "just working through",
                )
            ),
            msg=pr.reasoning_brief,
        )

    def test_surfaced_answer_uses_carryover_replacement_stance_after_wording_off(self):
        """After wording-off feedback replaces the carryover stance, the next
        respond-like-me on the same thread should surface that replacement
        snippet — not the original stitched answer."""
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        replacement = "I already told you no. I can't take that on right now."
        awkward = (
            "I was already overloaded. I'd hold the line on what I already said."
        )
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet=awkward,
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text=replacement,
            confidence_shown=0.55,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet=awkward[:200],
            feedback_target="wording",
        )
        q2 = "same coworker is still pushing after i already said no"
        pr = generate_personal_response(
            q2, self.db, debug_phase44_carryover=True
        )
        la = (pr.likely_answer or "").strip()
        self.assertIn("already told you no", la.lower())
        self.assertNotIn("already overloaded", la.lower())
        dbg = pr.phase44_carryover_debug
        self.assertIsNotNone(dbg)
        self.assertTrue(
            dbg["surfaced_answer_used_carryover_replacement_stance"]
        )
        self.assertTrue(dbg["feedback_replacement_available"])
        self.assertTrue(dbg["feedback_replacement_same_thread_relevant"])
        self.assertTrue(dbg["feedback_replacement_selected_for_surface"])
        self.assertEqual(dbg["feedback_replacement_block_reason"], "")
        self.assertTrue(
            dbg.get("older_phrasing_demoted_due_to_feedback"),
            msg=repr(dbg),
        )

    def test_phase46_close_same_thread_variant_prefers_replacement(self):
        """Near same-thread wording (not identical hash) still surfaces approved line."""
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        replacement = "I already told you no. I can't take that on right now."
        awkward = "I was already overloaded. I'd hold the line on what I already said."
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet=awkward,
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text=replacement,
            confidence_shown=0.55,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet=awkward[:200],
            feedback_target="wording",
        )
        q2 = "same coworker keeps pushing after i already said no"
        pr = generate_personal_response(
            q2, self.db, debug_phase44_carryover=True
        )
        la = (pr.likely_answer or "").strip().lower()
        self.assertIn("already told you no", la)
        self.assertNotIn("already overloaded", la)
        dbg = pr.phase44_carryover_debug
        self.assertIsNotNone(dbg)
        self.assertTrue(dbg["phase46_feedback_merge"]["merge_applied"])

    def test_phase46_unrelated_obligation_prompt_does_not_force_replacement(self):
        """Same family but no continuation cues: do not inject prior thread replacement."""
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        replacement = "I already told you no. I can't take that on right now."
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet="say you cannot cover and keep it short",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text=replacement,
            confidence_shown=0.55,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet="x",
            feedback_target="wording",
        )
        q2 = "my coworker is pushing me to cover shifts what would i say"
        pr = generate_personal_response(
            q2, self.db, debug_phase44_carryover=True
        )
        la = (pr.likely_answer or "").strip().lower()
        self.assertNotIn("already told you no", la)
        dbg = pr.phase44_carryover_debug
        self.assertIsNotNone(dbg)
        self.assertFalse(dbg["phase46_feedback_merge"]["merge_applied"])

    def test_phase46_obligation_replacement_does_not_bleed_into_conflict_prompt(self):
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        replacement = "I already told you no. I can't take that on right now."
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet="say you cannot cover",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text=replacement,
            confidence_shown=0.55,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet="x",
            feedback_target="wording",
        )
        q_conflict = (
            "what would i say if someone is being passive aggressive at work "
            "and keeps making digs"
        )
        pr = generate_personal_response(
            q_conflict, self.db, debug_phase44_carryover=True
        )
        la = (pr.likely_answer or "").strip().lower()
        self.assertNotIn("already told you no", la)
        self.assertNotIn("can't take that on right now", la)

    def test_phase46_newer_approved_replacement_beats_older_feedback_row(self):
        prev = normalize_input(
            "coworker keeps asking cover shifts after i said no what would i say"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet="stale awkward carryover stance line here",
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text="older replacement line should not win",
            confidence_shown=0.5,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet="a",
            feedback_target="wording",
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text="newer approved replacement line wins here",
            confidence_shown=0.5,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet="b",
            feedback_target="wording",
        )
        q2 = "same coworker still pushing after i said no what would i say"
        pr = generate_personal_response(q2, self.db)
        low = (pr.likely_answer or "").lower()
        self.assertIn("newer approved replacement line wins here", low)
        self.assertNotIn("older replacement line should not win", low)

    def test_phase47_exact_replay_stable_after_wording_off(self):
        """Repeated same prompt must keep user-approved replacement on the surface."""
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        replacement = "I already told you no. I can't take that on right now."
        awkward = (
            "I was already overloaded. I'd hold the line on what I already said."
        )
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet=awkward,
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text=replacement,
            confidence_shown=0.55,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet=awkward[:200],
            feedback_target="wording",
        )
        pr1 = generate_personal_response(prev, self.db, debug_phase44_carryover=True)
        pr2 = generate_personal_response(prev, self.db, debug_phase44_carryover=True)
        for pr in (pr1, pr2):
            la = (pr.likely_answer or "").lower()
            self.assertIn("already told you no", la)
            self.assertNotIn("already overloaded", la)
        self.assertTrue(pr2.phase44_carryover_debug["replay_exact_prompt_match"])
        self.assertTrue(pr2.phase44_carryover_debug["replay_used_existing_corrected_stance"])

    def test_phase47_my_coworker_pushing_again_variant_uses_approved_line(self):
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        replacement = "I already told you no. I can't take that on right now."
        awkward = "I was already overloaded. I'd hold the line on what I already said."
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet=awkward,
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text=replacement,
            confidence_shown=0.55,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet=awkward[:200],
            feedback_target="wording",
        )
        q2 = normalize_input(
            "my coworker is pushing again after i already told them no"
        )
        pr = generate_personal_response(q2, self.db, debug_phase44_carryover=True)
        la = (pr.likely_answer or "").lower()
        self.assertIn("already told you no", la)
        self.assertNotIn("already overloaded", la)
        dbg = pr.phase44_carryover_debug
        self.assertTrue(dbg["phase46_feedback_merge"]["merge_applied"])
        self.assertEqual(
            dbg["phase46_feedback_merge"]["merge_block_reason"],
            "",
        )
        self.assertTrue(dbg["replay_same_thread_variant_match"])
        self.assertTrue(
            dbg["phase46_feedback_merge"]["replacement_thread_gate_ok"]
        )

    def test_phase47_near_variant_uses_stm_stance_when_variant_hash_has_no_feedback(
        self,
    ):
        """
        Carryover winner can be a thread variant whose stance_snippet already holds
        the user-approved line (from prior replay) while personal_response_feedback
        rows still live only on an older prompt hash. Phase 47 must inherit from
        stance_snippet under the same-thread gate.
        """
        prev_variant = normalize_input(
            "same coworker keeps pushing after i already said no"
        )
        ph_var = hashlib.sha256(prev_variant.encode("utf-8")).hexdigest()
        stored_line = "I already told you no. I can't take that on right now."
        self.db.record_short_term_situation(
            prompt_norm=prev_variant,
            prompt_norm_hash=ph_var,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev_variant, OBLIGATION_OVERLOAD),
            stance_snippet=stored_line,
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        infl = build_respond_feedback_influence(
            self.db, ph_var, OBLIGATION_OVERLOAD
        )
        self.assertFalse(infl.action_ok_wording_off)
        q2 = normalize_input(
            "my coworker is pushing again after i already told them no"
        )
        pr = generate_personal_response(q2, self.db, debug_phase44_carryover=True)
        la = (pr.likely_answer or "").lower()
        self.assertIn("already told you no", la)
        self.assertNotIn("same way as before", la)
        dbg = pr.phase44_carryover_debug
        self.assertTrue(dbg["phase46_feedback_merge"]["merge_applied"])
        self.assertTrue(dbg["phase46_feedback_merge"]["stance_snippet_carryover_merge"])
        self.assertEqual(
            dbg["phase46_feedback_merge"]["merge_source"],
            "carryover_row_stored_stance_snippet_phase47",
        )
        self.assertIn(
            "stance",
            dbg["phase46_feedback_merge"]["corrected_stance_inheritance_reason"],
        )

    def test_phase47_stm_stance_prefers_resolved_replacement_over_wrong_inject(self):
        """STM recording must not prefer wrong+replacement inject over action-ok line."""
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        approved = "USER APPROVED REPLACEMENT LINE FOR STM RECORD"
        awkward = "awkward stitched surface line for test"
        self.db.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
            shape_key=carryover_shape_key(prev, OBLIGATION_OVERLOAD),
            stance_snippet=awkward,
            source="respond_like_me",
            asked_slots_json=json.dumps([]),
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="wrong",
            partial_aspect="",
            replacement_text="WRONG INJECT LINE SHOULD NOT WIN STM",
            confidence_shown=0.5,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet="x",
            feedback_target="wording",
        )
        self.db.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text=approved,
            confidence_shown=0.55,
            effective_family=OBLIGATION_OVERLOAD,
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet=awkward[:200],
            feedback_target="wording",
        )
        generate_personal_response(prev, self.db)
        latest = self.db.get_latest_unresolved_short_term_situation_row(
            prompt_norm_hash=ph,
            effective_family=OBLIGATION_OVERLOAD,
        )
        self.assertIsNotNone(latest)
        st = (latest.get("stance_snippet") or "").strip()
        self.assertIn("USER APPROVED", st)
        self.assertNotIn("WRONG INJECT", st)

    def test_spending_realism_pass_no_mixed_pronouns_on_needs_clause(self):
        raw = (
            "You'd likely separate what you truly need from what you want right now "
            "cover the roof first"
        )
        out = _phase41_style_realism_pass(raw, answer_focus="both")
        ol = out.lower()
        self.assertNotIn("what you truly need", ol)

    def test_coworker_continuation_still_obligation_primary(self):
        ordered, _ = rank_families(
            normalize_input(
                "same coworker is still pushing today what should i do now"
            )
        )
        self.assertEqual(ordered[0][0], OBLIGATION_OVERLOAD)


if __name__ == "__main__":
    unittest.main()
