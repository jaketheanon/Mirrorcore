"""Phase 44: short-term situation carryover (deterministic continuation)."""

import hashlib
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore
from mirrorcore.decision.routed_clarification import pick_next_question
from mirrorcore.persona.respond import build_respond_feedback_influence
from mirrorcore.decision.situation_carryover import (
    boundary_carryover_aligned,
    carryover_safe_stance_fallback,
    carryover_shape_key,
    carryover_slots_prefix,
    combined_shape_key,
    conflict_escalation_carryover_thread_ok,
    continuation_strength,
    diagnose_ask_carryover_candidates,
    feedback_replacement_same_thread_gate,
    low_information_carryover_row_exclusion_reason,
    pick_best_carryover,
    pick_best_carryover_for_ask,
    reference_continuation_cues,
    respond_route_keys_skip_short_term_situation_record,
    significant_tokens,
)
from mirrorcore.decision.ontology import GENERAL, OBLIGATION_OVERLOAD
from mirrorcore.router import normalize_input


class TestSituationCarryover(unittest.TestCase):
    def test_same_hash_strong_match(self):
        now = datetime(2026, 4, 1, 12, 0, 0)
        row = {
            "state": "unresolved",
            "updated_at": now.isoformat(),
            "effective_family": "spending",
            "prompt_norm": "should i buy laptop rent is late",
            "prompt_norm_hash": "abc",
            "shape_key": "spending|slots:x",
            "stance_snippet": "check bills first",
        }
        s = continuation_strength(
            "should i buy laptop rent is late",
            "abc",
            "spending",
            "spending|slots:",
            row,
            now=now,
        )
        self.assertGreaterEqual(s, 0.8)

    def test_same_family_unrelated_tokens_no_match(self):
        now = datetime(2026, 4, 1, 12, 0, 0)
        row = {
            "state": "unresolved",
            "updated_at": now.isoformat(),
            "effective_family": "spending",
            "prompt_norm": "should i buy a new laptop when rent is late",
            "prompt_norm_hash": "h1",
            "shape_key": "spending|slots:a,b",
            "stance_snippet": "basics first",
        }
        s = continuation_strength(
            "should i buy fancy coffee every morning",
            "h2",
            "spending",
            "spending|slots:c",
            row,
            now=now,
        )
        self.assertEqual(s, 0.0)

    def test_timing_contradiction_weakens(self):
        now = datetime(2026, 4, 1, 12, 0, 0)
        row = {
            "state": "unresolved",
            "updated_at": now.isoformat(),
            "effective_family": "risk_timing",
            "prompt_norm": (
                "still deciding whether to wait on the job offer or act slowly"
            ),
            "prompt_norm_hash": "x1",
            "shape_key": "risk_timing|slots:",
            "stance_snippet": "waiting a week felt right before",
        }
        s = continuation_strength(
            "have to act on the job offer today cannot wait anymore same offer",
            "x2",
            "risk_timing",
            "risk_timing|slots:",
            row,
            now=now,
        )
        self.assertGreater(s, 0.05)
        self.assertLess(s, 0.55)

    def test_pick_best_respects_unresolved_only(self):
        rows = [
            {
                "id": "1",
                "state": "superseded",
                "updated_at": "2026-04-01T12:00:00",
                "effective_family": "spending",
                "prompt_norm": "should i buy laptop rent is late",
                "prompt_norm_hash": "a",
                "shape_key": "spending|slots:",
                "stance_snippet": "",
            },
            {
                "id": "2",
                "state": "unresolved",
                "updated_at": "2026-04-01T12:00:00",
                "effective_family": "spending",
                "prompt_norm": "should i buy laptop rent is late",
                "prompt_norm_hash": "b",
                "shape_key": "spending|slots:",
                "stance_snippet": "",
            },
        ]
        best, strength = pick_best_carryover(
            "should i buy laptop rent is late",
            "b",
            "spending",
            "spending|slots:",
            rows,
            now=datetime(2026, 4, 1, 12, 0, 0),
        )
        self.assertIsNotNone(best)
        self.assertEqual(best["id"], "2")
        self.assertGreater(strength, 0.5)

    def test_store_record_and_supersede(self):
        p = Path(__file__).parent / "_tmp_phase44_carry.db"
        if p.exists():
            p.unlink()
        store = DatabaseStore(p)
        store.initialize_database()
        store.record_short_term_situation(
            prompt_norm="coworker keeps pushing boundaries exhausted",
            prompt_norm_hash="h1",
            effective_family="obligation_overload",
            shape_key=carryover_shape_key(
                "coworker keeps pushing boundaries exhausted", "obligation_overload"
            ),
            stance_snippet="energy check",
            source="ask",
            asked_slots_json=json.dumps(["energy_capacity"]),
        )
        rows = store.list_recent_short_term_situations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "unresolved")
        store.record_short_term_situation(
            prompt_norm="coworker still pushing boundaries and im exhausted",
            prompt_norm_hash="h2",
            effective_family="obligation_overload",
            shape_key=carryover_shape_key(
                "coworker still pushing boundaries and im exhausted",
                "obligation_overload",
            ),
            stance_snippet="follow up",
            source="ask",
            asked_slots_json=json.dumps(["energy_capacity", "guilt_axis"]),
        )
        rows2 = store.list_recent_short_term_situations()
        states = {r["id"]: r["state"] for r in rows2}
        self.assertIn("superseded", states.values())
        self.assertIn("unresolved", states.values())
        store.close()
        if p.exists():
            p.unlink()

    def test_store_low_information_insert_does_not_supersede_prior_thread_row(self):
        """Fallback respond shape must not mark a real unresolved row superseded."""
        p = Path(__file__).parent / "_tmp_phase45_lowinfo.db"
        if p.exists():
            p.unlink()
        store = DatabaseStore(p)
        store.initialize_database()
        prev_norm = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        cur_norm = normalize_input(
            "same coworker is still pushing after i already said no what would i say"
        )
        store.record_short_term_situation(
            prompt_norm=prev_norm,
            prompt_norm_hash="h_prev",
            effective_family="obligation_overload",
            shape_key=carryover_shape_key(prev_norm, "obligation_overload"),
            stance_snippet="say you cannot cover",
            source="respond_like_me",
        )
        sk_bad = combined_shape_key(
            cur_norm,
            "obligation_overload",
            {"route_keys": ["insufficient_evidence"], "clarif_slot_keys": []},
        )
        store.record_short_term_situation(
            prompt_norm=cur_norm,
            prompt_norm_hash="h_cur",
            effective_family="obligation_overload",
            shape_key=sk_bad,
            stance_snippet="I don't have enough saved decisions",
            source="respond_like_me",
        )
        rows = store.list_recent_short_term_situations()
        by_stance = {r["stance_snippet"][:24]: r["state"] for r in rows}
        self.assertEqual(by_stance.get("say you cannot cover"), "unresolved")
        store.close()
        if p.exists():
            p.unlink()

    def test_pick_next_question_deprioritizes_prior_slots(self):
        slot = pick_next_question(
            context_parts=["colleague asked me to cover a shift tomorrow im drained"],
            asked_ids=[],
            domain_order=[OBLIGATION_OVERLOAD, GENERAL],
            deprioritize_slot_ids=["energy_capacity"],
        )
        self.assertIsNotNone(slot)
        self.assertNotEqual(slot.slot_id, "energy_capacity")

    def test_significant_tokens_filters_short(self):
        t = significant_tokens("a an the cat")
        self.assertIn("cat", t)
        self.assertNotIn("the", t)

    def test_reference_cues_same_coworker_still_pushing(self):
        n = normalize_input(
            "same coworker is still pushing today what should i do now"
        )
        self.assertTrue(reference_continuation_cues(n))

    def test_carryover_slots_prefix_ignores_evidence_tail(self):
        full = "conflict|slots:a,b|r:foo;bar|c:x"
        cur = "conflict|slots:a,b"
        self.assertEqual(
            carryover_slots_prefix(full),
            carryover_slots_prefix(cur),
        )

    def test_pick_best_carryover_for_ask_scores_with_row_family_not_timing_guess(self):
        now = datetime(2026, 4, 2, 14, 0, 0)
        row_norm = normalize_input(
            "my coworker keeps pushing after i already said no"
        )
        row = {
            "id": "r1",
            "state": "unresolved",
            "updated_at": "2026-04-02T12:00:00",
            "effective_family": "obligation_overload",
            "prompt_norm": row_norm,
            "prompt_norm_hash": "hprev",
            "shape_key": carryover_shape_key(row_norm, "obligation_overload"),
            "stance_snippet": "hold the line on what you already said",
        }
        cur = normalize_input(
            "same coworker is still pushing today what should i do now"
        )
        h = hashlib.sha256(cur.encode("utf-8")).hexdigest()
        best, strength, fam = pick_best_carryover_for_ask(cur, h, [row], now=now)
        self.assertIsNotNone(best)
        self.assertGreaterEqual(strength, 0.38)
        self.assertEqual(fam, "obligation_overload")
        self.assertEqual(best["id"], "r1")

    def test_pick_best_for_ask_prefers_engaged_stance_for_pa_same_person_continuation(
        self,
    ):
        now = datetime(2026, 4, 2, 15, 0, 0)
        row_norm_avoid = normalize_input(
            "what would i say if someone is being passive aggressive at work"
        )
        row_norm_direct = normalize_input(
            "passive aggressive digs at work need to address calmly not ignore"
        )
        row_avoid = {
            "id": "avoid",
            "state": "unresolved",
            "updated_at": "2026-04-02T14:00:00",
            "effective_family": "conflict",
            "prompt_norm": row_norm_avoid,
            "prompt_norm_hash": "ha",
            "shape_key": carryover_shape_key(row_norm_avoid, "conflict"),
            "stance_snippet": "let it go and move on professionally",
        }
        row_direct = {
            "id": "direct",
            "state": "unresolved",
            "updated_at": "2026-04-02T14:05:00",
            "effective_family": "conflict",
            "prompt_norm": row_norm_direct,
            "prompt_norm_hash": "hd",
            "shape_key": carryover_shape_key(row_norm_direct, "conflict"),
            "stance_snippet": "address it directly name what you noticed stay calm",
        }
        cur = normalize_input(
            "what would i say if this same passive aggressive person keeps doing it"
        )
        h = hashlib.sha256(cur.encode("utf-8")).hexdigest()
        best, strength, fam = pick_best_carryover_for_ask(
            cur, h, [row_avoid, row_direct], now=now
        )
        self.assertIsNotNone(best)
        self.assertGreaterEqual(strength, 0.38)
        self.assertEqual(fam, "conflict")
        self.assertEqual(best["id"], "direct")

    def test_boundary_carryover_aligned_same_coworker_push_thread(self):
        prev = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        cur = normalize_input(
            "same coworker is still pushing after i already said no what would i say"
        )
        self.assertTrue(boundary_carryover_aligned(cur, prev))

    def test_boundary_carryover_aligned_false_without_explicit_continuation(self):
        prev = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        cur = normalize_input(
            "my coworker is pushing me to cover shifts what would i say"
        )
        self.assertFalse(boundary_carryover_aligned(cur, prev))

    def test_feedback_replacement_same_thread_gate_same_hash(self):
        prev = normalize_input("coworker cover shifts after i said no")
        h = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        row = {
            "state": "unresolved",
            "updated_at": "2026-04-05T12:00:00",
            "effective_family": "obligation_overload",
            "prompt_norm": prev,
            "prompt_norm_hash": h,
            "shape_key": carryover_shape_key(prev, "obligation_overload"),
            "stance_snippet": "x",
        }
        ok, reason = feedback_replacement_same_thread_gate(
            prev,
            prev,
            h,
            h,
            carryover_shape_key(prev, "obligation_overload"),
            row["shape_key"],
            row,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "same_prompt_hash")

    def test_feedback_replacement_same_thread_gate_false_without_continuation(self):
        prev = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        cur = normalize_input(
            "my coworker is pushing me to cover shifts what would i say"
        )
        h_prev = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        h_cur = hashlib.sha256(cur.encode("utf-8")).hexdigest()
        row = {
            "state": "unresolved",
            "updated_at": "2026-04-05T12:00:00",
            "effective_family": "obligation_overload",
            "prompt_norm": prev,
            "prompt_norm_hash": h_prev,
            "shape_key": carryover_shape_key(prev, "obligation_overload"),
            "stance_snippet": "y",
        }
        ok, reason = feedback_replacement_same_thread_gate(
            cur,
            prev,
            h_cur,
            h_prev,
            carryover_shape_key(cur, "obligation_overload"),
            row["shape_key"],
            row,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "no_continuation_cues")

    def test_feedback_replacement_same_thread_gate_true_for_close_obligation_thread(
        self,
    ):
        prev = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        cur = normalize_input(
            "same coworker is still pushing after i already said no what would i say"
        )
        h_prev = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        h_cur = hashlib.sha256(cur.encode("utf-8")).hexdigest()
        row = {
            "state": "unresolved",
            "updated_at": "2026-04-05T12:00:00",
            "effective_family": "obligation_overload",
            "prompt_norm": prev,
            "prompt_norm_hash": h_prev,
            "shape_key": carryover_shape_key(prev, "obligation_overload"),
            "stance_snippet": "z",
        }
        ok, reason = feedback_replacement_same_thread_gate(
            cur,
            prev,
            h_cur,
            h_prev,
            carryover_shape_key(cur, "obligation_overload"),
            row["shape_key"],
            row,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "thread_gate_escalation_ok")

    def test_conflict_escalation_thread_ok_false_across_unrelated_domains(self):
        prev = normalize_input("should i buy a laptop when rent is late")
        cur = normalize_input(
            "what would i say if this same passive aggressive person keeps doing it"
        )
        self.assertFalse(
            conflict_escalation_carryover_thread_ok(cur, prev, "h1", "h2")
        )

    def test_continuation_strength_zero_for_insufficient_evidence_shape(self):
        now = datetime(2026, 4, 3, 10, 0, 0)
        q = normalize_input(
            "same coworker is still pushing after i already said no what would i say"
        )
        h = hashlib.sha256(q.encode("utf-8")).hexdigest()
        sk_bad = combined_shape_key(
            q,
            "obligation_overload",
            {"route_keys": ["insufficient_evidence"], "clarif_slot_keys": []},
        )
        row = {
            "state": "unresolved",
            "updated_at": now.isoformat(),
            "effective_family": "obligation_overload",
            "prompt_norm": q,
            "prompt_norm_hash": h,
            "shape_key": sk_bad,
            "stance_snippet": (
                "I don't have enough saved decisions or style picks to say what you'd probably do here."
            ),
        }
        self.assertEqual(
            low_information_carryover_row_exclusion_reason(row),
            "excluded_low_information_route:insufficient_evidence",
        )
        s = continuation_strength(
            q,
            h,
            "obligation_overload",
            carryover_shape_key(q, "obligation_overload"),
            row,
            now=now,
        )
        self.assertEqual(s, 0.0)

    def test_pick_best_carryover_prefers_useful_row_over_same_hash_fallback(self):
        """Same-hash insufficient-evidence row must not beat a real thread row."""
        now = datetime(2026, 4, 3, 11, 0, 0)
        q = normalize_input(
            "same coworker is still pushing after i already said no what would i say"
        )
        h = hashlib.sha256(q.encode("utf-8")).hexdigest()
        prev = normalize_input(
            "what would i say when my coworker keeps asking me to cover shifts after i said no"
        )
        sk_bad = combined_shape_key(
            q,
            "obligation_overload",
            {"route_keys": ["insufficient_evidence"], "clarif_slot_keys": []},
        )
        row_bad = {
            "id": "bad",
            "state": "unresolved",
            "updated_at": now.isoformat(),
            "effective_family": "obligation_overload",
            "prompt_norm": q,
            "prompt_norm_hash": h,
            "shape_key": sk_bad,
            "stance_snippet": "I don't have enough saved decisions",
        }
        row_good = {
            "id": "good",
            "state": "unresolved",
            "updated_at": now.isoformat(),
            "effective_family": "obligation_overload",
            "prompt_norm": prev,
            "prompt_norm_hash": "otherhash",
            "shape_key": carryover_shape_key(prev, "obligation_overload"),
            "stance_snippet": "say you cannot cover and keep it short",
        }
        best, strength = pick_best_carryover(
            q,
            h,
            "obligation_overload",
            carryover_shape_key(q, "obligation_overload"),
            [row_bad, row_good],
            now=now,
        )
        self.assertIsNotNone(best)
        self.assertEqual(best["id"], "good")
        self.assertGreater(strength, 0.25)

    def test_diagnose_carryover_marks_excluded_fallback_row(self):
        now = datetime(2026, 4, 3, 12, 0, 0)
        q = normalize_input("same coworker is still pushing after i already said no")
        h = hashlib.sha256(q.encode("utf-8")).hexdigest()
        sk_bad = combined_shape_key(
            q,
            "obligation_overload",
            {"route_keys": ["insufficient_evidence"], "clarif_slot_keys": []},
        )
        row_bad = {
            "id": "x1",
            "state": "unresolved",
            "updated_at": now.isoformat(),
            "effective_family": "obligation_overload",
            "prompt_norm": q,
            "prompt_norm_hash": h,
            "shape_key": sk_bad,
            "stance_snippet": "generic",
        }
        diag = diagnose_ask_carryover_candidates(q, h, [row_bad], now=now)
        c0 = diag["candidates"][0]
        self.assertIn("excluded_low_information_route:insufficient_evidence", c0["zero_score_hints"])
        self.assertFalse(diag["any_candidate_meets_threshold"])

    def test_respond_skip_short_term_record_reason_for_insufficient_evidence(self):
        r = respond_route_keys_skip_short_term_situation_record(
            ("insufficient_evidence",)
        )
        self.assertEqual(r, "skip_short_term_record_low_information_route:insufficient_evidence")
        self.assertIsNone(
            respond_route_keys_skip_short_term_situation_record(("strong_decision",))
        )

    def test_carryover_safe_stance_fallback_avoids_hold_the_line_phrasing(self):
        pn = normalize_input(
            "same coworker is still pushing after i already said no what would i say"
        )
        s = carryover_safe_stance_fallback(pn, "obligation_overload").lower()
        self.assertNotIn("hold the line", s)
        self.assertIn("no", s)

    def test_action_ok_word_bad_feedback_rewrites_short_term_stance_snippet(self):
        p = Path(__file__).parent / "_tmp_phase45_wordoff.db"
        if p.exists():
            p.unlink()
        store = DatabaseStore(p)
        store.initialize_database()
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        awkward = (
            "I was already overloaded. I'd hold the line on what I already said."
        )
        store.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="obligation_overload",
            shape_key=carryover_shape_key(prev, "obligation_overload"),
            stance_snippet=awkward,
            source="respond_like_me",
        )
        store.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text="No, I can't cover that shift.",
            confidence_shown=0.55,
            effective_family="obligation_overload",
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet=awkward[:200],
            feedback_target="wording",
        )
        rows = store.list_recent_short_term_situations()
        hit = [r for r in rows if r.get("prompt_norm_hash") == ph]
        self.assertEqual(len(hit), 1)
        st = (hit[0].get("stance_snippet") or "").lower()
        self.assertNotIn("already overloaded", st)
        self.assertNotIn("hold the line", st)
        self.assertIn("can't cover", st)
        infl = build_respond_feedback_influence(store, ph, "obligation_overload")
        self.assertTrue(infl.action_ok_wording_off)
        self.assertIn(
            "can't cover",
            (infl.action_ok_wording_replacement_line or "").lower(),
        )
        store.close()
        if p.exists():
            p.unlink()

    def test_continuation_pick_uses_clean_stance_after_word_off_rewrite(self):
        """Repeated-boundary follow-up must not top-rank awkward prior surface wording."""
        p = Path(__file__).parent / "_tmp_phase45_wordoff2.db"
        if p.exists():
            p.unlink()
        store = DatabaseStore(p)
        store.initialize_database()
        prev = normalize_input(
            "what would i say coworker keeps asking me to cover shifts after i said no"
        )
        ph = hashlib.sha256(prev.encode("utf-8")).hexdigest()
        awkward = (
            "I was already overloaded. I'd hold the line on what I already said."
        )
        store.record_short_term_situation(
            prompt_norm=prev,
            prompt_norm_hash=ph,
            effective_family="obligation_overload",
            shape_key=carryover_shape_key(prev, "obligation_overload"),
            stance_snippet=awkward,
            source="respond_like_me",
        )
        store.record_personal_response_feedback(
            scenario_snippet=prev[:220],
            prompt_norm_hash=ph,
            rating="partly",
            partial_aspect="action_ok_word_bad",
            replacement_text="",
            confidence_shown=0.55,
            effective_family="obligation_overload",
            evidence_path={"route_keys": ["strong_decision"]},
            likely_answer_snippet=awkward[:200],
            feedback_target="wording",
        )
        q = normalize_input(
            "same coworker is still pushing after i already said no what would i say"
        )
        qh = hashlib.sha256(q.encode("utf-8")).hexdigest()
        rows = store.list_recent_short_term_situations()
        best, strength = pick_best_carryover(
            q,
            qh,
            "obligation_overload",
            carryover_shape_key(q, "obligation_overload"),
            rows,
        )
        self.assertIsNotNone(best)
        self.assertGreater(strength, 0.25)
        st = (best.get("stance_snippet") or "").lower()
        self.assertNotIn("already overloaded", st)
        self.assertNotIn("hold the line", st)
        self.assertIn("same clear no", st)
        store.close()
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    unittest.main()
