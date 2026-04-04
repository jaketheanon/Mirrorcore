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
from mirrorcore.decision.situation_carryover import (
    carryover_shape_key,
    carryover_slots_prefix,
    continuation_strength,
    pick_best_carryover,
    pick_best_carryover_for_ask,
    reference_continuation_cues,
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


if __name__ == "__main__":
    unittest.main()
