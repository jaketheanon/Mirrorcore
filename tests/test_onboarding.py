"""
Tests for Phase 30: guided onboarding, progress summary, and menu routing.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.db.store import DatabaseStore
from mirrorcore.main import create_parser, handle_analyze_followup, handle_analyze_log, handle_calibrate_style, handle_interview, handle_respond_like_me, handle_start
from mirrorcore.onboarding import (
    OnboardingSnapshot,
    continuation_hints,
    decision_model_label,
    format_progress_summary,
    load_onboarding_snapshot,
    personal_response_label,
    primary_next_suggestion,
    run_start_menu,
    style_model_label,
    troubleshooting_label,
)


class TestOnboardingLabels(unittest.TestCase):
    def test_decision_model_tiers(self):
        self.assertEqual(decision_model_label(0), "not started")
        self.assertEqual(decision_model_label(1), "started")
        self.assertEqual(decision_model_label(3), "started")
        self.assertEqual(decision_model_label(4), "usable")

    def test_style_model_tiers(self):
        self.assertEqual(style_model_label(0), "not started")
        self.assertEqual(style_model_label(3), "started")
        self.assertEqual(style_model_label(4), "usable")

    def test_personal_response_label(self):
        self.assertEqual(personal_response_label(0, 0), "not ready")
        self.assertEqual(personal_response_label(1, 1), "not ready")
        self.assertEqual(personal_response_label(2, 2), "ready to try")
        self.assertEqual(personal_response_label(4, 0), "ready to try")

    def test_troubleshooting_label(self):
        self.assertEqual(troubleshooting_label(0), "not yet used")
        self.assertEqual(troubleshooting_label(1), "available")


class TestOnboardingSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db_path = Path(self.tmp.name)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    def test_load_snapshot_counts(self):
        store = DatabaseStore(self.db_path)
        try:
            store.initialize_database()
            store.record_decision_memory(
                scenario_id="s1",
                scenario_text="t",
                choice_label="A",
                choice_value="a",
                reasoning_label="R",
                reasoning_value="r",
                value_tags=[],
                trait_signals={},
            )
            store.record_style_memory(
                prompt_id="p1",
                prompt_text="x",
                selected_label="L",
                selected_value="v",
                style_tags=[],
                tone_signals={},
            )
            store.save_terminal_incident(
                command="c",
                error_message="e",
                root_cause="rc",
                solution="sol",
                success=False,
            )
            snap = load_onboarding_snapshot(store)
            self.assertEqual(snap.decision_count, 1)
            self.assertEqual(snap.style_count, 1)
            self.assertEqual(snap.terminal_incidents, 1)
            self.assertFalse(snap.has_open_analysis_session)
        finally:
            store.close()

    def test_store_row_counts(self):
        store = DatabaseStore(self.db_path)
        try:
            store.initialize_database()
            self.assertEqual(store.count_decision_memory_rows(), 0)
            self.assertEqual(store.count_style_memory_rows(), 0)
            store.record_decision_memory(
                scenario_id="s1",
                scenario_text="t",
                choice_label="A",
                choice_value="a",
                reasoning_label="R",
                reasoning_value="r",
                value_tags=[],
                trait_signals={},
            )
            self.assertEqual(store.count_decision_memory_rows(), 1)
        finally:
            store.close()


class TestProgressAndHints(unittest.TestCase):
    def test_format_progress_includes_baseline_when_no_persona(self):
        snap = OnboardingSnapshot(
            persona_exists=False,
            decision_count=0,
            style_count=0,
            terminal_incidents=0,
            has_open_analysis_session=False,
        )
        text = "\n".join(format_progress_summary(snap))
        self.assertIn("not started", text)
        self.assertIn("Baseline profile", text)

    def test_primary_next_suggestion_open_session(self):
        snap = OnboardingSnapshot(
            persona_exists=True,
            decision_count=10,
            style_count=10,
            terminal_incidents=1,
            has_open_analysis_session=True,
        )
        self.assertIn("investigation", (primary_next_suggestion(snap) or "").lower())

    def test_continuation_partial_decision(self):
        snap = OnboardingSnapshot(
            persona_exists=True,
            decision_count=2,
            style_count=0,
            terminal_incidents=0,
            has_open_analysis_session=False,
        )
        h = continuation_hints(snap)
        self.assertTrue(any("decision" in x.lower() for x in h))


class TestStartMenuRouting(unittest.TestCase):
    def test_menu_routes_to_interview(self):
        calls = []

        def fake_interview(a):
            calls.append("interview")

        def fake_calibrate(a):
            calls.append("calibrate")

        def fake_respond(a):
            calls.append("respond")

        def fake_log(a):
            calls.append("log")

        def fake_followup(a):
            calls.append("followup")

        store = MagicMock()
        store.get_database_stats.return_value = {"terminal_incidents": 0}
        store.get_latest_analysis_session.return_value = None
        store.persona_exists.return_value = True
        store.count_decision_memory_rows.return_value = 0
        store.count_style_memory_rows.return_value = 0

        args = MagicMock()
        choices = iter(["1", "6"])

        def read_choice(_):
            return next(choices)

        run_start_menu(
            store,
            handle_interview=fake_interview,
            handle_calibrate_style=fake_calibrate,
            handle_respond_like_me=fake_respond,
            handle_analyze_log=fake_log,
            handle_analyze_followup=fake_followup,
            args_namespace=args,
            read_choice=read_choice,
        )
        self.assertEqual(calls, ["interview"])

    def test_menu_routes_respond(self):
        calls = []

        store = MagicMock()
        store.get_database_stats.return_value = {"terminal_incidents": 0}
        store.get_latest_analysis_session.return_value = None
        store.persona_exists.return_value = True
        store.count_decision_memory_rows.return_value = 0
        store.count_style_memory_rows.return_value = 0

        choices = iter(["3", "6"])

        def read_choice(_):
            return next(choices)

        run_start_menu(
            store,
            handle_interview=lambda a: calls.append("i"),
            handle_calibrate_style=lambda a: calls.append("c"),
            handle_respond_like_me=lambda a: calls.append("r"),
            handle_analyze_log=lambda a: calls.append("l"),
            handle_analyze_followup=lambda a: calls.append("f"),
            args_namespace=MagicMock(),
            read_choice=read_choice,
        )
        self.assertEqual(calls, ["r"])

    def test_continue_submenu_followup(self):
        calls = []
        store = MagicMock()
        store.get_database_stats.return_value = {"terminal_incidents": 0}
        store.get_latest_analysis_session.return_value = {"id": "x", "session_status": "active"}
        store.persona_exists.return_value = True
        store.count_decision_memory_rows.return_value = 0
        store.count_style_memory_rows.return_value = 0

        choices = iter(["5", "1", "6"])

        def read_choice(_):
            return next(choices)

        run_start_menu(
            store,
            handle_interview=lambda a: calls.append("i"),
            handle_calibrate_style=lambda a: calls.append("c"),
            handle_respond_like_me=lambda a: calls.append("r"),
            handle_analyze_log=lambda a: calls.append("l"),
            handle_analyze_followup=lambda a: calls.append("followup"),
            args_namespace=MagicMock(),
            read_choice=read_choice,
        )
        self.assertIn("followup", calls)


class TestCliIntegration(unittest.TestCase):
    def test_parser_has_start_subcommand(self):
        p = create_parser()
        args = p.parse_args(["start"])
        self.assertEqual(args.command, "start")

    def test_existing_commands_unchanged(self):
        p = create_parser()
        for cmd, extra in [
            ("interview", []),
            ("calibrate-style", []),
            ("respond-like-me", ["hello"]),
            ("analyze-log", []),
            ("analyze-followup", []),
        ]:
            with self.subTest(cmd=cmd):
                a = p.parse_args([cmd, *extra])
                self.assertEqual(a.command, cmd)

    def test_handle_start_invokes_menu(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "mirrorcore.db"
            store = DatabaseStore(db_path)
            try:
                store.initialize_database()
            finally:
                store.close()
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(td)
                with patch("mirrorcore.onboarding.run_start_menu") as m:
                    handle_start(MagicMock())
                    m.assert_called_once()
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
