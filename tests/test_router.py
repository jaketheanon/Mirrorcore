"""
Tests for Phase 31: deterministic router (classify_intent, routing, disambiguation).
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.main import create_parser, handle_ask
from mirrorcore.router import (
    DEBUG_HELP,
    DECISION_HELP,
    ONBOARDING_OR_HELP,
    PERSONAL_RESPONSE,
    PROFILE_BUILDING,
    PROFILE_INTERVIEW,
    PROFILE_STYLE,
    classify_intent,
    disambiguate_intent,
    resolve_full_route,
    resolve_route_after_classification,
    routing_feedback,
)


class TestClassifyIntent(unittest.TestCase):
    def test_decision_buy_now(self):
        c = classify_intent("Should I buy this right now?")
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_personal_likely_say(self):
        c = classify_intent("What would I probably say in this situation?")
        self.assertEqual(c.ordered[0][0], PERSONAL_RESPONSE)

    def test_debug_permission_denied(self):
        c = classify_intent("This service fails with permission denied")
        self.assertEqual(c.ordered[0][0], DEBUG_HELP)

    def test_figure_out_what_to_do(self):
        c = classify_intent("Help me figure out what to do")
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_onboarding_what_can_you_do(self):
        c = classify_intent("what can you do")
        self.assertEqual(c.ordered[0][0], ONBOARDING_OR_HELP)

    def test_profile_calibrate_style(self):
        c = classify_intent("I want to calibrate how I talk")
        self.assertEqual(c.ordered[0][0], PROFILE_BUILDING)
        r = resolve_route_after_classification(c)
        self.assertEqual(r.profile_target, PROFILE_STYLE)

    def test_profile_learn_decide(self):
        c = classify_intent("learn how I decide")
        self.assertEqual(c.ordered[0][0], PROFILE_BUILDING)
        r = resolve_route_after_classification(c)
        self.assertEqual(r.profile_target, PROFILE_INTERVIEW)

    def test_weak_input(self):
        c = classify_intent("   ")
        self.assertTrue(c.weak_input)

    def test_wait_or_act_not_weak_decision(self):
        c = classify_intent("I don't know whether to wait or act now.")
        self.assertFalse(c.weak_input)
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_what_do_i_do_phrase_decision(self):
        c = classify_intent("A coworker wants me to cover, but I'm exhausted. What do I do?")
        self.assertFalse(c.weak_input)
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_shift_cover_fatigue_without_should_i(self):
        c = classify_intent(
            "colleague asked me to pick up a shift tomorrow and I'm drained"
        )
        self.assertFalse(c.weak_input)
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_should_i_cover_with_overload_cues(self):
        c = classify_intent(
            "should I cover for my coworker? I'm overwhelmed with hours already"
        )
        self.assertFalse(c.weak_input)
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_wants_help_favor_bias_decision(self):
        c = classify_intent(
            "my manager wants me to do a huge favor covering tonight - help?"
        )
        self.assertFalse(c.weak_input)
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_coworker_keeps_pushing_after_no_is_decision_not_weak(self):
        c = classify_intent("my coworker keeps pushing after i already said no")
        self.assertFalse(c.weak_input)
        self.assertEqual(c.ordered[0][0], DECISION_HELP)
        r = resolve_route_after_classification(c)
        self.assertEqual(r.category, DECISION_HELP)

    def test_same_coworker_still_pushing_with_what_should_i_do(self):
        c = classify_intent(
            "same coworker is still pushing today. what should i do now"
        )
        self.assertFalse(c.weak_input)
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_same_coworker_still_pushing_without_explicit_question(self):
        c = classify_intent("same coworker is still pushing today")
        self.assertFalse(c.weak_input)
        self.assertEqual(c.ordered[0][0], DECISION_HELP)

    def test_ambiguous_top_two(self):
        c = classify_intent("help me decide stack trace")
        self.assertTrue(c.needs_intent_disambiguation)
        self.assertEqual(len(c.ambiguous_options), 2)
        self.assertIn(DEBUG_HELP, c.ambiguous_options)
        self.assertIn(DECISION_HELP, c.ambiguous_options)


class TestDisambiguation(unittest.TestCase):
    def test_disambiguate_picks_decision(self):
        c = classify_intent("help me decide stack trace")
        # ambiguous_options are sorted; 1 = first label, 2 = second
        opts = list(c.ambiguous_options)
        self.assertEqual(len(opts), 2)

        def read(_):
            return "2"

        r = disambiguate_intent(c, read)
        self.assertEqual(r.category, opts[1])

    def test_disambiguate_picks_menu(self):
        c = classify_intent("help me decide stack trace")

        def read(_):
            return "3"

        r = disambiguate_intent(c, read)
        self.assertEqual(r.category, ONBOARDING_OR_HELP)
        self.assertTrue(r.from_disambiguation)

    def test_resolve_full_route_profile_split(self):
        c0 = classify_intent("I want to keep building my profile")
        self.assertTrue(c0.needs_profile_split)

        def read2(p):
            if "Pick 1 or 2" in p:
                return "2"
            return ""

        _c, route = resolve_full_route("I want to keep building my profile", read2)
        self.assertEqual(route.category, PROFILE_BUILDING)
        self.assertEqual(route.profile_target, PROFILE_STYLE)


class TestRoutingFeedback(unittest.TestCase):
    def test_quiet_decision_and_profile_routes(self):
        from mirrorcore.router import ResolvedRoute

        self.assertEqual(
            routing_feedback(ResolvedRoute(category=DECISION_HELP, profile_target=None)),
            "",
        )
        self.assertEqual(
            routing_feedback(
                ResolvedRoute(category=PROFILE_BUILDING, profile_target=PROFILE_INTERVIEW)
            ),
            "",
        )

    def test_debug_hint_stays_short(self):
        from mirrorcore.router import ResolvedRoute

        fb = routing_feedback(ResolvedRoute(category=DEBUG_HELP, profile_target=None))
        self.assertTrue(fb)
        self.assertIn("paste", fb.lower())

    def test_onboarding_menu_line(self):
        from mirrorcore.router import ResolvedRoute

        fb = routing_feedback(
            ResolvedRoute(category=ONBOARDING_OR_HELP, profile_target=None)
        )
        self.assertIn("can do", fb.lower())

    def test_after_disambiguation_menu(self):
        from mirrorcore.router import ResolvedRoute

        fb = routing_feedback(
            ResolvedRoute(
                category=ONBOARDING_OR_HELP,
                profile_target=None,
                from_disambiguation=True,
            )
        )
        self.assertIn("menu", fb.lower())


class TestAskCommand(unittest.TestCase):
    def test_parser_ask_with_words(self):
        p = create_parser()
        a = p.parse_args(["ask", "Should", "I", "go?"])
        self.assertEqual(a.command, "ask")
        self.assertEqual(a.query, ["Should", "I", "go?"])

    def test_parser_ask_no_args(self):
        p = create_parser()
        a = p.parse_args(["ask"])
        self.assertEqual(a.query, [])

    @patch("mirrorcore.main.PROMPT_TOOLKIT_AVAILABLE", False)
    @patch("mirrorcore.main.handle_start")
    @patch("builtins.input", return_value="what can you do")
    def test_handle_ask_prompt_mode(self, mock_input, mock_start):
        with patch("mirrorcore.main.DatabaseStore", MagicMock()):
            handle_ask(MagicMock(query=[]))
        mock_start.assert_called_once()

    @patch("mirrorcore.main.DatabaseStore")
    @patch("mirrorcore.main.handle_respond_like_me")
    def test_handle_ask_routes_respond(self, mock_r, _mock_store_cls):
        args = MagicMock(query=["What would I probably say?"])
        handle_ask(args)
        mock_r.assert_called_once()
        call_ns = mock_r.call_args[0][0]
        self.assertIn("probably say", call_ns.scenario.lower())

    @patch("mirrorcore.decision.routed_clarification.run_routed_decision_guidance")
    @patch("mirrorcore.main.DatabaseStore", MagicMock())
    @patch("mirrorcore.main.handle_start")
    def test_handle_ask_coworker_boundary_runs_decision_not_menu(
        self, mock_start, mock_guidance
    ):
        mock_guidance.return_value = "guidance"
        args = MagicMock(
            query=[
                "my",
                "coworker",
                "keeps",
                "pushing",
                "after",
                "i",
                "already",
                "said",
                "no",
            ]
        )
        handle_ask(args)
        mock_guidance.assert_called_once()
        mock_start.assert_not_called()

    @patch("mirrorcore.decision.routed_clarification.run_routed_decision_guidance")
    @patch("mirrorcore.main.DatabaseStore", MagicMock())
    @patch("mirrorcore.main.handle_start")
    def test_handle_ask_same_coworker_continuation_runs_decision_not_menu(
        self, mock_start, mock_guidance
    ):
        mock_guidance.return_value = "guidance"
        args = MagicMock(
            query=[
                "same",
                "coworker",
                "is",
                "still",
                "pushing",
                "today.",
                "what",
                "should",
                "i",
                "do",
                "now",
            ]
        )
        handle_ask(args)
        mock_guidance.assert_called_once()
        mock_start.assert_not_called()

    def test_ask_parser_accepts_debug_trace(self):
        parser = create_parser()
        args = parser.parse_args(["ask", "--debug-trace", "hello"])
        self.assertTrue(args.debug_trace)

    def test_respond_like_me_parser_accepts_debug_trace(self):
        parser = create_parser()
        args = parser.parse_args(["respond-like-me", "--debug-trace", "hello"])
        self.assertTrue(args.debug_trace)


if __name__ == "__main__":
    unittest.main()
