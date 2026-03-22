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
    def test_lines(self):
        from mirrorcore.router import ResolvedRoute

        self.assertIn(
            "decision",
            routing_feedback(
                ResolvedRoute(category=DECISION_HELP, profile_target=None)
            ).lower(),
        )


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


if __name__ == "__main__":
    unittest.main()
