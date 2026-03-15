"""
Tests for Intake Agent

Tests the questionnaire, scenario engine, and profiler components.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.intake.questionnaire import get_assessment_definition, QuestionType, Response
from mirrorcore.intake.scenario_engine import ScenarioEngine
from mirrorcore.intake.profiler import Profiler


class TestQuestionnaire(unittest.TestCase):
    """Test the questionnaire definition."""
    
    def test_assessment_definition_loads(self):
        """Test that assessment definition loads successfully."""
        definition = get_assessment_definition()
        
        self.assertIsNotNone(definition)
        self.assertIn("core_questions", definition)
        self.assertIn("followup_questions", definition)
        self.assertIn("target_traits", definition)
        self.assertIn("max_total_questions", definition)
    
    def test_core_questions_exist(self):
        """Test that core questions exist and have required fields."""
        definition = get_assessment_definition()
        core_questions = definition["core_questions"]
        
        self.assertGreater(len(core_questions), 0)
        
        for question in core_questions:
            self.assertIsNotNone(question.id)
            self.assertIsNotNone(question.scenario_text)
            self.assertIsNotNone(question.question_text)
            self.assertIsInstance(question.type, QuestionType)
            self.assertIsNotNone(question.answer_options)
            self.assertGreater(len(question.answer_options), 0)
    
    def test_followup_questions_exist(self):
        """Test that follow-up questions exist and have required fields."""
        definition = get_assessment_definition()
        followup_questions = definition["followup_questions"]
        
        self.assertGreater(len(followup_questions), 0)
        
        for question in followup_questions:
            self.assertIsNotNone(question.id)
            self.assertIsNotNone(question.scenario_text)
            self.assertIsNotNone(question.question_text)
            self.assertIsInstance(question.type, QuestionType)
            self.assertIsNotNone(question.answer_options)
            # Open-ended questions may have no answer options, but multiple choice must have them
            if "MULTIPLE_CHOICE" in question.type.name:
                self.assertGreater(len(question.answer_options), 0)
    
    def test_target_traits_defined(self):
        """Test that target traits are defined."""
        definition = get_assessment_definition()
        target_traits = definition["target_traits"]
        
        self.assertGreater(len(target_traits), 0)
        expected_traits = [
            "troubleshooting_style", "risk_tolerance", "ambiguity_tolerance",
            "decision_speed", "evidence_threshold", "trust_verification_style",
            "communication_preference", "action_bias", "setback_response",
            "priority_resolution_style", "advice_filtering_style",
            "self_reliance_level", "confidence_calibration",
            "speed_vs_accuracy_preference", "tool_dependence"
        ]
        
        for trait in expected_traits:
            self.assertIn(trait, target_traits)


class TestScenarioEngine(unittest.TestCase):
    """Test the scenario engine."""
    
    def test_initialization(self):
        """Test scenario engine initialization."""
        engine = ScenarioEngine()
        
        self.assertIsNotNone(engine.definition)
        self.assertIsNotNone(engine.core_questions)
        self.assertIsNotNone(engine.followup_questions)
        self.assertIsNotNone(engine.state)
        self.assertIsNotNone(engine.state.assessment_session_id)
    
    def test_session_id_generation(self):
        """Test that assessment session ID is generated."""
        engine = ScenarioEngine()
        
        self.assertIsInstance(engine.state.assessment_session_id, str)
        self.assertGreater(len(engine.state.assessment_session_id), 0)
    
    def test_question_availability(self):
        """Test that questions are available for assessment."""
        engine = ScenarioEngine()
        
        self.assertGreater(len(engine.core_questions), 0)
        self.assertGreater(len(engine.followup_questions), 0)
    
    def test_followup_triggers_with_explanation_only(self):
        """Test that keyword_triggers is initialized even when only explanation is provided."""
        engine = ScenarioEngine()
        
        # Create response with explanation but no custom response
        response = Response(
            question_id="test_question",
            question_type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            selected_option_id="option_1",
            explanation="This approach involves some risk and expert guidance"
        )
        
        # This should not crash due to undefined keyword_triggers
        try:
            followup_ids = engine._get_followup_triggers(response)
            # Should return list (possibly empty)
            self.assertIsInstance(followup_ids, list)
        except NameError as e:
            self.fail(f"NameError occurred: {e}")
    
    def test_followup_triggers_with_empty_explanation(self):
        """Test that empty explanations are handled safely."""
        engine = ScenarioEngine()
        
        # Create response with empty explanation
        response = Response(
            question_id="test_question",
            question_type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            selected_option_id="option_1",
            explanation="   "  # Only whitespace
        )
        
        # This should not crash
        try:
            followup_ids = engine._get_followup_triggers(response)
            self.assertIsInstance(followup_ids, list)
        except Exception as e:
            self.fail(f"Exception occurred with empty explanation: {e}")
    
    def test_followup_triggers_with_keyword_in_explanation(self):
        """Test that keywords in explanations trigger followups correctly."""
        engine = ScenarioEngine()
        
        # Create response with explanation containing keyword
        response = Response(
            question_id="test_question",
            question_type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
            selected_option_id="option_1",
            explanation="I consider the risk factors carefully"
        )
        
        followup_ids = engine._get_followup_triggers(response)
        self.assertIsInstance(followup_ids, list)
        # Should contain risk_assessment trigger due to "risk" keyword
        self.assertIn("risk_assessment", followup_ids)


class TestProfiler(unittest.TestCase):
    """Test the profiler component."""
    
    def test_initialization(self):
        """Test profiler initialization."""
        profiler = Profiler()
        
        self.assertIsNotNone(profiler.definition)
        self.assertIsNotNone(profiler.trait_descriptions)
        self.assertIsNotNone(profiler.keyword_themes)
        self.assertGreater(len(profiler.keyword_themes), 0)
    
    def test_analyze_mock_responses(self):
        """Test analyzing minimal mock response set."""
        profiler = Profiler()
        
        # Create minimal mock responses
        mock_responses = [
            Response(
                question_id="test_question_1",
                question_type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
                selected_option_id="option_1",
                custom_response=None,
                explanation=None,
                trigger_reason="core_question",
                order_index=1,
                assessment_session_id="test-session"
            ),
            Response(
                question_id="test_question_2", 
                question_type=QuestionType.SCENARIO_MULTIPLE_CHOICE,
                selected_option_id="option_2",
                custom_response="I prefer to verify carefully",
                explanation="Safety is important",
                trigger_reason="core_question",
                order_index=2,
                assessment_session_id="test-session"
            )
        ]
        
        try:
            persona_traits = profiler.analyze_responses(mock_responses)
            self.assertIsNotNone(persona_traits)
            self.assertIsInstance(persona_traits, dict)
        except Exception as e:
            # Expected since test questions don't exist in definition
            # This tests the analysis framework works
            self.assertIn("test_question", str(e))
    
    def test_keyword_themes_structure(self):
        """Test that keyword themes have proper structure."""
        profiler = Profiler()
        
        for theme_name, theme_config in profiler.keyword_themes.items():
            self.assertIn("keywords", theme_config)
            self.assertIn("trait_impacts", theme_config)
            self.assertIn("direction", theme_config)
            
            self.assertIsInstance(theme_config["keywords"], list)
            self.assertGreater(len(theme_config["keywords"]), 0)
            self.assertIsInstance(theme_config["trait_impacts"], dict)
            self.assertIn(theme_config["direction"], [1.0, -1.0])
    
    def test_assessment_summary_generation(self):
        """Test assessment summary generation."""
        profiler = Profiler()
        
        # Mock persona traits
        from mirrorcore.intake.profiler import PersonaTrait, TraitEvidence
        mock_traits = {
            "test_trait": PersonaTrait(
                name="test_trait",
                value=0.5,
                confidence=0.7,
                evidence_count=2,
                contributing_responses=["q1", "q2"],
                evidence=TraitEvidence(
                    trait_name="test_trait",
                    value=0.5,
                    confidence=0.7,
                    evidence_count=2,
                    contributing_question_ids=["q1", "q2"],
                    supporting_signals=["Test signal"],
                    conflict_signals=[],
                    evidence_summary="Test summary",
                    raw_signals=[]
                )
            )
        }
        
        summary = profiler.generate_assessment_summary(mock_traits, [])
        self.assertIsNotNone(summary)
        self.assertIn("ASSESSMENT RESULTS", summary)
        self.assertIn("Traits inferred", summary)


if __name__ == "__main__":
    unittest.main()
