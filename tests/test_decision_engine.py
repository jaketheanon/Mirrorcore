"""
Tests for Decision Engine

Tests decision-making and analysis components.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mirrorcore.decision.engine import DecisionEngine, DecisionOption, DecisionResult
from mirrorcore.decision.tradeoffs import TradeoffAnalyzer
from mirrorcore.decision.confidence import ConfidenceCalculator


class TestDecisionEngine(unittest.TestCase):
    """Test the decision engine."""
    
    def setUp(self):
        self.engine = DecisionEngine()
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine.tradeoff_analyzer)
        self.assertIsNotNone(self.engine.confidence_calculator)
    
    def test_analyze_decision(self):
        """Test decision analysis."""
        options = [
            DecisionOption(
                name="Option A",
                description="First option",
                properties={"cost_score": 0.3, "performance_score": 0.8},
                pros=["Fast", "Reliable"],
                cons=["Expensive"]
            ),
            DecisionOption(
                name="Option B", 
                description="Second option",
                properties={"cost_score": 0.8, "performance_score": 0.5},
                pros=["Cheap", "Simple"],
                cons=["Slow"]
            )
        ]
        
        context = {"user_values": {"cost": 0.7, "performance": 0.3}}
        
        result = self.engine.analyze_decision(context, options)
        
        self.assertIsNotNone(result)
        self.assertIn(result.recommended_option, ["Option A", "Option B"])
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_calculate_option_score(self):
        """Test option score calculation."""
        option = DecisionOption(
            name="Test Option",
            description="Test",
            properties={"cost_score": 0.5, "performance_score": 0.7},
            pros=["Good"],
            cons=["Bad"]
        )
        
        context = {"user_values": {"cost": 0.5, "performance": 0.5}}
        tradeoffs = {}
        
        score = self.engine._calculate_option_score(option, context, tradeoffs)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestTradeoffAnalyzer(unittest.TestCase):
    """Test the tradeoff analyzer."""
    
    def setUp(self):
        self.analyzer = TradeoffAnalyzer()
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # No specific initialization to test
        self.assertIsNotNone(self.analyzer)
    
    def test_analyze_tradeoffs(self):
        """Test tradeoff analysis."""
        options = [
            {
                "name": "Option A",
                "properties": {"cost_score": 0.3, "performance_score": 0.8}
            },
            {
                "name": "Option B",
                "properties": {"cost_score": 0.8, "performance_score": 0.5}
            }
        ]
        
        context = {}
        result = self.analyzer.analyze_tradeoffs(options, context)
        
        self.assertIn("tradeoffs", result)
        self.assertIn("summary", result)
        self.assertIsInstance(result["tradeoffs"], list)
        self.assertEqual(len(result["tradeoffs"]), 2)  # Two options = one comparison
    
    def test_compare_options(self):
        """Test option comparison."""
        option1 = {
            "name": "Option A",
            "properties": {"cost_score": 0.3, "performance_score": 0.8}
        }
        option2 = {
            "name": "Option B", 
            "properties": {"cost_score": 0.8, "performance_score": 0.5}
        }
        
        tradeoffs = self.analyzer._compare_options(option1, option2)
        
        self.assertIsInstance(tradeoffs, list)
        # Should find differences in cost and performance
        self.assertGreater(len(tradeoffs), 0)


class TestConfidenceCalculator(unittest.TestCase):
    """Test the confidence calculator."""
    
    def setUp(self):
        self.calculator = ConfidenceCalculator()
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        scores = {"Option A": 0.8, "Option B": 0.6, "Option C": 0.4}
        context = {"data_quality": 0.8}
        
        confidence = self.calculator.calculate_confidence(scores, context)
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_confidence_empty_scores(self):
        """Test confidence with empty scores."""
        confidence = self.calculator.calculate_confidence({}, {})
        self.assertEqual(confidence, 0.0)
    
    def test_calculate_option_confidence(self):
        """Test option confidence calculation."""
        option_score = 0.8
        all_scores = [0.3, 0.5, 0.8, 0.9]
        
        confidence = self.calculator.calculate_option_confidence(option_score, all_scores)
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_option_confidence_single_option(self):
        """Test option confidence with single option."""
        option_score = 0.8
        all_scores = [0.8]
        
        confidence = self.calculator.calculate_option_confidence(option_score, all_scores)
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
