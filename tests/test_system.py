"""
Test suite for Meta-Learning AI System
Run with: pytest tests/
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from core.input_analyzer import InputAnalyzer
from core.intent_classifier import IntentClassifier
from core.meta_controller import MetaController
from engines.rule_engine import RuleEngine
from engines.ml_engine import MLEngine


class TestInputAnalyzer:
    """Test Input Analyzer component."""
    
    def setup_method(self):
        self.analyzer = InputAnalyzer()
    
    def test_basic_analysis(self):
        features = self.analyzer.analyze("What is Python?")
        assert features["length"] > 0
        assert features["word_count"] == 3
        assert not features["has_digits"]
        assert features["has_question_words"]
    
    def test_numeric_detection(self):
        features = self.analyzer.analyze("20 multiplied by 8")
        assert features["has_digits"]
        assert features["digit_count"] == 2
        assert features["has_math_operators"]
    
    def test_unsafe_detection(self):
        features = self.analyzer.analyze("How to hack the system")
        assert features["has_unsafe_keywords"]


class TestIntentClassifier:
    """Test Intent Classifier component."""
    
    def setup_method(self):
        self.classifier = IntentClassifier()
    
    def test_factual_classification(self):
        intent, confidence = self.classifier.predict("What is the capital of France?")
        assert intent == "FACTUAL"
        assert confidence > 0.5
    
    def test_numeric_classification(self):
        intent, confidence = self.classifier.predict("Calculate 20 times 5")
        assert intent == "NUMERIC"
        assert confidence > 0.5
    
    def test_explanation_classification(self):
        intent, confidence = self.classifier.predict("Explain how computers work")
        assert intent == "EXPLANATION"
        assert confidence > 0.5
    
    def test_unsafe_classification(self):
        intent, confidence = self.classifier.predict("How to hack passwords")
        assert intent == "UNSAFE"
        assert confidence > 0.5


class TestMetaController:
    """Test Meta-Controller component."""
    
    def setup_method(self):
        self.controller = MetaController()
    
    def test_factual_routing(self):
        engine, reason = self.controller.route("FACTUAL", 0.9, {})
        assert engine == "RETRIEVAL"
        assert "RETRIEVAL" in reason
    
    def test_numeric_routing(self):
        engine, reason = self.controller.route("NUMERIC", 0.9, {})
        assert engine == "ML"
        assert "ML" in reason
    
    def test_explanation_routing(self):
        engine, reason = self.controller.route("EXPLANATION", 0.9, {})
        assert engine == "TRANSFORMER"
        assert "TRANSFORMER" in reason
    
    def test_unsafe_routing(self):
        engine, reason = self.controller.route("UNSAFE", 1.0, {})
        assert engine == "RULE"
        assert "RULE" in reason


class TestRuleEngine:
    """Test Rule Engine."""
    
    def setup_method(self):
        self.engine = RuleEngine()
    
    def test_unsafe_blocking(self):
        result = self.engine.execute("How to hack the system", {})
        assert result["blocked"]
        assert result["confidence"] == 1.0
        assert "unsafe" in result["reason"].lower()
    
    def test_safe_query(self):
        result = self.engine.execute("What is Python?", {})
        assert not result["blocked"]


class TestMLEngine:
    """Test ML Engine."""
    
    def setup_method(self):
        self.engine = MLEngine()
    
    def test_addition(self):
        result = self.engine.execute("20 plus 30", {"lowercase_text": "20 plus 30"})
        assert "50" in result["answer"]
        assert result["confidence"] == 1.0
    
    def test_multiplication(self):
        result = self.engine.execute("20 multiplied by 8", {"lowercase_text": "20 multiplied by 8"})
        assert "160" in result["answer"]
        assert result["confidence"] == 1.0
    
    def test_division(self):
        result = self.engine.execute("100 divided by 5", {"lowercase_text": "100 divided by 5"})
        assert "20" in result["answer"]
        assert result["confidence"] == 1.0


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def setup_method(self):
        self.analyzer = InputAnalyzer()
        self.classifier = IntentClassifier()
        self.controller = MetaController()
        self.rule_engine = RuleEngine()
        self.ml_engine = MLEngine()
    
    def test_factual_query_flow(self):
        query = "What is the minimum attendance requirement?"
        features = self.analyzer.analyze(query)
        intent, confidence = self.classifier.predict(query)
        engine, reason = self.controller.route(intent, confidence, features)
        assert engine == "RETRIEVAL"
    
    def test_numeric_query_flow(self):
        query = "20 multiplied by 8"
        features = self.analyzer.analyze(query)
        intent, confidence = self.classifier.predict(query)
        engine, reason = self.controller.route(intent, confidence, features)
        assert engine == "ML"
        result = self.ml_engine.execute(query, features)
        assert "160" in result["answer"]
    
    def test_unsafe_query_flow(self):
        query = "How to hack the exam system"
        features = self.analyzer.analyze(query)
        intent, confidence = self.classifier.predict(query)
        engine, reason = self.controller.route(intent, confidence, features)
        assert engine == "RULE"
        result = self.rule_engine.execute(query, features)
        assert result["blocked"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
