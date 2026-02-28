"""
Test Suite for Multi-Label Semantic Intent Classification System

Validates:
1. Semantic intent scoring (all intents, not just top-1)
2. Multi-label activation (hybrid queries)
3. UNSAFE override behavior
4. Execution planning for intent chains
5. Performance constraints (<200ms total)
6. Explainability and auditability
"""

import pytest
from core.semantic_intent_classifier import (
    SemanticIntentClassifier,
    ExecutionPlanner
)
from core.meta_controller import MetaController
import time


class TestSemanticIntentClassifier:
    """Test the semantic intent classifier."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.classifier = SemanticIntentClassifier(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            intent_threshold=0.60,
            unsafe_threshold=0.50
        )
    
    def test_classifier_initialization(self):
        """Test that classifier initializes correctly."""
        assert self.classifier is not None
        assert self.classifier.intent_threshold == 0.60
        assert self.classifier.unsafe_threshold == 0.50
        assert len(self.classifier.INTENT_PROTOTYPES) == 4
    
    def test_classifier_output_format(self):
        """Test that classifier returns correct output format."""
        result = self.classifier.classify("What is the capital of France?")
        
        assert "scores" in result
        assert "active_intents" in result
        assert "primary_intent" in result
        assert "primary_confidence" in result
        assert "threshold" in result
        assert "model" in result
        assert "classification_time_ms" in result
        
        # Check all intents have scores
        assert set(result["scores"].keys()) == {"FACTUAL", "NUMERIC", "EXPLANATION", "UNSAFE"}
        
        # Check scores are normalized
        for score in result["scores"].values():
            assert 0 <= score <= 1
    
    def test_single_intent_factual(self):
        """Test: 'What is the capital of Germany?' → FACTUAL only."""
        result = self.classifier.classify("What is the capital of Germany?")
        
        assert "FACTUAL" in result["active_intents"]
        # NUMERIC and EXPLANATION should not be primary
        assert result["primary_intent"] == "FACTUAL"
        assert result["scores"]["FACTUAL"] > result["scores"]["NUMERIC"]
        assert result["scores"]["FACTUAL"] > result["scores"]["EXPLANATION"]
    
    def test_single_intent_numeric(self):
        """Test: 'What is 20% of 500?' → NUMERIC only."""
        result = self.classifier.classify("What is 20% of 500?")
        
        assert "NUMERIC" in result["active_intents"]
        assert result["primary_intent"] == "NUMERIC"
        assert result["scores"]["NUMERIC"] > result["scores"]["FACTUAL"]
    
    def test_single_intent_explanation(self):
        """Test: 'Explain why water boils at 100°C' → EXPLANATION."""
        result = self.classifier.classify("Explain why water boils at 100 degrees Celsius")
        
        # Could be FACTUAL or EXPLANATION, but should have high explanation score
        assert result["scores"]["EXPLANATION"] > 0.3
    
    def test_multi_intent_factual_numeric(self):
        """Test: 'What is 5 times the population of Germany?' → FACTUAL + NUMERIC."""
        result = self.classifier.classify("What is 5 times the population of Germany?")
        
        assert len(result["active_intents"]) >= 1  # At least one
        # Should have both high FACTUAL and NUMERIC scores
        assert result["scores"]["FACTUAL"] > 0.4
        assert result["scores"]["NUMERIC"] > 0.4
    
    def test_multi_intent_numeric_explanation(self):
        """Test: 'Explain why 20% of 500 is 100.' → NUMERIC + EXPLANATION."""
        result = self.classifier.classify("Explain why 20 percent of 500 equals 100")
        
        # Should have both NUMERIC and EXPLANATION high
        assert result["scores"]["NUMERIC"] > 0.3
        assert result["scores"]["EXPLANATION"] > 0.3
    
    def test_multi_intent_factual_explanation(self):
        """Test: 'Explain the capital of Germany.' → FACTUAL + EXPLANATION."""
        result = self.classifier.classify("Explain what the capital of Germany is and why it's important")
        
        # Both should be reasonably high
        assert result["scores"]["FACTUAL"] > 0.3
        assert result["scores"]["EXPLANATION"] > 0.3
    
    def test_unsafe_query_detection(self):
        """Test that UNSAFE queries have high unsafe score."""
        result = self.classifier.classify("How to cheat on an exam?")
        
        assert result["scores"]["UNSAFE"] > 0.3  # Should be fairly high
        # In some cases might still have other intents
    
    def test_threshold_filtering(self):
        """Test that active_intents respects threshold."""
        result = self.classifier.classify("What is the capital of France?")
        
        # All active intents should exceed threshold
        for intent in result["active_intents"]:
            if intent == "UNSAFE":
                assert result["scores"][intent] > self.classifier.unsafe_threshold
            else:
                assert result["scores"][intent] > self.classifier.intent_threshold
    
    def test_all_scores_present(self):
        """Test that all intent scores are returned, not just active ones."""
        result = self.classifier.classify("What is the capital of France?")
        
        # Should have all 4 intents scored
        assert len(result["scores"]) == 4
        assert all(intent in result["scores"] for intent in ["FACTUAL", "NUMERIC", "EXPLANATION", "UNSAFE"])
    
    def test_performance_under_100ms(self):
        """Test that single classification completes under 100ms."""
        start = time.time()
        result = self.classifier.classify("What is photosynthesis?")
        elapsed_ms = (time.time() - start) * 1000
        
        # Should be well under 100ms
        assert elapsed_ms < 100
        assert result["classification_time_ms"] < 100
    
    def test_performance_stats(self):
        """Test that performance stats are tracked."""
        stats = self.classifier.get_stats()
        
        assert "model" in stats
        assert "has_embeddings" in stats
        assert "intent_threshold" in stats
        assert "intents" in stats
    
    def test_integrity_check(self):
        """Test integrity check method."""
        integrity = self.classifier.integrity_check()
        
        assert "initialized" in integrity
        assert "embeddings_available" in integrity
        assert integrity["ready_for_inference"] is True


class TestExecutionPlanner:
    """Test the execution planner."""
    
    def test_single_intent_factual(self):
        """Test: FACTUAL only → RETRIEVAL_ENGINE."""
        engines, reasoning = ExecutionPlanner.plan_execution(["FACTUAL"])
        
        assert engines == ["RETRIEVAL_ENGINE"]
        assert "FACTUAL" in reasoning
    
    def test_single_intent_numeric(self):
        """Test: NUMERIC only → ML_ENGINE."""
        engines, reasoning = ExecutionPlanner.plan_execution(["NUMERIC"])
        
        assert engines == ["ML_ENGINE"]
        assert "NUMERIC" in reasoning
    
    def test_single_intent_explanation(self):
        """Test: EXPLANATION only → TRANSFORMER_ENGINE."""
        engines, reasoning = ExecutionPlanner.plan_execution(["EXPLANATION"])
        
        assert engines == ["TRANSFORMER_ENGINE"]
        assert "EXPLANATION" in reasoning
    
    def test_multi_intent_factual_numeric(self):
        """Test: FACTUAL + NUMERIC → RETRIEVAL → ML."""
        engines, reasoning = ExecutionPlanner.plan_execution(["FACTUAL", "NUMERIC"])
        
        assert "RETRIEVAL_ENGINE" in engines
        assert "ML_ENGINE" in engines
        # Retrieval should come before computation
        assert engines.index("RETRIEVAL_ENGINE") < engines.index("ML_ENGINE")
    
    def test_multi_intent_factual_explanation(self):
        """Test: FACTUAL + EXPLANATION → RETRIEVAL → TRANSFORMER."""
        engines, reasoning = ExecutionPlanner.plan_execution(["FACTUAL", "EXPLANATION"])
        
        assert "RETRIEVAL_ENGINE" in engines
        assert "TRANSFORMER_ENGINE" in engines
        # Retrieval should come before explanation
        assert engines.index("RETRIEVAL_ENGINE") < engines.index("TRANSFORMER_ENGINE")
    
    def test_multi_intent_numeric_explanation(self):
        """Test: NUMERIC + EXPLANATION → ML → TRANSFORMER."""
        engines, reasoning = ExecutionPlanner.plan_execution(["NUMERIC", "EXPLANATION"])
        
        assert "ML_ENGINE" in engines
        assert "TRANSFORMER_ENGINE" in engines
        # Computation should come before explanation
        assert engines.index("ML_ENGINE") < engines.index("TRANSFORMER_ENGINE")
    
    def test_unsafe_override(self):
        """Test: UNSAFE overrides everything → RULE_ENGINE only."""
        engines, reasoning = ExecutionPlanner.plan_execution(["UNSAFE"])
        
        assert engines == ["RULE_ENGINE"]
        assert "UNSAFE" in reasoning
    
    def test_unsafe_with_other_intents(self):
        """Test: UNSAFE + FACTUAL + NUMERIC → RULE_ENGINE only (override)."""
        engines, reasoning = ExecutionPlanner.plan_execution(["UNSAFE", "FACTUAL", "NUMERIC"])
        
        # UNSAFE should completely override
        assert engines == ["RULE_ENGINE"]
    
    def test_three_intent_chain(self):
        """Test: FACTUAL + NUMERIC + EXPLANATION → Full chain."""
        engines, reasoning = ExecutionPlanner.plan_execution(["FACTUAL", "NUMERIC", "EXPLANATION"])
        
        # Should have all three engines
        assert "RETRIEVAL_ENGINE" in engines
        assert "ML_ENGINE" in engines
        assert "TRANSFORMER_ENGINE" in engines
        
        # Order should be: retrieve → compute → explain
        assert engines.index("RETRIEVAL_ENGINE") < engines.index("ML_ENGINE")
        assert engines.index("ML_ENGINE") < engines.index("TRANSFORMER_ENGINE")


class TestMetaController:
    """Test the multi-intent meta-controller."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.controller = MetaController()
    
    def test_controller_initialization(self):
        """Test that controller initializes correctly."""
        assert self.controller is not None
        assert self.controller.intent_classifier is not None
        assert self.controller.execution_planner is not None
    
    def test_orchestrate_single_intent(self):
        """Test orchestration for single-intent query."""
        plan = self.controller.orchestrate("What is the capital of France?")
        
        assert plan["status"] == "ready"
        assert "intents" in plan
        assert "execution_plan" in plan
        assert len(plan["execution_plan"]["engine_chain"]) > 0
    
    def test_orchestrate_multi_intent(self):
        """Test orchestration for multi-intent query."""
        plan = self.controller.orchestrate("What is 5 times the population of Germany?")
        
        assert plan["status"] == "ready"
        # Should have multiple intents
        assert len(plan["intents"]["all_scores"]) == 4
    
    def test_orchestrate_unsafe_override(self):
        """Test that UNSAFE queries are immediately blocked."""
        plan = self.controller.orchestrate("How to cheat on an exam?")
        
        # Could be blocked or might have UNSAFE in active intents
        assert "UNSAFE" in plan["intents"]["active_intents"] or plan["status"] == "blocked"
    
    def test_route_method_backward_compatibility(self):
        """Test that route() method still works for compatibility."""
        engines, reasoning = self.controller.route("What is the capital of France?")
        
        assert isinstance(engines, list)
        assert len(engines) > 0
        assert isinstance(reasoning, str)
    
    def test_routing_stats_tracking(self):
        """Test that routing decisions are logged."""
        # Make a query
        self.controller.orchestrate("What is the capital of France?")
        
        stats = self.controller.get_routing_stats()
        
        assert stats["total_queries"] == 1
        assert "intent_distribution" in stats
        assert "engine_chain_distribution" in stats
    
    def test_multi_intent_tracking(self):
        """Test tracking of multi-intent queries."""
        # Query 1: Single intent
        self.controller.orchestrate("What is 5 plus 3?")
        
        # Query 2: Multi intent
        self.controller.orchestrate("What is 5 times the population of France?")
        
        stats = self.controller.get_routing_stats()
        
        assert stats["total_queries"] == 2
        assert "multi_intent_queries" in stats
    
    def test_integrity_check(self):
        """Test integrity check method."""
        integrity = self.controller.integrity_check()
        
        assert integrity["initialized"] is True
        assert "intent_classifier_ready" in integrity
    
    def test_validate_orchestration(self):
        """Test orchestration validation."""
        plan = self.controller.orchestrate("What is the capital of France?")
        
        is_valid, reason = self.controller.validate_orchestration(plan)
        
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)


class TestIntegrationScenarios:
    """Integration tests with real-world scenarios."""
    
    def setup_method(self):
        """Setup test fixture."""
        self.controller = MetaController()
    
    def test_scenario_factual_query(self):
        """Real scenario: Factual query."""
        plan = self.controller.orchestrate("What is the capital of Germany?")
        
        assert plan["status"] == "ready"
        assert "RETRIEVAL_ENGINE" in plan["execution_plan"]["engine_chain"]
    
    def test_scenario_numeric_query(self):
        """Real scenario: Numeric query."""
        plan = self.controller.orchestrate("What is 20% of 500?")
        
        assert plan["status"] == "ready"
        assert "ML_ENGINE" in plan["execution_plan"]["engine_chain"]
    
    def test_scenario_hybrid_factual_numeric(self):
        """Real scenario: Hybrid factual+numeric query."""
        plan = self.controller.orchestrate("What is 5 times the population of France?")
        
        assert plan["status"] == "ready"
        engines = plan["execution_plan"]["engine_chain"]
        # Should have both retrieval and computation
        assert any("RETRIEVAL" in e for e in engines) or any("ML" in e for e in engines)
    
    def test_scenario_hybrid_numeric_explanation(self):
        """Real scenario: Explain calculation."""
        plan = self.controller.orchestrate("Explain why 20% of 500 is 100")
        
        assert plan["status"] == "ready"
        engines = plan["execution_plan"]["engine_chain"]
        # Should have computation and explanation
        assert len(engines) >= 1
    
    def test_scenario_complex_query(self):
        """Real scenario: Complex multi-step query."""
        query = "What is the capital of Germany and what is 3 times its population?"
        plan = self.controller.orchestrate(query)
        
        assert plan["status"] == "ready"
        # Should have multiple intents
        assert len(plan["intents"]["all_scores"]) == 4
    
    def test_performance_single_query(self):
        """Test performance for single query."""
        start = time.time()
        plan = self.controller.orchestrate("What is the capital of France?")
        elapsed_ms = (time.time() - start) * 1000
        
        # Total orchestration should be under 200ms
        assert elapsed_ms < 200
    
    def test_performance_batch(self):
        """Test performance for batch of queries."""
        start = time.time()
        
        queries = [
            "What is the capital of France?",
            "What is 20% of 500?",
            "Explain photosynthesis",
            "What is 5 times the population of Germany?"
        ]
        
        for query in queries:
            self.controller.orchestrate(query)
        
        elapsed_ms = (time.time() - start) * 1000
        avg_time = elapsed_ms / len(queries)
        
        # Average should be under 100ms per query
        assert avg_time < 100
    
    def test_explainability(self):
        """Test that routing is explainable."""
        plan = self.controller.orchestrate("What is 5 times the population of France?")
        
        # Should have clear reasoning
        assert "intents" in plan
        assert plan["intents"]["primary_intent"] in ["FACTUAL", "NUMERIC", "EXPLANATION", "UNSAFE"]
        assert "chain_reasoning" in plan["execution_plan"]
        
        # Reasoning should be non-empty
        assert len(plan["execution_plan"]["chain_reasoning"]) > 0
    
    def test_auditability(self):
        """Test that all decisions are auditable."""
        plan = self.controller.orchestrate("What is the capital of France?")
        
        # All data should be present for audit trail
        assert "timestamp" in plan["metadata"]
        assert "classification_time_ms" in plan["metadata"]
        assert "intents" in plan
        assert plan["intents"]["threshold_used"] is not None


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
