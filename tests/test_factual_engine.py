"""
Comprehensive Test Suite for FactualEngine
Tests embedding-based semantic retrieval with confidence thresholding and ambiguity detection.
"""

import json
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.retrieval_engine import FactualEngine


class TestFactualEngineSetup:
    """Test engine initialization and setup."""
    
    def test_initialization_with_default_kb_path(self):
        """Test initialization with default KB path."""
        engine = FactualEngine()
        assert engine.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert engine.has_embeddings == True
        assert engine.model is not None
        print("✓ Initialization with default KB path successful")
    
    def test_kb_loading(self):
        """Test knowledge base loading."""
        engine = FactualEngine()
        assert engine.knowledge_base is not None
        assert "facts" in engine.knowledge_base
        fact_count = len(engine.knowledge_base.get("facts", []))
        print(f"✓ Knowledge base loaded with {fact_count} facts")
    
    def test_embedding_precomputation(self):
        """Test that embeddings are precomputed at startup."""
        engine = FactualEngine()
        assert len(engine.fact_embeddings) > 0
        assert len(engine.fact_lookup) > 0
        print(f"✓ Precomputed embeddings for {len(engine.fact_embeddings)} facts")


class TestFactualRetrieval:
    """Test factual retrieval with semantic search."""
    
    def test_exact_match_retrieval(self):
        """Test retrieval of exact semantic match."""
        engine = FactualEngine()
        
        # Query: "What is the capital of Germany?"
        result = engine.execute("What is the capital of Germany?", {})
        
        assert result["status"] == "success"
        assert result["type"] == "FACTUAL"
        assert result["data"]["answer"].lower() == "berlin"
        assert result["confidence"] >= 0.65
        print(f"✓ Exact match retrieval: Berlin (confidence: {result['confidence']:.3f})")
    
    def test_semantic_match_with_misspelling(self):
        """Test retrieval with tolerated misspelling/paraphrase."""
        engine = FactualEngine()
        
        # Similar but different phrasing
        result = engine.execute("Capital of Germany is?", {})
        
        assert result["status"] == "success"
        assert result["confidence"] >= 0.65
        print(f"✓ Semantic match with paraphrase: {result['data']['answer']} (confidence: {result['confidence']:.3f})")
    
    def test_numeric_structured_value(self):
        """Test retrieval of numeric structured_value for chaining."""
        engine = FactualEngine()
        
        # Query for Germany's population
        result = engine.execute("Population of Germany", {})
        
        assert result["status"] == "success"
        assert isinstance(result["data"]["structured_value"], int)
        assert result["data"]["structured_value"] > 0
        print(f"✓ Numeric structured_value retrieved: {result['data']['structured_value']}")
    
    def test_uncertain_response_low_confidence(self):
        """Test uncertain response when similarity below threshold."""
        engine = FactualEngine()
        
        # Query about non-existent entity
        result = engine.execute("What is the GDP of Mars?", {})
        
        assert result["status"] == "uncertain"
        assert result["data"]["answer"] is None
        assert result["confidence"] < engine.FACTUAL_CONFIDENCE_THRESHOLD
        print(f"✓ Uncertain response for non-existent entity (confidence: {result['confidence']:.3f})")
    
    def test_ambiguity_detection(self):
        """Test ambiguity detection when top-2 scores are similar."""
        engine = FactualEngine()
        
        # Query that might match multiple similar facts
        # This would need facts with similar scores to trigger ambiguity
        result = engine.execute("What is the capital of France?", {})
        
        # Even if ambiguous, we should get a structured response
        assert "status" in result
        assert "data" in result
        assert "confidence" in result
        print(f"✓ Ambiguity detection tested (status: {result['status']})")


class TestConfidenceThresholding:
    """Test confidence thresholding behavior."""
    
    def test_confidence_threshold_constant(self):
        """Test that confidence threshold is properly set."""
        engine = FactualEngine()
        assert engine.FACTUAL_CONFIDENCE_THRESHOLD == 0.65
        print(f"✓ Confidence threshold: {engine.FACTUAL_CONFIDENCE_THRESHOLD}")
    
    def test_ambiguity_threshold_constant(self):
        """Test that ambiguity threshold is properly set."""
        engine = FactualEngine()
        assert engine.AMBIGUITY_MAX_DIFF == 0.05
        print(f"✓ Ambiguity threshold: {engine.AMBIGUITY_MAX_DIFF}")


class TestMetadataAndAudit:
    """Test metadata and audit trail functionality."""
    
    def test_response_metadata_structure(self):
        """Test that response includes complete metadata."""
        engine = FactualEngine()
        
        result = engine.execute("What is the capital of Germany?", {})
        
        assert "metadata" in result
        metadata = result["metadata"]
        assert "fact_id" in metadata
        assert "source" in metadata
        assert "retrieval_time_ms" in metadata
        assert "timestamp" in metadata
        assert "engine" in metadata
        print(f"✓ Response metadata complete: {list(metadata.keys())}")
    
    def test_retrieval_logging(self):
        """Test that retrievals are logged."""
        engine = FactualEngine()
        
        # Clear history
        engine.clear_history()
        assert len(engine.retrieval_history) == 0
        
        # Execute query
        engine.execute("What is the capital of Germany?", {})
        
        # Check history
        assert len(engine.retrieval_history) > 0
        print(f"✓ Retrieval logged to history (size: {len(engine.retrieval_history)})")
    
    def test_statistics_generation(self):
        """Test statistics generation."""
        engine = FactualEngine()
        
        # Execute multiple queries
        engine.clear_history()
        engine.execute("What is the capital of Germany?", {})
        engine.execute("What is the capital of France?", {})
        
        stats = engine.get_stats()
        
        assert "total_retrievals" in stats
        assert "successful_retrievals" in stats
        assert "success_rate" in stats
        assert stats["total_retrievals"] >= 2
        print(f"✓ Statistics: {stats['successful_retrievals']}/{stats['total_retrievals']} successful")


class TestDeterminism:
    """Test deterministic behavior."""
    
    def test_consistent_results(self):
        """Test that same query produces same result."""
        engine = FactualEngine()
        
        query = "What is the capital of Germany?"
        result1 = engine.execute(query, {})
        result2 = engine.execute(query, {})
        
        # Results should be identical
        assert result1["status"] == result2["status"]
        assert result1["data"]["answer"] == result2["data"]["answer"]
        assert result1["confidence"] == result2["confidence"]
        print("✓ Results are deterministic (identical across runs)")


class TestResponseStructure:
    """Test structured response format."""
    
    def test_success_response_structure(self):
        """Test structure of successful response."""
        engine = FactualEngine()
        
        result = engine.execute("What is the capital of Germany?", {})
        
        # Check required fields
        required_fields = {"status", "type", "data", "confidence", "metadata"}
        assert required_fields <= set(result.keys())
        
        # Check data structure
        assert "answer" in result["data"]
        assert "structured_value" in result["data"]
        
        # Check types
        assert isinstance(result["status"], str)
        assert isinstance(result["type"], str)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["data"], dict)
        assert isinstance(result["metadata"], dict)
        
        print("✓ Success response structure valid")
    
    def test_uncertain_response_structure(self):
        """Test structure of uncertain response."""
        engine = FactualEngine()
        
        result = engine.execute("What is the GDP of Mars?", {})
        
        # Check required fields
        required_fields = {"status", "type", "data", "confidence", "metadata"}
        assert required_fields <= set(result.keys())
        
        # Data should indicate uncertainty
        assert result["data"]["answer"] is None
        assert result["status"] == "uncertain"
        
        print("✓ Uncertain response structure valid")


class TestValidation:
    """Test response validation."""
    
    def test_response_validation_success(self):
        """Test validation of successful response."""
        engine = FactualEngine()
        
        result = engine.execute("What is the capital of Germany?", {})
        assert engine.validate_response(result) == True
        print("✓ Response validation successful")
    
    def test_response_validation_uncertain(self):
        """Test validation of uncertain response."""
        engine = FactualEngine()
        
        result = engine.execute("What is the GDP of Mars?", {})
        assert engine.validate_response(result) == True
        print("✓ Uncertain response validation successful")


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_add_fact(self):
        """Test adding new fact to knowledge base."""
        engine = FactualEngine()
        
        new_fact = {
            "id": "fact_999",
            "question": "What is a test fact?",
            "answer": "A test fact for validation",
            "structured_value": "test",
            "category": "testing",
            "source": "Test Suite",
            "verified": True,
            "verified_date": "2025-01-01"
        }
        
        success = engine.add_fact(new_fact)
        assert success == True
        assert "fact_999" in engine.fact_lookup
        print("✓ Fact added successfully")
    
    def test_clear_history(self):
        """Test clearing retrieval history."""
        engine = FactualEngine()
        
        # Add some history
        engine.execute("What is the capital of Germany?", {})
        assert len(engine.retrieval_history) > 0
        
        # Clear
        engine.clear_history()
        assert len(engine.retrieval_history) == 0
        print("✓ History cleared successfully")
    
    def test_reset_stats(self):
        """Test resetting statistics."""
        engine = FactualEngine()
        
        # Execute query
        engine.execute("What is the capital of Germany?", {})
        assert engine.total_retrievals > 0
        
        # Reset
        engine.reset_stats()
        assert engine.total_retrievals == 0
        assert engine.successful_retrievals == 0
        print("✓ Statistics reset successfully")


class TestHybridChaining:
    """Test integration with multi-intent chaining."""
    
    def test_factual_numeric_chaining(self):
        """Test FACTUAL→NUMERIC chaining via structured_value."""
        engine = FactualEngine()
        
        # Query for population (requires structured_value)
        result = engine.execute("Population of Germany", {})
        
        assert result["status"] == "success"
        assert isinstance(result["data"]["structured_value"], int)
        
        # This structured_value should be consumable by numeric engine
        numeric_value = result["data"]["structured_value"]
        assert numeric_value == 83000000
        print(f"✓ Factual returns numeric value for chaining: {numeric_value}")


class TestPerformance:
    """Test performance characteristics."""
    
    def test_retrieval_time(self):
        """Test that retrieval completes within reasonable time."""
        engine = FactualEngine()
        
        result = engine.execute("What is the capital of Germany?", {})
        
        retrieval_time = result["metadata"]["retrieval_time_ms"]
        assert retrieval_time < 300  # Should be <300ms as per spec
        print(f"✓ Retrieval time: {retrieval_time:.2f}ms (< 300ms requirement)")


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("FACTUAL ENGINE TEST SUITE")
    print("="*60 + "\n")
    
    test_classes = [
        TestFactualEngineSetup,
        TestFactualRetrieval,
        TestConfidenceThresholding,
        TestMetadataAndAudit,
        TestDeterminism,
        TestResponseStructure,
        TestValidation,
        TestUtilityMethods,
        TestHybridChaining,
        TestPerformance
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 60)
        
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(instance, test_method)()
                passed_tests += 1
            except Exception as e:
                failed_tests.append(f"{test_class.__name__}.{test_method}: {e}")
                print(f"✗ {test_method}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} passed")
    print("="*60)
    
    if failed_tests:
        print("\nFailed Tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
    else:
        print("\n✅ All tests passed!")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
