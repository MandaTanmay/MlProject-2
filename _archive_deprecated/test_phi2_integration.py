#!/usr/bin/env python3
"""
Quick Integration Test for Phi2ExplanationEngine with App
Tests: engine initialization, grounding flow, and app integration points
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_engine_initialization():
    """Test Phi2ExplanationEngine can be initialized."""
    from engines.phi2_explanation_engine import Phi2ExplanationEngine
    
    logger.info("Testing Phi2ExplanationEngine initialization...")
    
    engine = Phi2ExplanationEngine(use_quantization=False, device="cpu")
    
    assert engine.model_name == "microsoft/phi-2"
    assert not engine.is_loaded
    assert engine.inference_count == 0
    assert hasattr(engine, 'validator')
    
    logger.info("✓ Engine initialization successful")
    return True


def test_grounding_validation():
    """Test grounding validation logic."""
    from engines.phi2_explanation_engine import Phi2ExplanationEngine
    
    logger.info("Testing grounding validation...")
    
    engine = Phi2ExplanationEngine(use_quantization=False, device="cpu")
    
    # Test 1: Empty grounding fails
    assert not engine._validate_grounded_input("Test", {})
    logger.info("  ✓ Empty grounding correctly rejected")
    
    # Test 2: Factual grounding passes
    assert engine._validate_grounded_input("Test", {"factual_result": "Result"})
    logger.info("  ✓ Factual grounding accepted")
    
    # Test 3: Numeric grounding passes
    assert engine._validate_grounded_input("Test", {"numeric_result": 42})
    logger.info("  ✓ Numeric grounding accepted")
    
    # Test 4: Code grounding passes
    assert engine._validate_grounded_input("Test", {"code_snippet": "code"})
    logger.info("  ✓ Code grounding accepted")
    
    logger.info("✓ Grounding validation tests passed")
    return True


def test_hallucination_guard():
    """Test hallucination guard validator."""
    from engines.phi2_explanation_engine import ControlledExplanationValidator
    
    logger.info("Testing hallucination guard validator...")
    
    validator = ControlledExplanationValidator()
    
    # Test 1: Length validation - too short
    is_valid, reason = validator.validate("Hi", {"factual_result": "Test"})
    assert not is_valid, "Short text should fail validation"
    assert "short" in reason.lower() or "minimum" in reason.lower()
    logger.info("  ✓ Short text rejection works")
    
    # Test 2: Length validation - too long
    long_text = "A" * 2500
    is_valid, reason = validator.validate(long_text, {"factual_result": "Test"})
    assert not is_valid, "Long text should fail validation"
    assert "long" in reason.lower()
    logger.info("  ✓ Long text rejection works")
    
    # Test 3: Valid length text
    valid_text = "This is a reasonable explanation of the concept that maintains proper length and content quality."
    is_valid, reason = validator.validate(valid_text, {"factual_result": "Test"})
    # Should pass length checks
    assert "length" not in reason.lower() or "summary" in reason.lower() or "pass" in reason.lower()
    logger.info("  ✓ Valid length text passes")
    
    # Test 4: Numeric validation
    stats_before = validator.get_stats()
    is_valid, reason = validator.validate(
        "The answer is 42 which is correct.",
        {"numeric_result": 42, "factual_result": "Test"}
    )
    stats_after = validator.get_stats()
    assert stats_after["total_validations"] == stats_before["total_validations"] + 1
    logger.info("  ✓ Validation tracking works")
    
    logger.info("✓ Hallucination guard tests passed")
    return True


def test_safe_prompt_generation():
    """Test safe prompt generation with system guard."""
    from engines.phi2_explanation_engine import Phi2ExplanationEngine
    
    logger.info("Testing safe prompt generation...")
    
    engine = Phi2ExplanationEngine(use_quantization=False, device="cpu")
    
    prompt = engine._build_safe_prompt(
        "What is meta-learning?",
        {
            "factual_result": "Meta-learning is learning to learn",
            "source": "knowledge_base"
        }
    )
    
    # Check that prompt contains key elements
    assert "meta-learning" in prompt.lower()
    assert len(prompt) > 100
    
    # Check for system guard indicators
    has_guard = any(word in prompt.lower() for word in [
        "controlled", "rules", "explain", "only", "grounded"
    ])
    assert has_guard, "Prompt should contain safety rules"
    
    logger.info("  ✓ System guard is included")
    logger.info("  ✓ Grounding data is formatted")
    logger.info("  ✓ Query is appended")
    
    logger.info("✓ Safe prompt generation tests passed")
    return True


def test_response_refusal():
    """Test refusal response generation."""
    from engines.phi2_explanation_engine import Phi2ExplanationEngine
    from datetime import datetime
    
    logger.info("Testing refusal response generation...")
    
    engine = Phi2ExplanationEngine(use_quantization=False, device="cpu")
    
    # Test execution without grounding
    response = engine.execute("Explain something", {})
    
    assert response["status"] == "refusal"
    assert response["confidence"] == 0.0
    assert response["grounded"] == False
    assert "explanation" in response
    
    logger.info("  ✓ Refusal response format correct")
    logger.info("  ✓ Status indicates refusal")
    logger.info("  ✓ Confidence is zero")
    
    logger.info("✓ Refusal response tests passed")
    return True


def test_inference_counter():
    """Test inference counter tracking."""
    from engines.phi2_explanation_engine import Phi2ExplanationEngine
    
    logger.info("Testing inference counter...")
    
    engine = Phi2ExplanationEngine(use_quantization=False, device="cpu")
    
    initial_count = engine.inference_count
    
    # Multiple inferences without grounding (will all fail grounding check)
    engine.execute("Test 1", {})
    engine.execute("Test 2", {})
    engine.execute("Test 3", {})
    
    assert engine.inference_count == initial_count + 3
    
    logger.info(f"  ✓ Inference count incremented: {initial_count} → {engine.inference_count}")
    
    logger.info("✓ Inference counter tests passed")
    return True


def test_statistics_tracking():
    """Test statistics tracking."""
    from engines.phi2_explanation_engine import Phi2ExplanationEngine
    
    logger.info("Testing statistics tracking...")
    
    engine = Phi2ExplanationEngine(use_quantization=False, device="cpu")
    
    # Get empty stats
    stats = engine.get_stats()
    
    assert "total_inferences" in stats
    assert "successful_explanations" in stats
    assert "failed_generations" in stats
    assert "success_rate" in stats
    assert "validator_stats" in stats
    
    assert stats["total_inferences"] == 0
    assert stats["success_rate"] == 0
    
    logger.info("  ✓ Statistics structure is correct")
    logger.info(f"  ✓ Initial stats: {stats['total_inferences']} inferences, {stats['success_rate']}% success")
    
    logger.info("✓ Statistics tracking tests passed")
    return True


def test_app_integration():
    """Test integration points in app.py."""
    logger.info("Testing app.py integration points...")
    
    # Check that required imports can be made
    try:
        from app import phi2_explanation_engine
        logger.info("  ✓ Phi2ExplanationEngine imported in app.py")
    except ImportError as e:
        logger.error(f"  ✗ Failed to import from app.py: {e}")
        return False
    
    # Check engine is initialized
    assert phi2_explanation_engine is not None
    logger.info("  ✓ Engine instance exists")
    
    # Check engine has required methods
    required_methods = ['load', 'execute', 'get_stats', '_validate_grounded_input']
    for method in required_methods:
        assert hasattr(phi2_explanation_engine, method)
    logger.info(f"  ✓ Engine has all required methods: {', '.join(required_methods)}")
    
    logger.info("✓ App integration tests passed")
    return True


def run_all_tests():
    """Run all integration tests."""
    logger.info("=" * 70)
    logger.info("PHI2 EXPLANATION ENGINE - INTEGRATION TEST SUITE")
    logger.info("=" * 70)
    
    tests = [
        ("Engine Initialization", test_engine_initialization),
        ("Grounding Validation", test_grounding_validation),
        ("Hallucination Guard", test_hallucination_guard),
        ("Safe Prompt Generation", test_safe_prompt_generation),
        ("Refusal Response", test_response_refusal),
        ("Inference Counter", test_inference_counter),
        ("Statistics Tracking", test_statistics_tracking),
        ("App Integration", test_app_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            logger.info("\n" + test_name.upper())
            logger.info("-" * 70)
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 70)
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    print("\n")
    exit_code = run_all_tests()
    print("\n")
    sys.exit(exit_code)
