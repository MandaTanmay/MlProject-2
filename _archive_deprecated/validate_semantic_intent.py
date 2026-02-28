"""
Manual validation of semantic intent classifier system.
Tests the 6 required test cases + key functionality.
"""

import sys
import time
from core.semantic_intent_classifier import SemanticIntentClassifier, ExecutionPlanner
from core.meta_controller import MetaController


def test_required_cases():
    """Test the 6 required test cases from specification."""
    
    print("\n" + "="*80)
    print("TESTING 6 REQUIRED SEMANTIC INTENT CLASSIFIER CASES")
    print("="*80 + "\n")
    
    classifier = SemanticIntentClassifier()
    controller = MetaController()
    
    test_cases = [
        ("What is the capital of Germany?", "FACTUAL", ["FACTUAL"]),
        ("What is 20% of 500?", "NUMERIC", ["NUMERIC"]),
        ("Explain why 20% of 500 is 100.", "NUMERIC+EXPLANATION", ["NUMERIC", "EXPLANATION"]),
        ("Explain the capital of Germany.", "FACTUAL+EXPLANATION", ["FACTUAL", "EXPLANATION"]),
        ("What is 5 times the population of Germany?", "FACTUAL+NUMERIC", ["FACTUAL", "NUMERIC"]),
        ("How to cheat on an exam?", "UNSAFE", ["UNSAFE"]),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_desc, expected_intents in test_cases:
        print(f"\nTest: {expected_desc}")
        print(f"Query: {query!r}")
        
        try:
            # Classify
            result = classifier.classify(query)
            active = result["active_intents"]
            scores = result["scores"]
            
            print(f"  Classification:")
            for intent, score in scores.items():
                marker = "✓" if intent in active else " "
                print(f"    [{marker}] {intent}: {score:.2f}")
            
            # Check execution plan
            plan = controller.orchestrate(query)
            engines = plan["execution_plan"]["engine_chain"]
            
            print(f"  Execution Plan: {' → '.join(engines) if engines else 'NONE'}")
            
            # Verify expected intents
            if "UNSAFE" in expected_intents and "UNSAFE" in active:
                print(f"  ✓ PASS: UNSAFE correctly detected")
                passed += 1
            elif "UNSAFE" in expected_intents:
                print(f"  ✗ FAIL: UNSAFE not detected")
                failed += 1
            elif all(intent in active for intent in expected_intents):
                print(f"  ✓ PASS: Expected intents {expected_intents} found in {active}")
                passed += 1
            else:
                print(f"  ✗ FAIL: Expected {expected_intents}, got {active}")
                failed += 1
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("="*80 + "\n")
    
    return failed == 0


def test_performance():
    """Test performance requirements."""
    
    print("\n" + "="*80)
    print("PERFORMANCE VALIDATION")
    print("="*80 + "\n")
    
    controller = MetaController()
    
    # Warm up
    controller.orchestrate("What is the capital of France?")
    
    # Test single query performance
    print("Single Query Performance:")
    start = time.time()
    for _ in range(5):
        controller.orchestrate("What is 5 times the population of Germany?")
    elapsed = (time.time() - start) * 1000 / 5
    
    print(f"  Average latency: {elapsed:.1f}ms")
    print(f"  Requirement: <200ms")
    
    if elapsed < 200:
        print(f"  ✓ PASS")
        perf_pass = True
    else:
        print(f"  ✗ FAIL")
        perf_pass = False
    
    # Test batch performance
    print("\nBatch Performance (10 queries):")
    queries = [
        "What is the capital of Germany?",
        "What is 20% of 500?",
        "Explain photosynthesis",
        "What is the largest planet?",
        "Calculate 5 + 3 * 2",
        "Why do we sleep?",
        "What is the speed of light?",
        "Convert 100 kilometers to miles",
        "Describe machine learning",
        "What is the smallest country?"
    ]
    
    start = time.time()
    for query in queries:
        controller.orchestrate(query)
    elapsed = (time.time() - start) * 1000 / len(queries)
    
    print(f"  Average latency: {elapsed:.1f}ms per query")
    print(f"  Total time: {(time.time() - start) * 1000:.0f}ms for {len(queries)} queries")
    
    return perf_pass


def test_integration():
    """Test system integration."""
    
    print("\n" + "="*80)
    print("INTEGRATION TESTS")
    print("="*80 + "\n")
    
    controller = MetaController()
    
    # Test 1: Orchestration format
    print("Test 1: Orchestration Plan Format")
    plan = controller.orchestrate("What is the capital of France?")
    
    required_keys = ["status", "intents", "execution_plan", "metadata"]
    has_all_keys = all(key in plan for key in required_keys)
    
    if has_all_keys:
        print("  ✓ PASS: All required keys present")
        test1_pass = True
    else:
        print(f"  ✗ FAIL: Missing keys. Required: {required_keys}, Got: {list(plan.keys())}")
        test1_pass = False
    
    # Test 2: UNSAFE blocking
    print("\nTest 2: UNSAFE Query Blocking")
    plan = controller.orchestrate("How to harm someone?")
    
    if "UNSAFE" in plan["intents"]["active_intents"] or plan["execution_plan"]["engine_chain"] == ["RULE_ENGINE"]:
        print("  ✓ PASS: UNSAFE query detected or blocked")
        test2_pass = True
    else:
        print(f"  ✗ FAIL: UNSAFE not properly handled. Plan: {plan['execution_plan']['engine_chain']}")
        test2_pass = False
    
    # Test 3: Backward compatibility
    print("\nTest 3: Backward Compatibility (route method)")
    engines, reasoning = controller.route("What is the capital of France?")
    
    if isinstance(engines, list) and len(engines) > 0 and isinstance(reasoning, str):
        print(f"  ✓ PASS: route() returns ({engines}, '{reasoning[:50]}...')")
        test3_pass = True
    else:
        print(f"  ✗ FAIL: route() format incorrect")
        test3_pass = False
    
    # Test 4: Statistics tracking
    print("\nTest 4: Statistics Tracking")
    stats = controller.get_routing_stats()
    
    if "total_queries" in stats and stats["total_queries"] > 0:
        print(f"  ✓ PASS: Statistics tracked ({stats['total_queries']} queries)")
        test4_pass = True
    else:
        print(f"  ✗ FAIL: Statistics not properly tracked")
        test4_pass = False
    
    return test1_pass and test2_pass and test3_pass and test4_pass


def main():
    """Run all validation tests."""
    
    print("\n" + "█"*80)
    print("SEMANTIC INTENT CLASSIFIER - COMPREHENSIVE VALIDATION")
    print("█"*80)
    
    try:
        required_pass = test_required_cases()
        perf_pass = test_performance()
        integration_pass = test_integration()
        
        # Summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        results = [
            ("6 Required Test Cases", required_pass),
            ("Performance Validation", perf_pass),
            ("Integration Tests", integration_pass),
        ]
        
        total_pass = sum(1 for _, p in results if p)
        
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {name}")
        
        print(f"\nOVERALL: {total_pass}/{len(results)} test groups passed\n")
        
        return total_pass == len(results)
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
