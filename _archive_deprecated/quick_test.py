#!/usr/bin/env python
"""Quick test runner to verify fixes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.input_analyzer import InputAnalyzer
from core.semantic_intent_classifier import SemanticIntentClassifier
from core.meta_controller import MetaController
from engines.rule_engine import RuleEngine

print("=" * 70)
print("QUICK TEST: Verifying Test Fixes")
print("=" * 70)

# Test 1: Input Analyzer - Numeric Detection
print("\n[1] InputAnalyzer - Numeric Detection")
analyzer = InputAnalyzer()
features = analyzer.analyze("20 multiplied by 8")
print(f"  has_digits: {features['has_digits']} (expected: True)")
print(f"  digit_count: {features['digit_count']} (expected: 2)")
print(f"  has_math_operators: {features['has_math_operators']} (expected: True)")
test1_pass = features["has_digits"] and features["digit_count"] == 2 and features["has_math_operators"]
print(f"  Result: {'✓ PASS' if test1_pass else '✗ FAIL'}")

# Test 2: Intent Classifier
print("\n[2] SemanticIntentClassifier - Basic Classification")
classifier = SemanticIntentClassifier()
result = classifier.classify("What is the capital of France?")
intent = result["primary_intent"]
confidence = result["scores"].get(intent, 0.5)
print(f"  Query: 'What is the capital of France?'")
print(f"  Intent: {intent} (expected: FACTUAL or EXPLANATION)")
print(f"  Confidence: {confidence:.3f} (expected: > 0.4)")
test2_pass = intent in ["FACTUAL", "EXPLANATION"] and confidence > 0.4
print(f"  Result: {'✓ PASS' if test2_pass else '✗ FAIL'}")

# Test 3: MetaController Routing
print("\n[3] MetaController - Routing")
controller = MetaController()
engine_chain, reason = controller.route("What is the capital of France?", {})
print(f"  Query: 'What is the capital of France?'")
print(f"  Engine Chain: {engine_chain}")
print(f"  Engine Type: {type(engine_chain)} (expected: list)")
valid_engines = ["RETRIEVAL", "FACTUAL", "ML_ENGINE", "TRANSFORMER", "RULE"]
test3_pass = isinstance(engine_chain, list) and len(engine_chain) > 0 and engine_chain[0] in valid_engines
print(f"  Result: {'✓ PASS' if test3_pass else '✗ FAIL'}")

# Test 4: RuleEngine - Unsafe Blocking
print("\n[4] RuleEngine - Unsafe Blocking")
rule_engine = RuleEngine()
result = rule_engine.execute("How to hack the system", {})
print(f"  Query: 'How to hack the system'")
print(f"  Blocked: {result['blocked']} (expected: True)")
print(f"  Status: {result['status']} (expected: blocked)")
print(f"  Confidence: {result['confidence']} (expected: 1.0)")
test4_pass = result["blocked"] and result["confidence"] == 1.0
print(f"  Result: {'✓ PASS' if test4_pass else '✗ FAIL'}")

# Summary
print("\n" + "=" * 70)
all_pass = test1_pass and test2_pass and test3_pass and test4_pass
print(f"OVERALL: {'✓ ALL TESTS PASS' if all_pass else '✗ SOME TESTS FAIL'}")
print("=" * 70)

sys.exit(0 if all_pass else 1)
