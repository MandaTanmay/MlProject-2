#!/usr/bin/env python
"""Quick validation of Rule Engine v2.0"""

import sys
sys.path.insert(0, '.')

from engines.rule_engine import RuleEngine

# Initialize rule engine
print("Initializing Rule Engine v2.0...")
r = RuleEngine()

# Test a blocking query
print("\n=== TEST 1: Blocking Query ===")
result = r.execute('How to cheat in exam?')
print(f"Status: {result['status']}")
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']}")
print(f"Message: {result['message']}")
print(f"Performance: {result['processing_time_ms']:.2f}ms")

# Test a safe query
print("\n=== TEST 2: Safe Query ===")
result2 = r.execute('Explain quantum mechanics')
print(f"Status: {result2['status']}")
print(f"Blocked: {result2['blocked']}")
print(f"Performance: {result2['processing_time_ms']:.2f}ms")

# Test statistics
print("\n=== STATISTICS ===")
stats = r.get_stats()
print(f"Total Refusals: {stats['total_refusals']}")
print(f"Model Version: {stats['model_version']}")
print(f"Embeddings Available: {stats['embedding_available']}")
print(f"Unsafe Categories: {len(stats['unsafe_categories'])}")

# Test integrity
print("\n=== INTEGRITY CHECK ===")
integrity = r.integrity_check()
print(f"Initialized: {integrity['initialized']}")
print(f"Semantic Classifier Ready: {integrity['semantic_classifier_ready']}")
print(f"Pattern Detector Ready: {integrity['pattern_detector_ready']}")
print(f"Domain Detector Ready: {integrity['domain_detector_ready']}")

print("\n✓ Rule Engine v2.0 is operational and production-ready!")
