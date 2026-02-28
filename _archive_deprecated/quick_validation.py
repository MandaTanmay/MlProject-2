#!/usr/bin/env python
"""Quick validation - checks if semantic intent system loads."""

print("Starting validation...")

try:
    print("1. Importing SemanticIntentClassifier...")
    from core.semantic_intent_classifier import SemanticIntentClassifier, ExecutionPlanner
    print("   ✓ Import successful")
    
    print("\n2. Importing MetaController...")
    from core.meta_controller import MetaController
    print("   ✓ Import successful")
    
    print("\n3. Initializing ExecutionPlanner...")
    chains = ExecutionPlanner.EXECUTION_CHAINS
    print(f"   ✓ ExecutionPlanner has {len(chains)} defined chains")
    
    print("\n4. Testing ExecutionPlanner.plan_execution()...")
    engines, reason = ExecutionPlanner.plan_execution(["FACTUAL"])
    print(f"   ✓ FACTUAL → {engines}")
    
    engines, reason = ExecutionPlanner.plan_execution(["NUMERIC", "FACTUAL"])
    print(f"   ✓ NUMERIC+FACTUAL → {engines}")
    
    engines, reason = ExecutionPlanner.plan_execution(["UNSAFE"])
    print(f"   ✓ UNSAFE → {engines}")
    
    print("\n5. Creating SemanticIntentClassifier (loading model)...")
    classifier = SemanticIntentClassifier()
    print("   ✓ Classifier initialized")
    
    print("\n✓ ALL VALIDATION CHECKS PASSED")
    print("\nSystem is ready for full testing with pytest.")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
