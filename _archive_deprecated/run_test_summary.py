"""
Quick test summary script - shows test results without verbose pytest output.
"""

import subprocess
import sys

def run_test_suite():
    """Run test suite and provide summary."""
    
    print("\n" + "="*80)
    print("SEMANTIC INTENT CLASSIFIER - TEST SUITE SUMMARY")
    print("="*80 + "\n")
    
    # Test groups
    test_groups = [
        ("ExecutionPlanner Tests", "tests/test_semantic_intent_classifier.py::TestExecutionPlanner"),
        ("SemanticIntentClassifier Tests", "tests/test_semantic_intent_classifier.py::TestSemanticIntentClassifier"),
        ("MetaController Tests", "tests/test_semantic_intent_classifier.py::TestMetaController"),
        ("Integration Tests", "tests/test_semantic_intent_classifier.py::TestIntegrationScenarios"),
    ]
    
    all_passed = 0
    all_failed = 0
    
    for group_name, test_path in test_groups:
        print(f"\n{group_name}:")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-q", "--tb=no"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            output = result.stdout + result.stderr
            
            # Extract pass/fail count
            for line in output.split('\n'):
                if 'passed' in line or 'failed' in line:
                    print(f"  {line.strip()}")
                    # Parse numbers
                    if 'passed' in line:
                        try:
                            passed = int(line.split()[0])
                            all_passed += passed
                        except:
                            pass
                    if 'failed' in line:
                        try:
                            import re
                            match = re.search(r'(\d+) failed', line)
                            if match:
                                all_failed += int(match.group(1))
                        except:
                            pass
        except subprocess.TimeoutExpired:
            print("  ⚠ Tests timed out")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*80)
    print(f"OVERALL SUMMARY: {all_passed} PASSED, {all_failed} FAILED")
    print("="*80 + "\n")
    
    return all_failed == 0

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
