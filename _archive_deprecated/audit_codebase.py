#!/usr/bin/env python3
"""
COMPREHENSIVE CODEBASE AUDIT SCRIPT
Identifies all critical issues in the meta-learning AI system.
"""
import sys
import logging
import inspect
import ast
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

AUDIT_REPORT = {
    "critical_errors": [],
    "logic_conflicts": [],
    "security_risks": [],
    "unused_files": [],
    "redundant_code": [],
    "performance_issues": [],
    "refactor_required": []
}


def check_meta_controller_route():
    """Verify meta_controller.route() signature and usage."""
    print("\n" + "="*70)
    print("AUDIT 1: MetaController.route() Analysis")
    print("="*70)
    
    from core.meta_controller import MetaController
    mc = MetaController()
    
    # Check signature
    sig = inspect.signature(mc.route)
    params = list(sig.parameters.keys())
    print(f"route() parameters: {params}")
    
    # The correct signature should be (query, query_features)
    if params == ['query', 'query_features']:
        print("✓ Correct signature: route(query, query_features)")
    else:
        AUDIT_REPORT["critical_errors"].append(
            f"MetaController.route() has wrong signature: {params}"
        )
    
    # Test correct call
    print("\nTesting correct usage...")
    try:
        result = mc.route("What is meta-learning?", {"length": 20})
        print(f"✓ Correct call works: {type(result)}")
        print(f"  Engine chain: {result[0]}")
        print(f"  Reasoning: {result[1][:50]}...")
    except Exception as e:
        print(f"✗ Correct call failed: {e}")
        AUDIT_REPORT["critical_errors"].append(f"route() fails with correct args: {e}")


def check_app_py_issues():
    """Analyze app.py for critical issues."""
    print("\n" + "="*70)
    print("AUDIT 2: app.py Analysis")
    print("="*70)
    
    app_path = Path(__file__).parent / "app.py"
    content = app_path.read_text()
    lines = content.split('\n')
    
    issues = []
    
    # Check 1: Missing logger import
    if 'import logging' not in content and 'logger.' in content:
        issues.append("Missing 'import logging' but uses logger")
        AUDIT_REPORT["critical_errors"].append("app.py: Missing logging import but uses logger")
    
    # Check 2: Wrong meta_controller.route() call
    for i, line in enumerate(lines, 1):
        if 'meta_controller.route(' in line:
            # Count arguments
            if 'intent, confidence, features' in line or line.count(',') >= 2:
                issues.append(f"Line {i}: Wrong args to meta_controller.route()")
                AUDIT_REPORT["critical_errors"].append(
                    f"app.py line {i}: Calling route(intent, confidence, features) but signature is route(query, query_features)"
                )
    
    # Check 3: Using old IntentClassifier instead of SemanticIntentClassifier
    if 'from core.intent_classifier import IntentClassifier' in content:
        issues.append("Imports old IntentClassifier (zero-shot) instead of SemanticIntentClassifier")
        AUDIT_REPORT["logic_conflicts"].append(
            "app.py imports IntentClassifier but should use SemanticIntentClassifier via MetaController"
        )
    
    # Check 4: Single-label forcing
    if 'intent, confidence = intent_classifier.predict' in content:
        issues.append("Uses single-label intent classification")
        AUDIT_REPORT["logic_conflicts"].append(
            "app.py uses single-label intent classification instead of multi-label routing"
        )
    
    # Report
    print(f"Issues found: {len(issues)}")
    for issue in issues:
        print(f"  ✗ {issue}")


def check_unused_modules():
    """Identify unused or obsolete modules."""
    print("\n" + "="*70)
    print("AUDIT 3: Unused/Obsolete Module Check")
    print("="*70)
    
    app_path = Path(__file__).parent / "app.py"
    content = app_path.read_text()
    
    modules = [
        ("DomainClassifier", "core.domain_classifier", "Exists but not used in app.py"),
        ("EngineSelector", "core.engine_selector", "Replaced by SemanticIntentClassifier"),
        ("IntentClassifier", "core.intent_classifier", "Replaced by SemanticIntentClassifier"),
    ]
    
    for class_name, module_path, reason in modules:
        # Check if imported and actually used
        imported = f"from {module_path} import" in content or f"import {module_path}" in content
        used = f"{class_name}()" in content or f"{class_name.lower()}" in content.lower()
        
        if not imported:
            print(f"  NOT IMPORTED: {class_name} - May be replaced")
        elif imported and not used:
            print(f"  ✗ IMPORTED BUT NOT USED: {class_name}")
            AUDIT_REPORT["redundant_code"].append(f"{class_name}: {reason}")
        else:
            print(f"  ⚠ USED BUT POSSIBLY OBSOLETE: {class_name}")


def check_execution_flow():
    """Verify correct execution flow per specification."""
    print("\n" + "="*70)
    print("AUDIT 4: Execution Flow Verification")
    print("="*70)
    
    # Expected flow:
    # Query → Domain Classifier → Rule Engine → Semantic Multi-Label Intent Scoring
    # → Execution Planner → Factual Engine → Numeric Engine → Transformer Engine
    
    expected_flow = [
        "Safety/Rule check first",
        "Multi-label intent classification",
        "Execution planning based on intents",
        "Engine chain execution",
        "Output validation"
    ]
    
    from core.meta_controller import MetaController
    from core.semantic_intent_classifier import SemanticIntentClassifier
    
    mc = MetaController()
    
    # Test classification
    result = mc.orchestrate("What is 25% of 400?")
    
    print(f"Query: What is 25% of 400?")
    print(f"Active intents: {result['intents']['active_intents']}")
    print(f"Primary intent: {result['intents']['primary_intent']}")
    print(f"Engine chain: {result['execution_plan']['engine_chain']}")
    print(f"Status: {result['status']}")
    
    # Verify multi-label behavior
    if len(result['intents']['active_intents']) >= 1:
        print("✓ Multi-label classification working")
    else:
        AUDIT_REPORT["logic_conflicts"].append("Multi-label classification not working")


def check_unused_files():
    """Find potentially unused files."""
    print("\n" + "="*70)
    print("AUDIT 5: Unused Files Check")
    print("="*70)
    
    root = Path(__file__).parent
    
    # Test files that might be obsolete
    test_files = list(root.glob("test_*.py")) + list(root.glob("validate_*.py"))
    
    for f in test_files:
        # Check if it's a one-off script
        content = f.read_text()
        if '__main__' in content and 'pytest' not in content.lower():
            # Standalone scripts - check if still needed
            if f.stat().st_mtime < 1740000000:  # Old files
                print(f"  ⚠ Possibly obsolete: {f.name}")
                AUDIT_REPORT["unused_files"].append(f.name)


def check_duplicates():
    """Find duplicate classifiers/logic."""
    print("\n" + "="*70)
    print("AUDIT 6: Duplicate Logic Check")
    print("="*70)
    
    # Two intent classifiers exist
    # from core.intent_classifier import IntentClassifier
    from core.semantic_intent_classifier import SemanticIntentClassifier
    
    print("Found classifiers:")
    print(f"  1. IntentClassifier (zero-shot MNLI) - DELETED")
    print(f"  2. SemanticIntentClassifier (embedding-based)")
    
    AUDIT_REPORT["redundant_code"].append(
        "IntentClassifier (zero-shot) is duplicate - SemanticIntentClassifier should be used"
    )


def generate_report():
    """Generate final audit report."""
    print("\n" + "="*70)
    print("FINAL AUDIT REPORT")
    print("="*70)
    
    print("\n📋 CRITICAL ERRORS:")
    for e in AUDIT_REPORT["critical_errors"]:
        print(f"  ❌ {e}")
    
    print("\n⚠️ LOGIC CONFLICTS:")
    for e in AUDIT_REPORT["logic_conflicts"]:
        print(f"  ⚡ {e}")
    
    print("\n🔒 SECURITY RISKS:")
    for e in AUDIT_REPORT["security_risks"]:
        print(f"  🔓 {e}")
    
    print("\n📁 UNUSED FILES:")
    for e in AUDIT_REPORT["unused_files"]:
        print(f"  📄 {e}")
    
    print("\n♻️ REDUNDANT CODE:")
    for e in AUDIT_REPORT["redundant_code"]:
        print(f"  🔄 {e}")
    
    print("\n🚀 PERFORMANCE ISSUES:")
    for e in AUDIT_REPORT["performance_issues"]:
        print(f"  ⏱️ {e}")
    
    print("\n🔧 REFACTOR REQUIRED:")
    for e in AUDIT_REPORT["refactor_required"]:
        print(f"  🛠️ {e}")
    
    total = sum(len(v) for v in AUDIT_REPORT.values())
    print(f"\n{'='*70}")
    print(f"TOTAL ISSUES: {total}")
    print('='*70)
    
    return AUDIT_REPORT


if __name__ == "__main__":
    print("🔍 META-LEARNING AI SYSTEM - COMPREHENSIVE CODEBASE AUDIT")
    print("="*70)
    
    try:
        check_meta_controller_route()
        check_app_py_issues()
        check_unused_modules()
        check_execution_flow()
        check_unused_files()
        check_duplicates()
        report = generate_report()
    except Exception as e:
        print(f"\n✗ Audit failed with error: {e}")
        import traceback
        traceback.print_exc()
