#!/usr/bin/env python3
"""
Final validation script for Phi2ExplanationEngine integration
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from engines.phi2_explanation_engine import Phi2ExplanationEngine, ControlledExplanationValidator

print('\n' + '='*70)
print('PHI2 EXPLANATION ENGINE - FINAL VALIDATION')
print('='*70)

# Test 1: Engine initialization
print('\n1. Testing engine initialization...')
engine = Phi2ExplanationEngine(use_quantization=False, device='cpu')
assert engine.model_name == 'microsoft/phi-2'
assert not engine.is_loaded
print('   ✓ Engine initialized correctly')

# Test 2: Validator initialization
print('\n2. Testing validator initialization...')
validator = ControlledExplanationValidator()
assert validator.passed_validations == 0
assert validator.failed_validations == 0
print('   ✓ Validator initialized correctly')

# Test 3: Grounding validation
print('\n3. Testing grounding validation...')
assert not engine._validate_grounded_input('Test', {})
assert engine._validate_grounded_input('Test', {'factual_result': 'Result'})
assert engine._validate_grounded_input('Test', {'numeric_result': 42})
assert engine._validate_grounded_input('Test', {'code_snippet': 'code'})
print('   ✓ Grounding validation working')

# Test 4: Safe prompt generation
print('\n4. Testing safe prompt generation...')
prompt = engine._build_safe_prompt('Test', {'factual_result': 'Test fact'})
assert 'Test' in prompt
assert len(prompt) > 100
print('   ✓ Safe prompt generation working')

# Test 5: Execution without grounding
print('\n5. Testing execution without grounding...')
response = engine.execute('Test', {})
assert response['status'] == 'refusal'
assert response['confidence'] == 0.0
print('   ✓ Refusal response working')

# Test 6: Statistics tracking
print('\n6. Testing statistics tracking...')
stats = engine.get_stats()
assert 'total_inferences' in stats
assert stats['total_inferences'] == 1
print('   ✓ Statistics tracking working')

# Test 7: Safety mechanisms
print('\n7. Testing safety mechanisms...')
assert engine.SAFE_GENERATION_CONFIG['do_sample'] == False
assert engine.SAFE_GENERATION_CONFIG['temperature'] == 0.2
assert 'ONLY explain' in engine.SYSTEM_GUARD
print('   ✓ Safety mechanisms in place')

print('\n' + '='*70)
print('✅ ALL PHI2 VALIDATION TESTS PASSED!')
print('='*70)
print('\nComponent Status:')
print('  Engine:      Ready (not loaded - model not needed for tests)')
print('  Validator:   Ready')
print('  Grounding:   Ready')
print('  Safety:      Ready')
print('  Integration: Ready')
print('\nNext: Run full test suite with pytest or deploy to production')
print('='*70 + '\n')
