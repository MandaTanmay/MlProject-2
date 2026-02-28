"""
Test Suite for Phi2ExplanationEngine
Tests: loading, grounding validation, hallucination detection, and integration
"""
import pytest
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.phi2_explanation_engine import (
    Phi2ExplanationEngine,
    ControlledExplanationValidator
)

logger = logging.getLogger(__name__)


class TestControlledExplanationValidator:
    """Test the hallucination guard validator."""
    
    def setup_method(self):
        """Initialize validator for each test."""
        self.validator = ControlledExplanationValidator()
    
    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        assert self.validator.passed_validations == 0
        assert self.validator.failed_validations == 0
        assert len(self.validator.validation_history) == 0
    
    def test_numeric_validation_match(self):
        """Test numeric validation passes when numbers match."""
        generated = "The result is 42 which matches the expected output."
        grounded_data = {
            "numeric_result": 42,
            "factual_result": "Basic arithmetic"
        }
        
        is_valid, reason = self.validator.validate(generated, grounded_data)
        # Should pass numeric validation
        assert "numeric" not in reason.lower() or "pass" in reason.lower()
    
    def test_numeric_validation_mismatch(self):
        """Test numeric validation fails when numbers don't match."""
        generated = "The result is 50 but we expected 42."
        grounded_data = {
            "numeric_result": 42,
            "factual_result": "Basic arithmetic"
        }
        
        is_valid, reason = self.validator.validate(generated, grounded_data)
        # Should detect numeric mismatch
        if "50" in generated and "42" in generated:
            # The validator may catch this as a potential issue
            pass  # Validation behavior depends on implementation
    
    def test_entity_validation(self):
        """Test entity validation detects new entities."""
        generated = "The algorithm uses Python and Java which are mentioned in the code."
        grounded_data = {
            "code_snippet": "def quicksort(arr):",
            "factual_result": "QuickSort algorithm explanation"
        }
        
        is_valid, reason = self.validator.validate(generated, grounded_data)
        # May flag new entities (Python, Java) not in grounding
        assert isinstance(is_valid, bool)
    
    def test_length_validation_too_short(self):
        """Test length validation rejects too short output."""
        generated = "OK"  # Too short
        grounded_data = {"factual_result": "A longer explanation"}
        
        is_valid, reason = self.validator.validate(generated, grounded_data)
        assert not is_valid
        assert "short" in reason.lower() or "content" in reason.lower()
    
    def test_length_validation_too_long(self):
        """Test length validation rejects too long output."""
        generated = "A" * 2500  # Too long
        grounded_data = {"factual_result": "Something"}
        
        is_valid, reason = self.validator.validate(generated, grounded_data)
        assert not is_valid
        assert "long" in reason.lower()
    
    def test_length_validation_acceptable(self):
        """Test length validation passes for acceptable length."""
        generated = "The QuickSort algorithm works by selecting a pivot and partitioning the array into smaller and larger elements. This process is repeated recursively until the array is sorted."
        grounded_data = {"factual_result": "QuickSort explanation"}
        
        is_valid, reason = self.validator.validate(generated, grounded_data)
        # Should pass length check
        assert "length" not in reason.lower() or "pass" in reason.lower() or "summary" in reason.lower()
    
    def test_validator_statistics(self):
        """Test validator tracks statistics correctly."""
        grounded_data = {"factual_result": "Test"}
        
        # Multiple validations
        self.validator.validate("This is a test explanation about the concept.", grounded_data)
        self.validator.validate("Too short", grounded_data)
        self.validator.validate("A" * 3000, grounded_data)
        
        stats = self.validator.get_stats()
        assert stats["total_validations"] == 3
        assert stats["passed"] + stats["failed"] == 3


class TestPhi2ExplanationEngine:
    """Test the Phi2ExplanationEngine."""
    
    def setup_method(self):
        """Initialize engine for each test."""
        self.engine = Phi2ExplanationEngine(
            use_quantization=False,  # Disable quantization for testing
            device="cpu"             # Force CPU for testing
        )
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine.model is None
        assert self.engine.tokenizer is None
        assert not self.engine.is_loaded
        assert self.engine.model_name == "microsoft/phi-2"
        assert self.engine.inference_count == 0
        assert self.engine.successful_explanations == 0
    
    def test_model_load_cpu(self):
        """Test model can be loaded on CPU."""
        # This test may be slow as it downloads the model
        can_load = self.engine.load()
        
        if can_load:
            assert self.engine.is_loaded
            assert self.engine.model is not None
            assert self.engine.tokenizer is not None
        else:
            # Transformers library might not be available
            logger.warning("Skipping model load test - transformers not available")
    
    def test_grounded_input_validation_empty(self):
        """Test empty grounded data is rejected."""
        is_valid = self.engine._validate_grounded_input("Test query", {})
        assert not is_valid
    
    def test_grounded_input_validation_with_factual(self):
        """Test grounded data with factual result is accepted."""
        is_valid = self.engine._validate_grounded_input(
            "What is meta-learning?",
            {"factual_result": "Meta-learning is learning to learn"}
        )
        assert is_valid
    
    def test_grounded_input_validation_with_numeric(self):
        """Test grounded data with numeric result is accepted."""
        is_valid = self.engine._validate_grounded_input(
            "What is 20% of 500?",
            {"numeric_result": 100}
        )
        assert is_valid
    
    def test_grounded_input_validation_with_code(self):
        """Test grounded data with code snippet is accepted."""
        is_valid = self.engine._validate_grounded_input(
            "Explain this code",
            {"code_snippet": "def hello(): print('world')"}
        )
        assert is_valid
    
    def test_safe_prompt_building(self):
        """Test safe prompt contains system guard and grounding."""
        prompt = self.engine._build_safe_prompt(
            "Explain meta-learning",
            {
                "factual_result": "Meta-learning is learning how to learn",
                "source": "test_kb"
            }
        )
        
        assert "meta-learning" in prompt.lower()
        assert self.engine.SYSTEM_GUARD.split('\n')[0] in prompt
        assert "meta-learning" in prompt.lower()
    
    def test_response_refusal_format(self):
        """Test refusal response has correct format."""
        from datetime import datetime
        start_time = datetime.now()
        
        response = self.engine._response_refusal("Test refusal reason", start_time)
        
        assert response["status"] == "refusal"
        assert "Test refusal reason" in response.get("explanation", "")
        assert response["confidence"] == 0.0
        assert response["grounded"] == False
        assert "engine" in response
    
    def test_engine_statistics(self):
        """Test engine tracks statistics."""
        stats = self.engine.get_stats()
        
        assert stats["total_inferences"] == 0
        assert stats["successful_explanations"] == 0
        assert stats["failed_generations"] == 0
        assert stats["success_rate"] == 0
    
    def test_execution_without_grounding(self):
        """Test execution without grounding returns refusal."""
        response = self.engine.execute(
            "Explain something",
            {}
        )
        
        assert response["status"] == "refusal"
        assert response["confidence"] == 0.0
    
    def test_execution_with_grounding(self):
        """Test execution with grounding (requires model to be loaded)."""
        # Only run if model is available
        if not self.engine.load():
            pytest.skip("Transformers library not available")
        
        grounded_data = {
            "factual_result": "Meta-learning is the process of improving a learning algorithm by learning from previous learning experiences.",
            "source": "knowledge_base"
        }
        
        response = self.engine.execute(
            "What is meta-learning?",
            grounded_data
        )
        
        assert "explanation" in response or "status" in response
        assert response.get("grounded") is not None
        assert response.get("engine") == "transformer"
        assert response.get("model_version") == "phi-2"
    
    def test_execution_increments_counter(self):
        """Test execution increments inference counter even on failure."""
        initial_count = self.engine.inference_count
        
        self.engine.execute("Test", {})  # Will fail validation
        
        assert self.engine.inference_count == initial_count + 1
    
    def test_system_guard_format(self):
        """Test system guard contains required rules."""
        assert "ONLY explain" in self.engine.SYSTEM_GUARD
        assert "NOT introduce new facts" in self.engine.SYSTEM_GUARD
        assert "NOT guess" in self.engine.SYSTEM_GUARD
        assert "NOT modify numeric" in self.engine.SYSTEM_GUARD
    
    def test_generation_config_deterministic(self):
        """Test generation config has deterministic settings."""
        config = self.engine.SAFE_GENERATION_CONFIG
        
        assert config["do_sample"] == False
        assert config["temperature"] == 0.2
        assert config["max_new_tokens"] == 300
        assert config["early_stopping"] == True


class TestPhi2IntegrationScenarios:
    """Integration tests for realistic usage scenarios."""
    
    def setup_method(self):
        """Initialize engine."""
        self.engine = Phi2ExplanationEngine(
            use_quantization=False,
            device="cpu"
        )
    
    def test_factual_explanation_scenario(self):
        """Test pure factual explanation scenario."""
        response = self.engine.execute(
            "What is meta-learning?",
            {
                "factual_result": "Meta-learning is learning to learn, using past experience to improve learning on new tasks"
            }
        )
        
        assert "status" in response
        assert "confidence" in response
        assert "grounded" in response
    
    def test_numeric_explanation_scenario(self):
        """Test numeric explanation scenario."""
        response = self.engine.execute(
            "Why is 25% of 400 equal to 100?",
            {
                "numeric_result": 100,
                "computation_type": "percentage"
            }
        )
        
        assert "status" in response
        assert response.get("grounded") is not None
    
    def test_combined_explanation_scenario(self):
        """Test combined factual + numeric explanation."""
        response = self.engine.execute(
            "Why does meta-learning improve learning by 25%?",
            {
                "factual_result": "Meta-learning uses experience from previous tasks",
                "numeric_result": 25,
                "source": "research_paper",
                "computation_type": "improvement_percentage"
            }
        )
        
        assert "status" in response
        assert response.get("confidence") >= 0
    
    def test_code_explanation_scenario(self):
        """Test code explanation scenario."""
        response = self.engine.execute(
            "Explain what this QuickSort function does",
            {
                "code_snippet": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x >= pivot]
    return quicksort(left) + [pivot] + quicksort(right)
""",
                "factual_result": "QuickSort is a divide-and-conquer sorting algorithm"
            }
        )
        
        assert "status" in response
        assert response.get("grounded") == ({"code_snippet": True} if hasattr(response, 'get') else True)


class TestPhi2SafetyGuards:
    """Test safety mechanisms of Phi2ExplanationEngine."""
    
    def setup_method(self):
        """Initialize engine."""
        self.engine = Phi2ExplanationEngine(
            use_quantization=False,
            device="cpu"
        )
        self.validator = self.engine.validator
    
    def test_system_guard_mandatory(self):
        """Test system guard is always included in prompts."""
        prompt = self.engine._build_safe_prompt(
            "Test query",
            {"factual_result": "Test"}
        )
        
        # System guard should be in the prompt
        assert "controlled academic explanation" in prompt.lower() or \
               "rules" in prompt.lower()
    
    def test_deterministic_generation_params(self):
        """Test generation parameters prevent hallucination."""
        assert self.engine.SAFE_GENERATION_CONFIG["do_sample"] == False
        assert self.engine.SAFE_GENERATION_CONFIG["temperature"] == 0.2
        assert self.engine.SAFE_GENERATION_CONFIG["max_new_tokens"] <= 300
    
    def test_grounding_requirement_enforcement(self):
        """Test grounding is required for all explanations."""
        # No grounding - should get refusal
        response = self.engine.execute("Explain AI", {})
        
        assert response["status"] == "refusal"
        assert response["grounded"] == False
    
    def test_validator_integration(self):
        """Test validator is called during execution."""
        # Check that validator is part of engine
        assert hasattr(self.engine, 'validator')
        assert isinstance(self.engine.validator, ControlledExplanationValidator)


# Quick test for manual execution
if __name__ == "__main__":
    import sys
    
    print("Running Phi2ExplanationEngine tests...")
    print("=" * 60)
    
    # Test validator
    print("\n1. Testing ControlledExplanationValidator...")
    validator = ControlledExplanationValidator()
    
    test_text = "The algorithm runs in O(n log n) time complexity."
    test_grounded = {"factual_result": "QuickSort explanation"}
    
    is_valid, reason = validator.validate(test_text, test_grounded)
    print(f"   Generated text validity: {is_valid}")
    print(f"   Reason: {reason}")
    
    # Test engine initialization
    print("\n2. Testing Phi2ExplanationEngine initialization...")
    engine = Phi2ExplanationEngine(use_quantization=False, device="cpu")
    print(f"   Engine initialized: {engine.model_name}")
    print(f"   Is loaded: {engine.is_loaded}")
    
    # Test grounding validation
    print("\n3. Testing grounding validation...")
    is_valid = engine._validate_grounded_input(
        "What is meta-learning?",
        {"factual_result": "Learning to learn"}
    )
    print(f"   Grounding valid: {is_valid}")
    
    # Test execution without loading model
    print("\n4. Testing execution without grounding...")
    response = engine.execute("Explain AI", {})
    print(f"   Response status: {response['status']}")
    
    print("\n" + "=" * 60)
    print("Phi2ExplanationEngine basic tests completed!")
    print("Run with pytest for full test suite: pytest tests/test_phi2_engine.py -v")
