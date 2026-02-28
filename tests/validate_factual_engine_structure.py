"""
Quick validation of FactualEngine implementation structure.
Does not require model download.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_factual_engine_code():
    """Validate that FactualEngine has all required methods."""
    
    print("\n" + "="*60)
    print("FACTUAL ENGINE STRUCTURE VALIDATION")
    print("="*60 + "\n")
    
    # Check if file exists
    engine_file = Path(__file__).parent.parent / "engines" / "retrieval_engine.py"
    assert engine_file.exists(), f"Engine file not found at {engine_file}"
    print(f"✓ Engine file found: {engine_file}")
    
    # Read the file
    with open(engine_file, 'r') as f:
        content = f.read()
    
    # Check for required class
    assert "class FactualEngine" in content, "FactualEngine class not found"
    print("✓ FactualEngine class defined")
    
    # Check for required methods
    required_methods = [
        "__init__",
        "_load_knowledge_base",
        "_precompute_embeddings",
        "execute",
        "_semantic_search",
        "_response_success",
        "_response_uncertain",
        "_response_ambiguous",
        "_response_error",
        "_log_retrieval",
        "get_stats",
        "add_fact",
        "validate_response",
        "clear_history",
        "reset_stats"
    ]
    
    for method in required_methods:
        assert f"def {method}" in content, f"Method {method} not found"
        print(f"✓ Method {method} defined")
    
    # Check for required constants
    assert "FACTUAL_CONFIDENCE_THRESHOLD = 0.65" in content, "Confidence threshold not found"
    print("✓ Confidence threshold = 0.65")
    
    assert "AMBIGUITY_MAX_DIFF = 0.05" in content, "Ambiguity threshold not found"
    print("✓ Ambiguity threshold = 0.05")
    
    # Check KB structure
    kb_file = Path(__file__).parent.parent / "data" / "knowledge_base.json"
    assert kb_file.exists(), f"Knowledge base not found at {kb_file}"
    with open(kb_file, 'r') as f:
        kb = json.load(f)
    
    assert "facts" in kb, "KB missing 'facts' field"
    assert len(kb["facts"]) > 0, "KB has no facts"
    print(f"✓ Knowledge base loaded with {len(kb['facts'])} facts")
    
    # Check fact structure
    sample_fact = kb["facts"][0]
    required_fields = ["id", "question", "answer", "structured_value", "category", "source", "verified", "verified_date"]
    for field in required_fields:
        assert field in sample_fact, f"Fact missing field: {field}"
    print(f"✓ Fact structure valid: {list(sample_fact.keys())}")
    
    # Check imports in file
    assert "from sentence_transformers import SentenceTransformer" in content, "SentenceTransformer import missing"
    assert "import numpy as np" in content, "numpy import missing"
    print("✓ Required imports present")
    
    # Check response structure methods
    assert '"status"' in content, "Status field in response structure"
    assert '"type": "FACTUAL"' in content, "Type field in response structure"
    assert '"confidence"' in content, "Confidence field in response structure"
    assert '"metadata"' in content, "Metadata field in response structure"
    print("✓ Response structure methods present")
    
    # Check semantic search signature
    assert "def _semantic_search(self, query: str)" in content, "Semantic search method signature incorrect"
    print("✓ Semantic search method signature correct")
    
    # Check confidence thresholding
    assert "self.FACTUAL_CONFIDENCE_THRESHOLD" in content, "Confidence thresholding logic missing"
    print("✓ Confidence thresholding logic present")
    
    # Check ambiguity detection
    assert "self.AMBIGUITY_MAX_DIFF" in content, "Ambiguity detection logic missing"
    print("✓ Ambiguity detection logic present")
    
    # Verify no old methods remain
    assert "def _search_wikipedia" not in content, "Old Wikipedia search method still present"
    assert "def _search_duckduckgo" not in content, "Old DuckDuckGo search method still present"
    assert "def _safe_refusal" not in content, "Old safe refusal method still present"
    print("✓ Old implementation methods removed")
    
    print("\n" + "="*60)
    print("✅ ALL STRUCTURE VALIDATIONS PASSED")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = validate_factual_engine_code()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
