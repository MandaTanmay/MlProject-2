
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from core.input_analyzer import InputAnalyzer
from engines.transformer_engine import TransformerEngine
from core.output_validator import OutputValidator

def test_transformer():
    analyzer = InputAnalyzer()
    engine = TransformerEngine()
    validator = OutputValidator()
    
    query = "explian c++"
    features = analyzer.analyze(query)
    
    print(f"Query: {query}")
    print(f"Is loaded: {engine.is_loaded}")
    
    if not engine.is_loaded:
        print("Transformer not loaded, skipping generation.")
        return

    result = engine.execute(query, features)
    print(f"Result answer: '{result['answer']}'")
    print(f"Result confidence: {result['confidence']}")
    print(f"Result strategy: {result['strategy']}")
    
    is_valid, validated_answer, details = validator.validate(
        answer=result['answer'],
        strategy=result['strategy'],
        confidence=result['confidence'],
        query=query
    )
    
    print(f"Is valid: {is_valid}")
    print(f"Validated answer: '{validated_answer}'")
    print(f"Issues: {details.get('issues')}")

if __name__ == "__main__":
    test_transformer()
