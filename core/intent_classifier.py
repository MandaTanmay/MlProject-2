"""
Intent Classifier - Machine Learning Component
Uses DistilBERT MNLI for zero-shot classification of query intent.
This is the ONLY ML component that learns routing decisions.
"""
import os
from typing import Tuple, Optional
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠ transformers library not installed. Install with: pip install transformers torch")


class IntentClassifier:
    """
    Zero-shot classifier using DistilBERT MNLI to classify query intent.
    Decides which engine should handle the query.
    """
    
    INTENTS = ["FACTUAL", "NUMERIC", "EXPLANATION", "UNSAFE"]
    
    # Intent labels for zero-shot classification
    INTENT_LABELS = [
        "factual information query",
        "numerical calculation or math problem",
        "explanation or conceptual question",
        "unsafe or malicious request"
    ]
    
    def __init__(self, model_name: str = "typeform/distilbert-base-uncased-mnli"):
        """
        Initialize the intent classifier with DistilBERT MNLI.
        
        Args:
            model_name: HuggingFace model name for zero-shot classification
        """
        self.model_name = model_name
        self.classifier = None
        self.is_loaded = False
        
        # Try to load the model
        if TRANSFORMERS_AVAILABLE:
            self.load_model()
        else:
            print("⚠ Intent classifier disabled - transformers library not available")
    
    def load_model(self) -> bool:
        """
        Load DistilBERT MNLI zero-shot classifier.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            print(f"Loading intent classifier: {self.model_name}...")
            self.classifier = pipeline("zero-shot-classification", model=self.model_name)
            self.is_loaded = True
            print(f"✓ Intent classifier loaded ({self.model_name})")
            return True
        except Exception as e:
            print(f"✗ Failed to load intent classifier: {e}")
            return False
    
    def predict(self, query: str) -> Tuple[str, float]:
        """
        Predict the intent of a query using zero-shot classification.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        if not self.is_loaded:
            # Fallback to rule-based classification if model not loaded
            return self._fallback_prediction(query)
        
        try:
            # Zero-shot classification
            result = self.classifier(query, self.INTENT_LABELS)
            
            # Map predicted label back to intent
            predicted_label = result['labels'][0]
            confidence = result['scores'][0]
            
            # Map label to intent
            label_to_intent = {
                "factual information query": "FACTUAL",
                "numerical calculation or math problem": "NUMERIC",
                "explanation or conceptual question": "EXPLANATION",
                "unsafe or malicious request": "UNSAFE"
            }
            
            intent = label_to_intent.get(predicted_label, "FACTUAL")
            
            return intent, float(confidence)
            
        except Exception as e:
            print(f"✗ Prediction error: {e}")
            return self._fallback_prediction(query)
    
    def _fallback_prediction(self, query: str) -> Tuple[str, float]:
        """
        Fallback rule-based classification when model isn't available.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        query_lower = query.lower()
        
        # Check for unsafe patterns
        unsafe_keywords = ['hack', 'cheat', 'bypass', 'crack', 'exploit', 'steal', 'illegal', 'break into']
        if any(keyword in query_lower for keyword in unsafe_keywords):
            return "UNSAFE", 1.0
        
        # Check for numeric patterns
        math_operators = ['+', '-', '*', '/', 'multiply', 'divide', 'add', 'subtract', 'plus', 'minus', 'times', 'average', 'sum']
        has_math = any(op in query_lower for op in math_operators)
        has_numbers = any(char.isdigit() for char in query)
        if has_math and has_numbers:
            return "NUMERIC", 0.9
        
        # Check for explanation patterns
        explanation_words = ['why', 'how', 'explain', 'describe', 'what is', 'what are', 'tell me about']
        if any(word in query_lower for word in explanation_words):
            # Further check if it's asking for explanation or fact
            if query_lower.startswith('why') or query_lower.startswith('how') or 'explain' in query_lower:
                return "EXPLANATION", 0.85
            else:
                return "FACTUAL", 0.85
        
        # Default to factual
        return "FACTUAL", 0.7
    
    def get_all_intents(self):
        """Return list of all possible intents."""
        return self.INTENTS.copy()
