"""
Engine Selector - Meta-ML Model
Uses Random Forest to intelligently select the best execution engine.
Learns from historical routing decisions and success rates.
Target Accuracy: > 85%
"""
from typing import Dict, Any, Tuple
from pathlib import Path
import joblib
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class EngineSelector:
    """
    Meta-ML model that predicts which engine should handle a query.
    Uses Random Forest trained on query features and historical performance.
    
    Engines: RETRIEVAL, ML, TRANSFORMER, RULE
    """
    
    ENGINES = ["RETRIEVAL", "ML", "TRANSFORMER", "RULE"]
    
    def __init__(self, model_dir: str = None):
        """
        Initialize engine selector.
        
        Args:
            model_dir: Directory containing trained models
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "training" / "models"
        
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False
        
        # Try to load trained model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load trained engine selection model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            model_path = self.model_dir / "engine_selector.joblib"
            
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.is_loaded = True
                print(f"✓ Engine selector loaded (Random Forest)")
                return True
            else:
                print(f"⚠ Engine selector model not found. Using rule-based routing.")
                print(f"   Expected: {model_path}")
                print(f"   Model will be trained automatically from routing logs.")
                return False
                
        except Exception as e:
            print(f"✗ Failed to load engine selector: {e}")
            return False
    
    def extract_features(self, intent: str, confidence: float, 
                        query_features: Dict[str, Any]) -> np.ndarray:
        """
        Extract features for engine selection.
        
        Args:
            intent: Classified intent (FACTUAL, NUMERIC, EXPLANATION, UNSAFE)
            confidence: Intent classification confidence
            query_features: Features from input analyzer
            
        Returns:
            Feature vector as numpy array
        """
        # Intent one-hot encoding
        intent_factual = 1 if intent == "FACTUAL" else 0
        intent_numeric = 1 if intent == "NUMERIC" else 0
        intent_explanation = 1 if intent == "EXPLANATION" else 0
        intent_unsafe = 1 if intent == "UNSAFE" else 0
        
        # Extract query features
        query_length = query_features.get("length", 0)
        word_count = query_features.get("word_count", 0)
        has_digits = 1 if query_features.get("has_digits", False) else 0
        digit_count = query_features.get("digit_count", 0)
        has_math_operators = 1 if query_features.get("has_math_operators", False) else 0
        has_question_words = 1 if query_features.get("has_question_words", False) else 0
        has_unsafe_keywords = 1 if query_features.get("has_unsafe_keywords", False) else 0
        
        # Derived features
        avg_word_length = query_length / max(word_count, 1)
        
        # Build feature vector
        features = np.array([
            intent_factual,
            intent_numeric,
            intent_explanation,
            intent_unsafe,
            confidence,
            query_length,
            word_count,
            has_digits,
            digit_count,
            has_math_operators,
            has_question_words,
            has_unsafe_keywords,
            avg_word_length
        ]).reshape(1, -1)
        
        return features
    
    def predict(self, intent: str, confidence: float, 
                query_features: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Predict which engine should handle the query.
        
        Args:
            intent: Classified intent
            confidence: Intent confidence score
            query_features: Query features
            
        Returns:
            Tuple of (engine_name, selection_confidence, reason)
        """
        if not self.is_loaded:
            # Fallback to rule-based selection
            return self._fallback_selection(intent, confidence, query_features)
        
        try:
            # Extract features
            features = self.extract_features(intent, confidence, query_features)
            
            # Predict engine
            engine = self.model.predict(features)[0]
            
            # Get confidence (probability of predicted class)
            probabilities = self.model.predict_proba(features)[0]
            engine_confidence = float(max(probabilities))
            
            # Generate reason
            reason = self._generate_reason(engine, intent, confidence, engine_confidence)
            
            return engine, engine_confidence, reason
            
        except Exception as e:
            print(f"✗ Engine selection error: {e}")
            return self._fallback_selection(intent, confidence, query_features)
    
    def _fallback_selection(self, intent: str, confidence: float, 
                           query_features: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Rule-based engine selection when ML model not available.
        This mimics the original meta_controller routing logic.
        
        Args:
            intent: Intent classification
            confidence: Intent confidence
            query_features: Query features
            
        Returns:
            Tuple of (engine, confidence, reason)
        """
        # Hard overrides
        if query_features.get("has_unsafe_keywords", False):
            return "RULE", 1.0, "Unsafe keywords detected - routing to RULE engine"
        
        # Intent-based routing
        routing_map = {
            "FACTUAL": "RETRIEVAL",
            "NUMERIC": "ML",
            "EXPLANATION": "TRANSFORMER",
            "UNSAFE": "RULE"
        }
        
        engine = routing_map.get(intent, "RULE")
        
        reasons = {
            "RETRIEVAL": f"Intent: {intent} (conf: {confidence:.2f}) → RETRIEVAL for verified facts",
            "ML": f"Intent: {intent} (conf: {confidence:.2f}) → ML for deterministic computation",
            "TRANSFORMER": f"Intent: {intent} (conf: {confidence:.2f}) → TRANSFORMER for explanations",
            "RULE": f"Intent: {intent} (conf: {confidence:.2f}) → RULE for safety filtering"
        }
        
        reason = reasons.get(engine, f"Default routing to {engine}")
        
        return engine, confidence, reason
    
    def _generate_reason(self, engine: str, intent: str, 
                        intent_confidence: float, 
                        engine_confidence: float) -> str:
        """
        Generate human-readable explanation for engine selection.
        
        Args:
            engine: Selected engine
            intent: Query intent
            intent_confidence: Intent classification confidence
            engine_confidence: Engine selection confidence
            
        Returns:
            Explanation string
        """
        reason = f"ML-based engine selection: {engine} "
        reason += f"(intent: {intent}, intent_conf: {intent_confidence:.2f}, "
        reason += f"engine_conf: {engine_confidence:.2f})"
        
        return reason
    
    def get_stats(self) -> dict:
        """
        Get engine selector statistics.
        
        Returns:
            Dictionary with selector stats
        """
        return {
            "model_loaded": self.is_loaded,
            "engines": self.ENGINES,
            "target_accuracy": "> 85%",
            "model_type": "Random Forest" if self.is_loaded else "Rule-based fallback",
            "feature_count": 13
        }
