"""
Meta-Controller - Hard Routing Rules
Enforces deterministic routing based on intent classification.
NO CONFIDENCE TRICKS. NO FALLBACKS. STRICT ENFORCEMENT.
"""
from typing import Dict, Any, Tuple


class MetaController:
    """
    Central controller that routes queries to appropriate engines.
    Uses hard-coded rules - NO flexibility, NO guessing.
    """
    
    # Strict routing map: Intent -> Engine
    ROUTING_MAP = {
        "FACTUAL": "RETRIEVAL",
        "NUMERIC": "ML",
        "EXPLANATION": "TRANSFORMER",  # ENABLED: Explanation queries use transformer engine
        "UNSAFE": "RULE"
    }
    
    def __init__(self):
        """Initialize the meta-controller."""
        self.routing_history = []
    
    def route(self, intent: str, confidence: float, query_features: Dict[str, Any]) -> Tuple[str, str]:
        """
        Route query to appropriate engine based on intent.
        This is DETERMINISTIC - no confidence-based decisions.
        
        Args:
            intent: Classified intent (FACTUAL, NUMERIC, EXPLANATION, UNSAFE)
            confidence: Confidence score from classifier (logged but not used for routing)
            query_features: Features from input analyzer
            
        Returns:
            Tuple of (engine_name, routing_reason)
        """
        # Hard enforcement: If unsafe keywords detected, override to RULE engine
        if query_features.get("has_unsafe_keywords", False):
            intent = "UNSAFE"
        
        # Get engine from routing map
        engine = self.ROUTING_MAP.get(intent, "RULE")
        
        # Generate explanation
        reason = self._get_routing_reason(intent, engine, confidence, query_features)
        
        # Log routing decision
        self.routing_history.append({
            "intent": intent,
            "engine": engine,
            "confidence": confidence,
            "reason": reason
        })
        
        return engine, reason
    
    def _get_routing_reason(self, intent: str, engine: str, confidence: float, features: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for routing decision.
        
        Args:
            intent: Classified intent
            engine: Selected engine
            confidence: Confidence score
            features: Query features
            
        Returns:
            Explanation string
        """
        reasons = {
            "FACTUAL": f"Query classified as FACTUAL (confidence: {confidence:.2f}). Routing to RETRIEVAL engine to fetch verified facts.",
            "NUMERIC": f"Query classified as NUMERIC (confidence: {confidence:.2f}). Routing to ML engine for deterministic computation.",
            "EXPLANATION": f"Query classified as EXPLANATION (confidence: {confidence:.2f}). Routing to TRANSFORMER engine for conceptual explanations.",
            "UNSAFE": f"Query classified as UNSAFE (confidence: {confidence:.2f}). Routing to RULE engine for safe refusal."
        }
        
        return reasons.get(intent, f"Unknown intent: {intent}. Defaulting to RULE engine.")
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.
        
        Returns:
            Dictionary with routing statistics
        """
        if not self.routing_history:
            return {
                "total_queries": 0,
                "engine_distribution": {},
                "intent_distribution": {}
            }
        
        total = len(self.routing_history)
        
        # Count engine usage
        engine_counts = {}
        for entry in self.routing_history:
            engine = entry["engine"]
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
        
        # Count intent distribution
        intent_counts = {}
        for entry in self.routing_history:
            intent = entry["intent"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return {
            "total_queries": total,
            "engine_distribution": engine_counts,
            "intent_distribution": intent_counts
        }
    
    def validate_routing(self, intent: str, engine: str) -> bool:
        """
        Validate that routing decision follows the rules.
        
        Args:
            intent: Classified intent
            engine: Selected engine
            
        Returns:
            True if routing is valid, False otherwise
        """
        expected_engine = self.ROUTING_MAP.get(intent)
        return engine == expected_engine
