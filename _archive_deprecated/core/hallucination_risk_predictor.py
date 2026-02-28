"""
Hallucination Risk Predictor
Predicts the risk of hallucination for a given query.
HIGH_RISK queries are blocked from transformer and routed to retrieval or refusal.
"""
from typing import Tuple, Dict, Any
import re


class HallucinationRiskPredictor:
    """
    Predicts hallucination risk before answering queries.
    Uses rule-based heuristics and pattern matching.
    
    Risk Levels: LOW, MEDIUM, HIGH
    """
    
    RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]
    
    def __init__(self):
        """Initialize hallucination risk predictor."""
        # High-risk patterns that should NEVER be answered by generative models
        self.high_risk_patterns = [
            # Factual queries with specific entities
            r'\b(who is|who are)\s+\w+',  # "who is X"
            r'\b(what is the|what are the)\s+(capital|population|president|leader)',
            r'\b(when was|when did)\s+\w+',  # "when was X born"
            r'\b(where is|where are)\s+\w+',  # "where is X located"
            r'\bname of\b',  # "name of X"
            
            # Numbers and statistics
            r'\b(how many|how much)\b',
            r'\bnumber of\b',
            r'\bpopulation\b',
            r'\bprice\b',
            r'\bcost\b',
            
            # Dates and times
            r'\b\d{4}\b',  # Years
            r'\bdate\b',
            r'\btime\b',
            r'\byear\b',
            
            # Specific facts
            r'\bcapital of\b',
            r'\bpresident of\b',
            r'\bprime minister of\b',
            r'\bCEO of\b',
            r'\bfounder of\b',
            
            # Definitions requiring precision
            r'\bdefine\b',
            r'\bdefinition of\b',
            r'\bwhat does .+ stand for\b',
            r'\babbreviation\b',
            r'\bacronym\b',
        ]
        
        # Medium-risk patterns
        self.medium_risk_patterns = [
            # Comparative questions
            r'\bcompare\b',
            r'\bdifference between\b',
            r'\bvs\b',
            r'\bversus\b',
            
            # Technical specifications
            r'\bspecifications?\b',
            r'\bfeatures?\b',
            r'\badvantages?\b',
            r'\bdisadvantages?\b',
        ]
        
        # Safe patterns (low risk)
        self.low_risk_patterns = [
            r'\bexplain\b',
            r'\bhow does .+ work\b',
            r'\bwhy\b',
            r'\bdescribe\b',
            r'\bwhat is .+ concept\b',
            r'\bwhat is .+ idea\b',
        ]
    
    def predict(self, query: str, intent: str, features: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Predict hallucination risk for a query.
        
        Args:
            query: User query
            intent: Classified intent
            features: Query features
            
        Returns:
            Tuple of (risk_level, confidence, reason)
        """
        query_lower = query.lower()
        
        # Rule 1: UNSAFE intent always HIGH risk (should be blocked anyway)
        if intent == "UNSAFE":
            return "HIGH", 1.0, "Unsafe query - high hallucination risk"
        
        # Rule 2: NUMERIC queries are LOW risk if handled by ML engine
        if intent == "NUMERIC" and features.get("has_math_operators"):
            return "LOW", 0.9, "Numeric computation - low hallucination risk with ML engine"
        
        # Rule 3: Check high-risk patterns
        for pattern in self.high_risk_patterns:
            if re.search(pattern, query_lower):
                return "HIGH", 0.95, f"High-risk pattern detected - should use retrieval not generation"
        
        # Rule 4: FACTUAL queries are HIGH risk for transformers
        if intent == "FACTUAL":
            return "HIGH", 0.9, "Factual query - high hallucination risk if answered by transformer"
        
        # Rule 5: Check medium-risk patterns
        for pattern in self.medium_risk_patterns:
            if re.search(pattern, query_lower):
                return "MEDIUM", 0.7, "Medium-risk pattern - requires careful validation"
        
        # Rule 6: EXPLANATION queries are generally LOW risk
        if intent == "EXPLANATION":
            # But check for specific entities
            if any(word in query_lower for word in ['name', 'who', 'when', 'where', 'which']):
                return "MEDIUM", 0.7, "Explanation with specific entities - medium risk"
            return "LOW", 0.8, "Conceptual explanation - low hallucination risk"
        
        # Rule 7: Queries with digits but no math operators (e.g., "Python 3")
        if features.get("has_digits") and not features.get("has_math_operators"):
            return "MEDIUM", 0.7, "Query contains numbers - medium hallucination risk"
        
        # Default: MEDIUM risk
        return "MEDIUM", 0.6, "Default medium risk - validation recommended"
    
    def should_block_transformer(self, risk_level: str) -> bool:
        """
        Determine if transformer should be blocked based on risk level.
        
        Args:
            risk_level: Predicted risk level
            
        Returns:
            True if transformer should be blocked
        """
        return risk_level == "HIGH"
    
    def get_safe_routing(self, risk_level: str, original_engine: str) -> str:
        """
        Get safe engine routing based on risk level.
        
        Args:
            risk_level: Predicted risk level
            original_engine: Originally selected engine
            
        Returns:
            Safe engine to use
        """
        if risk_level == "HIGH":
            # HIGH risk: Force retrieval or rule
            if original_engine == "TRANSFORMER":
                return "RETRIEVAL"
            return original_engine
        
        # LOW or MEDIUM risk: Use original engine
        return original_engine
    
    def get_stats(self) -> dict:
        """Get hallucination risk predictor statistics."""
        return {
            "risk_levels": self.RISK_LEVELS,
            "high_risk_patterns": len(self.high_risk_patterns),
            "medium_risk_patterns": len(self.medium_risk_patterns),
            "blocking_policy": "HIGH risk blocks transformer routing"
        }
