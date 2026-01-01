"""
Rule Engine - Safety and Restriction Enforcement
Blocks unsafe, restricted, or inappropriate queries.
Returns hard refusals with confidence 1.0.
"""
from typing import Dict, Any, List
import re


class RuleEngine:
    """
    Enforces safety rules and blocks restricted queries.
    This engine NEVER tries to answer - it only refuses.
    """
    
    def __init__(self):
        """Initialize the rule engine with safety patterns."""
        # Unsafe keywords and patterns
        self.unsafe_keywords = [
            'hack', 'cheat', 'bypass', 'crack', 'exploit', 
            'steal', 'illegal', 'break into', 'unauthorized',
            'password', 'phishing', 'malware', 'virus',
            'ddos', 'sql injection', 'xss', 'breach'
        ]
        
        # Academic integrity violations
        self.academic_violations = [
            'exam answers', 'test answers', 'homework answers',
            'solve my assignment', 'do my homework', 'complete my project',
            'write my essay', 'plagiarize', 'copy paste'
        ]
        
        # Harmful content
        self.harmful_patterns = [
            'how to hurt', 'how to harm', 'how to attack',
            'how to damage', 'make a weapon', 'create poison'
        ]
        
        self.refusal_count = 0
    
    def execute(self, query: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check query against safety rules and refuse if necessary.
        
        Args:
            query: User query
            features: Features from input analyzer
            
        Returns:
            Dictionary with answer, confidence, and reason
        """
        query_lower = query.lower()
        
        # Check for unsafe keywords
        matched_unsafe = [kw for kw in self.unsafe_keywords if kw in query_lower]
        if matched_unsafe:
            self.refusal_count += 1
            return {
                "answer": self._get_refusal_message("unsafe_content", matched_unsafe),
                "confidence": 1.0,
                "strategy": "RULE",
                "reason": f"Query blocked due to unsafe keywords: {', '.join(matched_unsafe)}",
                "blocked": True,
                "violation_type": "unsafe_content"
            }
        
        # Check for academic integrity violations
        matched_academic = [pattern for pattern in self.academic_violations if pattern in query_lower]
        if matched_academic:
            self.refusal_count += 1
            return {
                "answer": self._get_refusal_message("academic_integrity", matched_academic),
                "confidence": 1.0,
                "strategy": "RULE",
                "reason": f"Query blocked due to academic integrity concerns",
                "blocked": True,
                "violation_type": "academic_integrity"
            }
        
        # Check for harmful content
        matched_harmful = [pattern for pattern in self.harmful_patterns if pattern in query_lower]
        if matched_harmful:
            self.refusal_count += 1
            return {
                "answer": self._get_refusal_message("harmful_content", matched_harmful),
                "confidence": 1.0,
                "strategy": "RULE",
                "reason": f"Query blocked due to harmful content",
                "blocked": True,
                "violation_type": "harmful_content"
            }
        
        # If no violations found, this shouldn't have been routed here
        return {
            "answer": "Query routed to Rule Engine but no violations detected.",
            "confidence": 0.5,
            "strategy": "RULE",
            "reason": "No rule violations found",
            "blocked": False,
            "violation_type": None
        }
    
    def _get_refusal_message(self, violation_type: str, matched_patterns: List[str]) -> str:
        """
        Generate appropriate refusal message based on violation type.
        
        Args:
            violation_type: Type of violation
            matched_patterns: List of matched patterns
            
        Returns:
            Refusal message
        """
        messages = {
            "unsafe_content": (
                "I cannot assist with queries related to security exploits, "
                "unauthorized access, or potentially harmful activities. "
                "This type of content violates safety guidelines."
            ),
            "academic_integrity": (
                "I cannot help with completing assignments, exams, or homework directly. "
                "I can explain concepts and help you learn, but I won't provide direct answers "
                "that could be submitted as your own work."
            ),
            "harmful_content": (
                "I cannot provide information that could be used to cause harm to individuals "
                "or property. This request has been blocked for safety reasons."
            )
        }
        
        return messages.get(violation_type, "This query has been blocked by safety rules.")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about rule engine usage.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_refusals": self.refusal_count,
            "unsafe_keywords_count": len(self.unsafe_keywords),
            "academic_violations_count": len(self.academic_violations),
            "harmful_patterns_count": len(self.harmful_patterns)
        }
    
    def add_unsafe_keyword(self, keyword: str):
        """
        Add a new unsafe keyword to the blocklist.
        
        Args:
            keyword: Keyword to add
        """
        if keyword.lower() not in self.unsafe_keywords:
            self.unsafe_keywords.append(keyword.lower())
    
    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if query is safe without executing.
        
        Args:
            query: Query to check
            
        Returns:
            Dictionary with safety assessment
        """
        query_lower = query.lower()
        
        unsafe_matches = [kw for kw in self.unsafe_keywords if kw in query_lower]
        academic_matches = [p for p in self.academic_violations if p in query_lower]
        harmful_matches = [p for p in self.harmful_patterns if p in query_lower]
        
        is_safe = not (unsafe_matches or academic_matches or harmful_matches)
        
        return {
            "is_safe": is_safe,
            "unsafe_matches": unsafe_matches,
            "academic_matches": academic_matches,
            "harmful_matches": harmful_matches,
            "total_violations": len(unsafe_matches) + len(academic_matches) + len(harmful_matches)
        }
