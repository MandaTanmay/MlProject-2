"""
Input Analyzer - Pure Logic Only
Extracts features from user queries without ML.
"""
import re
from typing import Dict, Any


class InputAnalyzer:
    """Analyzes input queries using deterministic logic."""
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Extract features from query using pure logic.
        NO ML allowed in this component.
        
        Args:
            query: User input string
            
        Returns:
            Dictionary of extracted features
        """
        if not query or not isinstance(query, str):
            return {
                "length": 0,
                "word_count": 0,
                "has_digits": False,
                "digit_count": 0,
                "lowercase_text": "",
                "has_math_operators": False,
                "has_question_words": False,
                "question_type": None,
                "is_empty": True
            }
        
        cleaned = query.strip()
        lowercase = cleaned.lower()
        
        # Count features
        length = len(cleaned)
        words = cleaned.split()
        word_count = len(words)
        
        # Detect digits
        digits = re.findall(r'\d+', cleaned)
        has_digits = len(digits) > 0
        digit_count = len(digits)
        
        # Detect math operators
        math_operators = ['+', '-', '*', '/', 'multiply', 'multiplied', 'divide', 'divided', 'add', 'subtract', 'plus', 'minus', 'times']
        has_math_operators = any(op in lowercase for op in math_operators)
        
        # Detect question words
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'explain', 'describe', 'tell']
        has_question_words = any(word in lowercase for word in question_words)
        
        # Classify question type
        question_type = None
        if 'why' in lowercase or 'how' in lowercase or 'explain' in lowercase or 'describe' in lowercase:
            question_type = "EXPLANATION"
        elif 'what' in lowercase or 'which' in lowercase or 'who' in lowercase or 'when' in lowercase:
            question_type = "FACTUAL"
        elif has_math_operators and has_digits:
            question_type = "NUMERIC"
        
        # Detect unsafe patterns
        unsafe_keywords = ['hack', 'cheat', 'bypass', 'crack', 'exploit', 'steal', 'illegal', 'break into']
        has_unsafe_keywords = any(keyword in lowercase for keyword in unsafe_keywords)
        
        return {
            "length": length,
            "word_count": word_count,
            "has_digits": has_digits,
            "digit_count": digit_count,
            "lowercase_text": lowercase,
            "has_math_operators": has_math_operators,
            "has_question_words": has_question_words,
            "question_type": question_type,
            "is_empty": length == 0,
            "has_unsafe_keywords": has_unsafe_keywords,
            "original_text": cleaned
        }
