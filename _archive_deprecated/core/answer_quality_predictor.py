"""
Answer Quality Predictor
Predicts the quality of an answer before returning it to the user.
Rejects vague, repetitive, contradictory, low-confidence, or unsafe answers.
"""
from typing import Tuple, Dict, Any, List
import re
from difflib import SequenceMatcher


class AnswerQualityPredictor:
    """
    Predicts answer quality and blocks low-quality responses.
    
    Quality Levels: HIGH, MEDIUM, LOW, REJECT
    """
    
    QUALITY_LEVELS = ["HIGH", "MEDIUM", "LOW", "REJECT"]
    
    def __init__(self, min_acceptable_quality: str = "MEDIUM"):
        """
        Initialize answer quality predictor.
        
        Args:
            min_acceptable_quality: Minimum quality level to accept
        """
        self.min_acceptable_quality = min_acceptable_quality
        
        # Vague/generic phrases that indicate low quality
        self.vague_phrases = [
            "it depends",
            "there are many",
            "it varies",
            "generally speaking",
            "in most cases",
            "typically",
            "usually",
            "it's complicated",
            "it's complex",
            "hard to say",
            "difficult to determine",
            "not sure",
            "I don't know",
            "I cannot",
            "I'm not certain",
        ]
        
        # High-quality indicators
        self.quality_indicators = [
            "specifically",
            "precisely",
            "exactly",
            "for example",
            "such as",
            "including",
            "namely",
            "in particular",
        ]
    
    def predict(self, answer: str, strategy: str, confidence: float,
                query: str = "") -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict the quality of an answer.
        
        Args:
            answer: The answer to evaluate
            strategy: Strategy used (RETRIEVAL, ML, TRANSFORMER, RULE)
            confidence: Confidence score from engine
            query: Original query (for context)
            
        Returns:
            Tuple of (quality_level, quality_score, details)
        """
        issues = []
        quality_score = 1.0  # Start with perfect score
        
        # Check 1: Empty answer
        if not answer or len(answer.strip()) < 5:
            return "REJECT", 0.0, {
                "issues": ["Empty or too short"],
                "reason": "Answer must have substance"
            }
        
        # Check 2: Low confidence
        if confidence < 0.3:
            issues.append("Low confidence score")
            quality_score *= 0.5
        
        # Check 3: Vague/generic responses
        answer_lower = answer.lower()
        vague_count = sum(1 for phrase in self.vague_phrases if phrase in answer_lower)
        if vague_count > 2:
            issues.append(f"Too many vague phrases ({vague_count})")
            quality_score *= 0.6
        elif vague_count > 0:
            quality_score *= 0.9
        
        # Check 4: Repetition (for transformers)
        if strategy == "TRANSFORMER":
            has_repetition, rep_score = self._check_repetition(answer)
            if has_repetition:
                issues.append("Contains repeated sentences")
                quality_score *= rep_score
        
        # Check 5: Contradictions (for longer answers)
        if len(answer.split('.')) > 3:
            has_contradiction = self._check_contradictions(answer)
            if has_contradiction:
                issues.append("Contains contradictory statements")
                quality_score *= 0.4
        
        # Check 6: Unsafe content leaked
        unsafe_markers = ['hack', 'cheat', 'illegal', 'crack', 'steal']
        if any(marker in answer_lower for marker in unsafe_markers):
            return "REJECT", 0.0, {
                "issues": ["Unsafe content in answer"],
                "reason": "Answer contains unsafe content"
            }
        
        # Check 7: Refusal detection
        refusal_phrases = [
            "i cannot",
            "i can't",
            "i'm not able to",
            "i don't have",
            "restricted to",
            "outside my scope",
        ]
        is_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
        
        # Adjusted scoring for different strategies
        if strategy == "RETRIEVAL":
            # Retrieval should be high quality (sourced facts)
            if not issues:
                quality_score = min(quality_score, 0.95)
        
        elif strategy == "ML":
            # ML/Calculator should be perfect for correct computations
            if not issues:
                quality_score = 1.0
        
        elif strategy == "RULE":
            # Rule engine refusals are intentional
            if is_refusal:
                quality_score = 1.0
            else:
                quality_score *= 0.8
        
        elif strategy == "TRANSFORMER":
            # Transformers need extra scrutiny
            quality_score *= 0.9  # Inherent uncertainty in generation
            
            # Check for quality indicators
            indicator_count = sum(1 for ind in self.quality_indicators if ind in answer_lower)
            if indicator_count > 0:
                quality_score = min(1.0, quality_score * (1.0 + indicator_count * 0.05))
        
        # Determine quality level
        if quality_score >= 0.8:
            quality_level = "HIGH"
        elif quality_score >= 0.6:
            quality_level = "MEDIUM"
        elif quality_score >= 0.4:
            quality_level = "LOW"
        else:
            quality_level = "REJECT"
        
        details = {
            "quality_score": round(quality_score, 3),
            "issues": issues,
            "strategy": strategy,
            "confidence": confidence,
            "answer_length": len(answer),
            "vague_phrase_count": vague_count,
            "is_refusal": is_refusal
        }
        
        return quality_level, quality_score, details
    
    def should_reject(self, quality_level: str) -> bool:
        """
        Determine if answer should be rejected based on quality.
        
        Args:
            quality_level: Predicted quality level
            
        Returns:
            True if answer should be rejected
        """
        level_rank = {
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1,
            "REJECT": 0
        }
        
        min_rank = level_rank.get(self.min_acceptable_quality, 2)
        current_rank = level_rank.get(quality_level, 0)
        
        return current_rank < min_rank
    
    def _check_repetition(self, text: str, threshold: float = 0.85) -> Tuple[bool, float]:
        """
        Check for repeated sentences in text.
        
        Args:
            text: Text to check
            threshold: Similarity threshold for detecting repetition
            
        Returns:
            Tuple of (has_repetition, quality_multiplier)
        """
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return False, 1.0
        
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i+1:]:
                similarity = SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio()
                if similarity >= threshold:
                    return True, 0.5  # Severe penalty for repetition
        
        return False, 1.0
    
    def _check_contradictions(self, text: str) -> bool:
        """
        Check for contradictory statements (basic heuristic).
        
        Args:
            text: Text to check
            
        Returns:
            True if contradictions detected
        """
        text_lower = text.lower()
        
        # Check for explicit contradictions
        contradiction_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("always", "never"),
            ("all", "none"),
            ("is", "is not"),
            ("can", "cannot"),
            ("will", "will not"),
        ]
        
        for word1, word2 in contradiction_pairs:
            if word1 in text_lower and word2 in text_lower:
                # Simple check - could be refined with NLP
                return True
        
        return False
    
    def get_safe_fallback(self) -> str:
        """
        Get safe fallback answer for rejected responses.
        
        Returns:
            Safe fallback message
        """
        return "I cannot provide a reliable answer to this query. Please rephrase or ask a different question."
    
    def get_stats(self) -> dict:
        """Get answer quality predictor statistics."""
        return {
            "quality_levels": self.QUALITY_LEVELS,
            "min_acceptable": self.min_acceptable_quality,
            "vague_phrases": len(self.vague_phrases),
            "quality_indicators": len(self.quality_indicators),
            "rejection_policy": "Blocks LOW and REJECT quality answers"
        }
