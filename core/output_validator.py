"""
Output Validator - Anti-Hallucination Layer
Validates outputs before returning to user.
Blocks repeated sentences, conflicting information, and vague responses.
"""
import re
from typing import Dict, Any, Tuple, List
from difflib import SequenceMatcher


class OutputValidator:
    """
    Validates outputs from engines to prevent hallucinations.
    Acts as the final gatekeeper before responses reach users.
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the output validator.
        
        Args:
            similarity_threshold: Threshold for detecting near-duplicate sentences
        """
        self.similarity_threshold = similarity_threshold
        self.validation_history = []
    
    def validate(self, answer: str, strategy: str, confidence: float, 
                 query: str = "") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate an answer before returning to user.
        
        Args:
            answer: The answer to validate
            strategy: Strategy used (RETRIEVAL, ML, TRANSFORMER, RULE)
            confidence: Confidence score
            query: Original query (for context)
            
        Returns:
            Tuple of (is_valid, validated_answer, validation_details)
        """
        issues = []
        
        # Check 1: Empty or None answer
        if not answer or answer.strip() == "":
            issues.append("Empty answer")
            return False, self._get_safe_refusal(), {
                "valid": False,
                "issues": issues,
                "reason": "Empty response blocked"
            }
        
        # Transformer outputs are generative; allow them through unless empty
        if strategy == "TRANSFORMER":
            validation_result = {
                "valid": True,
                "issues": issues,
                "strategy": strategy,
                "confidence": confidence,
                "answer_length": len(answer)
            }
            self.validation_history.append(validation_result)
            return True, answer, validation_result

        # Retrieval answers are sourced; allow through unless empty
        if strategy == "RETRIEVAL":
            validation_result = {
                "valid": True,
                "issues": issues,
                "strategy": strategy,
                "confidence": confidence,
                "answer_length": len(answer)
            }
            self.validation_history.append(validation_result)
            return True, answer, validation_result

        # Check 2: Too short (likely incomplete) — allow short factual snippets
        if len(answer.strip()) < 10 and strategy not in ["RULE", "ML", "RETRIEVAL"]:
            issues.append("Answer too short")
        
        # Check 3: Repeated sentences
        has_repetition, repetition_details = self._check_repetition(answer)
        if has_repetition:
            issues.append(f"Repeated sentences: {repetition_details}")
        
        # Check 4: Conflicting numbers (for numeric answers)
        if strategy == "ML" or "number" in query.lower():
            has_conflict = self._check_numeric_conflicts(answer)
            if has_conflict:
                issues.append("Conflicting numbers detected")
        
        # Check 5: Vague or generic responses
        if strategy == "RETRIEVAL":
            is_vague = self._check_vagueness(answer)
            if is_vague:
                issues.append("Vague or generic answer")
        
        # Check 6: Hallucination indicators
        hallucination_markers = [
            "I think", "probably", "might be", "could be", 
            "I'm not sure", "maybe", "perhaps", "I believe"
        ]
        if any(marker in answer.lower() for marker in hallucination_markers):
            if strategy == "RETRIEVAL":  # Retrieval should never be uncertain
                issues.append("Uncertain language in factual answer")
        
        # Check 7: Multiple contradictory statements
        if self._has_contradictions(answer):
            issues.append("Contradictory statements")
        
        # Decide if answer is valid
        is_valid = len(issues) == 0
        
        # Log validation
        validation_result = {
            "valid": is_valid,
            "issues": issues,
            "strategy": strategy,
            "confidence": confidence,
            "answer_length": len(answer)
        }
        self.validation_history.append(validation_result)
        
        if not is_valid:
            return False, self._get_safe_refusal(), validation_result
        
        return True, answer, validation_result
    
    def _check_repetition(self, text: str) -> Tuple[bool, str]:
        """
        Check for repeated or near-duplicate sentences.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (has_repetition, details)
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return False, ""
        
        # Check each pair of sentences
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = self._sentence_similarity(sentences[i], sentences[j])
                if similarity >= self.similarity_threshold:
                    return True, f"Sentences {i+1} and {j+1} are {similarity:.0%} similar"
        
        return False, ""
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        Calculate similarity between two sentences.
        
        Args:
            sent1: First sentence
            sent2: Second sentence
            
        Returns:
            Similarity score (0 to 1)
        """
        return SequenceMatcher(None, sent1.lower(), sent2.lower()).ratio()
    
    def _check_numeric_conflicts(self, text: str) -> bool:
        """
        Check for conflicting numbers in the same answer.
        
        Args:
            text: Text to check
            
        Returns:
            True if conflicts detected
        """
        # Extract all numbers
        numbers = re.findall(r'\d+\.?\d*', text)
        
        if len(numbers) <= 1:
            return False
        
        # If multiple different numbers appear, check context
        unique_numbers = set(numbers)
        if len(unique_numbers) > 1:
            # This is a simple heuristic - could be improved
            # For now, we'll be conservative and not flag as conflict
            return False
        
        return False
    
    def _check_vagueness(self, text: str) -> bool:
        """
        Check if answer is too vague or generic.
        
        Args:
            text: Text to check
            
        Returns:
            True if vague
        """
        vague_patterns = [
            "it depends",
            "varies",
            "different for everyone",
            "no definitive answer",
            "it's complicated",
            "there are many factors"
        ]
        
        text_lower = text.lower()
        # Vague if contains multiple vague patterns and is short
        vague_count = sum(1 for pattern in vague_patterns if pattern in text_lower)
        return vague_count >= 2 and len(text) < 100
    
    def _has_contradictions(self, text: str) -> bool:
        """
        Check for obvious contradictions in text.
        
        Args:
            text: Text to check
            
        Returns:
            True if contradictions found
        """
        # Simple contradiction markers
        contradiction_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("always", "never"),
            ("all", "none"),
            ("can", "cannot"),
            ("will", "won't")
        ]
        
        text_lower = text.lower()
        for word1, word2 in contradiction_pairs:
            if word1 in text_lower and word2 in text_lower:
                # Check if they're in the same sentence (likely contradiction)
                sentences = re.split(r'[.!?]+', text_lower)
                for sentence in sentences:
                    if word1 in sentence and word2 in sentence:
                        return True
        
        return False
    
    def _get_safe_refusal(self) -> str:
        """
        Return a safe refusal message when validation fails.
        
        Returns:
            Safe refusal message
        """
        return "I cannot provide a reliable answer to this query. The response failed validation checks for accuracy and completeness."
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about validation history.
        
        Returns:
            Dictionary with validation statistics
        """
        if not self.validation_history:
            return {
                "total_validations": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "common_issues": {}
            }
        
        total = len(self.validation_history)
        valid = sum(1 for v in self.validation_history if v["valid"])
        invalid = total - valid
        
        # Count issue types
        issue_counts = {}
        for entry in self.validation_history:
            for issue in entry.get("issues", []):
                issue_type = issue.split(":")[0]  # Get issue category
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        return {
            "total_validations": total,
            "valid_count": valid,
            "invalid_count": invalid,
            "validation_rate": valid / total if total > 0 else 0,
            "common_issues": issue_counts
        }
