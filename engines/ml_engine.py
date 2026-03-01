"""
ML Engine - Numeric Computation Only
Handles arithmetic and numerical operations deterministically.
NO transformers. NO text generation. EXACT answers only.
"""
import re
import operator
from typing import Dict, Any, Optional, List
import statistics


class MLEngine:
    """
    Handles numerical computations deterministically.
    Transformers must NEVER be used for math.
    """
    
    def __init__(self):
        """Initialize ML engine with operators."""
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow,
        }
        
        self.computation_history = []
    
    def execute(self, query: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute numerical computation.
        Args:
            query: User query
            features: Features from input analyzer
        Returns:
            Dictionary with answer, confidence, strategy
        """
        query_lower = features.get("lowercase_text", query.lower())

        # Try direct math expression evaluation first
        math_expr = re.sub(r'[^0-9\+\-\*/\.\(\) ]', '', query_lower)
        direct_result = self.compute_expression(math_expr)
        if direct_result is not None:
            self.computation_history.append({
                "query": query,
                "result": direct_result,
                "type": "direct_expression"
            })
            return {
                "answer": f"The answer is {direct_result}",
                "confidence": 1.0,
                "strategy": "ML",
                "computation_type": "direct_expression",
                "reason": "Direct math expression evaluation"
            }

        # ...existing code...
        """
        Execute numerical computation.
        
        Args:
            query: User query
            features: Features from input analyzer
            
        Returns:
            Dictionary with answer, confidence, strategy
        """
        query_lower = features.get("lowercase_text", query.lower())
        
        # Try different computation strategies
        
        # 1. Basic arithmetic
        result = self._parse_arithmetic(query_lower)
        if result is not None:
            self.computation_history.append({
                "query": query,
                "result": result,
                "type": "arithmetic"
            })
            return {
                "answer": f"The answer is {result}",
                "confidence": 1.0,
                "strategy": "ML",
                "computation_type": "arithmetic",
                "reason": "Deterministic arithmetic computation"
            }
        
        # 2. Average calculation
        result = self._parse_average(query_lower)
        if result is not None:
            self.computation_history.append({
                "query": query,
                "result": result,
                "type": "average"
            })
            return {
                "answer": f"The average is {result}",
                "confidence": 1.0,
                "strategy": "ML",
                "computation_type": "average",
                "reason": "Deterministic average computation"
            }
        
        # 3. Sum calculation
        result = self._parse_sum(query_lower)
        if result is not None:
            self.computation_history.append({
                "query": query,
                "result": result,
                "type": "sum"
            })
            return {
                "answer": f"The sum is {result}",
                "confidence": 1.0,
                "strategy": "ML",
                "computation_type": "sum",
                "reason": "Deterministic sum computation"
            }
        
        # If no computation strategy worked
        return {
            "answer": "I can perform arithmetic operations, averages, and sums, but I could not parse a valid numerical operation from your query.",
            "confidence": 0.5,
            "strategy": "ML",
            "computation_type": "none",
            "reason": "Could not parse numerical operation"
        }
    
    def _parse_arithmetic(self, query: str) -> Optional[float]:
        """
        Parse and compute basic arithmetic expressions.
        
        Args:
            query: Query string (lowercase)
            
        Returns:
            Computation result or None
        """
        # Extract numbers
        numbers = re.findall(r'-?\d+\.?\d*', query)
        
        if len(numbers) < 2:
            return None
        
        # Convert to float
        try:
            nums = [float(n) for n in numbers]
        except ValueError:
            return None
        
        # Detect operation
        if any(word in query for word in ['add', 'plus', '+', 'sum of']):
            return nums[0] + nums[1]
        
        elif any(word in query for word in ['subtract', 'minus', '-', 'difference']):
            return nums[0] - nums[1]
        
        elif any(word in query for word in ['multiply', 'times', '*', 'multiplied', 'product']):
            return nums[0] * nums[1]
        
        elif any(word in query for word in ['divide', 'divided', '/', 'division']):
            if nums[1] != 0:
                return nums[0] / nums[1]
            else:
                return None  # Division by zero
        
        elif any(word in query for word in ['power', 'exponent', '**', '^', 'raised to']):
            return nums[0] ** nums[1]
        
        return None
    
    def _parse_average(self, query: str) -> Optional[float]:
        """
        Parse and compute average of numbers.
        
        Args:
            query: Query string (lowercase)
            
        Returns:
            Average or None
        """
        if 'average' not in query and 'mean' not in query:
            return None
        
        # Extract all numbers
        numbers = re.findall(r'-?\d+\.?\d*', query)
        
        if len(numbers) < 2:
            return None
        
        try:
            nums = [float(n) for n in numbers]
            return statistics.mean(nums)
        except (ValueError, statistics.StatisticsError):
            return None
    
    def _parse_sum(self, query: str) -> Optional[float]:
        """
        Parse and compute sum of numbers.
        
        Args:
            query: Query string (lowercase)
            
        Returns:
            Sum or None
        """
        if 'sum' not in query and 'total' not in query:
            return None
        
        # Extract all numbers
        numbers = re.findall(r'-?\d+\.?\d*', query)
        
        if len(numbers) < 2:
            return None
        
        try:
            nums = [float(n) for n in numbers]
            return sum(nums)
        except ValueError:
            return None
    
    def compute_expression(self, expression: str) -> Optional[float]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Result or None
        """
        # Sanitize expression - only allow numbers and operators
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return None
        
        try:
            # Use eval carefully (only after sanitization)
            result = eval(expression)
            return float(result)
        except Exception:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about computations.
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.computation_history)
        
        if total == 0:
            return {
                "total_computations": 0,
                "computation_types": {}
            }
        
        # Count by type
        type_counts = {}
        for entry in self.computation_history:
            comp_type = entry.get("type", "unknown")
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        return {
            "total_computations": total,
            "computation_types": type_counts
        }
