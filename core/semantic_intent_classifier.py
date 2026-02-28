"""
Semantic Intent Classifier - Multi-Label Intent Scoring
Replaces zero-shot classification with embedding-based semantic similarity.

Enables:
- Confidence scores for ALL intents (not single-label forcing)
- Multi-intent activation (hybrid queries)
- Deterministic semantic routing
- Fast inference (<100ms per query)
- Explainable scores and active intents

Architecture:
- Encodes intent prototypes once at startup
- Computes query similarity to all prototypes
- Returns scores + active intents list
- Threshold-based activation (default: 0.60)
"""

from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


logger = logging.getLogger(__name__)


class SemanticIntentClassifier:
    """
    Multi-label semantic intent classifier using embedding similarity.
    
    Replaces single-label zero-shot classification with deterministic,
    confidence-aware multi-intent scoring.
    """
    
    # Intent prototypes - semantic anchors for each intent
    INTENT_PROTOTYPES = {
        "FACTUAL": [
            "This query asks for factual academic information or verified data.",
            "The user wants to know factual details or retrieve specific information.",
            "This is a question about facts, definitions, or verifiable knowledge.",
            "This query asks about college regulations, policies, or institutional rules.",
            "The user wants to know about attendance requirements, credits, grading, or academic policies.",
            "This is asking about a specific institution, college, university, or programme details.",
            "What are the rules for admission, evaluation, or examination?",
            "Tell me about the college, its accreditation, affiliation, or regulations.",
        ],
        "NUMERIC": [
            "This query requires mathematical calculation, arithmetic computation, or numerical processing.",
            "The user asks for mathematical solving, numerical operations, or calculations.",
            "This involves math problems, numerical analysis, or quantitative operations.",
            "Calculate the sum, average, difference, or perform arithmetic on numbers.",
            "What is 2 plus 2? Solve this equation. Compute the total.",
        ],
        "EXPLANATION": [
            "This query asks for conceptual explanation or reasoning behind a result.",
            "The user wants to understand why something is true or how something works.",
            "This requires explaining concepts, mechanisms, or the logic behind facts.",
        ],
        "UNSAFE": [
            "This query requests harmful, unethical, illegal, or academic misconduct content.",
            "The user is asking for something that could cause harm or violate rules.",
            "This is a request for unsafe, illegal, or unethical information.",
        ]
    }
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        intent_threshold: float = 0.60,
        unsafe_threshold: float = 0.50
    ):
        """
        Initialize the semantic intent classifier.
        
        Args:
            model_name: Sentence transformer model to use
            intent_threshold: Minimum similarity threshold for intent activation (0-1)
            unsafe_threshold: Lower threshold for UNSAFE (more conservative)
        """
        self.model_name = model_name
        self.intent_threshold = intent_threshold
        self.unsafe_threshold = unsafe_threshold
        
        self.model = None
        self.has_embeddings = HAS_EMBEDDINGS
        self.prototype_embeddings = {}
        self.intents = list(self.INTENT_PROTOTYPES.keys())
        
        # Performance tracking
        self.total_classifications = 0
        self.avg_classification_time = 0.0
        
        # Initialize model
        if self.has_embeddings:
            try:
                self.model = SentenceTransformer(model_name)
                self._encode_prototypes()
                logger.info(f"✓ Semantic Intent Classifier initialized with {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embeddings model: {e}")
                self.has_embeddings = False
    
    def _encode_prototypes(self):
        """
        Pre-encode all intent prototypes once at startup.
        This ensures fast inference - no re-encoding per request.
        """
        try:
            for intent, statements in self.INTENT_PROTOTYPES.items():
                # Encode all statements for this intent
                embeddings = self.model.encode(statements, normalize_embeddings=True)
                
                # Average the embeddings (or could use max pooling)
                # Using mean is more stable
                self.prototype_embeddings[intent] = np.mean(embeddings, axis=0)
            
            logger.info(f"✓ Encoded {len(self.INTENT_PROTOTYPES)} intent prototypes")
        except Exception as e:
            logger.error(f"Error encoding prototypes: {e}")
            self.has_embeddings = False
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query into multi-label intents with confidence scores.
        
        Args:
            query: User query to classify
            
        Returns:
            Dictionary with:
            - scores: Dict[intent, float] with similarity scores
            - active_intents: List[intent] that exceed threshold
            - primary_intent: Intent with highest score
            - threshold: Threshold used
            - model: Model name
            - timestamp: Classification timestamp
        """
        start_time = datetime.now()
        
        if not self.has_embeddings or self.model is None:
            return self._fallback_classification(query)
        
        try:
            # Encode query using same model
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            
            # Compute similarity to each intent prototype
            scores = {}
            for intent, prototype_embedding in self.prototype_embeddings.items():
                # Cosine similarity (already normalized)
                similarity = float(np.dot(query_embedding, prototype_embedding))
                scores[intent] = similarity
            
            # Determine active intents based on thresholds
            active_intents = self._get_active_intents(scores)
            
            # Find primary intent (highest score)
            primary_intent = max(scores, key=scores.get)
            
            # Calculate classification time
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.total_classifications += 1
            self.avg_classification_time = (
                (self.avg_classification_time * (self.total_classifications - 1) + elapsed_ms)
                / self.total_classifications
            )
            
            return {
                "scores": {intent: round(score, 4) for intent, score in scores.items()},
                "active_intents": active_intents,
                "primary_intent": primary_intent,
                "primary_confidence": round(scores[primary_intent], 4),
                "threshold": self.intent_threshold,
                "model": self.model_name.split("/")[-1],
                "classification_time_ms": round(elapsed_ms, 2),
                "timestamp": start_time.isoformat(),
                "method": "semantic_embedding"
            }
        
        except Exception as e:
            logger.error(f"Error in semantic classification: {e}")
            return self._fallback_classification(query)
    
    def _get_active_intents(self, scores: Dict[str, float]) -> List[str]:
        """
        Determine which intents are active based on thresholds.
        
        UNSAFE has lower threshold (more conservative).
        All other intents use standard threshold.
        
        Args:
            scores: Dictionary of intent -> similarity score
            
        Returns:
            List of active intents
        """
        active = []
        
        # UNSAFE always checked with lower threshold
        if scores.get("UNSAFE", 0) > self.unsafe_threshold:
            active.append("UNSAFE")
        
        # Other intents use standard threshold
        for intent in ["FACTUAL", "NUMERIC", "EXPLANATION"]:
            if scores.get(intent, 0) > self.intent_threshold:
                active.append(intent)
        
        # Always return at least one intent (primary)
        if not active:
            # If nothing exceeds threshold, use primary intent
            primary = max(scores, key=scores.get)
            active = [primary]
        
        return active
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """
        Fallback classification when embeddings unavailable.
        Uses keyword heuristics for basic classification.
        
        Args:
            query: Query to classify
            
        Returns:
            Classification result with fallback method
        """
        query_lower = query.lower()
        
        # Heuristic keyword detection
        scores = {
            "FACTUAL": 0.4,
            "NUMERIC": 0.4,
            "EXPLANATION": 0.4,
            "UNSAFE": 0.0
        }
        
        # Numeric keywords
        if any(word in query_lower for word in ["calculate", "how much", "percentage", "multiply", "divide", "sum", "total"]):
            scores["NUMERIC"] += 0.3
        
        # Explanation keywords
        if any(word in query_lower for word in ["explain", "why", "how does", "what is", "describe", "elaborate"]):
            scores["EXPLANATION"] += 0.3
        
        # Factual keywords
        if any(word in query_lower for word in ["what is", "definition", "fact", "history", "when", "where", "who"]):
            scores["FACTUAL"] += 0.3
        
        # Normalize to roughly 0-1 range
        total = sum(scores.values())
        if total > 0:
            scores = {intent: score / total for intent, score in scores.items()}
        
        # Determine active intents
        active = self._get_active_intents(scores)
        primary = max(scores, key=scores.get)
        
        return {
            "scores": {intent: round(score, 4) for intent, score in scores.items()},
            "active_intents": active,
            "primary_intent": primary,
            "primary_confidence": round(scores[primary], 4),
            "threshold": self.intent_threshold,
            "model": "fallback_heuristic",
            "classification_time_ms": 1.0,
            "timestamp": datetime.now().isoformat(),
            "method": "fallback_keyword_heuristic"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get classifier statistics and performance metrics.
        
        Returns:
            Dictionary with classifier stats
        """
        return {
            "model": self.model_name,
            "has_embeddings": self.has_embeddings,
            "total_classifications": self.total_classifications,
            "avg_classification_time_ms": round(self.avg_classification_time, 2),
            "intent_threshold": self.intent_threshold,
            "unsafe_threshold": self.unsafe_threshold,
            "num_intents": len(self.intents),
            "intents": self.intents,
            "prototypes_encoded": len(self.prototype_embeddings) > 0
        }
    
    def integrity_check(self) -> Dict[str, bool]:
        """
        Verify classifier is initialized and ready.
        
        Returns:
            Dictionary with integrity check results
        """
        return {
            "initialized": self.has_embeddings or True,  # Always true (has fallback)
            "embeddings_available": self.has_embeddings,
            "prototypes_loaded": len(self.prototype_embeddings) == len(self.INTENT_PROTOTYPES),
            "model_loaded": self.model is not None,
            "ready_for_inference": True  # Always ready with fallback
        }


class ExecutionPlanner:
    """
    Deterministic execution planner for multi-intent queries.
    
    Chains engines based on active intents.
    Ensures proper order of execution.
    Prevents unsafe queries from running.
    """
    
    # Engine execution chains for intent combinations
    # Keys are sorted tuples to ensure consistent lookup
    EXECUTION_CHAINS = {
        # Single intents
        ("EXPLANATION",): ["TRANSFORMER_ENGINE"],
        ("FACTUAL",): ["RETRIEVAL_ENGINE"],
        ("NUMERIC",): ["ML_ENGINE"],  # Calculator
        
        # Two intents (sorted)
        ("EXPLANATION", "FACTUAL"): ["RETRIEVAL_ENGINE", "TRANSFORMER_ENGINE"],
        ("EXPLANATION", "NUMERIC"): ["ML_ENGINE", "TRANSFORMER_ENGINE"],
        ("FACTUAL", "NUMERIC"): ["RETRIEVAL_ENGINE", "ML_ENGINE"],
        
        # Three intents (sorted)
        ("EXPLANATION", "FACTUAL", "NUMERIC"): [
            "RETRIEVAL_ENGINE",  # Get facts
            "ML_ENGINE",         # Compute
            "TRANSFORMER_ENGINE" # Explain
        ],
    }
    
    @staticmethod
    def plan_execution(active_intents: List[str]) -> Tuple[List[str], str]:
        """
        Plan execution engine chain for active intents.
        
        Args:
            active_intents: List of active intent labels
            
        Returns:
            Tuple of (engine_chain, reasoning)
        """
        # UNSAFE always overrides
        if "UNSAFE" in active_intents:
            return ["RULE_ENGINE"], "UNSAFE query detected - immediate block."
        
        # Sort for consistent chain lookup
        intent_tuple = tuple(sorted(active_intents))
        
        # Get execution chain from map
        engine_chain = ExecutionPlanner.EXECUTION_CHAINS.get(
            intent_tuple,
            ["ML_ENGINE"]  # Default fallback
        )
        
        reasoning = ExecutionPlanner._get_reasoning(active_intents, engine_chain)
        
        return engine_chain, reasoning
    
    @staticmethod
    def _get_reasoning(active_intents: List[str], engine_chain: List[str]) -> str:
        """
        Generate human-readable explanation for execution plan.
        
        Args:
            active_intents: Active intents
            engine_chain: Engine execution chain
            
        Returns:
            Explanation string
        """
        intent_str = " + ".join(active_intents)
        engine_str = " → ".join(engine_chain)
        
        return f"Query contains intents: {intent_str}. Execution plan: {engine_str}"
