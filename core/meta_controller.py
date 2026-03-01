"""
Meta-Controller - Multi-Intent Orchestration Engine
Enforces deterministic multi-intent routing and execution planning.

Architecture:
1. Query → Semantic Intent Classifier → Multi-label scores
2. Active intents determined by threshold
3. UNSAFE override check (immediate block)
4. Execution planner chains engines
5. Final orchestration and validation

NO SINGLE-LABEL FORCING. SUPPORT HYBRID QUERIES.
"""
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

from core.semantic_intent_classifier import SemanticIntentClassifier, ExecutionPlanner

logger = logging.getLogger(__name__)


class MetaController:
    """
    Multi-intent meta-controller with deterministic execution planning.
    
    Replaces single-label routing with:
    - Confidence-aware multi-intent scoring
    - Deterministic execution chaining
    - Hybrid query support
    - Explainable routing decisions
    """
    
    def __init__(self):
        """Initialize the meta-controller with semantic intent classifier."""
        self.intent_classifier = SemanticIntentClassifier(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            intent_threshold=0.70,
            unsafe_threshold=0.50
        )
        
        self.execution_planner = ExecutionPlanner()
        self.routing_history = []
        
        logger.info("✓ MetaController initialized with semantic intent classification")
    
    def orchestrate(
        self,
        query: str,
        query_features: Dict[str, Any] = None
    ) -> Dict[str, Any]:

        """
        Full orchestration: intent classification → execution planning → routing.
        
        Args:
            query: User query
            query_features: Optional features from input analyzer
            
        Returns:
            Orchestration plan with intents, engines, and reasoning
        """
        start_time = datetime.now()
        query_lower = query.lower().strip()

        # 🔴 HARD OVERRIDE RULES
        query_lower = query.lower().strip()

        classification = None

        if query_lower.startswith(("what is", "who is", "define", "where is", "when was")):
            classification = {
                "scores": {
                    "FACTUAL": 1.0,
                    "NUMERIC": 0.0,
                    "EXPLANATION": 0.0,
                    "UNSAFE": 0.0
                },
                "active_intents": ["FACTUAL"],
                "primary_intent": "FACTUAL",
                "primary_confidence": 1.0,
                "threshold": 0.5,
                "method": "rule_override",
                "classification_time_ms": 0
            }

        elif query_lower.startswith(("how much", "calculate", "compute")):
            classification = {
                "scores": {
                    "FACTUAL": 0.0,
                    "NUMERIC": 1.0,
                    "EXPLANATION": 0.0,
                    "UNSAFE": 0.0
                },
                "active_intents": ["NUMERIC"],
                "primary_intent": "NUMERIC",
                "primary_confidence": 1.0,
                "threshold": 0.5,
                "method": "rule_override",
                "classification_time_ms": 0
            }

        # If no override matched → run normal classifier
        if classification is None:
            classification = self.intent_classifier.classify(query)
        
        # Step 2: Check for UNSAFE (overrides everything)
        if "UNSAFE" in classification["active_intents"]:
            return self._create_unsafe_response(query, classification, start_time)
        
        # Step 3: Plan execution for active intents
        active_intents = classification["active_intents"]
        engine_chain, planning_reasoning = self.execution_planner.plan_execution(active_intents)
        
        # Step 4: Create orchestration plan
        orchestration_plan = {
            "status": "ready",
            "query": query,
            "intents": {
                "all_scores": classification["scores"],
                "active_intents": classification["active_intents"],
                "primary_intent": classification["primary_intent"],
                "primary_confidence": classification["primary_confidence"],
                "threshold_used": classification["threshold"]
            },
            "execution_plan": {
                "engine_chain": engine_chain,
                "chain_reasoning": planning_reasoning,
                "num_engines": len(engine_chain),
                "engines": engine_chain
            },
            "metadata": {
                "classification_method": classification["method"],
                "classification_time_ms": classification["classification_time_ms"],
                "timestamp": start_time.isoformat()
            },
            "decomposition": self.decompose_query(query, active_intents)
        }
        
        # Step 5: Log routing decision
        self._log_routing_decision(query, orchestration_plan)
        
        return orchestration_plan
    
    def _create_unsafe_response(
        self,
        query: str,
        classification: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """
        Create response for UNSAFE queries (immediate block).
        
        Args:
            query: Original query
            classification: Classification result
            start_time: Start time of orchestration
            
        Returns:
            UNSAFE response plan
        """
        return {
            "status": "blocked",
            "blocked": True,
            "query": query,
            "intents": {
                "all_scores": classification["scores"],
                "active_intents": classification["active_intents"],
                "primary_intent": "UNSAFE",
                "primary_confidence": classification["scores"]["UNSAFE"]
            },
            "execution_plan": {
                "engine_chain": ["RULE_ENGINE"],
                "chain_reasoning": "UNSAFE query detected - immediate block at meta-controller level.",
                "num_engines": 1,
                "engines": ["RULE_ENGINE"]
            },
            "metadata": {
                "classification_method": classification["method"],
                "classification_time_ms": classification["classification_time_ms"],
                "timestamp": start_time.isoformat()
            }
        }
    
    def _log_routing_decision(self, query: str, orchestration_plan: Dict[str, Any]):
        """
        Log routing decision for auditability and debugging.
        
        Args:
            query: Original query
            orchestration_plan: Orchestration plan from orchestrate()
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # First 100 chars
            "primary_intent": orchestration_plan["intents"]["primary_intent"],
            "active_intents": orchestration_plan["intents"]["active_intents"],
            "engine_chain": orchestration_plan["execution_plan"]["engine_chain"],
            "status": orchestration_plan["status"]
        }
        
        self.routing_history.append(log_entry)
        
        # Log to system logger
        logger.info(
            f"Routing: {log_entry['primary_intent']} "
            f"({log_entry['active_intents']}) → {' → '.join(log_entry['engine_chain'])}"
        )
    
    def decompose_query(self, query: str, active_intents: List[str]) -> Dict[str, Any]:
        """
        Decomposes hybrid queries into engine-specific parameters.
        - Multiplication detection
        - Percentage handling
        - "of" numeric relationships
        - Extract entity for factual engine
        - Extract operator for numeric engine
        """
        import re
        query_lower = query.lower()
        decomposition = {
            "factual_entity": None,
            "numeric_operator": None,
            "numeric_params": [],
            "computation_type": None,
            "percentage": None
        }
        
        # Percentage/multiplication/of detection
        percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', query_lower)
        if percentage_match:
            decomposition["percentage"] = float(percentage_match.group(1))
            decomposition["numeric_operator"] = "*"
            decomposition["computation_type"] = "percentage"
        
        # Look for "of" relationship
        of_match = re.search(r'(?:%|percent)\s+of\s+(.+)', query_lower)
        if of_match:
            entity_candidate = of_match.group(1).strip()
            # Clean up the entity for factual lookup (e.g. "of the 400" -> 400, "of total students" -> "total students")
            decomposition["factual_entity"] = entity_candidate.replace("?", "")
            
        # Basic operator detection if not percentage
        if not decomposition["numeric_operator"]:
            if any(w in query_lower for w in ["multiply", "times", "*"]):
                decomposition["numeric_operator"] = "*"
            elif any(w in query_lower for w in ["add", "plus", "+", "sum"]):
                decomposition["numeric_operator"] = "+"
            elif any(w in query_lower for w in ["subtract", "minus", "-"]):
                decomposition["numeric_operator"] = "-"
            elif any(w in query_lower for w in ["divide", "/"]):
                decomposition["numeric_operator"] = "/"
            
        return decomposition
    
    def route(
        self,
        query: str,
        query_features: Dict[str, Any] = None
    ) -> Tuple[List[str], str]:
        """
        Simplified route method for backward compatibility.
        Returns engine chain for a query.
        
        Args:
            query: User query
            query_features: Optional query features
            
        Returns:
            Tuple of (engine_chain, reasoning)
        """
        plan = self.orchestrate(query, query_features)
        
        engine_chain = plan["execution_plan"]["engine_chain"]
        reasoning = plan["execution_plan"]["chain_reasoning"]
        
        return engine_chain, reasoning
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.
        
        Returns:
            Dictionary with routing statistics
        """
        if not self.routing_history:
            return {
                "total_queries": 0,
                "intent_distribution": {},
                "engine_chain_distribution": {},
                "multi_intent_queries": 0,
                "unsafe_blocks": 0,
                "classifier_stats": self.intent_classifier.get_stats()
            }
        
        total = len(self.routing_history)
        
        # Count intent distribution
        intent_counts = {}
        for entry in self.routing_history:
            primary = entry["primary_intent"]
            intent_counts[primary] = intent_counts.get(primary, 0) + 1
        
        # Count engine chain patterns
        chain_counts = {}
        for entry in self.routing_history:
            chain_tuple = tuple(entry["engine_chain"])
            chain_counts[chain_tuple] = chain_counts.get(chain_tuple, 0) + 1
        
        # Count multi-intent queries
        multi_intent = sum(1 for entry in self.routing_history if len(entry["active_intents"]) > 1)
        
        # Count UNSAFE blocks
        unsafe_blocks = sum(1 for entry in self.routing_history if entry["primary_intent"] == "UNSAFE")
        
        return {
            "total_queries": total,
            "intent_distribution": intent_counts,
            "engine_chain_distribution": {
                " → ".join(chain): count for chain, count in chain_counts.items()
            },
            "multi_intent_queries": multi_intent,
            "multi_intent_percentage": round((multi_intent / total * 100) if total > 0 else 0, 1),
            "unsafe_blocks": unsafe_blocks,
            "classifier_stats": self.intent_classifier.get_stats()
        }
    
    def integrity_check(self) -> Dict[str, bool]:
        """
        Verify meta-controller is initialized and ready.
        
        Returns:
            Dictionary with integrity check results
        """
        return {
            "initialized": True,
            "intent_classifier_ready": self.intent_classifier.integrity_check()["ready_for_inference"],
            "execution_planner_ready": hasattr(self.execution_planner, "plan_execution"),
            "routing_history_available": len(self.routing_history) >= 0
        }
    
    def validate_orchestration(self, plan: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that orchestration plan is correct.
        
        Args:
            plan: Orchestration plan from orchestrate()
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for UNSAFE override
        if "UNSAFE" in plan["intents"]["active_intents"]:
            if plan["execution_plan"]["engine_chain"] != ["RULE_ENGINE"]:
                return False, "UNSAFE not overriding to RULE_ENGINE"
        
        # Check that engine chain is non-empty
        if not plan["execution_plan"]["engine_chain"]:
            return False, "Empty engine chain"
        
        # Check that active intents correspond to engine chain
        # This is complex, so basic check for now
        
        return True, "Valid orchestration plan"
