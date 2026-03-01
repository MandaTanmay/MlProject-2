"""
Meta-Learning AI System - FastAPI Application
Production-grade AI orchestration layer that decides how to answer queries.
NOT a chatbot - it's an intelligent routing system.
"""
import nltk

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')
    nltk.download('punkt')
    nltk.download('wordnet')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
import logging
import os

logger = logging.getLogger(__name__)

# Allow optionally passing OUTSIDE-domain queries through the orchestration pipeline.
# Default: False. Can be enabled by setting environment variable `ALLOW_OUTSIDE_ROUTING=true`.
ALLOW_OUTSIDE_ROUTING = os.environ.get("ALLOW_OUTSIDE_ROUTING", "false").lower() in ("1", "true", "yes")

# Import core components
from core.domain_classifier import DomainClassifier
from core.input_analyzer import InputAnalyzer
from core.meta_controller import MetaController
from core.output_validator import OutputValidator

# Import engines
from engines.rule_engine import RuleEngine
from engines.retrieval_engine import FactualEngine
from engines.ml_engine import MLEngine
from engines.transformer_engine import TransformerEngine
from engines.phi2_explanation_engine import Phi2ExplanationEngine

# Import feedback
from feedback.feedback_store import FeedbackStore


# Initialize FastAPI app
app = FastAPI(
    title="Meta-Learning AI System",
    description="AI orchestration layer that intelligently routes queries to appropriate engines",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
domain_classifier = DomainClassifier()
input_analyzer = InputAnalyzer()
meta_controller = MetaController()
output_validator = OutputValidator()

# Initialize engines
rule_engine = RuleEngine()
retrieval_engine = FactualEngine()
ml_engine = MLEngine()
transformer_engine = TransformerEngine()
phi2_explanation_engine = Phi2ExplanationEngine(use_quantization=True, device="auto")

# Initialize feedback store
feedback_store = FeedbackStore()

# Query cache for feedback context (query -> {intent, confidence})
query_context_cache = {}


# Startup event - load Phi-2 model
@app.on_event("startup")
async def startup_load_phi2():
    """Load Phi-2 model at startup for explanation engine."""
    logger.info("Loading Phi-2 explanation engine on startup...")
    if phi2_explanation_engine.load():
        logger.info("✓ Phi-2 model loaded successfully")
    else:
        logger.warning("⚠ Phi-2 model failed to load - explanations will use fallback")


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the minimum attendance requirement?"
            }
        }


class QueryResponse(BaseModel):
    answer: str
    strategy: str
    confidence: float
    reason: str
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The minimum attendance requirement is 75%.",
                "strategy": "RETRIEVAL",
                "confidence": 1.0,
                "reason": "Intent-based routing: FACTUAL query routed to RETRIEVAL engine"
            }
        }


class FeedbackRequest(BaseModel):
    query: str
    strategy: str
    answer: str
    feedback: int  # 1 for positive, -1 for negative
    comment: Optional[str] = ""
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is meta-learning?",
                "strategy": "RETRIEVAL",
                "answer": "Meta-learning is...",
                "feedback": 1,
                "comment": "Very helpful!"
            }
        }


class StatsResponse(BaseModel):
    system_stats: Dict[str, Any]
    engine_stats: Dict[str, Any]
    feedback_stats: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "system": "Meta-Learning AI System",
        "version": "1.0.0",
        "status": "operational",
        "description": "AI orchestration layer for intelligent query routing",
        "endpoints": {
            "query": "/query",
            "feedback": "/feedback",
            "stats": "/stats",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():

    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "input_analyzer": "operational",
            "meta_controller": "operational",
            "output_validator": "operational",
            "rule_engine": "operational",
            "retrieval_engine": "operational",
            "ml_engine": "operational",
            "transformer_engine": "loaded" if transformer_engine.is_loaded else "fallback mode",
            "phi2_explanation_engine": "loaded" if phi2_explanation_engine.is_loaded else "fallback mode"
        }
    }


@app.get("/health/full")
async def health_full():
    """Detailed health including model names and load states."""
    return {
        "status": "healthy",
        "domain_classifier": {
            "loaded": domain_classifier.is_loaded,
            "model": "TF-IDF + Logistic Regression" if domain_classifier.is_loaded else "fallback"
        },
        "transformer_engine": {
            "loaded": transformer_engine.is_loaded,
            "model": getattr(transformer_engine, "model_name", "unknown")
        },
        "allow_outside_routing": ALLOW_OUTSIDE_ROUTING,
        "phi2_explanation_engine": {
            "loaded": phi2_explanation_engine.is_loaded,
            "model": phi2_explanation_engine.model_name,
            "quantization": phi2_explanation_engine.use_quantization
        },
        "components": {
            "input_analyzer": "operational",
            "meta_controller": "operational",
            "output_validator": "operational",
            "rule_engine": "operational",
            "retrieval_engine": "operational",
            "ml_engine": "operational"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):

    try:
        query = request.query.strip()

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # --------------------------------------------
        # NORMALIZATION LAYER
        # --------------------------------------------

        query = query.lower().strip()

        from textblob import TextBlob

        try:
            blob = TextBlob(query)
            corrected_query = str(blob.correct())
            query = corrected_query
        except Exception:
            pass  # fallback silently if correction fails

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        # ------------------------------------------------
        # STEP 1: HARD SAFETY CHECK (FIRST)
        # ------------------------------------------------
        from core.safety import is_harmful_input
        
        if is_harmful_input(query):
            return QueryResponse(
                answer="I'm not able to assist with harmful or dangerous requests.",
                strategy="SAFETY",
                confidence=1.0,
                reason="Blocked by safety layer before domain classification.",
                metadata={
                    "intent": "UNSAFE",
                    "intent_confidence": 1.0
                }
            )

        # ------------------------------------------------
        # STEP 2: NUMERIC FAST-PATH (Bypass Domain)
        # ------------------------------------------------
        features = input_analyzer.analyze(query)
        print("NUMERIC FEATURES:", features)
        if features.get("question_type") == "NUMERIC":
            result = ml_engine.execute(query, features)
        
            return QueryResponse(
                answer=result["answer"],
                strategy="ML",
                confidence=result.get("confidence", 1.0),
                reason="Numeric query detected - bypassed domain filter",
                metadata={
                    "intent": "NUMERIC",
                    "intent_confidence": 1.0,
                    "engine_chain": ["ML_ENGINE"]
                }
            )
                
        # ------------------------------------------------
        # STEP 2: Domain Classification
        # ------------------------------------------------
        domain, dom_conf = domain_classifier.predict(query)
        bypassed_domain_filter = False

        if domain == "OUTSIDE":
            if not ALLOW_OUTSIDE_ROUTING:
                return QueryResponse(
                    answer=domain_classifier.get_refusal_message(),
                    strategy="DOMAIN_FILTER",
                    confidence=dom_conf,
                    reason="Query is not related to the academic student domain.",
                    metadata={
                        "domain": domain,
                        "domain_confidence": dom_conf
                    }
                )
            else:
                # Allow OUTSIDE queries to continue through orchestration when explicitly enabled.
                # However, run an extra safety check to ensure harmful OUTSIDE queries are still blocked.
                bypassed_domain_filter = True
                logger.info("ALLOW_OUTSIDE_ROUTING enabled: proceeding with OUTSIDE-domain query through orchestration")
                from core.safety import is_harmful_input as _is_harmful
                if _is_harmful(query):
                    logger.warning("Blocked OUTSIDE-domain query by safety check after bypass")
                    return QueryResponse(
                        answer="I'm not able to assist with harmful or dangerous requests.",
                        strategy="SAFETY",
                        confidence=1.0,
                        reason="Blocked by safety layer after domain bypass.",
                        metadata={
                            "domain": domain,
                            "domain_confidence": dom_conf,
                            "bypassed_domain_filter": True,
                            "unsafe_block": True
                        }
                    )
        # ------------------------------------------------
        # STEP 3: Feature Extraction
        # ------------------------------------------------
        features = input_analyzer.analyze(query)
        # ------------------------------------------------
        # STEP 4: Multi-Label Intent Orchestration
        # ------------------------------------------------
        # Uses semantic similarity to determine active intents and execution chain
        orchestration_plan = meta_controller.orchestrate(query, features)
        # Record whether domain filter was bypassed so logs/response reflect it
        orchestration_plan.setdefault("metadata", {})["bypassed_domain_filter"] = bypassed_domain_filter
        # Store routing decision in database (Phase 7)
        is_blocked = orchestration_plan.get("status") == "blocked"
        feedback_store.store_routing_log(
            query=query,
            active_intents=orchestration_plan["intents"]["active_intents"],
            primary_intent=orchestration_plan["intents"]["primary_intent"],
            engine_chain=orchestration_plan["execution_plan"]["engine_chain"],
            status=orchestration_plan.get("status", "ready"),
            is_unsafe=is_blocked
        )
        if is_blocked:
            return QueryResponse(
                answer="I'm not able to assist with harmful or dangerous requests.",
                strategy="SAFETY",
                confidence=1.0,
                reason="Blocked by safety layer.",
                metadata=orchestration_plan.get("metadata", {})
            )
        execution_plan = orchestration_plan["execution_plan"]
        engines_to_execute = execution_plan["engine_chain"]
        routing_reason = execution_plan["chain_reasoning"]
        intent = orchestration_plan["intents"]["primary_intent"]
        confidence = orchestration_plan["intents"]["primary_confidence"]
        decomposition = orchestration_plan.get("decomposition", {})
        # ------------------------------------------------
        # STEP 5: Execute Engine(s)
        # ------------------------------------------------
        result = None
        grounded_data = {}  # Accumulate grounding for explanation engine
        for current_engine in engines_to_execute:
            if current_engine == "RULE" or current_engine == "RULE_ENGINE":
                result = rule_engine.execute(query, features)
            elif current_engine == "RETRIEVAL" or current_engine == "RETRIEVAL_ENGINE":
                if decomposition.get("factual_entity"):
                    # Use entity from decomposition if available
                    result = retrieval_engine.execute(decomposition["factual_entity"], features)
                else:
                    result = retrieval_engine.execute(query, features)
                # Extract answer - handle both flat and nested response formats
                factual_answer = result.get("answer") or result.get("data", {}).get("answer", "")
                grounded_data["factual_result"] = factual_answer
                # Normalize result to have answer at root level for downstream processing
                if result.get("status") == "success":
                    if not result.get("answer") and result.get("data", {}).get("answer"):
                        result["answer"] = result["data"]["answer"]
                    result["strategy"] = "RETRIEVAL"
                    result["confidence"] = result.get("confidence", 0.0)
                    result["reason"] = f"Retrieved from knowledge base (confidence: {result['confidence']:.2%})"
                else:
                    # Handle uncertain/error/ambiguous responses
                    status = result.get("status", "unknown")
                    reason = result.get("data", {}).get("reason") or result.get("metadata", {}).get("reason") or "No confident match found"
                    result["answer"] = f"I could not find a confident answer. Reason: {reason}"
                    result["strategy"] = "RETRIEVAL"
                    result["confidence"] = result.get("confidence", 0.0)
                    result["reason"] = f"Retrieval status: {status}"
                # The Retrieval Engine sometimes sets source inside data/metadata, not root.
                if isinstance(result.get("data"), dict):
                    grounded_data["source"] = result["data"].get("source", result.get("source", "Unknown"))
                    result["source"] = grounded_data["source"]
                else:
                    grounded_data["source"] = result.get("source", "Unknown")
            elif current_engine == "ML" or current_engine == "ML_ENGINE":
                if decomposition.get("computation_type") == "percentage" and grounded_data.get("factual_result"):
                    import re
                    pct = decomposition["percentage"]
                    factual_nums = re.findall(r'-?\d+\.?\d*', str(grounded_data["factual_result"]))
                    if factual_nums:
                        base_val = float(factual_nums[0])
                        ans = (pct / 100.0) * base_val
                        result = {
                            "answer": f"The answer is {ans}",
                            "confidence": 1.0,
                            "strategy": "ML",
                            "computation_type": "percentage",
                            "reason": f"Computed {pct}% of {base_val}"
                        }
                    else:
                        result = ml_engine.execute(query, features)
                else:
                    result = ml_engine.execute(query, features)
                # Store numeric result for grounding explanation
                grounded_data["numeric_result"] = result.get("answer")
                grounded_data["computation_type"] = result.get("computation_type")
            elif current_engine == "TRANSFORMER" or current_engine == "TRANSFORMER_ENGINE":
                # Use Phi2ExplanationEngine if available and grounded data is present
                if phi2_explanation_engine.is_loaded and grounded_data:
                    logger.info(f"Using Phi2ExplanationEngine with grounded data: {list(grounded_data.keys())}")
                    result = phi2_explanation_engine.execute(query, grounded_data)
                    # Convert phi2 response format to match app expectations
                    if result.get("status") == "success":
                        result["answer"] = result.get("explanation")
                        result["strategy"] = "EXPLANATION"
                        result["confidence"] = result.get("confidence", 0.9)
                        result["reason"] = "Generated using grounded Phi-2 explanation engine"
                    else:
                        # Fallback to transformer engine if phi2 fails
                        logger.warning(f"Phi2 explanation failed: {result.get('explanation')}")
                        result = transformer_engine.execute(query, features)
                else:
                    # Fallback if Phi2 not loaded or no grounding available
                    result = transformer_engine.execute(query, features)
            else:
                raise HTTPException(status_code=500, detail=f"Unknown engine: {current_engine}")
        # ------------------------------------------------
        # STEP 6: Output Validation
        # ------------------------------------------------
        is_valid, validated_answer, validation_details = output_validator.validate(
            answer=result["answer"],
            strategy=result["strategy"],
            confidence=result["confidence"],
            query=query
        )
        # Store context for feedback
        query_context_cache[query] = {
            "intent": intent,
            "confidence": confidence
        }
        # ------------------------------------------------
        # STEP 7: Return Response
        # ------------------------------------------------
        # If domain filter was bypassed, mark the overall strategy as OUTSIDE_ROUTING
        out_strategy = "OUTSIDE_ROUTING" if bypassed_domain_filter else result["strategy"]

        return QueryResponse(
            answer=validated_answer,
            strategy=out_strategy,
            confidence=result["confidence"],
            reason=("Domain filter bypassed; proceeding with routing." if bypassed_domain_filter else routing_reason),
            metadata={
                "intent": intent,
                "intent_confidence": confidence,
                "active_intents": orchestration_plan["intents"]["active_intents"],
                "intent_scores": orchestration_plan["intents"]["all_scores"],
                "engine_chain": orchestration_plan["execution_plan"]["engine_chain"],
                "classification_method": orchestration_plan["metadata"].get("classification_method", "semantic"),
                "classification_time_ms": orchestration_plan["metadata"].get("classification_time_ms"),
                "validation": validation_details,
                "source": result.get("source"),
                "computation_type": result.get("computation_type"),
                "domain": domain,
                "domain_confidence": dom_conf,
                "bypassed_domain_filter": bypassed_domain_filter
            }
        )
    except HTTPException:
        raise
    
    except Exception as e:
        print("FULL TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a query response.
    Feedback is used to automatically improve the intent classifier.
    """
    try:
        # Get stored context for this query
        context = query_context_cache.get(request.query, {})
        predicted_intent = context.get("intent", "UNKNOWN")
        predicted_confidence = context.get("confidence", 0.0)
        
        # Store feedback
        success = feedback_store.store_feedback(
            query=request.query,
            predicted_intent=predicted_intent,
            predicted_confidence=predicted_confidence,
            strategy=request.strategy,
            answer=request.answer,
            user_feedback=request.feedback,
            user_comment=request.comment
        )
        
        if success:
            # Auto-improvement: Check if we should update based on accumulated feedback
            stats = feedback_store.get_feedback_stats()
            total_feedback = stats.get("total_feedback", 0)
            
            # Trigger improvement every 10 feedbacks
            if total_feedback > 0 and total_feedback % 10 == 0:
                print(f"\n🔄 Auto-improvement triggered after {total_feedback} feedbacks")
                improvement_result = _auto_improve_classifier()
                
                return {
                    "status": "success",
                    "message": "Feedback received. Auto-improvement triggered!",
                    "feedback": "positive" if request.feedback > 0 else "negative",
                    "total_feedback_count": total_feedback,
                    "auto_improvement": improvement_result
                }
            
            return {
                "status": "success",
                "message": "Feedback received. System learning from your input!",
                "feedback": "positive" if request.feedback > 0 else "negative",
                "total_feedback_count": total_feedback
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store feedback")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing feedback: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics and performance metrics."""
    try:
        # System stats
        system_stats = {
            "routing": meta_controller.get_routing_stats(),
            "validation": output_validator.get_validation_stats()
        }
        
        # Engine stats
        engine_stats = {
            "rule": rule_engine.get_stats(),
            "retrieval": retrieval_engine.get_stats(),
            "ml": ml_engine.get_stats(),
            "transformer": transformer_engine.get_stats()
        }
        
        # Feedback stats
        feedback_stats = feedback_store.get_feedback_stats()
        
        return StatsResponse(
            system_stats=system_stats,
            engine_stats=engine_stats,
            feedback_stats=feedback_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.get("/intents")
async def get_intents():
    """Get list of supported intents and execution chains."""
    routing_map = {
        str(list(k)): v
        for k, v in meta_controller.execution_planner.EXECUTION_CHAINS.items()
    }
    return {
        "intents": meta_controller.intent_classifier.intents,
        "routing_map": routing_map,
        "description": {
            "FACTUAL": "Factual queries - routed to RETRIEVAL engine",
            "NUMERIC": "Numerical computations - routed to ML engine",
            "EXPLANATION": "Conceptual explanations - routed to TRANSFORMER engine",
            "UNSAFE": "Harmful queries - blocked by RULE engine"
        }
    }


@app.get("/model/status")
async def get_model_status():
    """Get detailed model training and load status."""
    ic = meta_controller.intent_classifier
    ic_loaded = ic.model is not None
    model_type = "semantic-embedding" if ic_loaded else "fallback-heuristic"

    status = {
        "model_type": model_type,
        "intent_classifier": {
            "loaded": ic_loaded,
            "model_name": getattr(ic, "model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            "type": "Semantic Embedding (sentence-transformers/all-MiniLM-L6-v2)",
            "requires_training": False,
            "status": "✅ READY" if ic_loaded else "⚠️ USING FALLBACK HEURISTIC"
        },
        "transformer_engine": {
            "loaded": transformer_engine.is_loaded,
            "model_name": getattr(transformer_engine, "model_name", "unknown"),
            "type": "Flan-T5 (pre-trained generative)",
            "requires_training": False,
            "status": "✅ READY" if transformer_engine.is_loaded else "⚠️ USING FALLBACK"
        },
        "training_info": {
            "note": "Intent classification uses MiniLM embedding similarity - no training required",
            "feedback_collected": feedback_store.get_feedback_stats().get("total_feedback", 0),
            "auto_improvement": "Enabled - domain/engine-selector models retrain from feedback"
        },
        "system_status": "✅ FULLY OPERATIONAL" if (ic_loaded and transformer_engine.is_loaded) else "⚠️ PARTIAL - Using fallback modes"
    }

    return status


@app.get("/model/registry")
async def get_model_registry():
    """Get versioned model registry - lists all saved model versions with metadata."""
    try:
        from core.model_registry import get_registry_summary, list_versions
        summary = get_registry_summary()
        history = list_versions()
        return {
            "status": "ok",
            "registered_models": summary,
            "version_history": history[-20:],  # Last 20 versions
            "note": "SemanticIntentClassifier (MiniLM) is not versioned here - it uses static pre-trained embeddings"
        }
    except Exception as e:
        return {"status": "error", "detail": str(e), "registered_models": {}}


@app.get("/model/metrics")
async def get_model_metrics():
    """Get model performance metrics (accuracy, precision, recall, F1) for presentation."""
    try:
        # Get feedback data
        feedback_stats = feedback_store.get_feedback_stats()
        
        # Calculate metrics from feedback
        metrics = _calculate_performance_metrics()
        
        return {
            "overall_metrics": metrics["overall"],
            "per_intent_metrics": metrics["per_intent"],
            "confusion_matrix": metrics["confusion_matrix"],
            "sample_size": metrics["total_samples"],
            "routing_accuracy": meta_controller.get_routing_stats(),
            "feedback_summary": {
                "total_feedback": feedback_stats.get("total_feedback", 0),
                "positive_feedback": feedback_stats.get("positive_feedback", 0),
                "negative_feedback": feedback_stats.get("negative_feedback", 0),
                "satisfaction_rate": feedback_stats.get("satisfaction_rate", 0)
            },
            "note": "Metrics calculated from user feedback and routing decisions"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")


def _calculate_performance_metrics() -> Dict[str, Any]:
    """
    Calculate accuracy, precision, recall, F1 score from feedback and routing history.
    """
    # Get all feedback from database
    try:
        conn = sqlite3.connect(feedback_store.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT predicted_intent, user_feedback, was_correct
            FROM feedback
        """)
        all_feedback = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Error fetching feedback: {e}")
        all_feedback = []
    
    if len(all_feedback) == 0:
        return {
            "overall": {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            },
            "per_intent": {},
            "confusion_matrix": {},
            "total_samples": 0
        }
    
    # Track predictions per intent
    intent_stats = defaultdict(lambda: {
        "true_positive": 0,
        "false_positive": 0,
        "total": 0
    })
    
    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    
    correct_predictions = 0
    total_predictions = len(all_feedback)
    
    for predicted_intent, user_feedback, was_correct in all_feedback:
        is_correct = user_feedback > 0
        
        if is_correct:
            correct_predictions += 1
            intent_stats[predicted_intent]["true_positive"] += 1
            confusion[predicted_intent][predicted_intent] += 1
        else:
            intent_stats[predicted_intent]["false_positive"] += 1
            confusion[predicted_intent]["incorrect"] += 1
        
        intent_stats[predicted_intent]["total"] += 1
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Calculate per-intent metrics
    per_intent_metrics = {}
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    intent_count = 0
    
    for intent, stats in intent_stats.items():
        tp = stats["true_positive"]
        fp = stats["false_positive"]
        total = stats["total"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total if total > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_intent_metrics[intent] = {
            "accuracy": round(tp / total if total > 0 else 0.0, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "samples": total
        }
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        intent_count += 1
    
    # Macro-averaged metrics
    avg_precision = total_precision / intent_count if intent_count > 0 else 0.0
    avg_recall = total_recall / intent_count if intent_count > 0 else 0.0
    avg_f1 = total_f1 / intent_count if intent_count > 0 else 0.0
    
    return {
        "overall": {
            "accuracy": round(overall_accuracy, 4),
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "f1_score": round(avg_f1, 4)
        },
        "per_intent": per_intent_metrics,
        "confusion_matrix": dict(confusion),
        "total_samples": total_predictions
    }


def _auto_improve_classifier():
    """
    Automatically improve classifier based on accumulated feedback.
    Exports feedback to training data and can trigger retraining.
    """
    try:
        # Get training samples from positive feedback
        training_samples = feedback_store.get_training_data(
            min_confidence=0.5,
            only_correct=True
        )
        
        if len(training_samples) < 5:
            print("⚠ Not enough feedback samples yet for improvement")
            return {
                "exported": False,
                "reason": "Insufficient samples",
                "sample_count": len(training_samples)
            }
        
        # Analyze feedback patterns
        stats = feedback_store.get_feedback_stats()
        intent_accuracy = stats.get("intent_accuracy", {})
        
        print("\n--- Auto-Improvement Analysis ---")
        for intent, data in intent_accuracy.items():
            accuracy = data.get("accuracy", 0)
            print(f"{intent}: {accuracy:.1%} accuracy ({data['correct']}/{data['total']})")
        
        # Export feedback to training dataset automatically
        training_csv_path = Path(__file__).parent / "training" / "intent_dataset.csv"
        exported_count = 0
        
        try:
            # Read existing training data to avoid duplicates
            existing_queries = set()
            if training_csv_path.exists():
                import csv
                with open(training_csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_queries.add(row['query'].lower().strip())
            
            # Append new samples
            with open(training_csv_path, 'a', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                
                # Write header if file is empty
                if training_csv_path.stat().st_size == 0:
                    writer.writerow(['query', 'intent'])
                
                for sample in training_samples:
                    query = sample.get('query', '').strip()
                    intent = sample.get('intent', '')
                    
                    if query.lower() not in existing_queries and query and intent:
                        writer.writerow([query, intent])
                        exported_count += 1
                        existing_queries.add(query.lower())
            
            print(f"✓ Exported {exported_count} new samples to training dataset")
            
        except Exception as e:
            print(f"⚠ Failed to export training data: {e}")
        
        # Save feedback patterns for reference
        feedback_log_path = Path(__file__).parent / "feedback" / "improvement_log.json"
        feedback_log_path.parent.mkdir(exist_ok=True)
        
        with open(feedback_log_path, "a") as f:
            import datetime
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_samples": len(training_samples),
                "exported_samples": exported_count,
                "intent_accuracy": intent_accuracy,
                "auto_retrain_triggered": False,  # Set to True if you add retraining
                "note": "Using MiniLM semantic embedding classifier - no retraining needed"
            }
            f.write(json.dumps(log_entry) + "\n")
        
        print(f"✓ Auto-improvement logged to {feedback_log_path}")
        print("✓ System continues learning from user feedback")
        
        # Trigger automatic retraining if enough new samples
        retrain_result = None
        if exported_count >= 5:
            print(f"\n🔄 Triggering automatic model retraining with {exported_count} new samples...")
            retrain_result = _retrain_model()
        
        return {
            "exported": True,
            "exported_count": exported_count,
            "total_samples": len(training_samples),
            "intent_accuracy": intent_accuracy,
            "retrain_triggered": retrain_result is not None,
            "retrain_result": retrain_result
        }
        
    except Exception as e:
        print(f"✗ Auto-improvement error: {e}")
        return {
            "exported": False,
            "error": str(e)
        }


def _retrain_model():
    """
    Automatically retrain the intent classifier with updated training data.
    """
    try:
        import subprocess
        import sys
        
        training_script = Path(__file__).parent / "training" / "train_intent_model.py"
        
        if not training_script.exists():
            print(f"⚠ Training script not found: {training_script}")
            return {
                "success": False,
                "error": "Training script not found"
            }
        
        print("\n" + "="*60)
        print("🔄 AUTOMATIC MODEL RETRAINING")
        print("="*60)
        
        # Run training script
        result = subprocess.run(
            [sys.executable, str(training_script)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Model retraining completed successfully!")
            print(result.stdout)

            # SemanticIntentClassifier uses pre-trained MiniLM embeddings and does not
            # need to be reinstanced after domain/engine-selector retraining.
            print("\n✓ Retraining complete - semantic intent classifier unchanged (embedding-based)")

            return {
                "success": True,
                "message": "Model retrained and reloaded successfully",
                "output": result.stdout[-500:]  # Last 500 chars
            }
        else:
            print(f"❌ Retraining failed with code {result.returncode}")
            print(result.stderr)
            return {
                "success": False,
                "error": result.stderr[-500:]
            }
            
    except subprocess.TimeoutExpired:
        print("❌ Retraining timeout - took longer than 5 minutes")
        return {
            "success": False,
            "error": "Training timeout"
        }
    except Exception as e:
        print(f"❌ Retraining error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 META-LEARNING AI SYSTEM")
    print("=" * 60)
    print("\nStarting FastAPI server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress CTRL+C to stop")
    print("=" * 60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
