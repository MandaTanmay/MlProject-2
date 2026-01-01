"""
Meta-Learning AI System - FastAPI Application
Production-grade AI orchestration layer that decides how to answer queries.
NOT a chatbot - it's an intelligent routing system.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

# Import core components
from core.input_analyzer import InputAnalyzer
from core.intent_classifier import IntentClassifier
from core.meta_controller import MetaController
from core.output_validator import OutputValidator

# Import engines
from engines.rule_engine import RuleEngine
from engines.retrieval_engine import RetrievalEngine
from engines.ml_engine import MLEngine
from engines.transformer_engine import TransformerEngine

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
input_analyzer = InputAnalyzer()
intent_classifier = IntentClassifier()
meta_controller = MetaController()
output_validator = OutputValidator()

# Initialize engines
rule_engine = RuleEngine()
retrieval_engine = RetrievalEngine()
ml_engine = MLEngine()
transformer_engine = TransformerEngine()

# Initialize feedback store
feedback_store = FeedbackStore()

# Query cache for feedback context (query -> {intent, confidence})
query_context_cache = {}


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
            "intent_classifier": "loaded" if intent_classifier.is_loaded else "fallback mode",
            "meta_controller": "operational",
            "output_validator": "operational",
            "rule_engine": "operational",
            "retrieval_engine": "operational",
            "ml_engine": "operational",
            "transformer_engine": "loaded" if transformer_engine.is_loaded else "fallback mode"
        }
    }


@app.get("/health/full")
async def health_full():
    """Detailed health including model names and load states."""
    return {
        "status": "healthy",
        "intent_classifier": {
            "loaded": intent_classifier.is_loaded,
            "model": getattr(intent_classifier, "model_name", "unknown")
        },
        "transformer_engine": {
            "loaded": transformer_engine.is_loaded,
            "model": getattr(transformer_engine, "model_name", "unknown")
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
    """
    Process a user query through the meta-learning pipeline.
    
    Pipeline:
    1. Input Analyzer - Extract features
    2. Intent Classifier - Classify intent
    3. Meta-Controller - Route to engine
    4. Engine Execution - Get answer
    5. Output Validator - Validate answer
    6. Return response
    """
    try:
        query = request.query.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Step 1: Analyze input
        features = input_analyzer.analyze(query)
        
        # Step 2: Classify intent (ML) then apply deterministic safety/math overrides
        intent, confidence = intent_classifier.predict(query)

        # Deterministic overrides to avoid misrouting simple math or unsafe content
        query_lower = query.lower()
        if features.get("has_unsafe_keywords"):
            intent, confidence = "UNSAFE", 1.0
        else:
            # Treat pure math/digit queries as NUMERIC even if classifier is uncertain
            simple_math_pattern = all(ch.isdigit() or ch in "+-*/ ." for ch in query)
            if (features.get("has_digits") and features.get("has_math_operators")) or simple_math_pattern:
                intent, confidence = "NUMERIC", max(confidence, 0.9)
            # Override prediction/fortune-telling queries to RULE engine (safe refusal)
            elif any(word in query_lower for word in ["predict my", "predict your", "will i", "fortune", "future of my", "my future"]):
                intent, confidence = "UNSAFE", 1.0
            # Comparative/versus questions are conceptual; treat as EXPLANATION
            elif " vs " in query_lower or " versus " in query_lower:
                intent = "EXPLANATION"
            # Override common factual phrasings to FACTUAL to avoid transformer on facts
            elif query_lower.startswith(("capital of", "who is", "what is", "where is", "define ", "definition of", "tell me about")):
                intent = "FACTUAL"
            # Override "how many" queries asking for specific facts to FACTUAL not EXPLANATION
            elif query_lower.startswith("how many") or query_lower.startswith("how much"):
                intent = "FACTUAL"
            elif features.get("question_type") == "EXPLANATION":
                intent = "EXPLANATION"
            elif features.get("question_type") == "FACTUAL":
                intent = "FACTUAL"
        
        # Step 3: Route to engine
        engine_name, routing_reason = meta_controller.route(intent, confidence, features)
        
        # Step 4: Execute appropriate engine
        if engine_name == "RULE":
            result = rule_engine.execute(query, features)
        elif engine_name == "RETRIEVAL":
            result = retrieval_engine.execute(query, features)
        elif engine_name == "ML":
            result = ml_engine.execute(query, features)
        elif engine_name == "TRANSFORMER":
            result = transformer_engine.execute(query, features)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown engine: {engine_name}")
        
        # Step 5: Validate output
        is_valid, validated_answer, validation_details = output_validator.validate(
            answer=result["answer"],
            strategy=result["strategy"],
            confidence=result["confidence"],
            query=query
        )
        
        # Store query context for feedback
        query_context_cache[query] = {
            "intent": intent,
            "confidence": confidence
        }
        
        # Prepare response
        response = QueryResponse(
            answer=validated_answer,
            strategy=result["strategy"],
            confidence=result["confidence"],
            reason=routing_reason,
            metadata={
                "intent": intent,
                "intent_confidence": confidence,
                "validation": validation_details,
                "source": result.get("source"),
                "computation_type": result.get("computation_type")
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


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
    """Get list of supported intents."""
    return {
        "intents": intent_classifier.get_all_intents(),
        "routing_map": meta_controller.ROUTING_MAP,
        "description": {
            "FACTUAL": "Factual queries - routed to RETRIEVAL engine",
            "NUMERIC": "Numerical computations - routed to ML engine",
            "EXPLANATION": "Conceptual explanations - routed to TRANSFORMER engine",
            "UNSAFE": "Unsafe/restricted queries - routed to RULE engine"
        }
    }


@app.get("/model/status")
async def get_model_status():
    """Get detailed model training and load status."""
    # Check if we're using trained model or zero-shot
    model_type = "zero-shot" if intent_classifier.is_loaded else "fallback"
    
    status = {
        "model_type": model_type,
        "intent_classifier": {
            "loaded": intent_classifier.is_loaded,
            "model_name": getattr(intent_classifier, "model_name", "unknown"),
            "type": "DistilBERT MNLI (pre-trained zero-shot)",
            "requires_training": False,
            "status": "✅ READY" if intent_classifier.is_loaded else "⚠️ USING FALLBACK"
        },
        "transformer_engine": {
            "loaded": transformer_engine.is_loaded,
            "model_name": getattr(transformer_engine, "model_name", "unknown"),
            "type": "Flan-T5 (pre-trained generative)",
            "requires_training": False,
            "status": "✅ READY" if transformer_engine.is_loaded else "⚠️ USING FALLBACK"
        },
        "training_info": {
            "note": "Current system uses pre-trained models that don't require training",
            "feedback_collected": feedback_store.get_feedback_stats().get("total_feedback", 0),
            "auto_improvement": "Enabled - triggers every 10 feedbacks"
        },
        "system_status": "✅ FULLY OPERATIONAL" if (intent_classifier.is_loaded and transformer_engine.is_loaded) else "⚠️ PARTIAL - Using fallback modes"
    }
    
    return status


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
                "note": "Using pre-trained zero-shot model - no retraining needed"
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
            
            # Reload the intent classifier
            print("\n🔄 Reloading intent classifier...")
            global intent_classifier
            intent_classifier = IntentClassifier()
            
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
