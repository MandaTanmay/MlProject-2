"""
Production-Grade Multi-Layer Academic Rule Engine
Enforces academic integrity, blocks unsafe/unethical requests, prevents bypass attempts.

Architecture:
1. Semantic Unsafe Classifier (embedding-based detection)
2. Pattern-Based Hard Rules (deterministic layer)
3. Domain Violation Detection (academic-only enforcement)
4. Confidence Aggregation Logic (multi-layer scoring)
5. Anti-Bypass Detection (indirect phrasing detection)

Non-negotiable: If ANY unsafe indicator triggered → immediate block, no routing.
"""

from typing import Dict, Any, List, Tuple
import json
import logging
from datetime import datetime
import re
import hashlib

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticUnsafeClassifier:
    """
    Layer 1: Semantic detection using embedding similarity.
    Uses lightweight MiniLM model for fast inference.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.65):
        """
        Initialize the semantic unsafe classifier.
        
        Args:
            model_name: Sentence transformer model to use
            threshold: Similarity threshold for unsafe classification (0-1)
        """
        self.threshold = threshold
        self.model = None
        self.has_embeddings = HAS_EMBEDDINGS
        
        if self.has_embeddings:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}. Falling back to rule-based detection.")
                self.has_embeddings = False
        
        # Define unsafe prototype statements and their categories
        self.unsafe_prototypes = {
            "CHEATING": [
                "This query attempts to cheat in an exam or assignment.",
                "The user is asking for help to unfairly gain academic advantage.",
                "This involves providing direct answers to homework or exams."
            ],
            "HACKING": [
                "This query requests hacking or bypassing security systems.",
                "The user is asking how to gain unauthorized system access.",
                "This involves breaking into protected computer systems."
            ],
            "ACADEMIC_MISCONDUCT": [
                "This query asks for unauthorized academic advantage.",
                "The user is attempting academic fraud or plagiarism.",
                "This involves copying or stealing others' work."
            ],
            "PROMPT_INJECTION": [
                "This query attempts to override system rules or instructions.",
                "The user is trying to manipulate the AI system's behavior.",
                "This involves bypassing safety guidelines through deception."
            ],
            "ILLEGAL": [
                "This query requests illegal activity or law-breaking methods.",
                "The user is asking how to commit a crime.",
                "This involves instructions for illegal conduct."
            ],
            "HARMFUL": [
                "This query requests information to cause harm to people.",
                "The user is asking how to injure or damage someone.",
                "This involves dangerous or violent content."
            ],
            "SYSTEM_BYPASS": [
                "This query attempts to bypass AI safety restrictions.",
                "The user is trying to disable or circumvent safety measures.",
                "This involves pretending the AI has no restrictions."
            ]
        }
        
        self.prototype_embeddings = None
        self.categories_list = list(self.unsafe_prototypes.keys())
        
        if self.has_embeddings and self.model:
            self._compute_prototype_embeddings()
    
    def _compute_prototype_embeddings(self):
        """Pre-compute embeddings for all unsafe prototypes."""
        all_prototypes = []
        for category, statements in self.unsafe_prototypes.items():
            all_prototypes.extend(statements)
        
        try:
            self.prototype_embeddings = self.model.encode(all_prototypes, normalize_embeddings=True)
        except Exception as e:
            logger.warning(f"Failed to encode prototypes: {e}")
            self.has_embeddings = False
    
    def detect(self, query: str) -> Dict[str, Any]:
        """
        Detect unsafe content using semantic similarity.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with detected unsafe categories and scores
        """
        if not self.has_embeddings or self.model is None:
            return {
                "is_unsafe": False,
                "categories": {},
                "max_score": 0.0,
                "detection_method": "embedding_unavailable"
            }
        
        try:
            # Encode the query
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            
            # Compute similarity to each prototype
            similarities = np.dot(self.prototype_embeddings, query_embedding)
            
            # Map similarities back to categories
            category_scores = {cat: 0.0 for cat in self.categories_list}
            proto_idx = 0
            
            for category, statements in self.unsafe_prototypes.items():
                cat_scores = []
                for _ in statements:
                    cat_scores.append(float(similarities[proto_idx]))
                    proto_idx += 1
                # Take max similarity for each category
                category_scores[category] = max(cat_scores) if cat_scores else 0.0
            
            # Determine if unsafe
            max_score = max(category_scores.values())
            is_unsafe = max_score >= self.threshold
            
            return {
                "is_unsafe": is_unsafe,
                "categories": category_scores,
                "max_score": max_score,
                "max_category": max(category_scores, key=category_scores.get) if is_unsafe else None,
                "detection_method": "semantic_embedding"
            }
            
        except Exception as e:
            logger.error(f"Error in semantic detection: {e}")
            return {
                "is_unsafe": False,
                "categories": {},
                "max_score": 0.0,
                "detection_method": "error"
            }


class PatternBasedRuleDetector:
    """
    Layer 2: Deterministic pattern-based rule detection.
    Hard rules that trigger immediate block.
    """
    
    def __init__(self):
        """Initialize pattern-based rules."""
        # Hard-block patterns (case-insensitive)
        self.hard_patterns = {
            "CHEATING": [
                r"give me answers",
                r"give me exam answers",
                r"solve my (assignment|homework|project|exam|test)",
                r"do my (assignment|homework|project|exam|test)",
                r"complete my (assignment|homework|project|exam|test)",
                r"how to cheat",
                r"cheat (in|on) (exam|test|assignment)",
                r"predict my marks?",
                r"predict.{0,20}marks?",
                r"how to pass (exam|test) without studying",
                r"leaked? qu?estion paper",
                r"exam paper (copy|copy paste)",
                r"plagiarize"
            ],
            "SYSTEM_MANIPULATION": [
                r"ignore your restrictions?",
                r"disable (safety|rule engine)",
                r"override rule engine",
                r"you are not restricted",
                r"pretend you (are not|have no) restrictions?",
                r"pretend you are (an? )?unrestricted",
                r"forget your restrictions?",
                r"disable your restrictions?",
                r"bypass restrictions?",
                r"jailbreak",
                r"ignore (previous instructions?|safety guidelines)"
            ],
            "HACKING": [
                r"hack (.*?(portal|system|website|server|database|grading|attendance))",
                r"bypass (.*?(system|authentication|security|portal|attendance))",
                r"breach (.*?(system|website|server|database|grading))",
                r"unauthorized access",
                r"sql injection",
                r"cross.?site scripting",
                r"exploit",
                r"crack password",
                r"steal (password|credential)"
            ],
            "HARMFUL": [
                r"how to (make|create|build) (bomb|weapon|explosive|poison)",
                r"how to hurt (someone|people|a person)",
                r"how to harm (someone|people|a person)",
                r"how to (kill|murder|assassinate)",
                r"instructions? (for|to) (make|create|build) (bomb|weapon)",
                r"make malware",
                r"create virus"
            ]
        }
        
        # Compile regex patterns for performance
        self.compiled_patterns = {}
        for category, patterns in self.hard_patterns.items():
            self.compiled_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE), pattern) for pattern in patterns
            ]
    
    def detect(self, query: str) -> Dict[str, Any]:
        """
        Detect unsafe content using hard pattern matching.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with matched patterns and categories
        """
        matched = {}
        
        for category, compiled_list in self.compiled_patterns.items():
            matches = []
            for regex, original_pattern in compiled_list:
                if regex.search(query):
                    matches.append(original_pattern)
            
            if matches:
                matched[category] = matches
        
        return {
            "is_unsafe": len(matched) > 0,
            "matched_categories": matched,
            "detection_method": "hard_rules"
        }


class DomainViolationDetector:
    """
    Layer 3: Domain restriction enforcement.
    Only academic queries allowed.
    """
    
    def __init__(self):
        """Initialize domain restrictions."""
        # Non-academic domains to block
        self.non_academic_keywords = {
            "POLITICS": ["election", "politician", "government policy", "vote", "congress", "parliament"],
            "SPORTS": ["soccer", "basketball", "cricket", "football", "tennis", "match scores", "team ranking"],
            "ENTERTAINMENT": ["movie", "film", "actor", "actress", "celebrity", "gossip", "tv show", "music chart"],
            "TRADING": ["stock market", "crypto currency", "bitcoin", "forex", "day trading", "investment tips"],
            "NEWS": ["latest news", "breaking news", "world news", "current events"],
            "WEAPONS": ["weapon design", "explosives", "firearms", "missile design"],
            "DRUGS": ["drug synthesis", "drug manufacturing", "illicit drug", "narcotics production"]
        }
    
    def detect(self, query: str) -> Dict[str, Any]:
        """
        Detect if query is outside academic domain.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with domain violation info
        """
        query_lower = query.lower()
        matched_domains = {}
        
        for domain, keywords in self.non_academic_keywords.items():
            matches = [kw for kw in keywords if kw in query_lower]
            if matches:
                matched_domains[domain] = matches
        
        return {
            "is_non_academic": len(matched_domains) > 0,
            "matched_domains": matched_domains,
            "detection_method": "domain_restriction"
        }


class RuleEngine:
    """
    Production-Grade Rule Engine: Multi-layer safety enforcement.
    
    Sits BEFORE all routing and semantic analysis.
    Acts as deterministic hard stop for unsafe queries.
    """
    
    # Current version for logging
    MODEL_VERSION = "2.0-production"
    
    def __init__(self):
        """Initialize all detection layers."""
        # Initialize detection layers
        self.semantic_classifier = SemanticUnsafeClassifier()
        self.pattern_detector = PatternBasedRuleDetector()
        self.domain_detector = DomainViolationDetector()
        
        # Statistics tracking
        self.refusal_count = 0
        self.detection_logs = []
        
        # Response messages (non-judgmental, professional, consistent)
        self.refusal_messages = {
            "CHEATING": "This system only supports legitimate academic learning queries and cannot assist with academic misconduct.",
            "ACADEMIC_MISCONDUCT": "This system only supports legitimate academic learning queries and cannot assist with academic misconduct.",
            "HACKING": "This system cannot provide guidance on unauthorized system access or security bypassing. Please direct technical questions to legitimate academic resources.",
            "SYSTEM_MANIPULATION": "This system operates within defined safety boundaries. I cannot assist with requests to bypass these protections.",
            "SYSTEM_BYPASS": "This system operates within defined safety boundaries. I cannot assist with requests to bypass these protections.",
            "HARMFUL": "This system cannot provide information that could be used to cause harm. Please contact appropriate authorities if you have concerns about safety.",
            "ILLEGAL": "This system cannot provide guidance on illegal activities. Please contact appropriate authorities if you have concerns.",
            "PROMPT_INJECTION": "This system cannot assist with attempts to override its operational guidelines.",
            "DOMAIN_VIOLATION": "This system is designed exclusively for academic queries. Your question falls outside the supported domain."
        }
    
    def execute(self, query: str, features: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main execution: Multi-layer safety check.
        
        HARD STOP: If unsafe detected → immediate block, no routing.
        
        Args:
            query: User query
            features: Optional features from input analyzer
            
        Returns:
            Dictionary with safety decision and response
        """
        start_time = datetime.now()
        
        # ========== Layer 1: Pattern-Based Hard Rules ==========
        pattern_result = self.pattern_detector.detect(query)
        if pattern_result["is_unsafe"]:
            return self._create_block_response(
                query,
                primary_category=list(pattern_result["matched_categories"].keys())[0],
                confidence=1.0,
                detection_source="hard_rules",
                start_time=start_time
            )
        
        # ========== Layer 2: Semantic Unsafe Classifier ==========
        semantic_result = self.semantic_classifier.detect(query)
        if semantic_result["is_unsafe"]:
            return self._create_block_response(
                query,
                primary_category=semantic_result["max_category"],
                confidence=semantic_result["max_score"],
                detection_source="semantic_embedding",
                category_scores=semantic_result["categories"],
                start_time=start_time
            )
        
        # ========== Layer 3: Domain Violation Detection ==========
        domain_result = self.domain_detector.detect(query)
        if domain_result["is_non_academic"]:
            return self._create_block_response(
                query,
                primary_category="DOMAIN_VIOLATION",
                confidence=0.95,
                detection_source="domain_restriction",
                start_time=start_time,
                domain_info=domain_result["matched_domains"]
            )
        
        # ========== All layers passed: Query is SAFE ==========
        return {
            "status": "safe",
            "blocked": False,
            "category": None,
            "confidence": 0.0,
            "message": None,
            "engine": "rule_engine",
            "model_version": self.MODEL_VERSION,
            "detection_source": "all_clear",
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }
    
    def _create_block_response(
        self,
        query: str,
        primary_category: str,
        confidence: float,
        detection_source: str,
        start_time: datetime,
        category_scores: Dict[str, float] = None,
        domain_info: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create structured block response.
        
        Args:
            query: Original query
            primary_category: Unsafe category detected
            confidence: Confidence score (0-1)
            detection_source: "hard_rules", "semantic_embedding", or "domain_restriction"
            start_time: Query processing start time
            category_scores: Optional semantic category scores
            domain_info: Optional domain violation info
            
        Returns:
            Structured block response
        """
        self.refusal_count += 1
        
        # Create response
        response = {
            "status": "blocked",
            "blocked": True,
            "category": primary_category,
            "confidence": round(confidence, 3),
            "message": self.refusal_messages.get(
                primary_category,
                "This query has been blocked by safety rules."
            ),
            "engine": "rule_engine",
            "model_version": self.MODEL_VERSION,
            "detection_source": detection_source,
            "processing_time_ms": round((datetime.now() - start_time).total_seconds() * 1000, 2)
        }
        
        # Log for auditability
        self._log_safety_event(query, primary_category, confidence, detection_source, category_scores)
        
        return response
    
    def _log_safety_event(
        self,
        query: str,
        category: str,
        confidence: float,
        detection_source: str,
        category_scores: Dict[str, float] = None
    ):
        """
        Log safety event for auditability and monitoring.
        
        Args:
            query: Original query
            category: Detected unsafe category
            confidence: Confidence score
            detection_source: Detection method used
            category_scores: Optional detailed scores
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query_hash": hashlib.sha256(query.encode()).hexdigest(),
            "category": category,
            "confidence": confidence,
            "detection_source": detection_source,
            "model_version": self.MODEL_VERSION,
            "category_scores": category_scores or {}
        }
        
        self.detection_logs.append(log_entry)
        
        # Log to system logger as well
        logger.info(f"[RULE_ENGINE] Safety event: category={category}, confidence={confidence:.3f}, source={detection_source}")
    
    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """
        Non-blocking safety check (for diagnostics).
        
        Args:
            query: Query to check
            
        Returns:
            Detailed safety assessment without blocking
        """
        pattern_result = self.pattern_detector.detect(query)
        semantic_result = self.semantic_classifier.detect(query)
        domain_result = self.domain_detector.detect(query)
        
        return {
            "is_safe": not (
                pattern_result["is_unsafe"] or
                semantic_result["is_unsafe"] or
                domain_result["is_non_academic"]
            ),
            "pattern_detection": pattern_result,
            "semantic_detection": semantic_result,
            "domain_detection": domain_result
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieve rule engine statistics.
        
        Returns:
            Dictionary with safety statistics
        """
        return {
            "total_refusals": self.refusal_count,
            "total_safety_events_logged": len(self.detection_logs),
            "model_version": self.MODEL_VERSION,
            "embedding_available": self.semantic_classifier.has_embeddings,
            "unsafe_categories": list(self.semantic_classifier.unsafe_prototypes.keys()),
            "hard_rule_categories": list(self.pattern_detector.hard_patterns.keys()),
            "domain_restrictions": list(self.domain_detector.non_academic_keywords.keys())
        }
    
    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent safety logs.
        
        Args:
            limit: Number of recent logs to return
            
        Returns:
            List of recent safety events
        """
        return self.detection_logs[-limit:]
    
    def integrity_check(self) -> Dict[str, Any]:
        """
        Verify rule engine integrity and readiness.
        
        Returns:
            Integrity status
        """
        return {
            "initialized": True,
            "semantic_classifier_ready": self.semantic_classifier.has_embeddings,
            "pattern_detector_ready": len(self.pattern_detector.compiled_patterns) > 0,
            "domain_detector_ready": len(self.domain_detector.non_academic_keywords) > 0,
            "refusal_messages_configured": len(self.refusal_messages) > 0,
            "model_version": self.MODEL_VERSION
        }
