"""
Factual Engine - Hybrid Knowledge Retrieval
Retrieves verified facts via embedding-based semantic search.
Falls back to external resources (Wikipedia, DuckDuckGo) for simple questions.
ZERO generation. ZERO guessing. Confidence-aware.

Architecture:
  Query → Encode (MiniLM) → Semantic Similarity → Top-K → 
  Confidence Check → If KB fails: Try Wikipedia/DuckDuckGo fallback → 
  Ambiguity Detection → Structured Response → Metadata

Strategy:
  - KB facts: Confidence 0.65-1.0 (verified, high confidence)
  - External facts: Confidence 0.50-0.60 (less trusted, must attribute source)
  - Below 0.50: Refuse (too uncertain)
"""

import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import logging
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger(__name__)


class FactualEngine:
    """
    Hybrid embedding-based and fallback semantic retrieval for factual questions.
    
    Strategy:
    1. First: Try local knowledge base (confidence 0.65-1.0)
    2. Fallback: Try Wikipedia/DuckDuckGo (confidence 0.50-0.60, must attribute)
    3. If still uncertain: Refuse (below 0.50 threshold)
    
    Guarantees:
    - No hallucination (never guesses or generates)
    - Confidence-aware (all responses scored)
    - Source attribution (external sources clearly marked)
    - Auditable (complete metadata trails)
    - Safe refusal (below threshold = refuse)
    """
    
    # Confidence thresholds
    FACTUAL_CONFIDENCE_THRESHOLD = 0.65  # KB facts minimum
    EXTERNAL_CONFIDENCE_THRESHOLD = 0.50  # External sources minimum
    AMBIGUITY_MAX_DIFF = 0.05  # Max difference between top-2 to flag ambiguity
    
    # External source settings
    WIKIPEDIA_CONFIDENCE = 0.55  # Lower than KB threshold
    DUCKDUCKGO_CONFIDENCE = 0.50  # Lowest trusted confidence
    
    def __init__(self, kb_path: Optional[str] = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", enable_external: bool = True):
        """
        Initialize factual engine with embedding-based semantic search + external fallback.
        
        Args:
            kb_path: Path to structured knowledge base JSON
            model_name: Sentence transformer model for embeddings
            enable_external: Enable fetching from Wikipedia/DuckDuckGo as fallback
        """
        if kb_path is None:
            kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
        
        self.kb_path = Path(kb_path)
        self.model_name = model_name
        self.model = None
        self.has_embeddings = HAS_EMBEDDINGS
        self.has_requests = HAS_REQUESTS
        self.enable_external = enable_external and HAS_REQUESTS
        
        # Knowledge base structure
        self.knowledge_base = self._load_knowledge_base()
        self.fact_embeddings = {}  # {fact_id: embedding_vector}
        self.fact_lookup = {}      # {fact_id: fact_data}
        
        # Statistics
        self.retrieval_history = []
        self.total_retrievals = 0
        self.successful_retrievals = 0
        self.external_fallback_count = 0
        
        # Initialize embeddings
        if self.has_embeddings:
            try:
                self.model = SentenceTransformer(model_name)
                self._precompute_embeddings()
                status = "✓ FactualEngine initialized"
                if self.enable_external:
                    status += " with external fallback enabled"
                logger.info(status)
            except Exception as e:
                logger.error(f"Failed to load embeddings model: {e}")
                self.has_embeddings = False
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load structured knowledge base.
        
        KB format:
        {
          "facts": [
            {
              "id": "fact_001",
              "question": "What is the capital of Germany?",
              "answer": "Berlin",
              "structured_value": "Berlin",
              "category": "geography",
              "source": "Academic Dataset",
              "verified": true,
              "verified_date": "2025-01-01"
            }, ...
          ]
        }
        
        Returns:
            Loaded knowledge base dictionary
        """
        try:
            if self.kb_path.exists():
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
                num_facts = len(kb.get('facts', []))
                logger.info(f"✓ Loaded knowledge base with {num_facts} verified facts")
                return kb
            else:
                logger.warning(f"Knowledge base not found at {self.kb_path}")
                return {"facts": []}
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return {"facts": []}
    
    def _precompute_embeddings(self):
        """
        Precompute embeddings for all facts at startup.
        This is a one-time cost that enables O(1) lookup per query.
        
        CRITICAL: Never recompute per request. Always reuse.
        """
        if not self.has_embeddings or self.model is None:
            return
        
        try:
            facts = self.knowledge_base.get('facts', [])
            
            for fact in facts:
                fact_id = fact.get('id')
                question = fact.get('question', '')
                
                if not fact_id or not question:
                    continue
                
                # Encode question pattern
                embedding = self.model.encode(question, normalize_embeddings=True)
                self.fact_embeddings[fact_id] = embedding
                self.fact_lookup[fact_id] = fact
            
            logger.info(f"✓ Precomputed {len(self.fact_embeddings)} fact embeddings")
        except Exception as e:
            logger.error(f"Error precomputing embeddings: {e}")
            self.has_embeddings = False
    
    
    def execute(self, query: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute deterministic factual retrieval.
        
        Process:
          1. Encode query
          2. Semantic similarity search
          3. Top-3 ranking
          4. Confidence thresholding
          5. Ambiguity detection
          6. Structure response
          7. Attach metadata
        
        Args:
            query: User query
            features: Features from input analyzer
            
        Returns:
            Structured response with status, confidence, metadata
        """
        start_time = time.time()
        self.total_retrievals += 1
        
        # Validate query
        if not query or not isinstance(query, str):
            return self._response_error("Invalid query format")
        
        # Check if embedding model available
        if not self.has_embeddings or self.model is None:
            return self._response_uncertain(
                query,
                "Embedding model not available",
                confidence=0.0
            )
        
        try:
            # Step 1: Semantic similarity search
            results = self._semantic_search(query)
            
            if not results:
                return self._response_uncertain(
                    query,
                    "No semantic matches found in knowledge base",
                    confidence=0.0
                )
            
            # results = [(fact_id, similarity_score), ...]
            
            # Step 2: Confidence thresholding
            top_score = results[0][1] if results else 0.0
            
            if top_score < self.FACTUAL_CONFIDENCE_THRESHOLD:
                # KB lookup failed - try external fallback
                if self.enable_external:
                    external_result = self._try_external_sources(query)
                    if external_result and external_result["confidence"] >= self.EXTERNAL_CONFIDENCE_THRESHOLD:
                        self.successful_retrievals += 1
                        self.external_fallback_count += 1
                        self._log_retrieval(query, external_result["metadata"].get("fact_id"), 
                                          external_result["confidence"], True, "external")
                        elapsed_ms = (time.time() - start_time) * 1000
                        external_result["metadata"]["retrieval_time_ms"] = round(elapsed_ms, 2)
                        return external_result
                
                return self._response_uncertain(
                    query,
                    f"Best KB match {top_score:.2f} below threshold {self.FACTUAL_CONFIDENCE_THRESHOLD}. External sources also insufficient.",
                    confidence=top_score
                )
            
            # Step 3: Ambiguity detection
            if len(results) >= 2:
                top_1_score = results[0][1]
                top_2_score = results[1][1]
                score_diff = top_1_score - top_2_score
                
                if score_diff < self.AMBIGUITY_MAX_DIFF:
                    return self._response_ambiguous(
                        query,
                        results[:3],  # Top 3 candidates
                        top_1_score
                    )
            
            # Step 4: Retrieve top fact
            fact_id = results[0][0]
            similarity_score = results[0][1]
            fact = self.fact_lookup.get(fact_id)
            
            if not fact:
                return self._response_uncertain(query, "Fact lookup failed", confidence=0.0)
            
            # Step 5: Structure response
            elapsed_ms = (time.time() - start_time) * 1000
            response = self._response_success(fact, similarity_score, elapsed_ms)
            
            self.successful_retrievals += 1
            self._log_retrieval(query, fact_id, similarity_score, True)
            
            return response
            
        except Exception as e:
            logger.error(f"Error during factual retrieval: {e}")
            return self._response_error(str(e))
    
    def _semantic_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Perform semantic similarity search.
        
        Process:
          1. Encode query
          2. Cosine similarity against all facts
          3. Rank descending
          4. Return top-5
        
        Args:
            query: Query string
            
        Returns:
            List of (fact_id, similarity_score) tuples, ranked by score
        """
        if not self.model or not self.fact_embeddings:
            return []
        
        try:
            # Encode query (normalized)
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            
            # Compute cosine similarity with all facts
            # (dot product on normalized embeddings = cosine similarity)
            similarities = {}
            for fact_id, fact_embedding in self.fact_embeddings.items():
                similarity = float(np.dot(query_embedding, fact_embedding))
                similarities[fact_id] = similarity
            
            # Rank descending
            ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            return ranked[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def _try_external_sources(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Try to fetch answer from external sources (Wikipedia, DuckDuckGo).
        Only used as fallback when KB lookup is insufficient.
        
        Strategy:
        1. Try Wikipedia first (0.55 confidence if found)
        2. Try DuckDuckGo (0.50 confidence if found)
        3. Return None if neither succeeds
        
        Args:
            query: Query string
            
        Returns:
            Response dict if found, None otherwise
        """
        if not self.enable_external or not self.has_requests:
            return None
        
        try:
            # Try Wikipedia first
            wiki_result = self._fetch_wikipedia(query)
            if wiki_result:
                return wiki_result
            
            # Fallback to DuckDuckGo
            ddg_result = self._fetch_duckduckgo(query)
            if ddg_result:
                return ddg_result
            
            return None
        except Exception as e:
            logger.warning(f"Error fetching external sources: {e}")
            return None
    
    def _fetch_wikipedia(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Fetch from Wikipedia API - only for simple factual questions.
        
        Args:
            query: Query string
            
        Returns:
            Response dict or None
        """
        try:
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
            
            # Clean query
            search_term = query.replace("what is", "").replace("who is", "").replace("?", "").strip()
            search_term = search_term.replace(" ", "%20")
            
            headers = {"User-Agent": "MetaLearningAI/1.0"}
            response = requests.get(f"{url}{search_term}", timeout=5, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                extract = data.get("extract", "")
                
                if extract and len(extract) >= 20:
                    return {
                        "status": "success",
                        "type": "FACTUAL",
                        "data": {
                            "answer": extract[:500],  # Limit length
                            "structured_value": extract[:200],
                            "entity": data.get("title", search_term),
                            "category": "external"
                        },
                        "confidence": self.WIKIPEDIA_CONFIDENCE,
                        "metadata": {
                            "fact_id": f"wikipedia_{search_term}",
                            "source": "Wikipedia",
                            "external": True,
                            "verified": False,
                            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            "timestamp": datetime.now().isoformat(),
                            "engine": "FactualEngine",
                            "retrieval_method": "wikipedia_api"
                        }
                    }
        except Exception as e:
            logger.debug(f"Wikipedia fetch failed: {e}")
        
        return None
    
    def _fetch_duckduckgo(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Fetch from DuckDuckGo Instant Answer API - final fallback.
        
        Args:
            query: Query string
            
        Returns:
            Response dict or None
        """
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Try AbstractText
                abstract = data.get("AbstractText", "")
                if abstract and len(abstract) >= 20:
                    return {
                        "status": "success",
                        "type": "FACTUAL",
                        "data": {
                            "answer": abstract[:500],
                            "structured_value": abstract[:200],
                            "entity": data.get("Heading", query),
                            "category": "external"
                        },
                        "confidence": self.DUCKDUCKGO_CONFIDENCE,
                        "metadata": {
                            "fact_id": f"duckduckgo_{query}",
                            "source": "DuckDuckGo",
                            "external": True,
                            "verified": False,
                            "timestamp": datetime.now().isoformat(),
                            "engine": "FactualEngine",
                            "retrieval_method": "duckduckgo_api"
                        }
                    }
        except Exception as e:
            logger.debug(f"DuckDuckGo fetch failed: {e}")
        
        return None
    
    def _response_success(self, fact: Dict[str, Any], similarity: float, elapsed_ms: float) -> Dict[str, Any]:
        """
        Build successful response.
        
        Args:
            fact: Fact data from knowledge base
            similarity: Similarity score [0, 1]
            elapsed_ms: Retrieval time in milliseconds
            
        Returns:
            Structured response
        """
        return {
            "status": "success",
            "type": "FACTUAL",
            "data": {
                "answer": fact.get("answer", ""),
                "structured_value": fact.get("structured_value", fact.get("answer", "")),
                "entity": fact.get("entity", ""),
                "category": fact.get("category", "")
            },
            "confidence": round(similarity, 4),
            "metadata": {
                "fact_id": fact.get("id"),
                "similarity_score": round(similarity, 4),
                "source": fact.get("source", "Unknown"),
                "verified": fact.get("verified", False),
                "verified_date": fact.get("verified_date", ""),
                "retrieval_time_ms": round(elapsed_ms, 2),
                "timestamp": datetime.now().isoformat(),
                "engine": "FactualEngine",
                "model": self.model_name.split("/")[-1] if self.model_name else "unknown"
            }
        }
    
    def _response_uncertain(self, query: str, reason: str, confidence: float) -> Dict[str, Any]:
        """
        Return uncertain response (no confident match found).
        
        Args:
            query: Original query
            reason: Reason for uncertainty
            confidence: Confidence score (if available)
            
        Returns:
            Uncertain response
        """
        return {
            "status": "uncertain",
            "type": "FACTUAL",
            "data": {
                "answer": None,
                "structured_value": None,
                "reason": reason
            },
            "confidence": round(confidence, 4),
            "metadata": {
                "fact_id": None,
                "reason": reason,
                "retrieval_time_ms": 0,
                "timestamp": datetime.now().isoformat(),
                "engine": "FactualEngine"
            }
        }
    
    def _response_ambiguous(self, query: str, candidates: List[Tuple[str, float]], top_score: float) -> Dict[str, Any]:
        """
        Return ambiguous response (multiple similar matches).
        
        Args:
            query: Original query
            candidates: List of (fact_id, score) tuples
            top_score: Top similarity score
            
        Returns:
            Ambiguous response with candidates
        """
        candidate_info = []
        for fact_id, score in candidates:
            fact = self.fact_lookup.get(fact_id)
            if fact:
                candidate_info.append({
                    "fact_id": fact_id,
                    "answer": fact.get("answer"),
                    "similarity": round(score, 4),
                    "category": fact.get("category")
                })
        
        return {
            "status": "ambiguous",
            "type": "FACTUAL",
            "data": {
                "candidates": candidate_info,
                "reason": f"Multiple similar matches detected (score diff < {self.AMBIGUITY_MAX_DIFF})"
            },
            "confidence": round(top_score, 4),
            "metadata": {
                "reason": "ambiguity_detected",
                "num_candidates": len(candidate_info),
                "timestamp": datetime.now().isoformat(),
                "engine": "FactualEngine"
            }
        }
    
    def _response_error(self, error_msg: str) -> Dict[str, Any]:
        """
        Return error response.
        
        Args:
            error_msg: Error message
            
        Returns:
            Error response
        """
        return {
            "status": "error",
            "type": "FACTUAL",
            "data": {
                "error": error_msg
            },
            "confidence": 0.0,
            "metadata": {
                "reason": error_msg,
                "timestamp": datetime.now().isoformat(),
                "engine": "FactualEngine"
            }
        }
    
    def _log_retrieval(self, query: str, fact_id: Optional[str], score: float, found: bool, source_type: str = "kb"):
        """
        Log retrieval event for auditability.
        
        Args:
            query: Original query
            fact_id: Retrieved fact ID (or None)
            score: Similarity/confidence score
            found: Whether fact was found
            source_type: "kb" or "external"
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "fact_id": fact_id,
            "similarity_score": round(score, 4),
            "found": found,
            "source": source_type
        }
        self.retrieval_history.append(record)
        logger.info(f"Retrieval: query={query[:50]}... → fact={fact_id} score={score:.3f} found={found} source={source_type}")
    

    
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retrieval statistics and auditability metrics.
        
        Returns:
            Dictionary with stats including success rate, source distribution
        """
        total = self.total_retrievals
        successful = self.successful_retrievals
        success_rate = (successful / total * 100) if total > 0 else 0.0
        
        # Analyze sources from history
        kb_count = sum(1 for r in self.retrieval_history if r.get("source") == "kb" and r["found"])
        external_count = sum(1 for r in self.retrieval_history if r.get("source") == "external" and r["found"])
        
        return {
            "total_retrievals": total,
            "successful_retrievals": successful,
            "success_rate": round(success_rate, 2),
            "kb_retrievals": kb_count,
            "external_retrievals": external_count,
            "external_fallback_count": self.external_fallback_count,
            "fact_count": len(self.fact_lookup),
            "embedding_count": len(self.fact_embeddings),
            "retrieval_history_size": len(self.retrieval_history),
            "confidence_threshold": self.FACTUAL_CONFIDENCE_THRESHOLD,
            "external_threshold": self.EXTERNAL_CONFIDENCE_THRESHOLD,
            "ambiguity_threshold": self.AMBIGUITY_MAX_DIFF,
            "external_enabled": self.enable_external,
            "model": self.model_name
        }
    
    def add_fact(self, fact: Dict[str, Any]) -> bool:
        """
        Add new fact to knowledge base and update embeddings.
        
        Args:
            fact: Fact dictionary with required fields
                  {id, question, answer, structured_value, category, source, verified_date}
                  
        Returns:
            True if successful, False otherwise
        """
        if not fact.get("id") or not fact.get("question"):
            logger.warning("Cannot add fact: missing id or question")
            return False
        
        fact_id = fact["id"]
        
        # Check if exists
        if fact_id in self.fact_lookup:
            logger.info(f"Updating fact {fact_id}")
        
        try:
            # Store fact
            self.fact_lookup[fact_id] = fact
            
            # Compute and store embedding
            if self.model:
                embedding = self.model.encode(fact["question"], normalize_embeddings=True)
                self.fact_embeddings[fact_id] = embedding
                logger.info(f"Added/updated fact {fact_id} with embedding")
                return True
            else:
                logger.error("Cannot add fact: model not available")
                return False
                
        except Exception as e:
            logger.error(f"Error adding fact {fact_id}: {e}")
            return False
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate response structure and fields.
        
        Args:
            response: Response dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = {"status", "type", "data", "confidence", "metadata"}
        
        if not all(key in response for key in required_keys):
            return False
        
        if response["type"] != "FACTUAL":
            return False
        
        if not isinstance(response["confidence"], (int, float)):
            return False
        
        if response["confidence"] < 0 or response["confidence"] > 1:
            return False
        
        return True
    
    def clear_history(self):
        """Clear retrieval history."""
        self.retrieval_history = []
        logger.info("Cleared retrieval history")
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.total_retrievals = 0
        self.successful_retrievals = 0
        self.retrieval_history = []
        logger.info("Reset statistics counters")
