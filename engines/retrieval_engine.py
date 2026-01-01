"""
Retrieval Engine - Fact Source
Retrieves facts from verified sources. NO GENERATION.
Search order: Local KB -> Wikipedia -> DuckDuckGo -> Safe Refusal
"""
import json
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import time


class RetrievalEngine:
    """
    Retrieves facts from verified sources.
    NEVER generates answers. NEVER hallucinates.
    """
    
    def __init__(self, kb_path: Optional[str] = None):
        """
        Initialize retrieval engine with knowledge base.
        
        Args:
            kb_path: Path to local knowledge base JSON file
        """
        if kb_path is None:
            kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
        
        self.kb_path = Path(kb_path)
        self.knowledge_base = self._load_knowledge_base()
        self.retrieval_history = []
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """
        Load local knowledge base from JSON file.
        
        Returns:
            Dictionary containing knowledge base
        """
        try:
            if self.kb_path.exists():
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    kb = json.load(f)
                print(f"✓ Loaded knowledge base with {len(kb.get('facts', []))} facts")
                return kb
            else:
                print(f"⚠ Knowledge base not found at {self.kb_path}")
                return {"facts": [], "metadata": {}}
        except Exception as e:
            print(f"✗ Failed to load knowledge base: {e}")
            return {"facts": [], "metadata": {}}
    
    def execute(self, query: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute retrieval query through multiple sources.
        
        Args:
            query: User query
            features: Features from input analyzer
            
        Returns:
            Dictionary with answer, confidence, strategy, and source
        """
        query_lower = features.get("lowercase_text", query.lower())
        
        # Step 1: Check local knowledge base
        kb_result = self._search_local_kb(query_lower)
        if kb_result:
            return kb_result
        
        # Step 2: Try Wikipedia
        wiki_result = self._search_wikipedia(query)
        if wiki_result:
            return wiki_result
        
        # Step 3: Try DuckDuckGo Instant Answer
        ddg_result = self._search_duckduckgo(query)
        if ddg_result:
            return ddg_result
        
        # Step 4: Safe refusal - fact not found
        return self._safe_refusal(query)
    
    def _search_local_kb(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search local knowledge base for exact or fuzzy matches.
        
        Args:
            query: Query string (lowercase)
            
        Returns:
            Result dictionary if found, None otherwise
        """
        if not self.knowledge_base or "facts" not in self.knowledge_base:
            return None
        
        # Search through facts
        query_tokens = set(query.split())
        for fact in self.knowledge_base["facts"]:
            keywords = [kw.lower() for kw in fact.get("keywords", [])]
            question = fact.get("question", "").lower()

            # Require either an exact question hit OR at least two keyword hits to avoid spurious matches
            keyword_hits = sum(1 for kw in keywords if kw in query_tokens or kw in query)
            if question in query or keyword_hits >= 2:
                self.retrieval_history.append({
                    "query": query,
                    "source": "local_kb",
                    "found": True
                })
                return {
                    "answer": fact.get("answer", ""),
                    "confidence": 1.0,
                    "strategy": "RETRIEVAL",
                    "source": "Local Knowledge Base",
                    "reason": "Match found in local knowledge base"
                }
        
        return None
    
    def _search_wikipedia(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search Wikipedia for factual information.
        
        Args:
            query: Query string
            
        Returns:
            Result dictionary if found, None otherwise
        """
        try:
            # Wikipedia API endpoint
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/"

            # Clean query for Wikipedia search with a few common mappings
            search_term = query.replace("what is", "").replace("who is", "").strip()
            search_term = search_term.replace("?", "").strip()

            normalized = search_term.lower().strip()
            wiki_overrides = {
                "c language": "C_(programming_language)",
                "c programming": "C_(programming_language)",
                "c programming language": "C_(programming_language)",
                "c++": "C++",
                "c++ language": "C++",
                "java": "Java_(programming_language)",
                "python": "Python_(programming_language)",
            }
            base_term = wiki_overrides.get(normalized, search_term)

            headers = {"User-Agent": "MetaLearningAI/1.0 (contact: dev@example.com)"}

            def try_fetch(term: str):
                full_url = f"{url}{requests.utils.quote(term)}?redirect=true"
                resp = requests.get(full_url, timeout=6, headers=headers)
                return resp, full_url

            def search_title(term: str) -> Optional[str]:
                search_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": term,
                    "format": "json",
                    "srlimit": 1
                }
                resp = requests.get(search_url, params=params, timeout=6, headers=headers)
                if resp.status_code != 200:
                    print(f"Wikipedia search API status: {resp.status_code}")
                    return None
                data = resp.json()
                hits = data.get("query", {}).get("search", [])
                if hits:
                    return hits[0].get("title")
                return None

            # Try raw cleaned term (spaces encoded)
            response, full_url = try_fetch(base_term)
            print(f"Wikipedia query term (raw): {base_term}")
            print(f"Wikipedia status: {response.status_code} for {full_url}")

            # If not found, try underscore form
            if (response.status_code != 200 or not response.text) and "(" not in base_term:
                underscore_term = base_term.replace(" ", "_")
                response, full_url = try_fetch(underscore_term)
                print(f"Wikipedia underscore term: {underscore_term}")
                print(f"Wikipedia status: {response.status_code} for {full_url}")

            # If still not found, try lowercase underscore form
            if (response.status_code != 200 or not response.text) and "(" not in base_term:
                lower_underscore_term = base_term.lower().replace(" ", "_")
                response, full_url = try_fetch(lower_underscore_term)
                print(f"Wikipedia lower underscore term: {lower_underscore_term}")
                print(f"Wikipedia status: {response.status_code} for {full_url}")

            # If still not found, try Title_With_Underscores
            if (response.status_code != 200 or not response.text) and "(" not in base_term:
                title_term = base_term.title().replace(" ", "_")
                response, full_url = try_fetch(title_term)
                print(f"Wikipedia title term: {title_term}")
                print(f"Wikipedia status: {response.status_code} for {full_url}")

            # If still not found, try the MediaWiki search API to get a canonical title
            if response.status_code != 200 or not response.text:
                search_hit_title = search_title(base_term)
                if search_hit_title:
                    response, full_url = try_fetch(search_hit_title)
                    print(f"Wikipedia search title term: {search_hit_title}")
                    print(f"Wikipedia status: {response.status_code} for {full_url}")

            if response.status_code == 200:
                data = response.json()
                extract = data.get("extract", "")

                # Accept shorter extracts (>=10 chars) to reduce false negatives
                if extract and len(extract) >= 10:
                    self.retrieval_history.append({
                        "query": query,
                        "source": "wikipedia",
                        "found": True
                    })
                    return {
                        "answer": extract,
                        "confidence": 0.9,
                        "strategy": "RETRIEVAL",
                        "source": "Wikipedia",
                        "reason": "Retrieved from Wikipedia API"
                    }
        except Exception as e:
            print(f"Wikipedia search error: {e}")

        return None
    
    def _search_duckduckgo(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search DuckDuckGo Instant Answer API.
        
        Args:
            query: Query string
            
        Returns:
            Result dictionary if found, None otherwise
        """
        try:
            # DuckDuckGo Instant Answer API
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
                
                # Try AbstractText first (accept any non-empty)
                abstract = data.get("AbstractText", "")
                if abstract and len(abstract) >= 10:
                    self.retrieval_history.append({
                        "query": query,
                        "source": "duckduckgo",
                        "found": True
                    })
                    return {
                        "answer": abstract,
                        "confidence": 0.85,
                        "strategy": "RETRIEVAL",
                        "source": "DuckDuckGo",
                        "reason": "Retrieved from DuckDuckGo Instant Answer API"
                    }
                
                # Try Answer field (accept any non-empty)
                answer = data.get("Answer", "")
                if answer:
                    self.retrieval_history.append({
                        "query": query,
                        "source": "duckduckgo",
                        "found": True
                    })
                    return {
                        "answer": answer,
                        "confidence": 0.85,
                        "strategy": "RETRIEVAL",
                        "source": "DuckDuckGo",
                        "reason": "Retrieved from DuckDuckGo Instant Answer API"
                    }
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        
        return None
    
    def _safe_refusal(self, query: str) -> Dict[str, Any]:
        """
        Return safe refusal when fact cannot be found.
        
        Args:
            query: Original query
            
        Returns:
            Refusal response
        """
        self.retrieval_history.append({
            "query": query,
            "source": "none",
            "found": False
        })
        
        return {
            "answer": (
                "I cannot find verified information to answer this query. "
                "The fact is not available in my knowledge base or external sources. "
                "I will not generate an answer to avoid providing incorrect information."
            ),
            "confidence": 1.0,  # Confident in the refusal
            "strategy": "RETRIEVAL",
            "source": "None",
            "reason": "Fact not found in any verified source - safe refusal"
        }
    
    def add_fact(self, question: str, answer: str, keywords: List[str]):
        """
        Add a new fact to the local knowledge base.
        
        Args:
            question: Question that this fact answers
            answer: The factual answer
            keywords: List of keywords for matching
        """
        new_fact = {
            "question": question,
            "answer": answer,
            "keywords": keywords
        }
        
        if "facts" not in self.knowledge_base:
            self.knowledge_base["facts"] = []
        
        self.knowledge_base["facts"].append(new_fact)
        
        # Save to file
        try:
            with open(self.kb_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            print(f"✓ Added new fact to knowledge base")
        except Exception as e:
            print(f"✗ Failed to save fact: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about retrieval operations.
        
        Returns:
            Dictionary with statistics
        """
        total = len(self.retrieval_history)
        if total == 0:
            return {
                "total_queries": 0,
                "successful_retrievals": 0,
                "failed_retrievals": 0,
                "source_distribution": {}
            }
        
        successful = sum(1 for r in self.retrieval_history if r["found"])
        failed = total - successful
        
        # Count by source
        source_counts = {}
        for entry in self.retrieval_history:
            if entry["found"]:
                source = entry["source"]
                source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_queries": total,
            "successful_retrievals": successful,
            "failed_retrievals": failed,
            "success_rate": successful / total if total > 0 else 0,
            "source_distribution": source_counts,
            "kb_size": len(self.knowledge_base.get("facts", []))
        }
