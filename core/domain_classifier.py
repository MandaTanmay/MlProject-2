"""
Domain Classifier - STUDENT vs OUTSIDE Domain Enforcement
First-level gatekeeper that blocks all non-academic queries.
This is MANDATORY - no queries pass without domain verification.
"""
from typing import Tuple
from pathlib import Path
import joblib
import warnings
import re
import difflib
import json

warnings.filterwarnings('ignore')


class DomainClassifier:
    """
    Binary classifier that determines if a query is academic (STUDENT) or not (OUTSIDE).
    Uses TF-IDF + Logistic Regression for fast, accurate domain classification.
    
    Target accuracy: > 95%
    """
    
    DOMAINS = ["STUDENT", "OUTSIDE"]
    
    # Strict refusal message for OUTSIDE domain
    REFUSAL_MESSAGE = "This system is restricted to academic student-related queries only."
    
    def __init__(self, model_dir: str = None):
        """
        Initialize domain classifier.
        
        Args:
            model_dir: Directory containing trained models
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "training" / "models"
        
        self.model_dir = Path(model_dir)
        self.vectorizer = None
        self.classifier = None
        self.is_loaded = False
        self._kb = None
        self._kb_tokens = set()
        
        # Try to load trained models
        self.load_models()
    
    def load_models(self) -> bool:
        """
        Load trained domain classification models.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            vectorizer_path = self.model_dir / "domain_vectorizer.joblib"
            classifier_path = self.model_dir / "domain_classifier.joblib"
            
            if vectorizer_path.exists() and classifier_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                self.classifier = joblib.load(classifier_path)
                self.is_loaded = True
                print(f"✓ Domain classifier loaded (TF-IDF + Logistic Regression)")
                return True
            else:
                print(f"⚠ Domain classifier models not found. Using fallback classification.")
                print(f"   Expected: {vectorizer_path}")
                print(f"   Run training/train_domain_model.py to create models.")
                return False
                
        except Exception as e:
            print(f"✗ Failed to load domain classifier: {e}")
            return False
    
    def predict(self, query: str) -> Tuple[str, float]:
        """
        Predict whether query is STUDENT (academic) or OUTSIDE domain.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (domain, confidence_score)
        """
        # SRKR whitelist - always classify as STUDENT with high confidence
        query_lower = query.lower()

        # Quick KB lookup: accept short/simple queries that match knowledge base
        try:
            if self._kb is None:
                kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
                if kb_path.exists():
                    with open(kb_path, "r", encoding="utf-8") as f:
                        self._kb = json.load(f)
                else:
                    self._kb = {"facts": []}
            # Build a token/alias set from KB (cache)
            if not self._kb_tokens:
                for fact in self._kb.get("facts", []):
                    for field in ("entity", "question", "id"):
                        val = fact.get(field, "")
                        if not val:
                            continue
                        val_l = val.lower()
                        # add full phrase
                        self._kb_tokens.add(val_l)
                        # add individual tokens
                        for tok in re.findall(r"\w+", val_l):
                            if len(tok) > 1:
                                self._kb_tokens.add(tok)

            # If query is a short token or phrase, check KB tokens/aliases
            if len(query.strip()) <= 30:
                qtok = query_lower.strip()
                # direct or substring match against KB phrases
                if qtok in self._kb_tokens:
                    return "STUDENT", 0.95
                for phrase in self._kb_tokens:
                    if qtok == phrase or qtok in phrase or phrase in qtok:
                        return "STUDENT", 0.95
                # fuzzy match on tokens
                try:
                    match = difflib.get_close_matches(qtok, list(self._kb_tokens), n=1, cutoff=0.78)
                    if match:
                        return "STUDENT", 0.9
                except Exception:
                    pass
        except Exception:
            # don't fail noisy KB checks
            pass
        srkr_keywords = [
            'srkr', 'b.tech', 'btech', 'jntuk', 'naac', 'aicte', 
            'r23', 'regulation', 'credits', 'cgpa', 'gpa', 'semester',
            'attendance', 'grading', 'evaluation', 'internship', 'project',
            'elective', 'mooc', 'honours', 'honors', 'minor', 'induction',
            'pass marks', 'revaluation', 'malpractice', 'promotion',
            'medium of instruction', 'programme duration', 'credit transfer'
        ]
        if any(kw in query_lower for kw in srkr_keywords):
            return "STUDENT", 0.95

        # Quick numeric/arithmetic detection: allow standalone math expressions
        if re.match(r'^[\d\s\+\-\*\/\^\.\%\(\)]+$', query.strip()):
            return "STUDENT", 0.99

        # Fuzzy keyword check: catch common academic keywords even if misspelled
        fuzzy_keywords = [
            'quantum', 'quantum computing', 'machine learning', 'artificial intelligence',
            'algorithm', 'mathematics', 'physics', 'chemistry', 'biology', 'computer', 'programming'
        ]
        tokens = re.findall(r"\w+", query_lower)
        for tok in tokens:
            match = difflib.get_close_matches(tok, fuzzy_keywords, n=1, cutoff=0.8)
            if match:
                return "STUDENT", 0.9
        
        if not self.is_loaded:
            # Fallback to rule-based classification
            return self._fallback_prediction(query)
        
        try:
            query_vec = self.vectorizer.transform([query])

            

            probabilities = self.classifier.predict_proba(query_vec)[0]

            # Extract probabilities properly
            student_index = list(self.classifier.classes_).index("STUDENT")
            outside_index = list(self.classifier.classes_).index("OUTSIDE")

            student_prob = probabilities[student_index]
            outside_prob = probabilities[outside_index]

            confidence = max(student_prob, outside_prob)

            # -----------------------------
            # STRICT ACADEMIC POLICY
            # -----------------------------

            STUDENT_THRESHOLD = 0.65
            MARGIN_THRESHOLD = 0.15

            margin = student_prob - outside_prob

            if student_prob >= STUDENT_THRESHOLD and margin >= MARGIN_THRESHOLD:
                return "STUDENT", float(student_prob)
            else:
                return "OUTSIDE", float(outside_prob)
            
        except Exception as e:
            print(f"✗ Domain prediction error: {e}")
            return self._fallback_prediction(query)
    
    def _fallback_prediction(self, query: str) -> Tuple[str, float]:
        """
        Fallback rule-based domain classification when model not available.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (domain, confidence_score)
        """
        query_lower = query.lower()
        
        # Academic keywords indicating STUDENT domain
        academic_keywords = [
            'course', 'class', 'lecture', 'professor', 'exam', 'test',
            'assignment', 'homework', 'grade', 'gpa', 'cgpa', 'credits',
            'semester', 'college', 'university', 'student', 'attendance',
            'library', 'lab', 'syllabus', 'curriculum', 'admission',
            'degree', 'major', 'minor', 'thesis', 'research', 'project',
            'scholarship', 'tuition', 'dean', 'faculty', 'department',
            'campus', 'hostel', 'cafeteria', 'sports complex',
            'placement', 'internship', 'career', 'coding', 'programming',
            'algorithm', 'data structure', 'machine learning', 'ai',
            'artificial intelligence', 'python', 'java', 'database',
            'web development', 'software', 'mathematics', 'physics',
            'chemistry', 'biology', 'engineering', 'science', 'study',
            'learning', 'education', 'academic', 'school'
        ]
        
        # Non-academic keywords indicating OUTSIDE domain
        outside_keywords = [
            'movie', 'film', 'cinema', 'actor', 'actress', 'director',
            'politics', 'politician', 'election', 'government', 'president',
            'prime minister', 'parliament', 'congress', 'party',
            'cricket', 'football', 'basketball', 'sports', 'match', 'tournament',
            'player', 'team', 'score', 'winner', 'champion',
            'recipe', 'cooking', 'restaurant', 'food', 'dish',
            'weather', 'forecast', 'temperature', 'rain', 'climate',
            'travel', 'vacation', 'hotel', 'flight', 'destination',
            'shopping', 'buy', 'price', 'discount', 'sale',
            'celebrity', 'gossip', 'entertainment', 'show', 'series',
            'stock market', 'shares', 'trading', 'investment',
            'medical diagnosis', 'disease', 'symptoms', 'medicine',
            'legal advice', 'lawyer', 'court', 'lawsuit'
        ]
        
        # Count keyword matches
        academic_score = sum(1 for kw in academic_keywords if kw in query_lower)
        outside_score = sum(1 for kw in outside_keywords if kw in query_lower)
        
        # Decision logic
        if outside_score > academic_score:
            return "OUTSIDE", 0.85
        elif academic_score > 0:
            return "STUDENT", 0.85
        else:
            # Ambiguous - default to STUDENT domain with low confidence
            # (allows academic queries without specific keywords)
            # NOTE: previously returned OUTSIDE here which caused simple
            # numeric/math queries (e.g. "2+2") to be blocked. Return
            # STUDENT by default to align with the comment and intended behavior.
            return "STUDENT", 0.6
    
    def get_refusal_message(self) -> str:
        """
        Get the standard refusal message for OUTSIDE domain queries.
        
        Returns:
            Refusal message string
        """
        return self.REFUSAL_MESSAGE
    
    def get_stats(self) -> dict:
        """
        Get domain classifier statistics.
        
        Returns:
            Dictionary with classifier stats
        """
        return {
            "model_loaded": self.is_loaded,
            "domains": self.DOMAINS,
            "target_accuracy": "> 95%",
            "model_type": "TF-IDF + Logistic Regression" if self.is_loaded else "Rule-based fallback"
        }
