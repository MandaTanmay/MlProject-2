"""
Domain Classifier - STUDENT vs OUTSIDE Domain Enforcement
First-level gatekeeper that blocks all non-academic queries.
This is MANDATORY - no queries pass without domain verification.
"""
from typing import Tuple
from pathlib import Path
import joblib
import warnings

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
            return "OUTSIDE", 0.6
    
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
