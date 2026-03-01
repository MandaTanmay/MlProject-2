"""
Semantic Domain Classifier - STRICT ACADEMIC MODE

Uses SentenceTransformer embeddings + Logistic Regression.
Pure semantic classification.
"""

from typing import Tuple
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer


class DomainClassifier:

    DOMAINS = ["STUDENT", "OUTSIDE"]
    REFUSAL_MESSAGE = "This system is restricted to academic student-related queries only."

    def __init__(self, model_dir: str = None):

        if model_dir is None:
            model_dir = Path(__file__).parent.parent / "training" / "models"

        self.model_dir = Path(model_dir)
        self.classifier_path = self.model_dir / "domain_classifier_semantic.joblib"

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.classifier = None
        self.is_loaded = False

        self._load()

    def _load(self):
        try:
            if self.classifier_path.exists():
                self.classifier = joblib.load(self.classifier_path)
                self.is_loaded = True
                print("✓ Semantic Domain Classifier loaded")
            else:
                print("✗ Semantic domain model not found. Train it first.")
        except Exception as e:
            print("✗ Failed to load domain classifier:", e)
            self.is_loaded = False

    def predict(self, query: str) -> Tuple[str, float]:

        if not self.is_loaded:
            return "OUTSIDE", 1.0

        try:
            embedding = self.embedding_model.encode([query])
            probabilities = self.classifier.predict_proba(embedding)[0]

            student_index = list(self.classifier.classes_).index("STUDENT")
            outside_index = list(self.classifier.classes_).index("OUTSIDE")

            student_prob = probabilities[student_index]
            outside_prob = probabilities[outside_index]

            if student_prob >= 0.60:
                return "STUDENT", float(student_prob)
            else:
                return "OUTSIDE", float(outside_prob)

        except Exception as e:
            print("Domain prediction error:", e)
            return "OUTSIDE", 1.0

    def get_refusal_message(self) -> str:
        return self.REFUSAL_MESSAGE