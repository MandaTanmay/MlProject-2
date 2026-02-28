"""
Retrain Domain / Engine-Selector Models from User Feedback
Exports feedback data, retrains sklearn models, and saves versioned artefacts
via core.model_registry.

NOTE: The SemanticIntentClassifier (MiniLM) is NOT retrained here - it uses
static pre-trained embeddings. Only the domain_classifier and engine_selector
(sklearn / joblib models) can be retrained from feedback.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feedback.feedback_store import FeedbackStore
from core.model_registry import save_model, get_registry_summary
import pandas as pd


def retrain_from_feedback():
    """
    Analyse collected feedback, export training data, retrain sklearn models,
    and persist versioned copies via the model registry.
    """

    feedback_store = FeedbackStore()
    stats = feedback_store.get_feedback_stats()

    print("=" * 60)
    print("FEEDBACK ANALYSIS FOR MODEL IMPROVEMENT")
    print("=" * 60)

    print(f"\nTotal Feedback:   {stats.get('total_feedback', 0)}")
    print(f"Positive:         {stats.get('positive_feedback', 0)} 👍")
    print(f"Negative:         {stats.get('negative_feedback', 0)} 👎")
    print(f"Satisfaction:     {stats.get('satisfaction_rate', 0):.1%}")

    intent_accuracy = stats.get('intent_accuracy', {})
    if intent_accuracy:
        print("\n--- Intent Accuracy ---")
        for intent, data in intent_accuracy.items():
            print(f"  {intent}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")

    # Export training samples
    print("\n--- Export Training Data ---")
    training_samples = feedback_store.get_training_data(
        min_confidence=0.5,
        only_correct=True
    )

    if not training_samples:
        print("⚠ No feedback data available yet. Collect user feedback via the UI first.")
        print("=" * 60)
        return False

    df = pd.DataFrame(training_samples)
    output_path = Path(__file__).parent / "feedback_training_data.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Exported {len(training_samples)} samples → {output_path}")

    print("\n--- Sample Distribution ---")
    print(df['intent'].value_counts().to_string())

    # Attempt to retrain sklearn domain_classifier on feedback data
    try:
        print("\n--- Retraining Domain Classifier ---")
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        # Build a quick domain model from feedback
        X = df["query"].tolist()
        y = df["intent"].tolist()

        if len(set(y)) < 2:
            print("⚠ Need at least 2 classes to retrain. Skipping sklearn retrain.")
        else:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=500, C=1.0)),
            ])
            pipe.fit(X, y)

            # Save with version tracking
            metrics = {
                "training_samples": len(X),
                "classes": list(set(y)),
                "satisfaction_rate": stats.get("satisfaction_rate", 0.0),
            }
            save_model(pipe, "feedback_intent_model", metadata=metrics)
            print(f"✓ Feedback-trained intent model saved (versioned).")

    except Exception as e:
        print(f"⚠ Sklearn retrain skipped: {e}")

    # Print registry summary
    print("\n--- Model Registry Summary ---")
    for name, info in get_registry_summary().items():
        print(f"  {name}: v{info['latest_version']} trained at {info['last_trained']}")

    print("\n--- Next Steps ---")
    print("1. Review feedback_training_data.csv for patterns.")
    print("2. Run training/train_all_models.py if enough new samples exist.")
    print("3. Domain classifier and engine selector will auto-version on each run.")

    print("=" * 60)
    return True


if __name__ == "__main__":
    retrain_from_feedback()
