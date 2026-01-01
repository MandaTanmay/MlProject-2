"""
Retrain Intent Classifier from User Feedback
Uses collected feedback to improve the zero-shot classifier's routing decisions.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feedback.feedback_store import FeedbackStore
import pandas as pd


def retrain_from_feedback():
    """
    Export feedback data for analysis and retraining.
    
    Since we're using DistilBERT MNLI (zero-shot), we can:
    1. Analyze misclassifications
    2. Adjust intent label definitions
    3. Add more specific label descriptions
    4. Fine-tune the model on feedback data
    """
    
    # Load feedback
    feedback_store = FeedbackStore()
    
    # Get statistics
    stats = feedback_store.get_feedback_stats()
    
    print("=" * 60)
    print("FEEDBACK ANALYSIS FOR MODEL IMPROVEMENT")
    print("=" * 60)
    
    print(f"\nTotal Feedback: {stats.get('total_feedback', 0)}")
    print(f"Positive: {stats.get('positive_feedback', 0)} 👍")
    print(f"Negative: {stats.get('negative_feedback', 0)} 👎")
    print(f"Satisfaction Rate: {stats.get('satisfaction_rate', 0):.1%}")
    
    print("\n--- Intent Accuracy ---")
    intent_accuracy = stats.get('intent_accuracy', {})
    for intent, data in intent_accuracy.items():
        print(f"{intent}: {data['correct']}/{data['total']} ({data['accuracy']:.1%})")
    
    # Get training data
    print("\n--- Export Training Data ---")
    training_samples = feedback_store.get_training_data(
        min_confidence=0.5,
        only_correct=True
    )
    
    if training_samples:
        # Export to CSV
        df = pd.DataFrame(training_samples)
        output_path = Path(__file__).parent / "feedback_training_data.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Exported {len(training_samples)} samples to {output_path}")
        
        print("\n--- Sample Distribution ---")
        print(df['intent'].value_counts())
        
        print("\n--- Next Steps ---")
        print("1. Review feedback_training_data.csv for patterns")
        print("2. Adjust zero-shot labels in intent_classifier.py if needed")
        print("3. Fine-tune DistilBERT on this data (advanced)")
        print("4. Or: Use feedback to create better training set for custom model")
    else:
        print("⚠ No feedback data available yet")
        print("Collect user feedback via the UI first!")
    
    print("=" * 60)


if __name__ == "__main__":
    retrain_from_feedback()
