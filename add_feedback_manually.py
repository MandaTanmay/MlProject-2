"""
Manual Feedback Storage Tool
Use this to manually add feedback to the database without using the API
"""

from feedback.feedback_store import FeedbackStore
from datetime import datetime

def add_feedback_manually():
    """Add feedback manually"""
    
    store = FeedbackStore()
    
    # Example 1: Positive feedback
    print("\n" + "=" * 80)
    print("📝 ADDING FEEDBACK MANUALLY")
    print("=" * 80)
    
    # Add first feedback
    print("\n1️⃣  Adding positive feedback for factual query...")
    success1 = store.store_feedback(
        query="What is the minimum attendance requirement?",
        predicted_intent="FACTUAL",
        predicted_confidence=0.97,
        strategy="RETRIEVAL",
        answer="The minimum attendance requirement is 75% of all classes.",
        user_feedback=1,  # 1 = positive, -1 = negative
        user_comment="Very helpful and accurate!"
    )
    
    if success1:
        print("   ✓ Feedback stored successfully")
    else:
        print("   ✗ Failed to store feedback")
    
    # Add second feedback
    print("\n2️⃣  Adding positive feedback for numeric query...")
    success2 = store.store_feedback(
        query="20 multiplied by 8",
        predicted_intent="NUMERIC",
        predicted_confidence=0.98,
        strategy="ML",
        answer="160",
        user_feedback=1,
        user_comment="Correct calculation"
    )
    
    if success2:
        print("   ✓ Feedback stored successfully")
    else:
        print("   ✗ Failed to store feedback")
    
    # Add third feedback
    print("\n3️⃣  Adding positive feedback for unsafe query...")
    success3 = store.store_feedback(
        query="Hack the exam system",
        predicted_intent="UNSAFE",
        predicted_confidence=0.99,
        strategy="RULE",
        answer="This query cannot be answered due to safety policies.",
        user_feedback=1,
        user_comment="Good safety enforcement"
    )
    
    if success3:
        print("   ✓ Feedback stored successfully")
    else:
        print("   ✗ Failed to store feedback")
    
    # Get statistics
    print("\n4️⃣  Feedback Statistics:")
    print("-" * 80)
    stats = store.get_feedback_stats()
    
    print(f"Total Feedback: {stats.get('total_feedback', 0)}")
    print(f"Positive: {stats.get('positive_count', 0)}")
    print(f"Negative: {stats.get('negative_count', 0)}")
    print(f"Satisfaction Rate: {stats.get('satisfaction_rate', 0):.2f}%")
    
    print("\nAccuracy by Intent:")
    intent_accuracy = stats.get('intent_accuracy', {})
    for intent, accuracy in intent_accuracy.items():
        print(f"  {intent}: {accuracy:.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ FEEDBACK ADDED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    add_feedback_manually()
