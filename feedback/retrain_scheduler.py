"""
Retrain Scheduler - Automatic Intent Model Retraining
Triggers retraining of intent classifier based on accumulated feedback.
ONLY the intent classifier is retrained - NEVER the transformer.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from feedback.feedback_store import FeedbackStore
from training.train_intent_model import train_intent_classifier


class RetrainScheduler:
    """
    Manages automatic retraining of the intent classifier.
    Transformer is NEVER retrained.
    """
    
    def __init__(self, feedback_store: Optional[FeedbackStore] = None,
                 min_samples: int = 50, min_accuracy_drop: float = 0.05):
        """
        Initialize retrain scheduler.
        
        Args:
            feedback_store: FeedbackStore instance
            min_samples: Minimum feedback samples before retraining
            min_accuracy_drop: Minimum accuracy drop to trigger retraining
        """
        self.feedback_store = feedback_store or FeedbackStore()
        self.min_samples = min_samples
        self.min_accuracy_drop = min_accuracy_drop
    
    def should_retrain(self) -> Dict[str, Any]:
        """
        Check if retraining should be triggered.
        
        Returns:
            Dictionary with decision and reasons
        """
        stats = self.feedback_store.get_feedback_stats()
        
        total_feedback = stats.get("total_feedback", 0)
        satisfaction_rate = stats.get("satisfaction_rate", 1.0)
        intent_accuracy = stats.get("intent_accuracy", {})
        
        reasons = []
        should_retrain = False
        
        # Check 1: Enough feedback samples
        if total_feedback >= self.min_samples:
            reasons.append(f"Sufficient feedback samples: {total_feedback} >= {self.min_samples}")
        else:
            reasons.append(f"Insufficient feedback samples: {total_feedback} < {self.min_samples}")
            return {
                "should_retrain": False,
                "reasons": reasons,
                "total_feedback": total_feedback,
                "satisfaction_rate": satisfaction_rate
            }
        
        # Check 2: Low satisfaction rate
        if satisfaction_rate < (1.0 - self.min_accuracy_drop):
            reasons.append(f"Low satisfaction rate: {satisfaction_rate:.2%}")
            should_retrain = True
        
        # Check 3: Intent-specific accuracy issues
        for intent, metrics in intent_accuracy.items():
            if metrics["accuracy"] < 0.7 and metrics["total"] >= 5:
                reasons.append(f"Low accuracy for {intent}: {metrics['accuracy']:.2%}")
                should_retrain = True
        
        return {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "total_feedback": total_feedback,
            "satisfaction_rate": satisfaction_rate,
            "intent_accuracy": intent_accuracy
        }
    
    def prepare_training_data(self, base_dataset_path: str = None) -> Optional[str]:
        """
        Prepare combined training dataset from base + feedback.
        
        Args:
            base_dataset_path: Path to base dataset CSV
            
        Returns:
            Path to combined dataset or None
        """
        if base_dataset_path is None:
            base_dataset_path = Path(__file__).parent.parent / "training" / "intent_dataset.csv"
        
        try:
            # Load base dataset
            base_df = pd.read_csv(base_dataset_path)
            print(f"✓ Loaded base dataset: {len(base_df)} samples")
            
            # Get feedback data
            feedback_samples = self.feedback_store.get_training_data(
                min_confidence=0.8,
                only_correct=True
            )
            
            if not feedback_samples:
                print("⚠ No valid feedback samples for training")
                return base_dataset_path
            
            # Convert feedback to DataFrame
            feedback_df = pd.DataFrame(feedback_samples)
            print(f"✓ Retrieved {len(feedback_df)} feedback samples")
            
            # Combine datasets
            combined_df = pd.concat([base_df, feedback_df], ignore_index=True)
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['query'])
            
            print(f"✓ Combined dataset: {len(combined_df)} samples")
            
            # Save combined dataset
            output_path = Path(__file__).parent / "combined_dataset.csv"
            combined_df.to_csv(output_path, index=False)
            
            print(f"✓ Saved combined dataset to {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"✗ Failed to prepare training data: {e}")
            return None
    
    def execute_retraining(self) -> Dict[str, Any]:
        """
        Execute the retraining process.
        
        Returns:
            Dictionary with retraining results
        """
        print("\n" + "=" * 60)
        print("RETRAINING SCHEDULER - INTENT CLASSIFIER UPDATE")
        print("=" * 60)
        
        # Check if retraining should happen
        decision = self.should_retrain()
        print(f"\n📊 Retraining Decision:")
        print(f"  Should Retrain: {decision['should_retrain']}")
        print(f"  Reasons:")
        for reason in decision['reasons']:
            print(f"    - {reason}")
        
        if not decision['should_retrain']:
            print("\n⏸ Retraining not needed at this time")
            return {
                "retrained": False,
                "reason": "Retraining criteria not met",
                "decision": decision
            }
        
        # Prepare training data
        print("\n📦 Preparing training data...")
        combined_dataset = self.prepare_training_data()
        
        if not combined_dataset:
            return {
                "retrained": False,
                "reason": "Failed to prepare training data",
                "decision": decision
            }
        
        # Get current accuracy (for comparison)
        stats = self.feedback_store.get_feedback_stats()
        accuracy_before = stats.get("satisfaction_rate", 0.0)
        
        # Execute training
        print("\n🔧 Starting retraining...")
        success = train_intent_classifier(
            dataset_path=combined_dataset,
            output_dir=Path(__file__).parent.parent / "training" / "models"
        )
        
        if not success:
            return {
                "retrained": False,
                "reason": "Training failed",
                "decision": decision
            }
        
        # Log retraining
        samples_used = stats.get("total_feedback", 0)
        self.feedback_store.log_retraining(
            samples_used=samples_used,
            accuracy_before=accuracy_before,
            accuracy_after=0.0,  # Would need to evaluate on test set
            notes="Automatic retraining triggered by feedback"
        )
        
        print("\n✓ Retraining complete!")
        print("=" * 60)
        
        return {
            "retrained": True,
            "samples_used": samples_used,
            "accuracy_before": accuracy_before,
            "decision": decision
        }
    
    def get_retraining_schedule_info(self) -> Dict[str, Any]:
        """
        Get information about retraining schedule.
        
        Returns:
            Dictionary with schedule information
        """
        decision = self.should_retrain()
        stats = self.feedback_store.get_feedback_stats()
        history = self.feedback_store.get_retraining_history()
        
        samples_needed = max(0, self.min_samples - stats.get("total_feedback", 0))
        
        return {
            "current_feedback_count": stats.get("total_feedback", 0),
            "min_samples_required": self.min_samples,
            "samples_until_eligible": samples_needed,
            "satisfaction_rate": stats.get("satisfaction_rate", 0.0),
            "should_retrain_now": decision["should_retrain"],
            "last_retraining": history[0] if history else None,
            "total_retrainings": len(history)
        }


def main():
    """Main function for manual retraining trigger."""
    scheduler = RetrainScheduler(min_samples=20)  # Lower threshold for testing
    
    print("🔄 Meta-Learning AI - Retrain Scheduler")
    print("=" * 60)
    
    # Show schedule info
    info = scheduler.get_retraining_schedule_info()
    print("\n📅 Current Schedule Status:")
    print(f"  Feedback Count: {info['current_feedback_count']}")
    print(f"  Samples Needed: {info['samples_until_eligible']}")
    print(f"  Satisfaction Rate: {info['satisfaction_rate']:.2%}")
    print(f"  Ready to Retrain: {info['should_retrain_now']}")
    
    if info['last_retraining']:
        print(f"\n  Last Retraining: {info['last_retraining']['timestamp']}")
        print(f"  Improvement: {info['last_retraining']['improvement']:.2%}")
    
    # Ask user
    print("\n")
    response = input("Do you want to trigger retraining now? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        result = scheduler.execute_retraining()
        
        if result['retrained']:
            print("\n✅ Retraining completed successfully!")
        else:
            print(f"\n❌ Retraining not performed: {result['reason']}")
    else:
        print("\n⏸ Retraining cancelled")


if __name__ == "__main__":
    main()
