"""
Train All Production Models
Trains all ML models for the production system:
1. Domain Classifier (TF-IDF + Logistic Regression) - Target: > 95%
2. Intent Classifier (TF-IDF + Logistic Regression) - Target: > 90%  
3. Engine Selector (Random Forest) - Target: > 85%
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from train_domain_model import train_domain_classifier
from train_intent_model import train_intent_classifier
from train_engine_selector import train_engine_selector


def train_all_models():
    """Train all production ML models."""
    print("\n" + "=" * 70)
    print("🚀 TRAINING ALL PRODUCTION ML MODELS")
    print("=" * 70)
    
    models_trained = []
    models_failed = []
    
    # 1. Train Domain Classifier
    print("\n\n[1/3] DOMAIN CLASSIFIER")
    print("-" * 70)
    try:
        success = train_domain_classifier()
        if success:
            models_trained.append("Domain Classifier (STUDENT vs OUTSIDE)")
        else:
            models_failed.append("Domain Classifier")
    except Exception as e:
        print(f"\n❌ Domain Classifier training failed: {e}")
        models_failed.append("Domain Classifier")
    
    # 2. Train Intent Classifier
    print("\n\n[2/3] INTENT CLASSIFIER")
    print("-" * 70)
    try:
        success = train_intent_classifier()
        if success:
            models_trained.append("Intent Classifier (FACTUAL/NUMERIC/EXPLANATION/UNSAFE)")
        else:
            models_failed.append("Intent Classifier")
    except Exception as e:
        print(f"\n❌ Intent Classifier training failed: {e}")
        models_failed.append("Intent Classifier")
    
    # 3. Train Engine Selector
    print("\n\n[3/3] ENGINE SELECTOR (Meta-ML Model)")
    print("-" * 70)
    try:
        success = train_engine_selector()
        if success:
            models_trained.append("Engine Selector (RETRIEVAL/ML/TRANSFORMER/RULE)")
        else:
            models_failed.append("Engine Selector")
    except Exception as e:
        print(f"\n❌ Engine Selector training failed: {e}")
        models_failed.append("Engine Selector")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Successfully Trained ({len(models_trained)}/{3}):")
    for model in models_trained:
        print(f"   ✓ {model}")
    
    if models_failed:
        print(f"\n❌ Failed ({len(models_failed)}/{3}):")
        for model in models_failed:
            print(f"   ✗ {model}")
    
    print("\n" + "=" * 70)
    
    if len(models_trained) == 3:
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("\n📋 Next Steps:")
        print("   1. Restart the application to load trained models")
        print("   2. Run tests to verify model performance")
        print("   3. Monitor accuracy metrics via /model/metrics endpoint")
        print("\n🚀 System is ready for production deployment!")
    else:
        print("⚠️ PARTIAL SUCCESS - Some models failed to train")
        print("\n📋 Action Required:")
        print("   1. Review error messages above")
        print("   2. Check dataset files in training/ directory")
        print("   3. Ensure sufficient training samples")
        print("   4. Re-run training for failed models")
    
    print("=" * 70 + "\n")
    
    return len(models_failed) == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train production ML models")
    parser.add_argument(
        "--model",
        choices=["all", "domain", "intent", "engine"],
        default="all",
        help="Which model to train (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.model == "all":
        success = train_all_models()
        sys.exit(0 if success else 1)
    elif args.model == "domain":
        train_domain_classifier()
    elif args.model == "intent":
        train_intent_classifier()
    elif args.model == "engine":
        train_engine_selector()
