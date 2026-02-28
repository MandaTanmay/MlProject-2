"""
Train Domain Classifier Model
Trains TF-IDF + Logistic Regression model for STUDENT vs OUTSIDE domain classification.
Target Accuracy: > 95%
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load domain training dataset from CSV.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with queries and domains
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} training examples")
        print(f"  Domain distribution:\n{df['domain'].value_counts()}")
        return df
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None


def train_domain_classifier(dataset_path: str = None, output_dir: str = None):
    """
    Train the domain classification model.
    
    Args:
        dataset_path: Path to training CSV
        output_dir: Directory to save trained models
    """
    # Set default paths
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "domain_dataset.csv"
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "models"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("DOMAIN CLASSIFIER TRAINING")
    print("Target: > 95% Accuracy")
    print("=" * 60)
    
    # Load dataset
    df = load_dataset(dataset_path)
    if df is None:
        return False
    
    # Validate minimum samples
    if len(df) < 300:
        print(f"⚠ WARNING: Dataset has only {len(df)} samples.")
        print(f"   Recommended: At least 300 samples for production accuracy.")
    
    # Check class balance
    domain_counts = df['domain'].value_counts()
    balance_ratio = domain_counts.min() / domain_counts.max()
    if balance_ratio < 0.7:
        print(f"⚠ WARNING: Dataset imbalance detected (ratio: {balance_ratio:.2f})")
        print(f"   Recommended: Balance ratio > 0.7 for best performance.")
    
    # Prepare data
    X = df['query'].values
    y = df['domain'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Create TF-IDF vectorizer
    print("\n🔧 Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
        min_df=1,
        max_df=0.95,
        lowercase=True,
        token_pattern=r'\b\w+\b'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✓ Vectorizer trained with {X_train_vec.shape[1]} features")
    
    # Train classifier
    print("\n🧠 Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=2000,
        random_state=42,
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',  # Handle any class imbalance
        multi_class='multinomial'
    )
    
    classifier.fit(X_train_vec, y_train)
    print(f"✓ Classifier trained")
    
    # Evaluate on training set
    y_train_pred = classifier.predict(X_train_vec)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\n📈 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Evaluate on test set
    y_test_pred = classifier.predict(X_test_vec)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"📈 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Cross-validation
    print("\n🔍 Performing 5-fold cross-validation...")
    X_all_vec = vectorizer.transform(X)
    cv_scores = cross_val_score(classifier, X_all_vec, y, cv=5, scoring='accuracy')
    print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Detailed metrics
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['OUTSIDE', 'STUDENT']))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print("\n  Matrix interpretation:")
    print(f"  True OUTSIDE predictions: {cm[0][0]}")
    print(f"  False STUDENT (should be OUTSIDE): {cm[0][1]} ⚠")
    print(f"  False OUTSIDE (should be STUDENT): {cm[1][0]} ⚠")
    print(f"  True STUDENT predictions: {cm[1][1]}")
    
    # Feature importance
    print("\n🔍 Top features for each domain:")
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients for binary classification
    if len(classifier.classes_) == 2:
        coef = classifier.coef_[0]
        
        # Top features for OUTSIDE domain (negative coefficients)
        outside_indices = np.argsort(coef)[:15]
        print("\n  OUTSIDE domain indicators:")
        for idx in outside_indices:
            print(f"    - {feature_names[idx]}: {coef[idx]:.4f}")
        
        # Top features for STUDENT domain (positive coefficients)
        student_indices = np.argsort(coef)[-15:][::-1]
        print("\n  STUDENT domain indicators:")
        for idx in student_indices:
            print(f"    - {feature_names[idx]}: {coef[idx]:.4f}")
    
    # Check if target accuracy met
    print("\n" + "=" * 60)
    if test_accuracy >= 0.95:
        print("✅ TARGET ACCURACY ACHIEVED!")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}% (Target: > 95%)")
    elif test_accuracy >= 0.90:
        print("⚠ APPROACHING TARGET")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}% (Target: > 95%)")
        print("   Consider adding more training samples or tuning hyperparameters.")
    else:
        print("❌ BELOW TARGET ACCURACY")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}% (Target: > 95%)")
        print("   Action required: Add more diverse training samples.")
    print("=" * 60)
    
    # Save models
    vectorizer_path = output_dir / "domain_vectorizer.joblib"
    classifier_path = output_dir / "domain_classifier.joblib"
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)

    print(f"\n💾 Models saved:")
    print(f"  Vectorizer: {vectorizer_path}")
    print(f"  Classifier: {classifier_path}")

    # Versioned registry save
    try:
        import sys
        sys.path.insert(0, str(output_dir.parent.parent))
        from core.model_registry import save_model as _reg_save
        _reg_save(classifier, "domain_classifier", metadata={
            "model_type": "TF-IDF + Logistic Regression",
            "test_accuracy": float(test_accuracy),
            "training_samples": len(X_train),
        })
        _reg_save(vectorizer, "domain_vectorizer", metadata={"model_type": "TF-IDF"})
        print("  ✓ Versioned copies saved to model registry.")
    except Exception as _e:
        print(f"  ⚠ Model registry save skipped: {_e}")
    
    # Save metadata
    metadata = {
        "model_type": "TF-IDF + Logistic Regression",
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X_train_vec.shape[1],
        "classes": classifier.classes_.tolist()
    }
    
    metadata_path = output_dir / "domain_model_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata: {metadata_path}")
    
    print("\n✓ Domain classifier training complete!")
    print("\n🔄 Restart the application to load the new model.")
    
    return True


if __name__ == "__main__":
    train_domain_classifier()
