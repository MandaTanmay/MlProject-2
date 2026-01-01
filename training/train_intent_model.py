"""
Train Intent Classifier Model
Trains TF-IDF + Logistic Regression model for intent classification.
This is the ONLY ML training component in the system.
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
    Load training dataset from CSV.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with queries and intents
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} training examples")
        print(f"  Intent distribution:\n{df['intent'].value_counts()}")
        return df
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return None


def train_intent_classifier(dataset_path: str = None, output_dir: str = None):
    """
    Train the intent classification model.
    
    Args:
        dataset_path: Path to training CSV
        output_dir: Directory to save trained models
    """
    # Set default paths
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "intent_dataset.csv"
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "models"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("INTENT CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Load dataset
    df = load_dataset(dataset_path)
    if df is None:
        return False
    
    # Prepare data
    X = df['query'].values
    y = df['intent'].values
    
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
        max_features=1000,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.9,
        lowercase=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"✓ Vectorizer trained with {X_train_vec.shape[1]} features")
    
    # Train classifier
    print("\n🧠 Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial'
    )
    
    classifier.fit(X_train_vec, y_train)
    print("✓ Classifier trained")
    
    # Evaluate on test set
    print("\n📈 Evaluation on test set:")
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("🔀 Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    # Cross-validation
    print("\n🔄 Cross-validation (5-fold):")
    X_all_vec = vectorizer.transform(X)
    cv_scores = cross_val_score(classifier, X_all_vec, y, cv=5)
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean CV Score: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
    
    # Save models
    print("\n💾 Saving models...")
    vectorizer_path = output_dir / "vectorizer.joblib"
    classifier_path = output_dir / "classifier.joblib"
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)
    
    print(f"✓ Vectorizer saved to: {vectorizer_path}")
    print(f"✓ Classifier saved to: {classifier_path}")
    
    # Test with sample queries
    print("\n🧪 Testing with sample queries:")
    test_queries = [
        "What is the attendance policy?",
        "Calculate 25 times 4",
        "Explain artificial intelligence",
        "How to hack the system?"
    ]
    
    for query in test_queries:
        query_vec = vectorizer.transform([query])
        prediction = classifier.predict(query_vec)[0]
        probabilities = classifier.predict_proba(query_vec)[0]
        confidence = np.max(probabilities)
        print(f"  '{query}'")
        print(f"    → {prediction} (confidence: {confidence:.2%})")
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    # Train the model
    success = train_intent_classifier()
    
    if success:
        print("\n✓ Intent classifier is ready to use!")
    else:
        print("\n✗ Training failed. Please check the error messages above.")
