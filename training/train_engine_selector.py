"""
Train Engine Selector Model
Trains Random Forest model for intelligent engine selection.
Learns from historical routing decisions and feedback.
Target Accuracy: > 85%
"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sqlite3


def extract_training_data_from_feedback(feedback_db_path: str) -> pd.DataFrame:
    """
    Extract training data from feedback database.
    
    Args:
        feedback_db_path: Path to feedback database
        
    Returns:
        DataFrame with features and engine labels
    """
    try:
        conn = sqlite3.connect(feedback_db_path)
        
        # Get feedback with positive ratings (correct routing)
        query = """
            SELECT 
                query,
                predicted_intent as intent,
                predicted_confidence as confidence,
                strategy_used as engine,
                user_feedback,
                was_correct
            FROM feedback
            WHERE was_correct = 1
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            print("⚠ No training data found in feedback database")
            return None
        
        print(f"✓ Extracted {len(df)} successful routing examples from feedback")
        
        # Extract features for each query
        from core.input_analyzer import InputAnalyzer
        analyzer = InputAnalyzer()
        
        features_list = []
        for _, row in df.iterrows():
            query = row['query']
            intent = row['intent']
            confidence = row['confidence']
            
            # Analyze query
            features = analyzer.analyze(query)
            
            # Build feature vector
            feature_dict = {
                'intent_FACTUAL': 1 if intent == 'FACTUAL' else 0,
                'intent_NUMERIC': 1 if intent == 'NUMERIC' else 0,
                'intent_EXPLANATION': 1 if intent == 'EXPLANATION' else 0,
                'intent_UNSAFE': 1 if intent == 'UNSAFE' else 0,
                'confidence': confidence,
                'query_length': features.get('length', 0),
                'word_count': features.get('word_count', 0),
                'has_digits': 1 if features.get('has_digits', False) else 0,
                'digit_count': features.get('digit_count', 0),
                'has_math_operators': 1 if features.get('has_math_operators', False) else 0,
                'has_question_words': 1 if features.get('has_question_words', False) else 0,
                'has_unsafe_keywords': 1 if features.get('has_unsafe_keywords', False) else 0,
                'avg_word_length': features.get('length', 0) / max(features.get('word_count', 1), 1),
                'engine': row['engine']
            }
            
            features_list.append(feature_dict)
        
        df_features = pd.DataFrame(features_list)
        
        print(f"\n📊 Engine distribution in training data:")
        print(df_features['engine'].value_counts())
        
        return df_features
        
    except Exception as e:
        print(f"✗ Failed to extract training data: {e}")
        return None


def create_synthetic_training_data() -> pd.DataFrame:
    """
    Create synthetic training data based on routing rules.
    Used when feedback data is insufficient.
    
    Returns:
        DataFrame with synthetic training examples
    """
    print("Creating synthetic training data based on routing rules...")
    
    # Synthetic examples for each intent->engine mapping
    examples = []
    
    # FACTUAL -> RETRIEVAL
    for i in range(100):
        examples.append({
            'intent_FACTUAL': 1,
            'intent_NUMERIC': 0,
            'intent_EXPLANATION': 0,
            'intent_UNSAFE': 0,
            'confidence': np.random.uniform(0.7, 1.0),
            'query_length': np.random.randint(20, 80),
            'word_count': np.random.randint(4, 15),
            'has_digits': 0,
            'digit_count': 0,
            'has_math_operators': 0,
            'has_question_words': 1,
            'has_unsafe_keywords': 0,
            'avg_word_length': np.random.uniform(4, 7),
            'engine': 'RETRIEVAL'
        })
    
    # NUMERIC -> ML
    for i in range(100):
        examples.append({
            'intent_FACTUAL': 0,
            'intent_NUMERIC': 1,
            'intent_EXPLANATION': 0,
            'intent_UNSAFE': 0,
            'confidence': np.random.uniform(0.8, 1.0),
            'query_length': np.random.randint(10, 50),
            'word_count': np.random.randint(3, 10),
            'has_digits': 1,
            'digit_count': np.random.randint(2, 5),
            'has_math_operators': 1,
            'has_question_words': 0,
            'has_unsafe_keywords': 0,
            'avg_word_length': np.random.uniform(3, 6),
            'engine': 'ML'
        })
    
    # EXPLANATION -> TRANSFORMER
    for i in range(100):
        examples.append({
            'intent_FACTUAL': 0,
            'intent_NUMERIC': 0,
            'intent_EXPLANATION': 1,
            'intent_UNSAFE': 0,
            'confidence': np.random.uniform(0.6, 0.95),
            'query_length': np.random.randint(30, 100),
            'word_count': np.random.randint(5, 20),
            'has_digits': 0,
            'digit_count': 0,
            'has_math_operators': 0,
            'has_question_words': 1,
            'has_unsafe_keywords': 0,
            'avg_word_length': np.random.uniform(5, 8),
            'engine': 'TRANSFORMER'
        })
    
    # UNSAFE -> RULE
    for i in range(50):
        examples.append({
            'intent_FACTUAL': 0,
            'intent_NUMERIC': 0,
            'intent_EXPLANATION': 0,
            'intent_UNSAFE': 1,
            'confidence': np.random.uniform(0.8, 1.0),
            'query_length': np.random.randint(15, 70),
            'word_count': np.random.randint(3, 15),
            'has_digits': 0,
            'digit_count': 0,
            'has_math_operators': 0,
            'has_question_words': 0,
            'has_unsafe_keywords': 1,
            'avg_word_length': np.random.uniform(4, 7),
            'engine': 'RULE'
        })
    
    df = pd.DataFrame(examples)
    print(f"✓ Created {len(df)} synthetic training examples")
    print(f"\n📊 Engine distribution:")
    print(df['engine'].value_counts())
    
    return df


def train_engine_selector(feedback_db_path: str = None, output_dir: str = None):
    """
    Train the engine selector model.
    
    Args:
        feedback_db_path: Path to feedback database
        output_dir: Directory to save trained model
    """
    # Set default paths
    if feedback_db_path is None:
        feedback_db_path = Path(__file__).parent.parent / "feedback" / "feedback.db"
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "models"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("ENGINE SELECTOR TRAINING (Meta-ML Model)")
    print("Target: > 85% Accuracy")
    print("=" * 60)
    
    # Try to load from feedback database
    df = None
    if Path(feedback_db_path).exists():
        df = extract_training_data_from_feedback(feedback_db_path)
    
    # If no feedback data or insufficient, create synthetic data
    if df is None or len(df) < 100:
        print("\n⚠ Insufficient feedback data. Creating synthetic training data...")
        df_synthetic = create_synthetic_training_data()
        
        if df is not None:
            # Combine real feedback with synthetic
            df = pd.concat([df, df_synthetic], ignore_index=True)
            print(f"✓ Combined real feedback ({len(df) - len(df_synthetic)}) with synthetic data")
        else:
            df = df_synthetic
    
    # Prepare training data
    feature_columns = [
        'intent_FACTUAL', 'intent_NUMERIC', 'intent_EXPLANATION', 'intent_UNSAFE',
        'confidence', 'query_length', 'word_count', 'has_digits', 'digit_count',
        'has_math_operators', 'has_question_words', 'has_unsafe_keywords',
        'avg_word_length'
    ]
    
    X = df[feature_columns].values
    y = df['engine'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Dataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Train Random Forest
    print("\n🌲 Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print(f"✓ Random Forest trained with {model.n_estimators} trees")
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"\n📈 Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"📈 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Cross-validation
    print("\n🔍 Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV Mean: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Detailed metrics
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred, labels=['RETRIEVAL', 'ML', 'TRANSFORMER', 'RULE'])
    print(cm)
    print("\n  Engines: [RETRIEVAL, ML, TRANSFORMER, RULE]")
    
    # Feature importance
    print("\n🔍 Feature Importance (Top 10):")
    feature_importance = sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Check if target accuracy met
    print("\n" + "=" * 60)
    if test_accuracy >= 0.85:
        print("✅ TARGET ACCURACY ACHIEVED!")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}% (Target: > 85%)")
    elif test_accuracy >= 0.80:
        print("⚠ APPROACHING TARGET")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}% (Target: > 85%)")
        print("   Model is usable but consider adding more training data.")
    else:
        print("❌ BELOW TARGET ACCURACY")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}% (Target: > 85%)")
        print("   Action required: Collect more feedback or adjust hyperparameters.")
    print("=" * 60)
    
    # Save model
    model_path = output_dir / "engine_selector.joblib"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Versioned registry save
    try:
        import sys
        sys.path.insert(0, str(output_dir.parent.parent))
        from core.model_registry import save_model as _reg_save
        _reg_save(model, "engine_selector", metadata={
            "model_type": "Random Forest",
            "test_accuracy": float(test_accuracy),
            "training_samples": len(X_train),
            "n_estimators": model.n_estimators,
        })
        print("  ✓ Versioned copy saved to model registry.")
    except Exception as _e:
        print(f"  ⚠ Model registry save skipped: {_e}")
    
    # Save metadata
    metadata = {
        "model_type": "Random Forest",
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_columns,
        "classes": model.classes_.tolist(),
        "n_estimators": model.n_estimators
    }
    
    metadata_path = output_dir / "engine_selector_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata: {metadata_path}")
    print("\n✓ Engine selector training complete!")
    print("\n🔄 Restart the application to load the new model.")
    
    return True


if __name__ == "__main__":
    train_engine_selector()
