import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer


print("="*60)
print("TRAINING SEMANTIC DOMAIN CLASSIFIER")
print("="*60)

dataset_path = Path(__file__).parent / "domain_dataset.csv"
df = pd.read_csv(dataset_path)

print(f"Loaded {len(df)} samples")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["query"].tolist())

X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    df["domain"],
    test_size=0.2,
    random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, preds))

model_dir = Path(__file__).parent / "models"
model_dir.mkdir(exist_ok=True)

save_path = model_dir / "domain_classifier_semantic.joblib"
joblib.dump(clf, save_path)

print("\n✓ Model saved:", save_path)
print("DONE.")