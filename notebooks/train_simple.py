"""
Simplified Training Script - Works with existing processed data
Trains ALL models and saves to backend/models/
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

BASE_DIR = Path("/home/ghost/fake-news-game-theory")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SIMPLIFIED TRAINING PIPELINE - Traditional ML Models")
print("=" * 80)
print(f"Saving to: {MODELS_DIR}\n")

# Load data
print("Loading data...")
X_train = pd.read_csv(PROCESSED_DIR / 'train/X_train.csv')
y_train = pd.read_csv(PROCESSED_DIR / 'train/y_train.csv').values.ravel()
X_val = pd.read_csv(PROCESSED_DIR / 'validation/X_val.csv')
y_val = pd.read_csv(PROCESSED_DIR / 'validation/y_val.csv').values.ravel()
X_test = pd.read_csv(PROCESSED_DIR / 'test/X_test.csv')
y_test = pd.read_csv(PROCESSED_DIR / 'test/y_test.csv').values.ravel()

print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

# Load preprocessing objects
with open(PROCESSED_DIR / 'features/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(PROCESSED_DIR / 'features/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open(PROCESSED_DIR / 'features/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Save preprocessing for backend
preprocessing = {
    'scaler': scaler,
    'vectorizer': vectorizer,
    'feature_names': feature_names
}
with open(MODELS_DIR / 'preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing, f)
print(f"✓ Saved preprocessing to: {MODELS_DIR}/preprocessing.pkl\n")

# Train models
models = {
    'logistic_regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42,
        class_weight='balanced', n_jobs=-1
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=150, max_depth=5, random_state=42
    ),
    'naive_bayes': GaussianNB()
}

print("Training models...")
print("=" * 80)

trained_models = {}
results = {}

for name, model in tqdm(list(models.items()), desc="Training"):
    print(f"\n→ {name}")
    start = time.time()
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    elapsed = time.time() - start
    
    print(f"  Val:  Acc={val_acc:.4f}, F1={val_f1:.4f}")
    print(f"  Test: Acc={test_acc:.4f}, F1={test_f1:.4f}")
    print(f"  Time: {elapsed:.2f}s")
    
    # Save
    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)
    print(f"  ✓ Saved: {model_path}")
    
    trained_models[name] = model
    results[name] = {'val_f1': val_f1, 'test_f1': test_f1}

# Create ensemble
print("\n" + "=" * 80)
print("Creating Ensemble")
print("=" * 80)

# Top 3 models by validation F1
sorted_models = sorted(results.items(), key=lambda x: x[1]['val_f1'], reverse=True)
top_3 = sorted_models[:3]

print("Top 3 models:")
for name, metrics in top_3:
    print(f"  {name}: Val F1={metrics['val_f1']:.4f}")

estimators = [(name, trained_models[name]) for name, _ in top_3]
ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

print("\nTraining ensemble...")
ensemble.fit(X_train, y_train)

val_pred = ensemble.predict(X_val)
test_pred = ensemble.predict(X_test)

val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred, average='weighted')
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average='weighted')

print(f"Ensemble Val:  Acc={val_acc:.4f}, F1={val_f1:.4f}")
print(f"Ensemble Test: Acc={test_acc:.4f}, F1={test_f1:.4f}")

# Save ensemble
ensemble_path = MODELS_DIR / "ensemble_config.joblib"
joblib.dump(ensemble, ensemble_path)
print(f"✓ Saved: {ensemble_path}")

# Summary
print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print("\nModels saved to:", MODELS_DIR)
print("\nFiles created:")
for f in sorted(MODELS_DIR.glob("*.joblib")) + sorted(MODELS_DIR.glob("*.pkl")):
    print(f"  ✓ {f.name}")

print("\n" + "=" * 80)
print("FINAL RESULTS ON TEST SET:")
print("=" * 80)
for name, metrics in sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True):
    print(f"{name:25} Test F1: {metrics['test_f1']:.4f}")
print(f"{'ensemble':25} Test F1: {test_f1:.4f}")

print("\n" + "=" * 80)
print("Next steps:")
print("1. Restart backend: cd backend && uvicorn app.main:app --reload")
print("2. Check logs for: '✓ Successfully loaded model'")
print("3. Test classifier at: http://localhost:3000/classifier")
print("=" * 80)
