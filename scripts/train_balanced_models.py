#!/usr/bin/env python
"""
Train balanced models using LIAR dataset which matches the use case better.
Fixes the bias issue by using proper data and class balancing.
"""

import sys
sys.path.insert(0, '/home/ghost/fake-news-game-theory/backend')

import pandas as pd
import numpy as np
import pickle
import joblib
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from collections import Counter

print("="*80)
print("TRAINING BALANCED MODELS - LIAR DATASET")
print("="*80)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_basic_features(texts):
    features = []
    for text in texts:
        features.append([
            len(text),
            len(text.split()),
            text.count('!'),
            text.count('?'),
            sum(1 for c in text if c.isupper()) / (len(text) + 1),
        ])
    return np.array(features)

# Load LIAR dataset
print("\n[1/6] Loading LIAR dataset...")
train_path = '/home/ghost/fake-news-game-theory/data/raw/liar_dataset/train.tsv'
valid_path = '/home/ghost/fake-news-game-theory/data/raw/liar_dataset/valid.tsv'
test_path = '/home/ghost/fake-news-game-theory/data/raw/liar_dataset/test.tsv'

# Load all splits
dfs = []
for path, name in [(train_path, 'train'), (valid_path, 'valid'), (test_path, 'test')]:
    df = pd.read_csv(path, sep='\t', header=None)
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party'] + [f'col_{i}' for i in range(8, df.shape[1])]
    df.columns = columns
    dfs.append(df)
    print(f"  {name}: {len(df)} samples")

# Combine all data
combined = pd.concat(dfs, ignore_index=True)

# Map labels to binary
def map_label(label):
    return 0 if label in ['true', 'mostly-true', 'half-true'] else 1

combined['binary_label'] = combined['label'].apply(map_label)

print(f"\nTotal samples: {len(combined)}")
print(f"Label distribution:")
print(f"  Real (0): {(combined['binary_label'] == 0).sum()}")
print(f"  Fake (1): {(combined['binary_label'] == 1).sum()}")

# Balance the dataset by undersampling majority class
print("\n[2/6] Balancing dataset...")
real_df = combined[combined['binary_label'] == 0]
fake_df = combined[combined['binary_label'] == 1]

min_samples = min(len(real_df), len(fake_df))
print(f"  Undersampling to {min_samples} samples per class...")

real_balanced = real_df.sample(n=min_samples, random_state=42)
fake_balanced = fake_df.sample(n=min_samples, random_state=42)

balanced_df = pd.concat([real_balanced, fake_balanced], ignore_index=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Balanced dataset: {len(balanced_df)} samples")
print(f"    Real: {(balanced_df['binary_label'] == 0).sum()}")
print(f"    Fake: {(balanced_df['binary_label'] == 1).sum()}")

# Split data
X_text = balanced_df['statement'].values
y = balanced_df['binary_label'].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nSplit: Train={len(X_train_text)}, Test={len(X_test_text)}")

# Clean text
print("\n[3/6] Cleaning text...")
X_train_clean = [clean_text(t) for t in X_train_text]
X_test_clean = [clean_text(t) for t in X_test_text]

# Extract features
print("\n[4/6] Extracting features...")
vectorizer = TfidfVectorizer(
    max_features=3000,  # Reduced for shorter texts
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train_clean)
X_test_tfidf = vectorizer.transform(X_test_clean)

train_basic = extract_basic_features(X_train_clean)
test_basic = extract_basic_features(X_test_clean)

X_train = hstack([X_train_tfidf, csr_matrix(train_basic)])
X_test = hstack([X_test_tfidf, csr_matrix(test_basic)])

print(f"  Features: {X_train.shape[1]}")

# Save preprocessing
print("\n[5/6] Saving preprocessing...")
models_dir = Path('/home/ghost/fake-news-game-theory/backend/models')
preprocessing = {
    'vectorizer': vectorizer,
    'scaler': None,
    'feature_names': list(vectorizer.get_feature_names_out()) + ['length', 'word_count', 'exclaim', 'question', 'upper_ratio']
}

with open(models_dir / 'preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing, f)

print("  ✓ Preprocessing saved")

# Train models with class balancing
print("\n[6/6] Training balanced models...")

models = {
    'logistic_regression': LogisticRegression(
        max_iter=2000,
        C=1.0,
        random_state=42,
        class_weight='balanced',  # Important!
        solver='saga',
        n_jobs=-1
    ),
    'naive_bayes': MultinomialNB(alpha=0.5),  # Adjusted smoothing
    'random_forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,  # Reduced depth
        min_samples_split=10,  # Increased to prevent overfitting
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',  # Important!
        n_jobs=-1
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,  # Increased
        min_samples_leaf=5,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Check predictions distribution
    fake_preds = (y_pred == 1).sum()
    real_preds = (y_pred == 0).sum()
    fake_ratio = fake_preds / len(y_pred)

    print(f"    Accuracy: {acc:.2%}, F1: {f1:.4f}")
    print(f"    Predictions: {real_preds} Real, {fake_preds} Fake ({fake_ratio:.1%} fake)")

    # Classification report
    print(f"    Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'], digits=3)
    print(f"    {report}")

    joblib.dump(model, models_dir / f'{name}.joblib')

    results[name] = {
        'accuracy': acc,
        'f1': f1,
        'fake_ratio': fake_ratio
    }

# Create balanced ensemble
print("\n  Creating ensemble...")

# SimpleEnsemble class
class SimpleEnsemble:
    def __init__(self, lr, nb, rf, gb):
        self.lr = lr
        self.nb = nb
        self.rf = rf
        self.gb = gb
        # Equal weights for balanced ensemble
        self.weights = [1.0, 1.0, 1.0, 1.0]

    def predict(self, X):
        pred_lr = self.lr.predict(X)
        pred_nb = self.nb.predict(X)
        pred_rf = self.rf.predict(X)
        pred_gb = self.gb.predict(X)

        weighted_sum = (
            pred_lr * self.weights[0] +
            pred_nb * self.weights[1] +
            pred_rf * self.weights[2] +
            pred_gb * self.weights[3]
        )

        threshold = sum(self.weights) / 2
        return (weighted_sum >= threshold).astype(int)

    def predict_proba(self, X):
        proba_lr = self.lr.predict_proba(X)
        proba_nb = self.nb.predict_proba(X)
        proba_rf = self.rf.predict_proba(X)
        proba_gb = self.gb.predict_proba(X)

        total_weight = sum(self.weights)
        proba = (
            proba_lr * self.weights[0] +
            proba_nb * self.weights[1] +
            proba_rf * self.weights[2] +
            proba_gb * self.weights[3]
        ) / total_weight

        return proba

lr = joblib.load(models_dir / 'logistic_regression.joblib')
nb = joblib.load(models_dir / 'naive_bayes.joblib')
rf = joblib.load(models_dir / 'random_forest.joblib')
gb = joblib.load(models_dir / 'gradient_boosting.joblib')

ensemble = SimpleEnsemble(lr, nb, rf, gb)

# Test ensemble
y_pred_ens = ensemble.predict(X_test)
ens_acc = accuracy_score(y_test, y_pred_ens)
ens_f1 = f1_score(y_test, y_pred_ens, average='weighted')
fake_preds = (y_pred_ens == 1).sum()
real_preds = (y_pred_ens == 0).sum()
fake_ratio = fake_preds / len(y_pred_ens)

print(f"  Ensemble: Accuracy={ens_acc:.2%}, F1={ens_f1:.4f}")
print(f"  Predictions: {real_preds} Real, {fake_preds} Fake ({fake_ratio:.1%} fake)")

joblib.dump(ensemble, models_dir / 'ensemble_config.joblib')

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\n📊 Model Performance on Balanced LIAR Dataset:")
for name, metrics in results.items():
    print(f"  {name:25s}: Acc={metrics['accuracy']:.2%}, F1={metrics['f1']:.4f}, Fake%={metrics['fake_ratio']:.1%}")
print(f"  {'ensemble':25s}: Acc={ens_acc:.2%}, F1={ens_f1:.4f}, Fake%={fake_ratio:.1%}")

print(f"\n💾 Models saved to: {models_dir}/")
print("\n✅ Models retrained with balanced data!")
print("="*80)
