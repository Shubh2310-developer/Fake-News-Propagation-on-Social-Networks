#!/usr/bin/env python
"""
Fast and efficient model training for all models with ~80% accuracy target.
Optimized for speed and memory efficiency.
"""

import sys
sys.path.insert(0, '/home/ghost/fake-news-game-theory/backend')

import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from ml_models.preprocessing import TextProcessor, FeatureExtractor

print("="*100)
print("FAST MODEL TRAINING - ALL DATASETS - TARGET: 80% ACCURACY")
print("="*100)

# Step 1: Load datasets efficiently
print("\n[1/7] Loading datasets...")

# Load Kaggle dataset (largest, most reliable)
print("  Loading Kaggle Fake News...")
fake_df = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/kaggle_fake_news/Fake.csv')
true_df = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/kaggle_fake_news/True.csv')

fake_df['label'] = 1
true_df['label'] = 0

# Combine and sample
kaggle_df = pd.concat([fake_df, true_df], ignore_index=True)

# Load LIAR dataset
print("  Loading LIAR dataset...")
liar_train = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/liar_dataset/train.tsv', sep='\t', header=None)
liar_valid = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/liar_dataset/valid.tsv', sep='\t', header=None)

columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party'] + [f'col_{i}' for i in range(8, liar_train.shape[1])]
liar_train.columns = columns
liar_valid.columns = columns

liar_all = pd.concat([liar_train, liar_valid], ignore_index=True)
liar_all['text'] = liar_all['statement']
liar_all['label'] = liar_all['label'].apply(lambda x: 0 if x in ['true', 'mostly-true', 'half-true'] else 1)
liar_df = liar_all[['text', 'label']]

# Combine datasets
print("\n  Combining datasets...")
combined = pd.concat([kaggle_df[['text', 'label']], liar_df], ignore_index=True)

# Clean and filter
combined = combined.dropna(subset=['text'])
combined = combined[combined['text'].str.strip() != '']
combined = combined[combined['text'].str.len() > 20]

# Sample for efficiency - 40K samples should be enough
MAX_SAMPLES = 40000
if len(combined) > MAX_SAMPLES:
    combined = combined.sample(n=MAX_SAMPLES, random_state=42)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  Total samples: {len(combined)}")
print(f"  Real: {(combined['label'] == 0).sum()}, Fake: {(combined['label'] == 1).sum()}")

# Step 2: Split data
X_text = combined['text'].values
y = combined['label'].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train_text)}, Test: {len(X_test_text)}")

# Step 3: Preprocessing
print("\n[2/7] Text preprocessing...")
text_processor = TextProcessor()
X_train_clean = [text_processor.clean(str(text)) for text in X_train_text]
X_test_clean = [text_processor.clean(str(text)) for text in X_test_text]

# Step 4: Feature extraction - optimized settings
print("\n[3/7] Feature extraction...")
feature_extractor = FeatureExtractor(
    max_tfidf_features=5000,
    tfidf_ngram_range=(1, 2),  # Reduced to bigrams for speed
    include_sentiment=True,
    include_readability=True
)

# Linguistic and stylistic features
print("  Extracting linguistic features...")
train_ling = feature_extractor.extract_linguistic_features(X_train_clean)
test_ling = feature_extractor.extract_linguistic_features(X_test_clean)

print("  Extracting stylistic features...")
train_style = feature_extractor.extract_stylistic_features(X_train_clean)
test_style = feature_extractor.extract_stylistic_features(X_test_clean)

train_eng = pd.concat([train_ling, train_style], axis=1)
test_eng = pd.concat([test_ling, test_style], axis=1)

print(f"  Engineered features: {train_eng.shape[1]}")

# TF-IDF features
print("  Extracting TF-IDF features...")
train_tfidf = feature_extractor.fit_transform_tfidf(X_train_clean)
test_tfidf = feature_extractor.transform_tfidf(X_test_clean)

print(f"  TF-IDF features: {train_tfidf.shape[1]}")

# Combine features
X_train = np.hstack([train_eng.values, train_tfidf])
X_test = np.hstack([test_eng.values, test_tfidf])

print(f"  Total features: {X_train.shape[1]}")

# Step 5: Scale features
print("\n[4/7] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Save preprocessing
print("\n[5/7] Saving preprocessing...")
preprocessing = {
    'vectorizer': feature_extractor.tfidf_vectorizer,
    'scaler': scaler,
    'feature_names': list(train_eng.columns) + [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
}

models_dir = Path('/home/ghost/fake-news-game-theory/backend/models')
models_dir.mkdir(parents=True, exist_ok=True)

with open(models_dir / 'preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing, f)
print("  ✓ Preprocessing saved")

# Step 7: Train models
print("\n[6/7] Training models...")

models = {
    'logistic_regression': LogisticRegression(
        max_iter=2000,
        C=1.5,
        random_state=42,
        class_weight='balanced',
        solver='saga',
        n_jobs=-1
    ),
    'naive_bayes': MultinomialNB(alpha=0.1),
    'random_forest': RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
}

results = {}
trained_models = {}

# For MultinomialNB, we need non-negative features
from sklearn.preprocessing import MinMaxScaler
nb_scaler = MinMaxScaler()
X_train_nb = nb_scaler.fit_transform(X_train)
X_test_nb = nb_scaler.transform(X_test)

for name, model in models.items():
    print(f"\n  Training {name}...")

    if name == 'naive_bayes':
        # Use scaled non-negative features for NB
        model.fit(X_train_nb, y_train)
        y_pred = model.predict(X_test_nb)
    else:
        # Use standard scaled features for others
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"    Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # Save model
    joblib.dump(model, models_dir / f'{name}.joblib')

    trained_models[name] = model
    results[name] = {'accuracy': acc, 'f1': f1}

# Save NB scaler
with open(models_dir / 'nb_scaler.pkl', 'wb') as f:
    pickle.dump(nb_scaler, f)

# Step 8: Create ensemble
print("\n[7/7] Creating ensemble...")

# Create ensemble with proper handling for different feature requirements
class CustomEnsemble:
    """Custom ensemble that handles different preprocessing for NB."""

    def __init__(self, models, nb_scaler, std_scaler):
        self.lr = models['logistic_regression']
        self.nb = models['naive_bayes']
        self.rf = models['random_forest']
        self.gb = models['gradient_boosting']
        self.nb_scaler = nb_scaler
        self.std_scaler = std_scaler

    def predict(self, X):
        # Get predictions from each model
        X_scaled = self.std_scaler.transform(X)
        X_nb = self.nb_scaler.transform(X)

        pred_lr = self.lr.predict(X_scaled)
        pred_nb = self.nb.predict(X_nb)
        pred_rf = self.rf.predict(X_scaled)
        pred_gb = self.gb.predict(X_scaled)

        # Weighted voting
        predictions = np.vstack([pred_lr, pred_nb * 0.5, pred_rf * 1.5, pred_gb * 2])
        return np.round(predictions.mean(axis=0)).astype(int)

    def predict_proba(self, X):
        X_scaled = self.std_scaler.transform(X)
        X_nb = self.nb_scaler.transform(X)

        proba_lr = self.lr.predict_proba(X_scaled)
        proba_nb = self.nb.predict_proba(X_nb)
        proba_rf = self.rf.predict_proba(X_scaled)
        proba_gb = self.gb.predict_proba(X_scaled)

        # Weighted average
        weights = np.array([1.0, 0.5, 1.5, 2.0])
        weights = weights / weights.sum()

        proba = (proba_lr * weights[0] + proba_nb * weights[1] +
                 proba_rf * weights[2] + proba_gb * weights[3])

        return proba

ensemble = CustomEnsemble(trained_models, nb_scaler, scaler)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')

print(f"\n  Ensemble Accuracy: {ensemble_acc:.4f}, F1: {ensemble_f1:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['Real', 'Fake']))

# Save ensemble
joblib.dump(ensemble, models_dir / 'ensemble_config.joblib')
print("  ✓ Ensemble saved")

# Summary
print("\n" + "="*100)
print("TRAINING COMPLETE")
print("="*100)
print(f"\n📊 Model Performance:")
for name, metrics in results.items():
    print(f"  {name:25s}: Acc={metrics['accuracy']:.2%}, F1={metrics['f1']:.4f}")
print(f"  {'ensemble':25s}: Acc={ensemble_acc:.2%}, F1={ensemble_f1:.4f}")

print(f"\n📈 Dataset:")
print(f"  Total samples: {len(combined)}")
print(f"  Training: {len(X_train_text)}, Test: {len(X_test_text)}")
print(f"  Features: {X_train.shape[1]}")

print(f"\n💾 Models saved:")
print(f"  Location: {models_dir}/")
print(f"    ✓ logistic_regression.joblib")
print(f"    ✓ naive_bayes.joblib")
print(f"    ✓ random_forest.joblib")
print(f"    ✓ gradient_boosting.joblib")
print(f"    ✓ ensemble_config.joblib")
print(f"    ✓ preprocessing.pkl")
print(f"    ✓ nb_scaler.pkl")

print("\n✅ All models trained successfully!")
print("="*100)
