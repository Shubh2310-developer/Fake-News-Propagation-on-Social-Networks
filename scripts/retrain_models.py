#!/usr/bin/env python
"""
Retrain all ML models with proper feature alignment.
This script ensures the preprocessing pipeline matches the model training perfectly.
"""

import sys
import os
sys.path.insert(0, '/home/ghost/fake-news-game-theory/backend')

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_models.preprocessing import TextProcessor, FeatureExtractor

print("="*100)
print("RETRAINING MODELS WITH CORRECT FEATURE PIPELINE")
print("="*100)

# Load dataset
print("\n[1/6] Loading dataset...")
train_path = '/home/ghost/fake-news-game-theory/data/raw/liar_dataset/train.tsv'
test_path = '/home/ghost/fake-news-game-theory/data/raw/liar_dataset/test.tsv'

train_df = pd.read_csv(train_path, sep='\t', header=None)
test_df = pd.read_csv(test_path, sep='\t', header=None)

# Set column names
columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party'] + [f'col_{i}' for i in range(8, train_df.shape[1])]
train_df.columns = columns
test_df.columns = columns

# Map labels to binary
def map_label(label):
    if label in ['true', 'mostly-true', 'half-true']:
        return 0  # real
    else:
        return 1  # fake

train_df['binary_label'] = train_df['label'].apply(map_label)
test_df['binary_label'] = test_df['label'].apply(map_label)

print(f"  Train samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"  Train label distribution:\n{train_df['binary_label'].value_counts()}")

# Extract statements
X_train_text = train_df['statement'].values
y_train = train_df['binary_label'].values
X_test_text = test_df['statement'].values
y_test = test_df['binary_label'].values

# Clean text
print("\n[2/6] Cleaning text...")
text_processor = TextProcessor()
X_train_clean = [text_processor.clean(text) for text in X_train_text]
X_test_clean = [text_processor.clean(text) for text in X_test_text]

# Extract features
print("\n[3/6] Extracting features...")
feature_extractor = FeatureExtractor(
    max_tfidf_features=2000,
    tfidf_ngram_range=(1, 2),
    include_sentiment=True,
    include_readability=True
)

# Extract linguistic and stylistic features
print("  - Extracting linguistic features...")
train_ling = feature_extractor.extract_linguistic_features(X_train_clean)
test_ling = feature_extractor.extract_linguistic_features(X_test_clean)

print("  - Extracting stylistic features...")
train_style = feature_extractor.extract_stylistic_features(X_train_clean)
test_style = feature_extractor.extract_stylistic_features(X_test_clean)

# Combine engineered features
train_engineered = pd.concat([train_ling, train_style], axis=1)
test_engineered = pd.concat([test_ling, test_style], axis=1)

print(f"  - Engineered features: {train_engineered.shape[1]}")
print(f"  - Feature names: {list(train_engineered.columns[:10])}...")

# Extract TF-IDF features
print("  - Extracting TF-IDF features...")
train_tfidf = feature_extractor.fit_transform_tfidf(X_train_clean)
test_tfidf = feature_extractor.transform_tfidf(X_test_clean)

print(f"  - TF-IDF features: {train_tfidf.shape[1]}")

# Combine all features
X_train_features = np.hstack([train_engineered.values, train_tfidf])
X_test_features = np.hstack([test_engineered.values, test_tfidf])

print(f"\n  Total features: {X_train_features.shape[1]}")
print(f"  Train shape: {X_train_features.shape}")
print(f"  Test shape: {X_test_features.shape}")

# Save preprocessing pipeline
print("\n[4/6] Saving preprocessing pipeline...")
preprocessing_data = {
    'vectorizer': feature_extractor.tfidf_vectorizer,
    'scaler': None,  # No scaling needed for tree-based models
    'feature_names': list(train_engineered.columns) + [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
}

preprocessing_path = '/home/ghost/fake-news-game-theory/backend/models/preprocessing.pkl'
with open(preprocessing_path, 'wb') as f:
    pickle.dump(preprocessing_data, f)
print(f"  Saved to: {preprocessing_path}")
print(f"  Total feature names: {len(preprocessing_data['feature_names'])}")

# Train models
print("\n[5/6] Training models...")
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
}

trained_models = {}
results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_features, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"    Test Accuracy: {accuracy:.4f}")
    print(f"    Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['real', 'fake'])
    print(f"    {report}")

    # Save model
    model_path = f'/home/ghost/fake-news-game-theory/backend/models/{name}.joblib'
    joblib.dump(model, model_path)
    print(f"    Saved to: {model_path}")

    trained_models[name] = model
    results[name] = accuracy

# Create ensemble
print("\n[6/6] Creating ensemble model...")
ensemble = VotingClassifier(
    estimators=[
        ('lr', trained_models['logistic_regression']),
        ('nb', trained_models['naive_bayes']),
        ('rf', trained_models['random_forest']),
        ('gb', trained_models['gradient_boosting'])
    ],
    voting='soft',
    weights=[1, 1, 1.5, 1.5]  # Give more weight to tree-based models
)

# Fit ensemble (required even though individual models are already fitted)
ensemble.fit(X_train_features, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test_features)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

print(f"\n  Ensemble Test Accuracy: {ensemble_accuracy:.4f}")
print(f"  Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['real', 'fake']))

# Save ensemble
ensemble_path = '/home/ghost/fake-news-game-theory/backend/models/ensemble_config.joblib'
joblib.dump(ensemble, ensemble_path)
print(f"  Saved to: {ensemble_path}")

# Summary
print("\n" + "="*100)
print("TRAINING COMPLETE - SUMMARY")
print("="*100)
print(f"\nModel Accuracies:")
for name, acc in results.items():
    print(f"  {name:25s}: {acc:.2%}")
print(f"  {'ensemble':25s}: {ensemble_accuracy:.2%}")

print(f"\nFeature Configuration:")
print(f"  Engineered features: {train_engineered.shape[1]}")
print(f"  TF-IDF features: {train_tfidf.shape[1]}")
print(f"  Total features: {X_train_features.shape[1]}")

print(f"\nAll models saved to: /home/ghost/fake-news-game-theory/backend/models/")
print(f"\n✅ Models are now properly aligned with preprocessing pipeline!")
print("="*100)
