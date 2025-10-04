#!/usr/bin/env python
"""
Enhanced model training with ALL datasets, advanced feature engineering, and optimized hyperparameters.
Target: ~80% accuracy across all models.
"""

import sys
sys.path.insert(0, '/home/ghost/fake-news-game-theory/backend')

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import warnings
warnings.filterwarnings('ignore')

from ml_models.preprocessing import TextProcessor, FeatureExtractor

print("="*100)
print("ENHANCED MODEL TRAINING - MULTI-DATASET APPROACH")
print("="*100)

# Load ALL available datasets
print("\n[1/8] Loading ALL datasets...")

# Dataset 1: LIAR (Political statements)
print("  Loading LIAR dataset...")
liar_train = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/liar_dataset/train.tsv', sep='\t', header=None)
liar_valid = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/liar_dataset/valid.tsv', sep='\t', header=None)
liar_test = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/liar_dataset/test.tsv', sep='\t', header=None)

columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party'] + [f'col_{i}' for i in range(8, liar_train.shape[1])]
liar_train.columns = columns
liar_valid.columns = columns
liar_test.columns = columns

liar_all = pd.concat([liar_train, liar_valid, liar_test], ignore_index=True)
liar_all['text'] = liar_all['statement']
liar_all['binary_label'] = liar_all['label'].apply(lambda x: 0 if x in ['true', 'mostly-true', 'half-true'] else 1)

# Dataset 2: Kaggle Fake News (Full articles)
print("  Loading Kaggle Fake News dataset...")
kaggle_fake = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/kaggle_fake_news/Fake.csv')
kaggle_true = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/kaggle_fake_news/True.csv')

kaggle_fake['binary_label'] = 1
kaggle_true['binary_label'] = 0
kaggle_all = pd.concat([kaggle_fake, kaggle_true], ignore_index=True)

# Dataset 3: FakeNewsNet (Social media news)
print("  Loading FakeNewsNet dataset...")
try:
    politifact_fake = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/fakenewsnet/politifact_fake.csv', encoding='utf-8', on_bad_lines='skip')
    politifact_real = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/fakenewsnet/politifact_real.csv', encoding='utf-8', on_bad_lines='skip')
    gossipcop_fake = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/fakenewsnet/gossipcop_fake.csv', encoding='utf-8', on_bad_lines='skip')
    gossipcop_real = pd.read_csv('/home/ghost/fake-news-game-theory/data/raw/fakenewsnet/gossipcop_real.csv', encoding='utf-8', on_bad_lines='skip')

    politifact_fake['binary_label'] = 1
    politifact_real['binary_label'] = 0
    gossipcop_fake['binary_label'] = 1
    gossipcop_real['binary_label'] = 0

    fakenewsnet_all = pd.concat([politifact_fake, politifact_real, gossipcop_fake, gossipcop_real], ignore_index=True)

    # Extract text column
    if 'title' in fakenewsnet_all.columns and 'text' not in fakenewsnet_all.columns:
        fakenewsnet_all['text'] = fakenewsnet_all['title']
    elif 'content' in fakenewsnet_all.columns:
        fakenewsnet_all['text'] = fakenewsnet_all['content']
except Exception as e:
    print(f"  Warning: Error loading FakeNewsNet: {e}")
    fakenewsnet_all = pd.DataFrame()

# Combine all datasets
print("\n  Combining datasets...")
all_data = []

# LIAR dataset
liar_subset = liar_all[['text', 'binary_label']].copy()
all_data.append(liar_subset)

# Kaggle dataset
kaggle_subset = kaggle_all[['text', 'binary_label']].copy()
all_data.append(kaggle_subset)

# FakeNewsNet dataset
if not fakenewsnet_all.empty and 'text' in fakenewsnet_all.columns:
    fakenewsnet_subset = fakenewsnet_all[['text', 'binary_label']].copy()
    all_data.append(fakenewsnet_subset)

# Merge all
combined_df = pd.concat(all_data, ignore_index=True)

# Remove NaN and empty texts
combined_df = combined_df.dropna(subset=['text'])
combined_df = combined_df[combined_df['text'].str.strip() != '']
combined_df = combined_df[combined_df['text'].str.len() > 10]  # Minimum length filter

# Shuffle
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Limit dataset size for memory efficiency (adjust based on your RAM)
MAX_SAMPLES = 50000  # Optimized for 16GB RAM
if len(combined_df) > MAX_SAMPLES:
    combined_df = combined_df.sample(n=MAX_SAMPLES, random_state=42)

print(f"\n  Total samples: {len(combined_df)}")
print(f"  Label distribution:")
print(f"    Real (0): {(combined_df['binary_label'] == 0).sum()}")
print(f"    Fake (1): {(combined_df['binary_label'] == 1).sum()}")

# Split into train and test
X_text = combined_df['text'].values
y = combined_df['binary_label'].values

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train: {len(X_train_text)}, Test: {len(X_test_text)}")

# Clean text
print("\n[2/8] Preprocessing text...")
text_processor = TextProcessor()
print("  Cleaning training texts...")
X_train_clean = [text_processor.clean(str(text)) for text in X_train_text]
print("  Cleaning test texts...")
X_test_clean = [text_processor.clean(str(text)) for text in X_test_text]

# Extract features with enhanced settings
print("\n[3/8] Extracting advanced features...")
feature_extractor = FeatureExtractor(
    max_tfidf_features=6000,  # Optimized for memory
    tfidf_ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    include_sentiment=True,
    include_readability=True
)

print("  - Linguistic features...")
train_ling = feature_extractor.extract_linguistic_features(X_train_clean)
test_ling = feature_extractor.extract_linguistic_features(X_test_clean)

print("  - Stylistic features...")
train_style = feature_extractor.extract_stylistic_features(X_train_clean)
test_style = feature_extractor.extract_stylistic_features(X_test_clean)

# Combine engineered features
train_engineered = pd.concat([train_ling, train_style], axis=1)
test_engineered = pd.concat([test_ling, test_style], axis=1)
print(f"    Engineered features: {train_engineered.shape[1]}")

print("  - TF-IDF features...")
train_tfidf = feature_extractor.fit_transform_tfidf(X_train_clean)
test_tfidf = feature_extractor.transform_tfidf(X_test_clean)
print(f"    TF-IDF features: {train_tfidf.shape[1]}")

# Combine all features
X_train_features = np.hstack([train_engineered.values, train_tfidf])
X_test_features = np.hstack([test_engineered.values, test_tfidf])
print(f"\n  Total features: {X_train_features.shape[1]}")

# Feature selection for top K features
print("\n[4/8] Feature selection...")
print(f"  Selecting top 4000 features using chi2...")
selector = SelectKBest(chi2, k=min(4000, X_train_features.shape[1]))

# Make features non-negative for chi2
min_val = X_train_features.min()
if min_val < 0:
    X_train_features = X_train_features - min_val
    X_test_features = X_test_features - min_val

X_train_selected = selector.fit_transform(X_train_features, y_train)
X_test_selected = selector.transform(X_test_features)
print(f"  Selected features: {X_train_selected.shape[1]}")

# Scale features
print("\n[5/8] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Save preprocessing
print("\n[6/8] Saving preprocessing pipeline...")
preprocessing_data = {
    'vectorizer': feature_extractor.tfidf_vectorizer,
    'scaler': scaler,
    'selector': selector,
    'feature_names': list(train_engineered.columns) + [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
}
with open('/home/ghost/fake-news-game-theory/backend/models/preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing_data, f)
print("  ✓ Preprocessing pipeline saved")

# Train models with highly optimized hyperparameters
print("\n[7/8] Training models with optimized hyperparameters...")

models = {
    'logistic_regression': LogisticRegression(
        max_iter=3000,
        C=2.0,  # Increased regularization
        random_state=42,
        class_weight='balanced',
        solver='saga',
        penalty='l2',
        n_jobs=-1
    ),
    'naive_bayes': GaussianNB(
        var_smoothing=1e-9  # Better fit for larger datasets
    ),
    'random_forest': RandomForestClassifier(
        n_estimators=400,  # More trees
        max_depth=40,  # Deeper trees
        min_samples_split=4,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        bootstrap=True
    ),
    'gradient_boosting': GradientBoostingClassifier(
        n_estimators=400,  # More estimators
        learning_rate=0.08,  # Adjusted learning rate
        max_depth=8,  # Deeper trees
        min_samples_split=4,
        min_samples_leaf=1,
        subsample=0.85,
        max_features='sqrt',
        random_state=42
    ),
    'adaboost': AdaBoostClassifier(
        n_estimators=200,
        learning_rate=1.0,
        random_state=42
    )
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')

    print(f"    Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

    # Save model
    joblib.dump(model, f'/home/ghost/fake-news-game-theory/backend/models/{name}.joblib')

    trained_models[name] = model
    results[name] = {'accuracy': test_acc, 'f1': test_f1}

# Create optimized ensemble with best models
print("\n[8/8] Creating optimized ensemble...")

# Sort models by F1 score
sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
print("\n  Model ranking by F1 score:")
for name, metrics in sorted_models:
    print(f"    {name:25s}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

# Create weighted ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', trained_models['logistic_regression']),
        ('nb', trained_models['naive_bayes']),
        ('rf', trained_models['random_forest']),
        ('gb', trained_models['gradient_boosting']),
        ('ab', trained_models['adaboost'])
    ],
    voting='soft',
    weights=[1.3, 0.7, 1.8, 2.2, 1.5],  # Optimized weights
    n_jobs=-1
)

print("\n  Fitting ensemble...")
ensemble.fit(X_train_scaled, y_train)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')

print(f"\n  Ensemble Performance:")
print(f"    Accuracy: {ensemble_accuracy:.4f}")
print(f"    F1 Score: {ensemble_f1:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_ensemble, target_names=['Real', 'Fake']))

# Save ensemble
joblib.dump(ensemble, '/home/ghost/fake-news-game-theory/backend/models/ensemble_config.joblib')
print("\n  ✓ Ensemble saved")

# Summary
print("\n" + "="*100)
print("ENHANCED TRAINING COMPLETE")
print("="*100)
print(f"\n📊 Final Model Accuracies:")
for name, metrics in results.items():
    print(f"  {name:25s}: Acc={metrics['accuracy']:.2%}, F1={metrics['f1']:.4f}")
print(f"  {'ensemble':25s}: Acc={ensemble_accuracy:.2%}, F1={ensemble_f1:.4f}")

print(f"\n📈 Dataset Statistics:")
print(f"  Total samples: {len(combined_df)}")
print(f"  Training samples: {len(X_train_text)}")
print(f"  Test samples: {len(X_test_text)}")

print(f"\n🔧 Feature Engineering:")
print(f"  Raw features: {X_train_features.shape[1]}")
print(f"  Selected features: {X_train_selected.shape[1]}")
print(f"  TF-IDF: {train_tfidf.shape[1]}, Engineered: {train_engineered.shape[1]}")

print(f"\n💾 Saved Models:")
print(f"  Location: /home/ghost/fake-news-game-theory/backend/models/")
for name in models.keys():
    print(f"    ✓ {name}.joblib")
print(f"    ✓ ensemble_config.joblib")
print(f"    ✓ preprocessing.pkl")

print("\n✅ All models trained successfully with enhanced multi-dataset approach!")
print("="*100)
