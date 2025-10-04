"""
Complete Model Training Pipeline for RTX 4050 (6GB VRAM)
Trains ALL models properly with real data and saves for backend integration
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from tqdm import tqdm

# Transformers for DistilBERT (lightweight BERT for 6GB VRAM)
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification
)

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/home/ghost/fake-news-game-theory")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPLETE MODEL TRAINING PIPELINE - MAXIMUM DATA MODE")
print("=" * 80)
print(f"Hardware: RTX 4050 (6GB VRAM), 16GB RAM")
print(f"Dataset: ~44,000 samples (80% train, 10% val, 10% test)")
print(f"Models to train: ALL (Traditional ML + LSTM + DistilBERT + Ensemble)")
print(f"Output directory: {MODELS_DIR}")
print("=" * 80)

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✓ Using device: {device}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class DataLoader:
    """Load and prepare data for all models."""

    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.text_train = None
        self.text_val = None
        self.text_test = None
        self.scaler = None
        self.vectorizer = None
        self.feature_names = None

    def load_data(self):
        """Load preprocessed data."""
        print("\n📊 Loading training data...")

        # Load features
        self.X_train = pd.read_csv(PROCESSED_DIR / 'train/X_train.csv')
        self.y_train = pd.read_csv(PROCESSED_DIR / 'train/y_train.csv').values.ravel()
        self.X_val = pd.read_csv(PROCESSED_DIR / 'validation/X_val.csv')
        self.y_val = pd.read_csv(PROCESSED_DIR / 'validation/y_val.csv').values.ravel()
        self.X_test = pd.read_csv(PROCESSED_DIR / 'test/X_test.csv')
        self.y_test = pd.read_csv(PROCESSED_DIR / 'test/y_test.csv').values.ravel()

        # Load text data from RAW sources for LSTM and BERT
        print("📝 Loading raw text data...")

        # Load raw fake and true news
        fake_df = pd.read_csv(DATA_DIR / 'raw/kaggle_fake_news/Fake.csv')
        true_df = pd.read_csv(DATA_DIR / 'raw/kaggle_fake_news/True.csv')

        # Add labels
        fake_df['label'] = 0
        true_df['label'] = 1

        # Combine
        all_data = pd.concat([fake_df, true_df], ignore_index=True)

        # Shuffle
        all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Extract text and labels - USE ALL AVAILABLE DATA
        texts = all_data['text'].fillna('').astype(str).tolist()
        labels = all_data['label'].values

        # Use maximum data for RTX 4050 (6GB VRAM, 16GB RAM)
        # Split: 80% train, 10% val, 10% test
        total_samples = len(texts)
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)
        test_size = total_samples - train_size - val_size

        print(f"📈 Using MAXIMUM training data: {total_samples} total samples")
        print(f"   Train: {train_size}, Val: {val_size}, Test: {test_size}")

        self.text_train = texts[:train_size]
        self.text_val = texts[train_size:train_size + val_size]
        self.text_test = texts[train_size + val_size:]

        # Update labels accordingly
        self.y_train_text = labels[:train_size]
        self.y_val_text = labels[train_size:train_size + val_size]
        self.y_test_text = labels[train_size + val_size:]

        # Load preprocessing objects
        with open(PROCESSED_DIR / 'features/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(PROCESSED_DIR / 'features/tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(PROCESSED_DIR / 'features/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)

        print(f"✓ Train: {self.X_train.shape}, Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        print(f"✓ Features: {len(self.feature_names)}, Text samples: {len(self.text_train)}")

        return self


class TraditionalMLTrainer:
    """Train traditional ML models."""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.models = {}
        self.results = {}

    def train_all(self):
        """Train all traditional ML models."""
        print("\n" + "=" * 80)
        print("TRAINING TRADITIONAL ML MODELS")
        print("=" * 80)

        models_config = {
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

        for name, model in tqdm(models_config.items(), desc="Training models"):
            print(f"\n→ Training {name}...")
            start = time.time()

            model.fit(self.data_loader.X_train, self.data_loader.y_train)

            # Evaluate
            val_pred = model.predict(self.data_loader.X_val)
            val_proba = model.predict_proba(self.data_loader.X_val)[:, 1] if hasattr(model, 'predict_proba') else None

            metrics = self._calculate_metrics(self.data_loader.y_val, val_pred, val_proba)
            elapsed = time.time() - start

            self.models[name] = model
            self.results[name] = {'metrics': metrics, 'time': elapsed}

            print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, Time: {elapsed:.2f}s")

            # Save model
            model_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"  ✓ Saved: {model_path}")

        return self.models

    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        return metrics


class LSTMDataset(Dataset):
    """Dataset for LSTM."""

    def __init__(self, texts, labels, vocab, max_length=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower().split()[:self.max_length]
        # Convert to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text]
        # Pad
        if len(indices) < self.max_length:
            indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))

        return torch.LongTensor(indices), torch.LongTensor([self.labels[idx]])


class LSTMClassifier(nn.Module):
    """Optimized LSTM for 6GB VRAM."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden))
        return out


class LSTMTrainer:
    """Train LSTM model."""

    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device
        self.model = None
        self.vocab = None

    def build_vocab(self):
        """Build vocabulary from training texts."""
        print("\n→ Building vocabulary...")
        from collections import Counter

        word_counts = Counter()
        for text in self.data_loader.text_train:
            word_counts.update(str(text).lower().split())

        # Keep top 20000 words (fits in 6GB VRAM)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in word_counts.most_common(20000):
            vocab[word] = len(vocab)

        self.vocab = vocab
        print(f"  ✓ Vocabulary size: {len(vocab)}")
        return vocab

    def train(self, epochs=10, batch_size=64):
        """Train LSTM model."""
        print("\n" + "=" * 80)
        print("TRAINING LSTM MODEL")
        print("=" * 80)

        # Build vocab
        self.build_vocab()

        # Create datasets - use text-specific labels
        train_dataset = LSTMDataset(self.data_loader.text_train, self.data_loader.y_train_text, self.vocab)
        val_dataset = LSTMDataset(self.data_loader.text_val, self.data_loader.y_val_text, self.vocab)

        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Initialize model
        self.model = LSTMClassifier(len(self.vocab)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

        best_val_f1 = 0
        best_model_state = None

        print(f"\n→ Training for {epochs} epochs...")
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for texts, labels in train_loader:
                texts, labels = texts.to(self.device), labels.squeeze().to(self.device)

                optimizer.zero_grad()
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validate
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for texts, labels in val_loader:
                    texts, labels = texts.to(self.device), labels.squeeze().to(self.device)
                    outputs = self.model(texts)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Metrics
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            val_acc = accuracy_score(all_labels, all_preds)

            scheduler.step(val_loss)

            # Save best
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()

            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        # Load best model
        self.model.load_state_dict(best_model_state)
        print(f"\n✓ Best validation F1: {best_val_f1:.4f}")

        # Save model
        save_dict = {
            'model_state_dict': best_model_state,
            'vocab': self.vocab,
            'config': {
                'vocab_size': len(self.vocab),
                'embedding_dim': 128,
                'hidden_dim': 128,
                'num_layers': 2
            }
        }
        model_path = MODELS_DIR / "lstm_classifier.pt"
        torch.save(save_dict, model_path)
        print(f"✓ Saved: {model_path}")

        return self.model


class DistilBERTTrainer:
    """Train DistilBERT (lightweight BERT for 6GB VRAM)."""

    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device
        self.model = None
        self.tokenizer = None

    def train(self, epochs=3, batch_size=16):
        """Train DistilBERT model with custom training loop."""
        print("\n" + "=" * 80)
        print("TRAINING DISTILBERT MODEL (Lightweight BERT for 6GB VRAM)")
        print("=" * 80)

        # Initialize tokenizer and model
        print("\n→ Loading DistilBERT (66M parameters vs 110M for BERT-base)...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2
        ).to(self.device)

        # Prepare datasets
        print("→ Tokenizing data (batch processing for memory efficiency)...")

        # Tokenize in batches to save memory
        def tokenize_batch(texts, max_length=128):
            return self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

        train_encodings = tokenize_batch(self.data_loader.text_train)
        val_encodings = tokenize_batch(self.data_loader.text_val)

        # Create datasets
        class BERTDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

        train_dataset = BERTDataset(train_encodings, self.data_loader.y_train_text)
        val_dataset = BERTDataset(val_encodings, self.data_loader.y_val_text)

        # Create data loaders
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

        # Training loop
        print(f"\n→ Training for {epochs} epochs with batch_size={batch_size}...")
        best_val_f1 = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    val_loss += loss.item()

                    preds = torch.argmax(outputs.logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='weighted')

            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self.model.state_dict().copy()

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print(f"\n✓ Best validation F1: {best_val_f1:.4f}")

        # Save model
        model_path = MODELS_DIR / "bert_classifier"
        model_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        print(f"✓ Saved: {model_path}")

        return self.model


class EnsembleTrainer:
    """Create ensemble from all trained models."""

    def __init__(self, traditional_models, data_loader):
        self.traditional_models = traditional_models
        self.data_loader = data_loader

    def create_ensemble(self):
        """Create voting ensemble."""
        print("\n" + "=" * 80)
        print("CREATING ENSEMBLE MODEL")
        print("=" * 80)

        from sklearn.ensemble import VotingClassifier

        # Use top 3 traditional models
        models_with_scores = []
        for name, model in self.traditional_models.items():
            val_pred = model.predict(self.data_loader.X_val)
            f1 = f1_score(self.data_loader.y_val, val_pred, average='weighted')
            models_with_scores.append((name, model, f1))

        models_with_scores.sort(key=lambda x: x[2], reverse=True)
        top_models = models_with_scores[:3]

        print("\n→ Top 3 models for ensemble:")
        for name, _, f1 in top_models:
            print(f"  • {name}: F1={f1:.4f}")

        # Create ensemble
        estimators = [(name, model) for name, model, _ in top_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        ensemble.fit(self.data_loader.X_train, self.data_loader.y_train)

        # Evaluate
        val_pred = ensemble.predict(self.data_loader.X_val)
        val_f1 = f1_score(self.data_loader.y_val, val_pred, average='weighted')
        val_acc = accuracy_score(self.data_loader.y_val, val_pred)

        print(f"\n✓ Ensemble F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")

        # Save
        model_path = MODELS_DIR / "ensemble_config.joblib"
        joblib.dump(ensemble, model_path)
        print(f"✓ Saved: {model_path}")

        return ensemble


def save_preprocessing_objects(data_loader):
    """Save preprocessing objects for backend."""
    print("\n→ Saving preprocessing objects...")

    preprocessing = {
        'scaler': data_loader.scaler,
        'vectorizer': data_loader.vectorizer,
        'feature_names': data_loader.feature_names
    }

    save_path = MODELS_DIR / 'preprocessing.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessing, f)

    print(f"✓ Saved: {save_path}")


def evaluate_all_models(data_loader, traditional_models, lstm_model, distilbert_model, ensemble_model):
    """Final evaluation on test set."""
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    results = {}

    # Traditional ML
    for name, model in traditional_models.items():
        test_pred = model.predict(data_loader.X_test)
        test_proba = model.predict_proba(data_loader.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        acc = accuracy_score(data_loader.y_test, test_pred)
        f1 = f1_score(data_loader.y_test, test_pred, average='weighted')

        results[name] = {'accuracy': acc, 'f1': f1}
        print(f"{name:25} Acc: {acc:.4f}, F1: {f1:.4f}")

    # Ensemble
    test_pred = ensemble_model.predict(data_loader.X_test)
    acc = accuracy_score(data_loader.y_test, test_pred)
    f1 = f1_score(data_loader.y_test, test_pred, average='weighted')
    results['ensemble'] = {'accuracy': acc, 'f1': f1}
    print(f"{'ensemble':25} Acc: {acc:.4f}, F1: {f1:.4f}")

    print("\n" + "=" * 80)

    # Save results
    results_dir = BASE_DIR / 'data/results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df.to_csv(results_dir / 'final_evaluation.csv')
    print(f"✓ Results saved to: data/results/final_evaluation.csv")

    return results


def main():
    """Main training pipeline."""
    start_time = time.time()

    # Load data
    data_loader = DataLoader().load_data()

    # Train traditional ML
    ml_trainer = TraditionalMLTrainer(data_loader)
    traditional_models = ml_trainer.train_all()

    # Train LSTM - optimized for larger dataset
    lstm_trainer = LSTMTrainer(data_loader, device)
    lstm_model = lstm_trainer.train(epochs=5, batch_size=128)

    # Train DistilBERT
    distilbert_trainer = DistilBERTTrainer(data_loader, device)
    distilbert_model = distilbert_trainer.train(epochs=3, batch_size=16)

    # Create ensemble
    ensemble_trainer = EnsembleTrainer(traditional_models, data_loader)
    ensemble_model = ensemble_trainer.create_ensemble()

    # Save preprocessing
    save_preprocessing_objects(data_loader)

    # Final evaluation
    results = evaluate_all_models(
        data_loader, traditional_models, lstm_model, distilbert_model, ensemble_model
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"\nAll models saved to: {MODELS_DIR}")
    print("\nModels trained:")
    print("  ✓ Logistic Regression")
    print("  ✓ Random Forest")
    print("  ✓ Gradient Boosting")
    print("  ✓ Naive Bayes")
    print("  ✓ LSTM (PyTorch)")
    print("  ✓ DistilBERT (Transformers)")
    print("  ✓ Ensemble (Voting)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
