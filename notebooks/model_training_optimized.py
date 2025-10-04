"""
Optimized Model Training Script for Fake News Detection
This version is optimized to prevent freezing and crashes
"""

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import time
from datetime import datetime

# sklearn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = "/home/ghost/fake-news-game-theory/data"
PROCESSED_PATH = os.path.join(BASE_PATH, "processed")
MODELS_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

for path in [MODELS_PATH, RESULTS_PATH]:
    os.makedirs(path, exist_ok=True)

print("✓ Optimized Model Training Pipeline Initialized")


class DataLoaderOptimized:
    """Load and prepare data efficiently"""

    def __init__(self, processed_path=PROCESSED_PATH):
        self.processed_path = processed_path

    def load_training_data(self):
        """Load preprocessed training data"""
        print("Loading training data...")

        X_train = pd.read_csv(os.path.join(self.processed_path, 'train/X_train.csv'))
        y_train = pd.read_csv(os.path.join(self.processed_path, 'train/y_train.csv'))
        X_val = pd.read_csv(os.path.join(self.processed_path, 'validation/X_val.csv'))
        y_val = pd.read_csv(os.path.join(self.processed_path, 'validation/y_val.csv'))
        X_test = pd.read_csv(os.path.join(self.processed_path, 'test/X_test.csv'))
        y_test = pd.read_csv(os.path.join(self.processed_path, 'test/y_test.csv'))

        y_train = y_train.values.ravel()
        y_val = y_val.values.ravel()
        y_test = y_test.values.ravel()

        print(f"✓ Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class FastMLTrainer:
    """Optimized traditional ML trainer"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def initialize_models(self):
        """Initialize baseline models (SVM removed for speed)"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=500, class_weight='balanced', solver='lbfgs', n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced',
                n_jobs=-1, max_depth=20
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42, max_depth=5
            ),
            'naive_bayes': GaussianNB()
        }
        print(f"✓ Initialized {len(self.models)} models (SVM excluded for performance)")

    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Train all baseline models quickly"""
        print("\nTraining baseline models...")
        trained_models = {}

        for name, model in tqdm(self.models.items(), desc="Training"):
            start_time = time.time()
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

            val_metrics = self._calculate_metrics(y_val, val_pred, val_proba)
            training_time = time.time() - start_time

            self.results[name] = {
                'model': model,
                'val_metrics': val_metrics,
                'training_time': training_time
            }
            trained_models[name] = model

            print(f"  {name}: Val F1={val_metrics['f1']:.4f}, Time={training_time:.2f}s")

        return trained_models

    def optimize_top_model(self, X_train, y_train, X_val, y_val):
        """Optimize only the best performing model"""
        print("\nOptimizing best model...")

        # Find best model
        best_model_name = max(self.results.items(),
                            key=lambda x: x[1]['val_metrics']['f1'])[0]
        print(f"Optimizing: {best_model_name}")

        if best_model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5]
            }
            base_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

        elif best_model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 150],
                'learning_rate': [0.1, 0.15],
                'max_depth': [3, 5]
            }
            base_model = GradientBoostingClassifier(random_state=42)

        else:
            print(f"  Skipping optimization for {best_model_name}")
            return {}

        search = RandomizedSearchCV(
            base_model, param_grid,
            n_iter=5, cv=2, scoring='f1',
            random_state=42, n_jobs=-1, verbose=0
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        val_pred = best_model.predict(X_val)
        val_proba = best_model.predict_proba(X_val)[:, 1]
        val_metrics = self._calculate_metrics(y_val, val_pred, val_proba)

        print(f"  Optimized {best_model_name}: Val F1={val_metrics['f1']:.4f}")

        return {
            f"{best_model_name}_optimized": {
                'model': best_model,
                'best_params': search.best_params_,
                'val_metrics': val_metrics
            }
        }

    def create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create voting ensemble from top 3 models"""
        print("\nCreating ensemble...")

        # Get top 3 models
        model_scores = [(name, res['val_metrics']['f1'], res['model'])
                       for name, res in self.results.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:min(3, len(model_scores))]

        print("  Top models:")
        for name, score, _ in top_models:
            print(f"    {name}: F1={score:.4f}")

        estimators = [(name, model) for name, _, model in top_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        ensemble.fit(X_train, y_train)

        val_pred = ensemble.predict(X_val)
        val_proba = ensemble.predict_proba(X_val)[:, 1]
        val_metrics = self._calculate_metrics(y_val, val_pred, val_proba)

        print(f"  Ensemble: Val F1={val_metrics['f1']:.4f}")
        return ensemble, val_metrics

    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        return metrics


class FakeNewsDataset(Dataset):
    """PyTorch Dataset"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features.values if hasattr(features, 'values') else features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CompactNeuralNetwork(nn.Module):
    """Smaller, faster neural network"""

    def __init__(self, input_dim, hidden_dims=[256, 128], dropout_rate=0.3):
        super(CompactNeuralNetwork, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FastDeepLearning:
    """Optimized deep learning trainer"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {self.device}")

    def train_neural_network(self, X_train, y_train, X_val, y_val, epochs=15, batch_size=128):
        """Train compact neural network"""
        print(f"\nTraining neural network ({epochs} epochs)...")

        train_dataset = FakeNewsDataset(X_train, y_train)
        val_dataset = FakeNewsDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        model = CompactNeuralNetwork(X_train.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_val_acc = 0
        best_model_state = None

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate
            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = correct / total

            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        model.load_state_dict(best_model_state)
        final_metrics = self._evaluate(model, val_loader)
        print(f"  Final Val F1={final_metrics['f1']:.4f}")

        return model, final_metrics

    def _evaluate(self, model, data_loader):
        """Evaluate model"""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device)
                outputs = model(batch_features)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())

        return {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'auc_roc': roc_auc_score(all_labels, all_probabilities)
        }


class ModelEvaluator:
    """Evaluate and compare models"""

    def evaluate_all_models(self, models_dict, X_test, y_test):
        """Evaluate all models on test set"""
        print("\nEvaluating models on test set...")
        results = {}

        for name, model_info in models_dict.items():
            if name == 'deep_nn':
                metrics = self._eval_deep(model_info, X_test, y_test)
            else:
                model = model_info['model'] if isinstance(model_info, dict) else model_info
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                }
                if y_proba is not None:
                    metrics['auc_roc'] = roc_auc_score(y_test, y_proba)

            results[name] = metrics
            print(f"  {name}: F1={metrics['f1']:.4f}")

        return results

    def _eval_deep(self, model_info, X_test, y_test):
        """Evaluate deep learning model"""
        model = model_info['model']
        device = next(model.parameters()).device
        model.eval()

        X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test).to(device)

        with torch.no_grad():
            outputs = model(X_test_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

        y_pred = predictions.cpu().numpy()
        y_proba = probabilities[:, 1].cpu().numpy()

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba)
        }

    def create_report(self, results):
        """Create comparison report"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON REPORT")
        print("=" * 70)

        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'AUC-ROC': metrics.get('auc_roc', 0)
            })

        df = pd.DataFrame(comparison_data).sort_values('F1 Score', ascending=False)
        print(df.to_string(index=False, float_format='%.4f'))

        best_model = df.iloc[0]['Model']
        best_f1 = df.iloc[0]['F1 Score']
        print(f"\n🏆 Best Model: {best_model} (F1={best_f1:.4f})")

        return df


def quick_train(optimize=False, deep_nn=False):
    """
    Quick optimized training

    Args:
        optimize: Enable hyperparameter optimization (slower)
        deep_nn: Train deep neural network (slower)
    """
    print("\n" + "=" * 70)
    print("OPTIMIZED TRAINING PIPELINE")
    print("=" * 70)
    print(f"Hyperparameter optimization: {optimize}")
    print(f"Deep neural network: {deep_nn}\n")

    # Load data
    loader = DataLoaderOptimized()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_training_data()

    # Train traditional ML
    ml_trainer = FastMLTrainer()
    ml_trainer.initialize_models()
    models = ml_trainer.train_baseline_models(X_train, y_train, X_val, y_val)

    # Optional optimization
    if optimize:
        optimized = ml_trainer.optimize_top_model(X_train, y_train, X_val, y_val)
        for name, info in optimized.items():
            models[name] = info['model']

    # Ensemble
    ensemble, _ = ml_trainer.create_ensemble(X_train, y_train, X_val, y_val)
    models['ensemble'] = ensemble

    # Optional deep learning
    if deep_nn:
        dl_trainer = FastDeepLearning()
        dnn_model, dnn_metrics = dl_trainer.train_neural_network(
            X_train, y_train, X_val, y_val, epochs=15
        )
        models['deep_nn'] = {'model': dnn_model}

    # Evaluate
    evaluator = ModelEvaluator()
    test_results = evaluator.evaluate_all_models(models, X_test, y_test)
    comparison_df = evaluator.create_report(test_results)

    # Save best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models[best_model_name]
    if isinstance(best_model, dict):
        best_model = best_model['model']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_PATH, f"best_{best_model_name}_{timestamp}.pkl")

    if isinstance(best_model, nn.Module):
        torch.save(best_model.state_dict(), model_path.replace('.pkl', '.pth'))
    else:
        joblib.dump(best_model, model_path)

    print(f"\n✓ Best model saved: {model_path}")

    # Save results
    results_file = os.path.join(RESULTS_PATH, f'comparison_{timestamp}.csv')
    comparison_df.to_csv(results_file, index=False)
    print(f"✓ Results saved: {results_file}")

    return {
        'models': models,
        'test_results': test_results,
        'comparison': comparison_df,
        'best_model_path': model_path
    }


def train_single_model(model_type='random_forest'):
    """Train single model (fastest option)"""
    print(f"\nTraining single {model_type} model...")

    loader = DataLoaderOptimized()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.load_training_data()

    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=500, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    metrics = {
        'accuracy': accuracy_score(y_val, val_pred),
        'precision': precision_score(y_val, val_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_val, val_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_val, val_pred, average='weighted', zero_division=0)
    }

    print(f"\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_PATH, f"{model_type}_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved: {model_path}")

    return model, metrics, model_path


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    mode = 'fast'
    model_type = 'random_forest'

    for i, arg in enumerate(sys.argv):
        if arg == '--mode' and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
        elif arg == '--model-type' and i + 1 < len(sys.argv):
            model_type = sys.argv[i + 1]

    print("\n" + "=" * 70)
    print("OPTIMIZED FAKE NEWS DETECTION TRAINING")
    print("=" * 70)

    if mode == 'single':
        print(f"\nMode: Single Model ({model_type})")
        print("=" * 70 + "\n")
        model, metrics, path = train_single_model(model_type)
    else:
        if mode == 'fast':
            print("\nMode: Fast (baseline models only)")
            optimize, deep_nn = False, False
        elif mode == 'deep':
            print("\nMode: With Deep Learning")
            optimize, deep_nn = False, True
        elif mode == 'optimize':
            print("\nMode: With Hyperparameter Optimization")
            optimize, deep_nn = True, False
        elif mode == 'full':
            print("\nMode: Full (all features)")
            optimize, deep_nn = True, True
        else:
            print(f"\nUnknown mode: {mode}, using fast mode")
            optimize, deep_nn = False, False

        print("=" * 70 + "\n")
        results = quick_train(optimize=optimize, deep_nn=deep_nn)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
