#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, TensorDataset
import torch.nn.functional as F

# sklearn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.calibration import CalibratedClassifierCV

# Additional imports
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
import json
import time
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

# Set up paths
BASE_PATH = "/home/ghost/fake-news-game-theory/data"
PROCESSED_PATH = os.path.join(BASE_PATH, "processed")
MODELS_PATH = os.path.join(BASE_PATH, "models")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

# Create directories
for path in [MODELS_PATH, RESULTS_PATH]:
    os.makedirs(path, exist_ok=True)

print("Model Training Pipeline Initialized")
print(f"Models will be saved to: {MODELS_PATH}")
print(f"Results will be saved to: {RESULTS_PATH}")

## 1. Data Loading and Preprocessing

class DataLoader:
    """Load and prepare data for model training"""

    def __init__(self, processed_path=PROCESSED_PATH):
        self.processed_path = processed_path
        self.scaler = None
        self.feature_names = None
        self.tfidf_vectorizer = None

    def load_training_data(self):
        """Load preprocessed training data"""
        print("Loading training data...")

        # Load features and labels
        X_train = pd.read_csv(os.path.join(self.processed_path, 'train/X_train.csv'))
        y_train = pd.read_csv(os.path.join(self.processed_path, 'train/y_train.csv'))

        X_val = pd.read_csv(os.path.join(self.processed_path, 'validation/X_val.csv'))
        y_val = pd.read_csv(os.path.join(self.processed_path, 'validation/y_val.csv'))

        X_test = pd.read_csv(os.path.join(self.processed_path, 'test/X_test.csv'))
        y_test = pd.read_csv(os.path.join(self.processed_path, 'test/y_test.csv'))

        # Load preprocessing objects
        with open(os.path.join(self.processed_path, 'features/scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(self.processed_path, 'features/feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)

        with open(os.path.join(self.processed_path, 'features/tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)

        # Convert to numpy arrays and flatten labels
        y_train = y_train.values.ravel()
        y_val = y_val.values.ravel()
        y_test = y_test.values.ravel()

        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features: {len(self.feature_names)}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

## 2. Traditional Machine Learning Models

class TraditionalMLTrainer:
    """Train and evaluate traditional ML models"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def initialize_models(self):
        """Initialize baseline models"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'svm': SVC(
                probability=True, random_state=42, class_weight='balanced'
            ),
            'naive_bayes': GaussianNB()
        }

        print(f"Initialized {len(self.models)} traditional ML models")

    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Train all baseline models"""
        print("Training baseline models...")

        trained_models = {}

        for name, model in tqdm(self.models.items(), desc="Training models"):
            start_time = time.time()

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
            val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, train_pred, train_proba)
            val_metrics = self._calculate_metrics(y_val, val_pred, val_proba)

            training_time = time.time() - start_time

            # Store results
            self.results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'training_time': training_time
            }

            trained_models[name] = model

            print(f"{name}: Val Accuracy = {val_metrics['accuracy']:.4f}, "
                  f"Val F1 = {val_metrics['f1']:.4f}, Time = {training_time:.2f}s")

        return trained_models

    def create_ensemble(self, models_dict, X_train, y_train, X_val, y_val):
        """Create ensemble of best models"""
        print("Creating ensemble model...")

        # Select top 3 models based on validation F1 score
        model_scores = []
        for name, result in self.results.items():
            model_scores.append((name, result['val_metrics']['f1'], result['model']))

        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:3]

        print("Top models for ensemble:")
        for name, score, _ in top_models:
            print(f"  {name}: F1 = {score:.4f}")

        # Create voting classifier
        estimators = [(name, model) for name, _, model in top_models]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')

        # Train ensemble
        ensemble.fit(X_train, y_train)

        # Evaluate ensemble
        val_pred = ensemble.predict(X_val)
        val_proba = ensemble.predict_proba(X_val)[:, 1]
        val_metrics = self._calculate_metrics(y_val, val_pred, val_proba)

        print(f"Ensemble - Val Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}")

        return ensemble, val_metrics

    def _calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)

        return metrics

## 3. Deep Learning Models

class FakeNewsDataset(Dataset):
    """Dataset class for PyTorch training"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features.values if hasattr(features, 'values') else features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DeepNeuralNetwork(nn.Module):
    """Deep neural network for fake news classification"""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(DeepNeuralNetwork, self).__init__()

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

        # Output layer
        layers.append(nn.Linear(prev_dim, 2))  # 2 classes: fake, real

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class DeepLearningTrainer:
    """Train deep learning models"""

    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def train_neural_network(self, X_train, y_train, X_val, y_val,
                           hidden_dims=[512, 256, 128], epochs=50, batch_size=64):
        """Train deep neural network"""
        print("Training deep neural network...")

        # Create datasets
        train_dataset = FakeNewsDataset(X_train, y_train)
        val_dataset = FakeNewsDataset(X_val, y_val)

        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_dim = X_train.shape[1]
        model = DeepNeuralNetwork(input_dim, hidden_dims).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        train_losses = []
        val_losses = []
        val_accuracies = []

        best_val_acc = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
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

            # Validation phase
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

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            scheduler.step(val_loss)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        # Load best model
        model.load_state_dict(best_model_state)

        # Final evaluation
        final_metrics = self._evaluate_deep_model(model, val_loader)

        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc
        }

        return model, final_metrics, training_history

    def _evaluate_deep_model(self, model, data_loader):
        """Evaluate deep learning model"""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_features, batch_labels in data_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)

                outputs = model(batch_features)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='weighted'),
            'recall': recall_score(all_labels, all_predictions, average='weighted'),
            'f1': f1_score(all_labels, all_predictions, average='weighted'),
            'auc_roc': roc_auc_score(all_labels, all_probabilities)
        }

        return metrics

## 4. Model Evaluation and Comparison

class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(self):
        self.results = {}

    def evaluate_all_models(self, models_dict, X_test, y_test):
        """Evaluate all trained models on test set"""
        print("Evaluating all models on test set...")

        evaluation_results = {}

        for model_name, model_info in models_dict.items():
            print(f"Evaluating {model_name}...")

            if model_name == 'deep_nn':
                # Special handling for deep neural network
                metrics = self._evaluate_deep_model(model_info, X_test, y_test)
            else:
                # Traditional ML models
                model = model_info['model'] if isinstance(model_info, dict) else model_info
                metrics = self._evaluate_traditional_model(model, X_test, y_test)

            evaluation_results[model_name] = metrics

        return evaluation_results

    def _evaluate_traditional_model(self, model, X_test, y_test):
        """Evaluate traditional ML model"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        if y_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_test, y_proba)

        return metrics

    def _evaluate_deep_model(self, model_info, X_test, y_test):
        """Evaluate deep learning model"""
        model = model_info['model']
        model.eval()

        # Get device from model
        device = next(model.parameters()).device

        # Convert to tensors and move to device
        X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test).to(device)

        with torch.no_grad():
            outputs = model(X_test_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

        y_pred = predictions.cpu().numpy()
        y_proba = probabilities[:, 1].cpu().numpy()

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        return metrics

    def create_comparison_report(self, results):
        """Create comprehensive comparison report"""
        print("\n" + "="*60)
        print("MODEL COMPARISON REPORT")
        print("="*60)

        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'AUC-ROC': metrics.get('auc_roc', 'N/A')
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1 Score', ascending=False)

        print(comparison_df.to_string(index=False, float_format='%.4f'))

        # Identify best model
        best_model = comparison_df.iloc[0]['Model']
        best_f1 = comparison_df.iloc[0]['F1 Score']

        print(f"\n🏆 Best Model: {best_model} (F1 Score: {best_f1:.4f})")

        return comparison_df

## 5. Model Persistence and Deployment

class ModelManager:
    """Manage model saving, loading, and versioning"""

    def __init__(self, models_path=MODELS_PATH):
        self.models_path = models_path

    def save_model(self, model, model_name, metrics=None, metadata=None):
        """Save trained model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.models_path, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_file = os.path.join(model_dir, 'model.pkl')

        if isinstance(model, nn.Module):
            # PyTorch model
            torch.save(model.state_dict(), model_file.replace('.pkl', '.pth'))
        else:
            # Scikit-learn or traditional model
            joblib.dump(model, model_file)

        # Save metrics
        if metrics:
            metrics_file = os.path.join(model_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

        # Save metadata
        full_metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_type': type(model).__name__,
            **(metadata or {})
        }

        metadata_file = os.path.join(model_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(full_metadata, f, indent=2)

        print(f"Model saved to: {model_dir}")
        return model_dir

    def save_best_model(self, models_dict, comparison_df, criteria='F1 Score'):
        """Save the best performing model"""
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = models_dict[best_model_name]

        # Extract model object if wrapped in dict
        if isinstance(best_model, dict):
            model = best_model.get('model', best_model)
            metrics = best_model.get('val_metrics', {})
        else:
            model = best_model
            metrics = {}

        metadata = {
            'selection_criteria': criteria,
            'comparison_rank': 1,
            'total_models_compared': len(models_dict)
        }

        save_path = self.save_model(model, f"best_{best_model_name}", metrics, metadata)

        print(f"\nBest model ({best_model_name}) saved successfully!")
        return save_path

## 6. Complete Training Pipeline

class CompletePipeline:
    """Complete end-to-end training pipeline"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.traditional_trainer = TraditionalMLTrainer()
        self.deep_trainer = DeepLearningTrainer()
        self.evaluator = ModelEvaluator()
        self.model_manager = ModelManager()

        self.all_models = {}
        self.results = {}

    def run_complete_pipeline(self, train_deep_nn=True):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print("STARTING COMPLETE MODEL TRAINING PIPELINE")
        print("="*70 + "\n")

        # Step 1: Load data
        print("STEP 1: Loading Data")
        print("-" * 70)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.data_loader.load_training_data()

        # Step 2: Train traditional ML models
        print("\nSTEP 2: Training Traditional ML Models")
        print("-" * 70)
        self.traditional_trainer.initialize_models()
        baseline_models = self.traditional_trainer.train_baseline_models(
            X_train, y_train, X_val, y_val
        )
        self.all_models.update(baseline_models)

        # Step 3: Create ensemble
        print("\nSTEP 3: Creating Ensemble Model")
        print("-" * 70)
        ensemble_model, ensemble_metrics = self.traditional_trainer.create_ensemble(
            self.all_models, X_train, y_train, X_val, y_val
        )
        self.all_models['ensemble'] = ensemble_model

        # Step 4: Train deep neural network
        if train_deep_nn:
            print("\nSTEP 4: Training Deep Neural Network")
            print("-" * 70)
            dnn_model, dnn_metrics, dnn_history = self.deep_trainer.train_neural_network(
                X_train, y_train, X_val, y_val
            )
            self.all_models['deep_nn'] = {'model': dnn_model, 'history': dnn_history}

        # Step 5: Evaluate all models
        print("\nSTEP 5: Evaluating All Models on Test Set")
        print("-" * 70)
        test_results = self.evaluator.evaluate_all_models(
            self.all_models, X_test, y_test
        )

        # Step 6: Create comparison report
        print("\nSTEP 6: Creating Comparison Report")
        print("-" * 70)
        comparison_df = self.evaluator.create_comparison_report(test_results)

        # Step 7: Save best model
        print("\nSTEP 7: Saving Best Model")
        print("-" * 70)
        best_model_path = self.model_manager.save_best_model(
            self.all_models, comparison_df
        )

        # Save comparison results
        comparison_file = os.path.join(RESULTS_PATH, 'model_comparison.csv')
        comparison_df.to_csv(comparison_file, index=False)
        print(f"Comparison results saved to: {comparison_file}")

        # Final summary
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print(f"Total models trained: {len(self.all_models)}")
        print(f"Best model: {comparison_df.iloc[0]['Model']}")
        print(f"Best F1 Score: {comparison_df.iloc[0]['F1 Score']:.4f}")
        print(f"Results saved to: {RESULTS_PATH}")
        print(f"Models saved to: {MODELS_PATH}")

        return {
            'models': self.all_models,
            'test_results': test_results,
            'comparison': comparison_df,
            'best_model_path': best_model_path
        }

def quick_train():
    """Quick training script for immediate use"""
    pipeline = CompletePipeline()

    # Run with default settings (no BERT for speed)
    results = pipeline.run_complete_pipeline(
        train_deep_nn=True
    )

    return results

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("="*70)
    print("Starting quick training...")
    print("="*70 + "\n")

    # Run quick training
    results = quick_train()

    print("\nTraining complete! Access results with:")
    print("  - results['models'] - All trained models")
    print("  - results['test_results'] - Test set evaluation")
    print("  - results['comparison'] - Model comparison dataframe")
    print("  - results['best_model_path'] - Path to best model")