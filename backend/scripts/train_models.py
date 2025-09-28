#!/usr/bin/env python3
"""
Model Training Pipeline Script

This script serves as a command-line interface to orchestrate the end-to-end
machine learning model training process. It automates data loading, preprocessing,
training, evaluation, and artifact storage.

Usage:
    python train_models.py --model-type bert --dataset-path data/training_data.csv
    python train_models.py --model-type ensemble --experiment-name exp_01 --epochs 5

Features:
    - Command-line interface with configurable parameters
    - Experiment tracking and model versioning
    - Modular training logic for different model types
    - Robust data handling with stratified splitting
    - Performance comparison and model registration
"""

import argparse
import logging
import sys
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class ModelTrainer:
    """Orchestrates the training process for different model types."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = config.get('experiment_name', f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir = Path(config['output_dir']) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize experiment tracking
        self.experiment_log = {
            'experiment_name': self.experiment_name,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'metrics': {},
            'artifacts': []
        }

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare training data with robust splitting."""
        logger.info(f"Loading data from {self.config['dataset_path']}")

        # Load data
        if self.config['dataset_path'].endswith('.csv'):
            data = pd.read_csv(self.config['dataset_path'])
        elif self.config['dataset_path'].endswith('.json'):
            data = pd.read_json(self.config['dataset_path'])
        else:
            raise ValueError("Unsupported data format. Use CSV or JSON.")

        # Validate required columns
        required_cols = ['text', 'label']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean and preprocess
        data = data.dropna(subset=['text', 'label'])
        data['text'] = data['text'].astype(str)

        # Handle label encoding if needed
        if data['label'].dtype == 'object':
            unique_labels = data['label'].unique()
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            data['label'] = data['label'].map(label_map)
            self.experiment_log['label_mapping'] = label_map
            logger.info(f"Label mapping: {label_map}")

        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'],
            data['label'],
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42),
            stratify=data['label']
        )

        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        logger.info(f"Label distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Label distribution - Test: {y_test.value_counts().to_dict()}")

        return pd.concat([X_train, X_test]), y_train, y_test

    def train_bert_model(self, X_train: pd.Series, y_train: pd.Series,
                        X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train a BERT-based model for text classification."""
        logger.info("Training BERT model")

        model_name = self.config.get('bert_model', 'bert-base-uncased')
        num_labels = len(y_train.unique())

        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Prepare datasets
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512
            )

        train_dataset = Dataset.from_dict({
            'text': X_train.tolist(),
            'labels': y_train.tolist()
        }).map(tokenize_function, batched=True)

        test_dataset = Dataset.from_dict({
            'text': X_test.tolist(),
            'labels': y_test.tolist()
        }).map(tokenize_function, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / 'bert_checkpoints'),
            learning_rate=self.config.get('learning_rate', 2e-5),
            per_device_train_batch_size=self.config.get('batch_size', 16),
            per_device_eval_batch_size=self.config.get('batch_size', 16),
            num_train_epochs=self.config.get('epochs', 3),
            weight_decay=0.01,
            logging_dir=str(self.output_dir / 'logs'),
            logging_steps=100,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            seed=self.config.get('random_state', 42)
        )

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Evaluate
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, predictions.predictions)

        # Save model artifacts
        model_path = self.output_dir / 'bert_model'
        trainer.save_model(str(model_path))
        tokenizer.save_pretrained(str(model_path))

        self.experiment_log['artifacts'].append(str(model_path))

        return {
            'model_type': 'bert',
            'model_path': str(model_path),
            'metrics': metrics,
            'predictions': y_pred.tolist()
        }

    def train_ensemble_model(self, X_train: pd.Series, y_train: pd.Series,
                           X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Train an ensemble model combining multiple algorithms."""
        logger.info("Training ensemble model")

        # Feature extraction
        vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_features', 10000),
            ngram_range=(1, 2),
            stop_words='english'
        )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Base models
        models = {
            'logistic': LogisticRegression(
                random_state=self.config.get('random_state', 42),
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 100),
                random_state=self.config.get('random_state', 42)
            )
        }

        # Train individual models
        individual_metrics = {}
        for name, model in models.items():
            logger.info(f"Training {name}")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            y_pred_proba = model.predict_proba(X_test_vec)
            individual_metrics[name] = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Ensemble model
        ensemble = VotingClassifier(
            estimators=list(models.items()),
            voting='soft'
        )
        ensemble.fit(X_train_vec, y_train)

        # Evaluate ensemble
        y_pred = ensemble.predict(X_test_vec)
        y_pred_proba = ensemble.predict_proba(X_test_vec)
        ensemble_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)

        # Save model artifacts
        model_path = self.output_dir / 'ensemble_model.pkl'
        vectorizer_path = self.output_dir / 'vectorizer.pkl'

        with open(model_path, 'wb') as f:
            pickle.dump(ensemble, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)

        self.experiment_log['artifacts'].extend([str(model_path), str(vectorizer_path)])

        return {
            'model_type': 'ensemble',
            'model_path': str(model_path),
            'vectorizer_path': str(vectorizer_path),
            'individual_metrics': individual_metrics,
            'ensemble_metrics': ensemble_metrics,
            'predictions': y_pred.tolist()
        }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            if y_pred_proba.ndim > 1:
                y_pred_proba = y_pred_proba[:, 1]  # Take positive class probability
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def compare_with_baseline(self, results: Dict[str, Any]) -> bool:
        """Compare model performance with baseline and determine if it should be promoted."""
        baseline_path = Path(self.config.get('baseline_path', 'models/baseline_metrics.json'))

        if not baseline_path.exists():
            logger.info("No baseline found. Current model will be set as baseline.")
            return True

        with open(baseline_path, 'r') as f:
            baseline_metrics = json.load(f)

        current_f1 = results['metrics'].get('f1_score', 0)
        baseline_f1 = baseline_metrics.get('f1_score', 0)

        improvement = current_f1 - baseline_f1
        logger.info(f"Current F1: {current_f1:.4f}, Baseline F1: {baseline_f1:.4f}")
        logger.info(f"Improvement: {improvement:.4f}")

        return improvement > self.config.get('min_improvement_threshold', 0.01)

    def save_experiment_log(self, results: Dict[str, Any]):
        """Save comprehensive experiment log."""
        self.experiment_log['end_time'] = datetime.now().isoformat()
        self.experiment_log['results'] = results

        log_path = self.output_dir / 'experiment_log.json'
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2)

        logger.info(f"Experiment log saved to {log_path}")

    def train(self) -> Dict[str, Any]:
        """Main training orchestration method."""
        try:
            # Load and prepare data
            _, y_train, y_test = self.load_and_prepare_data()

            # Get full dataset for splitting
            data = pd.read_csv(self.config['dataset_path'])
            X_train, X_test, y_train, y_test = train_test_split(
                data['text'], data['label'],
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                stratify=data['label']
            )

            # Train based on model type
            model_type = self.config['model_type'].lower()

            if model_type == 'bert':
                results = self.train_bert_model(X_train, y_train, X_test, y_test)
            elif model_type == 'ensemble':
                results = self.train_ensemble_model(X_train, y_train, X_test, y_test)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Compare with baseline and register if better
            if self.compare_with_baseline(results):
                logger.info("Model performance exceeds baseline. Registering as new production model.")
                self._register_model(results)
            else:
                logger.info("Model performance does not exceed baseline. Model saved for review.")

            # Save experiment log
            self.save_experiment_log(results)

            return results

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.experiment_log['error'] = str(e)
            self.experiment_log['end_time'] = datetime.now().isoformat()
            self.save_experiment_log({})
            raise

    def _register_model(self, results: Dict[str, Any]):
        """Register model as production candidate."""
        registry_dir = Path(self.config.get('model_registry', 'models/registry'))
        registry_dir.mkdir(parents=True, exist_ok=True)

        # Save model metadata
        metadata = {
            'model_type': results['model_type'],
            'experiment_name': self.experiment_name,
            'metrics': results.get('metrics', results.get('ensemble_metrics', {})),
            'registered_at': datetime.now().isoformat(),
            'model_path': results['model_path']
        }

        with open(registry_dir / 'latest_model.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model registered at {registry_dir / 'latest_model.json'}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train machine learning models for fake news detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['bert', 'ensemble'],
        help='Type of model to train'
    )

    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to training dataset (CSV or JSON)'
    )

    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/experiments',
        help='Output directory for model artifacts'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name for this experiment (auto-generated if not provided)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--bert-model',
        type=str,
        default='bert-base-uncased',
        help='BERT model to use for training'
    )

    parser.add_argument(
        '--max-features',
        type=int,
        default=10000,
        help='Maximum features for TF-IDF (ensemble model)'
    )

    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of estimators for Random Forest (ensemble model)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Convert arguments to config dictionary
    config = vars(args)
    config = {k.replace('_', '_'): v for k, v in config.items()}

    logger.info(f"Starting model training with config: {config}")

    # Initialize trainer and start training
    trainer = ModelTrainer(config)
    results = trainer.train()

    # Print summary
    logger.info("=== Training Complete ===")
    logger.info(f"Model type: {results['model_type']}")
    logger.info(f"Output directory: {trainer.output_dir}")

    if 'metrics' in results:
        metrics = results['metrics']
        logger.info(f"Final metrics:")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")

    if 'ensemble_metrics' in results:
        metrics = results['ensemble_metrics']
        logger.info(f"Ensemble metrics:")
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")


if __name__ == "__main__":
    main()