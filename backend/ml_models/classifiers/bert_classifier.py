# backend/ml_models/classifiers/bert_classifier.py

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .base_classifier import BaseClassifier
import logging
import os

logger = logging.getLogger(__name__)


class BERTClassifier(BaseClassifier):
    """
    BERT-based fake news classifier using the Hugging Face ecosystem.

    This implementation leverages pre-trained BERT models and fine-tunes them
    for fake news classification using the Transformers library.
    """

    def __init__(self,
                 model_name: str = "bert_classifier",
                 pretrained_model: str = 'bert-base-uncased',
                 num_labels: int = 2,
                 max_length: int = 512,
                 **kwargs):
        """
        Initialize the BERT classifier.

        Args:
            model_name: Name identifier for the model
            pretrained_model: Hugging Face model name (e.g., 'bert-base-uncased')
            num_labels: Number of classification labels (2 for binary)
            max_length: Maximum sequence length for tokenization
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        self.pretrained_model = pretrained_model
        self.num_labels = num_labels
        self.max_length = max_length

        # Initialize tokenizer and model
        logger.info(f"Loading tokenizer and model: {pretrained_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels
        )

        # Training components
        self.trainer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(self.device)

        logger.info(f"BERT classifier initialized on device: {self.device}")

    def _prepare_dataset(self, texts: List[str], labels: Optional[List[int]] = None) -> Dataset:
        """
        Prepare dataset by tokenizing texts and creating Dataset object.

        Args:
            texts: List of text samples
            labels: Optional list of labels for training

        Returns:
            Hugging Face Dataset object
        """
        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Create dataset dictionary
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }

        if labels is not None:
            dataset_dict['labels'] = torch.tensor(labels, dtype=torch.long)

        return Dataset.from_dict(dataset_dict)

    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for evaluation during training.

        Args:
            eval_pred: Evaluation prediction object

        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the BERT classifier using the Trainer API.

        Args:
            X_train: Training texts (DataFrame with text column or Series)
            y_train: Training labels
            X_val: Optional validation texts
            y_val: Optional validation labels

        Returns:
            Dictionary containing training metrics and history
        """
        logger.info(f"Training BERT classifier on {len(X_train)} samples")

        # Extract texts from input
        if isinstance(X_train, pd.DataFrame):
            if 'text' in X_train.columns:
                train_texts = X_train['text'].tolist()
            else:
                # Assume first column contains text
                train_texts = X_train.iloc[:, 0].tolist()
        else:
            train_texts = X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train)

        train_labels = y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train)

        # Prepare training dataset
        train_dataset = self._prepare_dataset(train_texts, train_labels)

        # Prepare validation dataset if provided
        eval_dataset = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                if 'text' in X_val.columns:
                    val_texts = X_val['text'].tolist()
                else:
                    val_texts = X_val.iloc[:, 0].tolist()
            else:
                val_texts = X_val.tolist() if hasattr(X_val, 'tolist') else list(X_val)

            val_labels = y_val.tolist() if hasattr(y_val, 'tolist') else list(y_val)
            eval_dataset = self._prepare_dataset(val_texts, val_labels)

        # Configure training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_accuracy" if eval_dataset else None,
            greater_is_better=True,
            save_total_limit=2,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics if eval_dataset else None,
        )

        # Train the model
        logger.info("Starting BERT training...")
        train_result = self.trainer.train()

        self.is_trained = True

        # Prepare return metrics
        metrics = {
            "training_loss": train_result.training_loss,
            "training_samples": len(train_texts),
            "training_steps": train_result.global_step,
            "epochs_completed": train_result.epoch
        }

        # Add evaluation metrics if validation was performed
        if eval_dataset:
            eval_result = self.trainer.evaluate()
            metrics.update({f"validation_{k}": v for k, v in eval_result.items()})

        logger.info(f"BERT training completed. Final training loss: {train_result.training_loss:.4f}")
        return metrics

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict class labels for input texts.

        Args:
            texts: List of text samples to classify

        Returns:
            Array of predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare dataset for prediction
        dataset = self._prepare_dataset(texts)

        # Get predictions using trainer
        predictions = self.trainer.predict(dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=1)

        return predicted_labels

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for input texts.

        Args:
            texts: List of text samples to classify

        Returns:
            Array of prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare dataset for prediction
        dataset = self._prepare_dataset(texts)

        # Get predictions using trainer
        predictions = self.trainer.predict(dataset)

        # Apply softmax to get probabilities
        logits = predictions.predictions
        probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()

        return probabilities

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test texts
            y_test: Test labels

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Extract texts from input
        if isinstance(X_test, pd.DataFrame):
            if 'text' in X_test.columns:
                test_texts = X_test['text'].tolist()
            else:
                test_texts = X_test.iloc[:, 0].tolist()
        else:
            test_texts = X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test)

        test_labels = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)

        # Prepare test dataset
        test_dataset = self._prepare_dataset(test_texts, test_labels)

        # Evaluate using trainer
        eval_result = self.trainer.evaluate(test_dataset)

        # Clean up metric names (remove 'eval_' prefix)
        metrics = {k.replace('eval_', ''): v for k, v in eval_result.items()}

        logger.info(f"Evaluation completed. Accuracy: {metrics.get('accuracy', 0):.4f}")
        return metrics

    def save(self, file_path: str) -> None:
        """
        Save the trained model to a directory.

        Args:
            file_path: Directory path where the model should be saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Create directory if it doesn't exist
        os.makedirs(file_path, exist_ok=True)

        # Save model and tokenizer
        self._model.save_pretrained(file_path)
        self.tokenizer.save_pretrained(file_path)

        # Save additional configuration
        import json
        config = {
            "model_name": self.model_name,
            "pretrained_model": self.pretrained_model,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "is_trained": self.is_trained,
            "config": self.config
        }

        with open(os.path.join(file_path, "classifier_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"BERT model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'BERTClassifier':
        """
        Load a trained model from a directory.

        Args:
            file_path: Directory path to the saved model

        Returns:
            Loaded classifier instance
        """
        # Load configuration
        import json
        config_path = os.path.join(file_path, "classifier_config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Create new instance
        classifier = cls(
            model_name=config["model_name"],
            pretrained_model=config["pretrained_model"],
            num_labels=config["num_labels"],
            max_length=config["max_length"],
            **config.get("config", {})
        )

        # Load the trained model and tokenizer
        classifier._model = AutoModelForSequenceClassification.from_pretrained(file_path)
        classifier.tokenizer = AutoTokenizer.from_pretrained(file_path)
        classifier.is_trained = config["is_trained"]

        # Move model to appropriate device
        classifier._model.to(classifier.device)

        # Initialize trainer for predictions (without training arguments)
        from transformers import Trainer
        classifier.trainer = Trainer(model=classifier._model)

        logger.info(f"BERT model loaded from {file_path}")
        return classifier

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary containing model metadata and configuration
        """
        base_info = super().get_model_info()
        bert_info = {
            "pretrained_model": self.pretrained_model,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self._model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        }

        return {**base_info, **bert_info}