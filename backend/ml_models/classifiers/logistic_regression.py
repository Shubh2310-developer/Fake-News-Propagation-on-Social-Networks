# backend/ml_models/classifiers/logistic_regression.py

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base_classifier import BaseClassifier
import logging

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier(BaseClassifier):
    """
    A baseline classifier using Scikit-learn's Logistic Regression.

    This provides a fast, interpretable baseline model for fake news classification.
    Ideal for establishing performance benchmarks against more complex models.
    """

    def __init__(self,
                 model_name: str = "logistic_regression",
                 max_iter: int = 1000,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize the Logistic Regression classifier.

        Args:
            model_name: Name identifier for the model
            max_iter: Maximum iterations for convergence
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for LogisticRegression
        """
        super().__init__(model_name, **kwargs)

        # Initialize the scikit-learn model with parameters
        self._model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        self.random_state = random_state
        logger.info(f"Initialized LogisticRegressionClassifier with max_iter={max_iter}")

    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: pd.DataFrame = None,
              y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Train the logistic regression model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Training {self.model_name} on {len(X_train)} samples")

        # Convert pandas to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train

        if isinstance(y_train, pd.Series):
            y_train_array = y_train.values
        else:
            y_train_array = y_train

        # Fit the model
        self._model.fit(X_train_array, y_train_array)
        self.is_trained = True

        # Calculate training metrics
        train_predictions = self._model.predict(X_train_array)
        train_accuracy = accuracy_score(y_train_array, train_predictions)

        metrics = {
            "training_accuracy": train_accuracy,
            "training_samples": len(X_train),
            "model_coefficients_shape": self._model.coef_.shape,
            "converged": self._model.n_iter_ < self._model.max_iter
        }

        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            val_predictions = self.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_predictions)
            metrics["validation_accuracy"] = val_accuracy
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

        logger.info(f"Training completed. Accuracy: {train_accuracy:.4f}")
        return metrics

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict class labels for input features.

        Args:
            texts: Input features (should be preprocessed features, not raw text)

        Returns:
            Array of predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Handle both pandas DataFrame and numpy array inputs
        if isinstance(texts, pd.DataFrame):
            features = texts.values
        else:
            features = np.array(texts)

        predictions = self._model.predict(features)
        return predictions

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for input features.

        Args:
            texts: Input features (should be preprocessed features, not raw text)

        Returns:
            Array of prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Handle both pandas DataFrame and numpy array inputs
        if isinstance(texts, pd.DataFrame):
            features = texts.values
        else:
            features = np.array(texts)

        probabilities = self._model.predict_proba(features)
        return probabilities

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        # Add AUC-ROC for binary classification
        if len(np.unique(y_test)) == 2:
            from sklearn.metrics import roc_auc_score
            metrics["auc_roc"] = roc_auc_score(y_test, y_proba[:, 1])

        logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics

    def save(self, file_path: str) -> None:
        """
        Save the trained model to a file.

        Args:
            file_path: Path where the model should be saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model": self._model,
            "model_name": self.model_name,
            "config": self.config,
            "is_trained": self.is_trained
        }

        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'LogisticRegressionClassifier':
        """
        Load a trained model from a file.

        Args:
            file_path: Path to the saved model

        Returns:
            Loaded classifier instance
        """
        model_data = joblib.load(file_path)

        # Create new instance
        classifier = cls(
            model_name=model_data["model_name"],
            **model_data.get("config", {})
        )

        # Restore model state
        classifier._model = model_data["model"]
        classifier.is_trained = model_data["is_trained"]

        logger.info(f"Model loaded from {file_path}")
        return classifier

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (coefficients) from the logistic regression model.

        Returns:
            Dictionary mapping feature indices to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        # For logistic regression, use absolute values of coefficients
        if self._model.coef_.ndim == 1:
            # Binary classification
            coefficients = np.abs(self._model.coef_)
        else:
            # Multi-class: average across classes
            coefficients = np.mean(np.abs(self._model.coef_), axis=0)

        # Create feature importance dictionary
        feature_importance = {
            f"feature_{i}": float(coef)
            for i, coef in enumerate(coefficients)
        }

        return feature_importance