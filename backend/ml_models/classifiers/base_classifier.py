# backend/ml_models/classifiers/base_classifier.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    """Abstract Base Class for all news classifiers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._model = None
        self.is_trained = False
        self.config = kwargs

    @abstractmethod
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Trains the classifier and returns training history or metrics.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Dictionary containing training metrics and history
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predicts class labels for a list of texts.

        Args:
            texts: List of text samples to classify

        Returns:
            Array of predicted class labels
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predicts class probabilities for a list of texts.

        Args:
            texts: List of text samples to classify

        Returns:
            Array of prediction probabilities for each class
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the model on test data and returns a dict of metrics.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing evaluation metrics (accuracy, precision, recall, f1)
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, file_path: str) -> None:
        """
        Saves the trained model to a file.

        Args:
            file_path: Path where the model should be saved
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, file_path: str) -> 'BaseClassifier':
        """
        Loads a trained model from a file.

        Args:
            file_path: Path to the saved model

        Returns:
            Loaded classifier instance
        """
        raise NotImplementedError

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Returns feature importance if supported by the model.

        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns information about the model configuration and status.

        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "config": self.config,
            "model_type": self.__class__.__name__
        }