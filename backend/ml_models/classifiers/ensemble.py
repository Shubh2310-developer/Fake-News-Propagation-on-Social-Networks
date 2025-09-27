# backend/ml_models/classifiers/ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.optimize import minimize
from .base_classifier import BaseClassifier
import logging
import joblib

logger = logging.getLogger(__name__)


class EnsembleClassifier(BaseClassifier):
    """
    Combines predictions from multiple classifiers using voting strategies.

    This ensemble approach leverages the strengths of diverse models to achieve
    higher accuracy and robustness than any single model alone.
    """

    def __init__(self,
                 models: List[BaseClassifier],
                 voting: str = 'soft',
                 weights: Optional[List[float]] = None,
                 model_name: str = "ensemble",
                 **kwargs):
        """
        Initialize the ensemble classifier.

        Args:
            models: List of trained BaseClassifier instances
            voting: Voting strategy ('hard' for majority vote, 'soft' for probability averaging)
            weights: Optional weights for each model (for weighted voting)
            model_name: Name identifier for the ensemble
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        if not models:
            raise ValueError("At least one model must be provided for ensemble")

        if not all(isinstance(m, BaseClassifier) for m in models):
            raise TypeError("All models in the ensemble must be BaseClassifier instances")

        self._models = models
        self.voting = voting.lower()
        self.weights = weights

        if self.voting not in ['hard', 'soft']:
            raise ValueError("Voting strategy must be 'hard' or 'soft'")

        # Validate weights
        if self.weights is not None:
            if len(self.weights) != len(self._models):
                raise ValueError("Number of weights must match number of models")
            if not np.allclose(sum(self.weights), 1.0):
                logger.warning("Weights do not sum to 1.0, normalizing...")
                self.weights = np.array(self.weights) / sum(self.weights)
        else:
            # Equal weights by default
            self.weights = np.ones(len(self._models)) / len(self._models)

        # Check if all models are trained
        self.is_trained = all(model.is_trained for model in self._models)

        logger.info(f"Ensemble initialized with {len(self._models)} models using {self.voting} voting")

    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train all base models in the ensemble.

        Note: In practice, base models are typically trained separately.
        This method trains any untrained models and can optimize ensemble weights.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Dictionary containing ensemble training metrics
        """
        logger.info("Training ensemble models...")

        training_results = {}
        trained_count = 0

        # Train any untrained models
        for i, model in enumerate(self._models):
            if not model.is_trained:
                logger.info(f"Training model {i+1}/{len(self._models)}: {model.model_name}")
                try:
                    result = model.train(X_train, y_train, X_val, y_val)
                    training_results[f"model_{i}_{model.model_name}"] = result
                    trained_count += 1
                except Exception as e:
                    logger.error(f"Failed to train model {model.model_name}: {e}")
                    raise

        # Optimize ensemble weights if validation data is provided
        optimized_weights = None
        if X_val is not None and y_val is not None and self.voting == 'soft':
            logger.info("Optimizing ensemble weights on validation data...")
            try:
                optimized_weights = self.optimize_weights(X_val, y_val)
                training_results["optimized_weights"] = optimized_weights.tolist()
            except Exception as e:
                logger.warning(f"Weight optimization failed: {e}")

        self.is_trained = True

        metrics = {
            "models_trained": trained_count,
            "total_models": len(self._models),
            "ensemble_weights": self.weights.tolist(),
            "voting_strategy": self.voting,
            "training_results": training_results
        }

        # Evaluate ensemble on validation data if available
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics["validation_metrics"] = val_metrics

        logger.info(f"Ensemble training completed. {trained_count} models trained.")
        return metrics

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict class labels using ensemble voting.

        Args:
            texts: List of text samples to classify

        Returns:
            Array of predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        if self.voting == 'hard':
            return self._predict_hard_voting(texts)
        else:  # soft voting
            probabilities = self.predict_proba(texts)
            return np.argmax(probabilities, axis=1)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities using weighted averaging.

        Args:
            texts: List of text samples to classify

        Returns:
            Array of prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        all_probas = []

        # Get probabilities from each model
        for i, model in enumerate(self._models):
            try:
                model_probas = model.predict_proba(texts)
                all_probas.append(model_probas)
            except Exception as e:
                logger.error(f"Error getting probabilities from model {model.model_name}: {e}")
                raise

        # Stack probabilities and apply weights
        stacked_probas = np.stack(all_probas)  # Shape: (n_models, n_samples, n_classes)
        weighted_probas = np.average(stacked_probas, axis=0, weights=self.weights)

        return weighted_probas

    def _predict_hard_voting(self, texts: List[str]) -> np.ndarray:
        """
        Predict using hard voting (majority vote).

        Args:
            texts: List of text samples to classify

        Returns:
            Array of predicted class labels
        """
        all_predictions = []

        # Get predictions from each model
        for model in self._models:
            predictions = model.predict(texts)
            all_predictions.append(predictions)

        # Stack predictions and find majority vote
        stacked_predictions = np.stack(all_predictions)  # Shape: (n_models, n_samples)

        # Apply weights to predictions (each model vote is weighted)
        weighted_votes = []
        for i in range(stacked_predictions.shape[1]):  # For each sample
            sample_votes = stacked_predictions[:, i]
            unique_classes, counts = np.unique(sample_votes, return_counts=True)

            # Weight the votes
            weighted_counts = {}
            for j, class_vote in enumerate(sample_votes):
                if class_vote not in weighted_counts:
                    weighted_counts[class_vote] = 0
                weighted_counts[class_vote] += self.weights[j]

            # Find class with highest weighted vote
            best_class = max(weighted_counts.items(), key=lambda x: x[1])[0]
            weighted_votes.append(best_class)

        return np.array(weighted_votes)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the ensemble on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before evaluation")

        # Get ensemble predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        # Add AUC-ROC for binary classification
        if y_proba.shape[1] == 2:
            from sklearn.metrics import roc_auc_score
            metrics["auc_roc"] = roc_auc_score(y_test, y_proba[:, 1])

        # Evaluate individual models for comparison
        individual_metrics = {}
        for i, model in enumerate(self._models):
            try:
                model_metrics = model.evaluate(X_test, y_test)
                individual_metrics[f"model_{i}_{model.model_name}"] = model_metrics
            except Exception as e:
                logger.warning(f"Could not evaluate model {model.model_name}: {e}")

        metrics["individual_model_metrics"] = individual_metrics

        logger.info(f"Ensemble evaluation completed. Accuracy: {accuracy:.4f}")
        return metrics

    def optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> np.ndarray:
        """
        Optimize ensemble weights to minimize validation loss.

        Args:
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Optimized weights array
        """
        logger.info("Optimizing ensemble weights...")

        # Get predictions from all models
        all_probas = []
        for model in self._models:
            probas = model.predict_proba(X_val)
            all_probas.append(probas)

        stacked_probas = np.stack(all_probas)  # Shape: (n_models, n_samples, n_classes)

        def objective(weights):
            """Objective function to minimize (negative log-likelihood)."""
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_probas = np.average(stacked_probas, axis=0, weights=weights)

            # Avoid log(0) by adding small epsilon
            epsilon = 1e-15
            ensemble_probas = np.clip(ensemble_probas, epsilon, 1 - epsilon)

            # Calculate negative log-likelihood
            log_likelihood = 0
            for i, true_label in enumerate(y_val):
                log_likelihood += np.log(ensemble_probas[i, true_label])

            return -log_likelihood

        # Constraints: weights must sum to 1 and be non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self._models))]

        # Initial guess: equal weights
        initial_weights = np.ones(len(self._models)) / len(self._models)

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            optimized_weights = result.x / np.sum(result.x)  # Normalize
            self.weights = optimized_weights
            logger.info(f"Weight optimization successful. New weights: {optimized_weights}")
            return optimized_weights
        else:
            logger.warning(f"Weight optimization failed: {result.message}")
            return self.weights

    def save(self, file_path: str) -> None:
        """
        Save the ensemble configuration and model references.

        Note: Individual models should be saved separately.

        Args:
            file_path: Path where the ensemble config should be saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")

        ensemble_data = {
            "model_name": self.model_name,
            "config": self.config,
            "voting": self.voting,
            "weights": self.weights.tolist(),
            "is_trained": self.is_trained,
            "model_names": [model.model_name for model in self._models],
            "model_types": [type(model).__name__ for model in self._models]
        }

        joblib.dump(ensemble_data, file_path)
        logger.info(f"Ensemble configuration saved to {file_path}")
        logger.warning("Individual models must be saved and loaded separately")

    @classmethod
    def load(cls, file_path: str, models: List[BaseClassifier]) -> 'EnsembleClassifier':
        """
        Load ensemble configuration and combine with provided models.

        Args:
            file_path: Path to the saved ensemble config
            models: List of pre-loaded BaseClassifier instances

        Returns:
            Loaded ensemble classifier instance
        """
        ensemble_data = joblib.load(file_path)

        # Validate that provided models match saved configuration
        if len(models) != len(ensemble_data["model_names"]):
            raise ValueError("Number of provided models doesn't match saved configuration")

        ensemble = cls(
            models=models,
            voting=ensemble_data["voting"],
            weights=ensemble_data["weights"],
            model_name=ensemble_data["model_name"],
            **ensemble_data.get("config", {})
        )

        ensemble.is_trained = ensemble_data["is_trained"]

        logger.info(f"Ensemble loaded from {file_path}")
        return ensemble

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the ensemble and its models.

        Returns:
            Dictionary containing ensemble metadata
        """
        base_info = super().get_model_info()

        ensemble_info = {
            "voting_strategy": self.voting,
            "weights": self.weights.tolist(),
            "num_models": len(self._models),
            "model_details": [
                {
                    "name": model.model_name,
                    "type": type(model).__name__,
                    "is_trained": model.is_trained
                }
                for model in self._models
            ]
        }

        return {**base_info, **ensemble_info}

    def get_individual_predictions(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model in the ensemble.

        Args:
            texts: List of text samples to classify

        Returns:
            Dictionary mapping model names to their predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        individual_predictions = {}

        for model in self._models:
            try:
                predictions = model.predict(texts)
                individual_predictions[model.model_name] = predictions
            except Exception as e:
                logger.error(f"Error getting predictions from {model.model_name}: {e}")

        return individual_predictions

    def get_prediction_agreement(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze agreement between models in the ensemble.

        Args:
            texts: List of text samples to classify

        Returns:
            Dictionary containing agreement statistics
        """
        individual_preds = self.get_individual_predictions(texts)

        if not individual_preds:
            return {"error": "No predictions available"}

        # Convert to array for analysis
        pred_array = np.array(list(individual_preds.values()))

        # Calculate agreement statistics
        unanimous_agreement = np.sum(np.all(pred_array == pred_array[0], axis=0))
        majority_agreement = np.sum(
            np.apply_along_axis(lambda x: np.max(np.bincount(x)) > len(self._models) / 2, 0, pred_array)
        )

        return {
            "total_samples": len(texts),
            "unanimous_agreement": int(unanimous_agreement),
            "majority_agreement": int(majority_agreement),
            "unanimous_agreement_rate": unanimous_agreement / len(texts),
            "majority_agreement_rate": majority_agreement / len(texts),
            "individual_predictions": {k: v.tolist() for k, v in individual_preds.items()}
        }