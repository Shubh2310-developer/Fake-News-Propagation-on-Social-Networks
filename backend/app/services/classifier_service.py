# backend/app/services/classifier_service.py

from typing import Dict, Any, List, Optional
import logging
import asyncio
from pathlib import Path
import joblib

from ml_models.classifiers import (
    BaseClassifier,
    LogisticRegressionClassifier,
    BERTClassifier,
    LSTMClassifier,
    EnsembleClassifier
)
from ml_models.preprocessing import TextProcessor, FeatureExtractor
from app.core.config import settings

logger = logging.getLogger(__name__)


class ClassifierService:
    """Orchestrates the ML classification workflow."""

    def __init__(self):
        """Initialize the classifier service."""
        self.models: Dict[str, BaseClassifier] = {}
        self.text_processor = TextProcessor()
        self.feature_extractor = FeatureExtractor()
        self.model_configs = self._get_model_configurations()
        self._model_loading_status = {}

        logger.info("ClassifierService initialized")

    def _get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations from settings."""
        return {
            'logistic_regression': {
                'class': LogisticRegressionClassifier,
                'model_path': getattr(settings, 'LOGISTIC_MODEL_PATH', 'models/logistic_regression.joblib'),
                'enabled': getattr(settings, 'ENABLE_LOGISTIC_MODEL', True)
            },
            'bert': {
                'class': BERTClassifier,
                'model_path': getattr(settings, 'BERT_MODEL_PATH', 'models/bert_classifier'),
                'enabled': getattr(settings, 'ENABLE_BERT_MODEL', True)
            },
            'lstm': {
                'class': LSTMClassifier,
                'model_path': getattr(settings, 'LSTM_MODEL_PATH', 'models/lstm_classifier.pt'),
                'enabled': getattr(settings, 'ENABLE_LSTM_MODEL', True)
            },
            'ensemble': {
                'class': EnsembleClassifier,
                'model_path': getattr(settings, 'ENSEMBLE_MODEL_PATH', 'models/ensemble_config.joblib'),
                'enabled': getattr(settings, 'ENABLE_ENSEMBLE_MODEL', True)
            }
        }

    async def load_models(self) -> Dict[str, bool]:
        """
        Loads all configured machine learning models into memory.

        Returns:
            Dictionary mapping model names to their loading success status
        """
        logger.info("Starting model loading process")
        loading_results = {}

        for model_name, config in self.model_configs.items():
            if not config['enabled']:
                logger.info(f"Model {model_name} is disabled, skipping")
                loading_results[model_name] = False
                continue

            try:
                await self._load_single_model(model_name, config)
                loading_results[model_name] = True
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                loading_results[model_name] = False

        # Load ensemble after individual models if enabled
        if 'ensemble' in loading_results and loading_results['ensemble']:
            try:
                await self._setup_ensemble_model()
                logger.info("Ensemble model setup completed")
            except Exception as e:
                logger.error(f"Failed to setup ensemble model: {e}")
                loading_results['ensemble'] = False

        logger.info(f"Model loading completed. Results: {loading_results}")
        return loading_results

    async def _load_single_model(self, model_name: str, config: Dict[str, Any]) -> None:
        """Load a single model."""
        model_path = Path(config['model_path'])

        if not model_path.exists():
            logger.warning(f"Model file not found for {model_name}: {model_path}")
            # Create a fresh model instance for training
            model_class = config['class']
            model = model_class(model_name=model_name)
            self.models[model_name] = model
            self._model_loading_status[model_name] = 'untrained'
            return

        try:
            # Load pre-trained model
            model_class = config['class']

            if model_name == 'ensemble':
                # Ensemble requires special loading with constituent models
                # For now, create an empty ensemble
                constituent_models = [
                    model for name, model in self.models.items()
                    if name != 'ensemble' and model.is_trained
                ]
                if constituent_models:
                    model = EnsembleClassifier(
                        models=constituent_models,
                        model_name='ensemble'
                    )
                else:
                    model = EnsembleClassifier(models=[], model_name='ensemble')
            else:
                model = model_class.load(str(model_path))

            self.models[model_name] = model
            self._model_loading_status[model_name] = 'loaded'

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            # Fall back to creating a fresh model
            model_class = config['class']
            model = model_class(model_name=model_name)
            self.models[model_name] = model
            self._model_loading_status[model_name] = 'fallback'

    async def _setup_ensemble_model(self) -> None:
        """Setup ensemble model with loaded constituent models."""
        if 'ensemble' not in self.models:
            return

        # Get all trained models for ensemble
        constituent_models = [
            model for name, model in self.models.items()
            if name != 'ensemble' and hasattr(model, 'is_trained') and model.is_trained
        ]

        if constituent_models:
            # Update ensemble with constituent models
            ensemble_model = self.models['ensemble']
            if hasattr(ensemble_model, '_models'):
                ensemble_model._models = constituent_models
                ensemble_model.is_trained = True
                logger.info(f"Ensemble configured with {len(constituent_models)} models")

    async def predict(self, text: str, model_type: str = "ensemble") -> Dict[str, Any]:
        """
        Cleans text, runs prediction, and formats the output.

        Args:
            text: Input text to classify
            model_type: Type of model to use for prediction

        Returns:
            Dictionary containing prediction results
        """
        try:
            if model_type not in self.models:
                available_models = list(self.models.keys())
                logger.error(f"Model '{model_type}' not found. Available models: {available_models}")
                raise ValueError(f"Model '{model_type}' not found. Available: {available_models}")

            model = self.models[model_type]

            if not hasattr(model, 'is_trained') or not model.is_trained:
                logger.error(f"Model '{model_type}' is not trained")
                raise ValueError(f"Model '{model_type}' is not trained")

            # Preprocess text
            cleaned_text = await self._preprocess_text(text)

            # Run prediction
            probabilities = model.predict_proba([cleaned_text])[0]
            prediction_index = int(probabilities.argmax())

            # Format results
            labels = ['real', 'fake']  # Assuming 0=real, 1=fake
            prediction_label = labels[prediction_index]
            confidence = float(probabilities[prediction_index])

            result = {
                "text": text,
                "cleaned_text": cleaned_text,
                "prediction": prediction_label,
                "confidence": confidence,
                "probabilities": {
                    "real": float(probabilities[0]),
                    "fake": float(probabilities[1])
                },
                "model_used": model_type,
                "model_info": model.get_model_info() if hasattr(model, 'get_model_info') else {}
            }

            logger.info(f"Prediction completed using {model_type}: {prediction_label} ({confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    async def _preprocess_text(self, text: str) -> str:
        """Preprocess text for model input."""
        try:
            # Basic cleaning
            cleaned_text = self.text_processor.clean(text)

            # Additional preprocessing could be added here
            # e.g., length checks, encoding validation, etc.

            return cleaned_text
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            raise

    async def predict_batch(self, texts: List[str], model_type: str = "ensemble") -> List[Dict[str, Any]]:
        """
        Run batch predictions on multiple texts.

        Args:
            texts: List of input texts
            model_type: Type of model to use

        Returns:
            List of prediction results
        """
        try:
            if model_type not in self.models:
                raise ValueError(f"Model '{model_type}' not found")

            model = self.models[model_type]

            if not hasattr(model, 'is_trained') or not model.is_trained:
                raise ValueError(f"Model '{model_type}' is not trained")

            # Preprocess all texts
            cleaned_texts = [await self._preprocess_text(text) for text in texts]

            # Run batch prediction
            all_probabilities = model.predict_proba(cleaned_texts)

            # Format results
            results = []
            labels = ['real', 'fake']

            for i, (original_text, cleaned_text, probabilities) in enumerate(zip(texts, cleaned_texts, all_probabilities)):
                prediction_index = int(probabilities.argmax())
                prediction_label = labels[prediction_index]
                confidence = float(probabilities[prediction_index])

                result = {
                    "text": original_text,
                    "cleaned_text": cleaned_text,
                    "prediction": prediction_label,
                    "confidence": confidence,
                    "probabilities": {
                        "real": float(probabilities[0]),
                        "fake": float(probabilities[1])
                    },
                    "model_used": model_type,
                    "batch_index": i
                }
                results.append(result)

            logger.info(f"Batch prediction completed for {len(texts)} texts using {model_type}")
            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

    async def get_model_info(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about loaded models.

        Args:
            model_type: Specific model to get info for, or None for all models

        Returns:
            Dictionary containing model information
        """
        try:
            if model_type:
                if model_type not in self.models:
                    raise ValueError(f"Model '{model_type}' not found")

                model = self.models[model_type]
                return {
                    "model_type": model_type,
                    "is_trained": getattr(model, 'is_trained', False),
                    "loading_status": self._model_loading_status.get(model_type, 'unknown'),
                    "model_info": model.get_model_info() if hasattr(model, 'get_model_info') else {}
                }
            else:
                # Return info for all models
                all_info = {}
                for name, model in self.models.items():
                    all_info[name] = {
                        "is_trained": getattr(model, 'is_trained', False),
                        "loading_status": self._model_loading_status.get(name, 'unknown'),
                        "model_info": model.get_model_info() if hasattr(model, 'get_model_info') else {}
                    }
                return all_info

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise

    async def get_model_metrics(self, model_type: str) -> Dict[str, Any]:
        """
        Get evaluation metrics for a specific model.

        Args:
            model_type: Type of model to get metrics for

        Returns:
            Dictionary containing model metrics
        """
        try:
            if model_type not in self.models:
                raise ValueError(f"Model '{model_type}' not found")

            model = self.models[model_type]

            # Try to load metrics from saved files
            metrics_path = Path(f"models/{model_type}_metrics.json")
            if metrics_path.exists():
                import json
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                return metrics
            else:
                # Return basic model info if no metrics file exists
                return {
                    "model_type": model_type,
                    "metrics_available": False,
                    "message": "No saved metrics found for this model"
                }

        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            raise

    async def train_model(self,
                         model_type: str,
                         training_data: Dict[str, Any],
                         save_model: bool = True) -> Dict[str, Any]:
        """
        Train a specific model with provided data.

        Args:
            model_type: Type of model to train
            training_data: Training data and configuration
            save_model: Whether to save the trained model

        Returns:
            Dictionary containing training results
        """
        try:
            if model_type not in self.models:
                raise ValueError(f"Model '{model_type}' not found")

            model = self.models[model_type]

            # Extract training data
            X_train = training_data['X_train']
            y_train = training_data['y_train']
            X_val = training_data.get('X_val')
            y_val = training_data.get('y_val')

            logger.info(f"Starting training for {model_type}")

            # Train the model
            training_results = model.train(X_train, y_train, X_val, y_val)

            # Save model if requested
            if save_model:
                model_path = self.model_configs[model_type]['model_path']
                model.save(model_path)
                logger.info(f"Model saved to {model_path}")

            # Update loading status
            self._model_loading_status[model_type] = 'trained'

            logger.info(f"Training completed for {model_type}")
            return {
                "model_type": model_type,
                "training_status": "completed",
                "training_results": training_results,
                "model_saved": save_model
            }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    async def reload_model(self, model_type: str) -> bool:
        """
        Reload a specific model from disk.

        Args:
            model_type: Type of model to reload

        Returns:
            True if reload was successful, False otherwise
        """
        try:
            if model_type not in self.model_configs:
                raise ValueError(f"Unknown model type: {model_type}")

            config = self.model_configs[model_type]
            await self._load_single_model(model_type, config)

            logger.info(f"Model {model_type} reloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to reload model {model_type}: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.models.keys())

    def is_model_ready(self, model_type: str) -> bool:
        """Check if a model is loaded and ready for predictions."""
        if model_type not in self.models:
            return False

        model = self.models[model_type]
        return hasattr(model, 'is_trained') and model.is_trained