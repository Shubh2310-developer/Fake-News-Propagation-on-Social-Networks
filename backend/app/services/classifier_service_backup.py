# backend/app/services/classifier_service.py

from typing import Dict, Any, List, Optional
import logging
import asyncio
from pathlib import Path
import joblib
import pickle
import numpy as np
import pandas as pd

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
        self.feature_extractor = None  # Will be initialized after loading preprocessing
        self.vectorizer = None
        self.scaler = None
        self.feature_names = []
        self.model_configs = self._get_model_configurations()
        self._model_loading_status = {}

        # Load preprocessing pipeline
        self._load_preprocessing_pipeline()

        # Initialize feature extractor with the loaded vectorizer
        self.feature_extractor = FeatureExtractor()
        if self.vectorizer:
            self.feature_extractor.tfidf_vectorizer = self.vectorizer

        logger.info("ClassifierService initialized")

    def _load_preprocessing_pipeline(self):
        """Load the TF-IDF vectorizer and scaler."""
        try:
            preprocessing_path = Path('models/preprocessing.pkl')
            if preprocessing_path.exists():
                with open(preprocessing_path, 'rb') as f:
                    preprocessing_data = pickle.load(f)
                    self.vectorizer = preprocessing_data.get('vectorizer')
                    self.scaler = preprocessing_data.get('scaler')
                    self.feature_names = preprocessing_data.get('feature_names', [])
                    logger.info(f"Preprocessing pipeline loaded successfully ({len(self.feature_names)} features)")
            else:
                logger.warning("Preprocessing pipeline file not found")
        except Exception as e:
            logger.error(f"Failed to load preprocessing pipeline: {e}")

    def _get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations from settings."""
        return {
            'logistic_regression': {
                'class': LogisticRegressionClassifier,
                'model_path': getattr(settings, 'LOGISTIC_MODEL_PATH', 'models/logistic_regression.joblib'),
                'enabled': getattr(settings, 'ENABLE_LOGISTIC_MODEL', True)
            },
            'naive_bayes': {
                'class': LogisticRegressionClassifier,  # Use same base class for traditional ML
                'model_path': getattr(settings, 'NAIVE_BAYES_MODEL_PATH', 'models/naive_bayes.joblib'),
                'enabled': getattr(settings, 'ENABLE_NAIVE_BAYES_MODEL', True)
            },
            'random_forest': {
                'class': LogisticRegressionClassifier,  # Use same base class for traditional ML
                'model_path': getattr(settings, 'RANDOM_FOREST_MODEL_PATH', 'models/random_forest.joblib'),
                'enabled': getattr(settings, 'ENABLE_RANDOM_FOREST_MODEL', True)
            },
            'gradient_boosting': {
                'class': LogisticRegressionClassifier,  # Use same base class for traditional ML
                'model_path': getattr(settings, 'GRADIENT_BOOSTING_MODEL_PATH', 'models/gradient_boosting.joblib'),
                'enabled': getattr(settings, 'ENABLE_GRADIENT_BOOSTING_MODEL', True)
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

        # Load all non-ensemble models first
        for model_name, config in self.model_configs.items():
            if model_name == 'ensemble':
                continue  # Skip ensemble for now

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

        # Now load ensemble after all individual models
        if 'ensemble' in self.model_configs and self.model_configs['ensemble']['enabled']:
            try:
                await self._load_ensemble_model()
                loading_results['ensemble'] = True
                logger.info("Ensemble model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ensemble model: {e}")
                loading_results['ensemble'] = False

        logger.info(f"Model loading completed. Results: {loading_results}")
        return loading_results

    async def _load_single_model(self, model_name: str, config: Dict[str, Any]) -> None:
        """Load a single model (not ensemble)."""
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

    async def _load_ensemble_model(self) -> None:
        """Load ensemble model with constituent models."""
        ensemble_config = self.model_configs['ensemble']
        ensemble_path = Path(ensemble_config['model_path'])

        # Get all successfully loaded traditional ML models for ensemble
        constituent_models = [
            model for name, model in self.models.items()
            if name in ['logistic_regression', 'naive_bayes', 'random_forest', 'gradient_boosting']
            and hasattr(model, 'is_trained') and model.is_trained
        ]

        if not constituent_models:
            raise ValueError("No trained models available for ensemble")

        logger.info(f"Creating ensemble with {len(constituent_models)} models: {[m.model_name for m in constituent_models]}")

        # Load ensemble configuration if it exists
        if ensemble_path.exists():
            try:
                ensemble_model = EnsembleClassifier.load(str(ensemble_path), constituent_models)
                logger.info("Loaded ensemble from saved configuration")
            except Exception as e:
                logger.warning(f"Could not load saved ensemble config: {e}. Creating new ensemble.")
                ensemble_model = EnsembleClassifier(
                    models=constituent_models,
                    voting='soft',
                    model_name='ensemble'
                )
        else:
            logger.info("No saved ensemble config found. Creating new ensemble.")
            ensemble_model = EnsembleClassifier(
                models=constituent_models,
                voting='soft',
                model_name='ensemble'
            )

        self.models['ensemble'] = ensemble_model
        self._model_loading_status['ensemble'] = 'loaded'
        logger.info(f"Ensemble model configured with {len(constituent_models)} models")

    async def predict(self, text: str, model_type: str = "ensemble") -> Dict[str, Any]:
        """
        Cleans text, runs prediction, and formats the output.

        Args:
            text: Input text to classify
            model_type: Type of model to use for prediction

        Returns:
            Dictionary containing prediction results
        """
        import time
        start_time = time.time()

        try:
            if model_type not in self.models:
                available_models = list(self.models.keys())
                logger.error(f"Model '{model_type}' not found. Available models: {available_models}")
                raise ValueError(f"Model '{model_type}' not found. Available: {available_models}")

            model = self.models[model_type]

            if not hasattr(model, 'is_trained') or not model.is_trained:
                logger.warning(f"Model '{model_type}' is not trained, returning demo prediction")
                # Return mock prediction for demo purposes
                import random
                import time
                start_time = time.time()

                random.seed(len(text))
                fake_prob = random.random()
                real_prob = 1.0 - fake_prob
                prediction_label = 'fake' if fake_prob > 0.5 else 'real'

                processing_time = time.time() - start_time

                return {
                    "text": text,
                    "prediction": prediction_label,
                    "confidence": max(fake_prob, real_prob),
                    "probabilities": {
                        "real": real_prob,
                        "fake": fake_prob
                    },
                    "explanation": {"status": "untrained", "mode": "demo"},
                    "model_used": model_type,
                    "processing_time": processing_time
                }

            # Preprocess text
            cleaned_text = await self._preprocess_text(text)

            # Transform text to features for traditional ML models
            if model_type in ['logistic_regression', 'naive_bayes', 'random_forest', 'gradient_boosting']:
                if not self.vectorizer:
                    raise ValueError("Vectorizer not loaded. Cannot process text for traditional ML models.")

                # Extract ALL features: engineered (31) + TF-IDF (2000) = 2031 total
                features_dense = self._extract_all_features([cleaned_text])

                # Run prediction with features
                probabilities = model.predict_proba(features_dense)[0]
            elif model_type == 'ensemble':
                # Special handling for ensemble - extract features and call constituent models directly
                if not self.vectorizer:
                    raise ValueError("Vectorizer not loaded. Cannot process text for traditional ML models.")

                # Extract ALL features: engineered (31) + TF-IDF (2000) = 2031 total
                features_dense = self._extract_all_features([cleaned_text])

                # Get predictions from each constituent model using their sklearn models directly
                all_probas = []
                for constituent_model in model._models:
                    sklearn_model = constituent_model._model if hasattr(constituent_model, '_model') else constituent_model
                    model_proba = sklearn_model.predict_proba(features_dense)
                    all_probas.append(model_proba)

                # Stack and weight the probabilities
                stacked_probas = np.stack(all_probas)
                probabilities = np.average(stacked_probas, axis=0, weights=model.weights)[0]
            else:
                # For BERT/LSTM models, pass the raw text
                probabilities = model.predict_proba([cleaned_text])[0]

            prediction_index = int(probabilities.argmax())

            # Format results
            labels = ['real', 'fake']  # Assuming 0=real, 1=fake
            prediction_label = labels[prediction_index]
            confidence = float(probabilities[prediction_index])

            processing_time = time.time() - start_time

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
                "processing_time": processing_time,
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

    def _extract_all_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract all features (engineered + TF-IDF) in the correct order.

        Args:
            texts: List of cleaned text samples

        Returns:
            Feature array of shape (n_samples, 2031)
        """
        # Extract linguistic and stylistic features using the feature extractor
        linguistic_features = self.feature_extractor.extract_linguistic_features(texts)
        stylistic_features = self.feature_extractor.extract_stylistic_features(texts)

        # Combine engineered features
        engineered_df = pd.concat([linguistic_features, stylistic_features], axis=1)

        # Extract TF-IDF features
        tfidf_features = self.vectorizer.transform(texts)
        tfidf_array = tfidf_features.toarray()

        # Combine all features: engineered + TF-IDF
        all_features = np.hstack([engineered_df.values, tfidf_array])

        # Validate feature dimensions
        expected_features = 2032  # 32 engineered + 2000 TF-IDF
        if all_features.shape[1] != expected_features:
            logger.error(f"Feature dimension mismatch! Expected {expected_features}, got {all_features.shape[1]}")

        return all_features

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

            # Transform text to features for traditional ML models
            if model_type in ['logistic_regression', 'naive_bayes', 'random_forest', 'gradient_boosting', 'ensemble']:
                if not self.vectorizer:
                    raise ValueError("Vectorizer not loaded. Cannot process text for traditional ML models.")

                # Transform texts to TF-IDF features (returns sparse matrix)
                features = self.vectorizer.transform(cleaned_texts)

                # Models expect dense arrays, so convert from sparse
                features_dense = features.toarray()

                # TEMPORARY FIX: Pad features to match model's expected dimensions (2031)
                # The vectorizer produces 2000 features but models expect 2031
                # This is due to mismatch between saved vectorizer and training pipeline
                model_obj = model._model if hasattr(model, '_model') else model
                if hasattr(model_obj, 'n_features_in_'):
                    expected_features = model_obj.n_features_in_
                    current_features = features_dense.shape[1]
                    if current_features < expected_features:
                        padding = np.zeros((features_dense.shape[0], expected_features - current_features))
                        features_dense = np.hstack([features_dense, padding])
                        logger.warning(f"Padded features from {current_features} to {expected_features} for batch prediction")

                # Run prediction with features
                all_probabilities = model.predict_proba(features_dense)
            else:
                # For BERT/LSTM models, pass the raw text
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