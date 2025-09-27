# backend/tests/services/test_classifier_service.py

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from app.services.classifier_service import ClassifierService
from ml_models.classifiers.base_classifier import BaseClassifier


@pytest.mark.unit
class TestClassifierService:
    """Test suite for ClassifierService business logic."""

    @pytest.fixture
    def classifier_service(self):
        """Create a ClassifierService instance for testing."""
        return ClassifierService()

    @pytest.fixture
    def mock_trained_model(self):
        """Create a mock trained model."""
        mock_model = MagicMock(spec=BaseClassifier)
        mock_model.is_trained = True
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        mock_model.predict.return_value = np.array([0])
        mock_model.get_model_info.return_value = {
            "type": "mock_model",
            "version": "1.0",
            "accuracy": 0.95
        }
        return mock_model

    @pytest.fixture
    def mock_untrained_model(self):
        """Create a mock untrained model."""
        mock_model = MagicMock(spec=BaseClassifier)
        mock_model.is_trained = False
        mock_model.get_model_info.return_value = {
            "type": "mock_model",
            "version": "1.0",
            "trained": False
        }
        return mock_model

    def test_init_creates_empty_models_dict(self, classifier_service):
        """Test that initialization creates empty models dictionary."""
        assert hasattr(classifier_service, 'models')
        assert isinstance(classifier_service.models, dict)

    def test_load_models_success(self, classifier_service, mock_trained_model):
        """Test successful model loading."""
        with patch('app.services.classifier_service.EnsembleClassifier') as mock_ensemble, \
             patch('app.services.classifier_service.BertClassifier') as mock_bert:

            mock_ensemble.return_value = mock_trained_model
            mock_bert.return_value = mock_trained_model

            result = classifier_service.load_models()

            assert "ensemble" in result
            assert "bert" in result
            assert result["ensemble"] is True
            assert result["bert"] is True

    def test_predict_with_valid_model(self, classifier_service, mock_trained_model):
        """Test prediction with a valid, trained model."""
        classifier_service.models = {"ensemble": mock_trained_model}

        result = classifier_service.predict("Test news article", "ensemble")

        assert "text" in result
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "model_used" in result

        assert result["text"] == "Test news article"
        assert result["prediction"] in ["real", "fake"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["model_used"] == "ensemble"

    def test_predict_with_invalid_model(self, classifier_service):
        """Test prediction with invalid model type."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            classifier_service.predict("Test text", "nonexistent")

    def test_predict_with_untrained_model(self, classifier_service, mock_untrained_model):
        """Test prediction with untrained model."""
        classifier_service.models = {"untrained": mock_untrained_model}

        with pytest.raises(ValueError, match="Model 'untrained' is not trained"):
            classifier_service.predict("Test text", "untrained")

    def test_predict_empty_text(self, classifier_service, mock_trained_model):
        """Test prediction with empty text."""
        classifier_service.models = {"ensemble": mock_trained_model}

        with pytest.raises(ValueError, match="Text cannot be empty"):
            classifier_service.predict("", "ensemble")

    def test_predict_batch_success(self, classifier_service, mock_trained_model):
        """Test batch prediction with valid inputs."""
        classifier_service.models = {"ensemble": mock_trained_model}
        mock_trained_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_trained_model.predict.return_value = np.array([0, 1])

        texts = ["Real news article", "Fake news article"]
        results = classifier_service.predict_batch(texts, "ensemble")

        assert len(results) == 2
        assert all("batch_index" in result for result in results)
        assert results[0]["batch_index"] == 0
        assert results[1]["batch_index"] == 1

    def test_predict_batch_empty_list(self, classifier_service):
        """Test batch prediction with empty list."""
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            classifier_service.predict_batch([], "ensemble")

    def test_get_model_metrics_success(self, classifier_service, mock_trained_model):
        """Test getting model metrics for trained model."""
        classifier_service.models = {"ensemble": mock_trained_model}

        result = classifier_service.get_model_metrics("ensemble")

        assert "model_type" in result
        assert "metrics_available" in result
        assert result["model_type"] == "ensemble"
        assert result["metrics_available"] is True

    def test_get_model_metrics_invalid_model(self, classifier_service):
        """Test getting metrics for invalid model."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            classifier_service.get_model_metrics("nonexistent")

    def test_get_available_models(self, classifier_service, mock_trained_model):
        """Test getting list of available models."""
        classifier_service.models = {
            "ensemble": mock_trained_model,
            "bert": mock_trained_model
        }

        models = classifier_service.get_available_models()

        assert "ensemble" in models
        assert "bert" in models
        assert len(models) == 2

    def test_model_training_workflow(self, classifier_service):
        """Test model training workflow."""
        with patch.object(classifier_service, '_train_model_async') as mock_train:
            mock_train.return_value = {
                "training_id": "test-123",
                "status": "started",
                "model_type": "logistic_regression"
            }

            training_data = {
                "X_train": [["feature1"], ["feature2"]],
                "y_train": [0, 1],
                "X_val": [["feature3"]],
                "y_val": [0]
            }

            result = classifier_service.train_model("logistic_regression", training_data)

            assert "training_id" in result
            assert "status" in result
            assert result["status"] == "started"

    def test_preprocess_text_basic(self, classifier_service):
        """Test basic text preprocessing."""
        with patch('app.services.classifier_service.clean_text') as mock_clean:
            mock_clean.return_value = "cleaned text"

            result = classifier_service._preprocess_text("Raw text with noise!")

            mock_clean.assert_called_once_with("Raw text with noise!")
            assert result == "cleaned text"

    def test_confidence_calculation(self, classifier_service):
        """Test confidence score calculation from probabilities."""
        proba = np.array([[0.8, 0.2]])
        confidence = classifier_service._calculate_confidence(proba)

        assert confidence == 0.8  # Max probability

        proba_close = np.array([[0.51, 0.49]])
        confidence_close = classifier_service._calculate_confidence(proba_close)

        assert confidence_close == 0.51

    def test_prediction_label_mapping(self, classifier_service):
        """Test mapping from numeric predictions to labels."""
        assert classifier_service._map_prediction_to_label(0) == "real"
        assert classifier_service._map_prediction_to_label(1) == "fake"

    def test_model_info_aggregation(self, classifier_service, mock_trained_model):
        """Test aggregating model information."""
        classifier_service.models = {"ensemble": mock_trained_model}

        info = classifier_service.get_models_info()

        assert "ensemble" in info
        assert "type" in info["ensemble"]
        assert "version" in info["ensemble"]

    @pytest.mark.slow
    def test_prediction_performance(self, classifier_service, mock_trained_model, performance_timer):
        """Test prediction performance requirements."""
        classifier_service.models = {"ensemble": mock_trained_model}
        timer = performance_timer(threshold_seconds=0.1)  # 100ms threshold

        classifier_service.predict("Test news article", "ensemble")

        elapsed = timer()
        assert elapsed < 0.1, f"Prediction took {elapsed:.3f}s, should be < 0.1s"

    def test_concurrent_predictions(self, classifier_service, mock_trained_model):
        """Test handling concurrent predictions."""
        classifier_service.models = {"ensemble": mock_trained_model}

        async def predict_async():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, classifier_service.predict, "Test text", "ensemble"
            )

        async def run_concurrent_predictions():
            tasks = [predict_async() for _ in range(5)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_concurrent_predictions())
        assert len(results) == 5
        assert all(result["prediction"] in ["real", "fake"] for result in results)

    def test_error_handling_model_loading_failure(self, classifier_service):
        """Test error handling when model loading fails."""
        with patch('app.services.classifier_service.EnsembleClassifier') as mock_ensemble:
            mock_ensemble.side_effect = Exception("Failed to load model")

            with pytest.raises(Exception, match="Failed to load model"):
                classifier_service.load_models()

    def test_memory_cleanup_after_prediction(self, classifier_service, mock_trained_model):
        """Test that memory is properly managed after predictions."""
        classifier_service.models = {"ensemble": mock_trained_model}

        # Simulate prediction
        result = classifier_service.predict("Test text", "ensemble")

        # Verify model is still accessible (not garbage collected)
        assert "ensemble" in classifier_service.models
        assert classifier_service.models["ensemble"] is not None

    def test_model_validation_on_load(self, classifier_service):
        """Test model validation during loading."""
        with patch('app.services.classifier_service.EnsembleClassifier') as mock_ensemble:
            mock_model = MagicMock()
            mock_model.is_trained = False  # Invalid state
            mock_ensemble.return_value = mock_model

            # Should handle untrained models gracefully
            result = classifier_service.load_models()
            assert result["ensemble"] is False  # Should indicate failure

    def test_prediction_consistency(self, classifier_service, mock_trained_model):
        """Test that predictions are consistent for same input."""
        classifier_service.models = {"ensemble": mock_trained_model}

        text = "Consistent test input"
        result1 = classifier_service.predict(text, "ensemble")
        result2 = classifier_service.predict(text, "ensemble")

        assert result1["prediction"] == result2["prediction"]
        assert result1["confidence"] == result2["confidence"]