# backend/tests/api/test_classifier.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.mark.api
class TestClassifierAPI:
    """Test suite for classifier API endpoints."""

    def test_predict_endpoint_success(self, client: TestClient, mock_model_loading):
        """
        Tests the /predict endpoint with a valid request.
        Verifies that the API correctly handles valid text input and returns
        properly formatted prediction results.
        """
        with patch('app.services.classifier_service.ClassifierService.predict') as mock_predict:
            mock_predict.return_value = {
                "text": "Scientists announce major breakthrough.",
                "prediction": "real",
                "confidence": 0.85,
                "probabilities": {"real": 0.85, "fake": 0.15},
                "model_used": "ensemble"
            }

            payload = {
                "text": "Scientists announce major breakthrough.",
                "model_type": "ensemble"
            }
            response = client.post("/api/v1/classifier/predict", json=payload)

            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "model_used" in data

            # Verify response values
            assert data["prediction"] in ["real", "fake"]
            assert 0.0 <= data["confidence"] <= 1.0
            assert "real" in data["probabilities"]
            assert "fake" in data["probabilities"]

    def test_predict_endpoint_empty_text(self, client: TestClient):
        """
        Tests the /predict endpoint with empty text input.
        Verifies that proper validation errors are returned.
        """
        payload = {"text": ""}
        response = client.post("/api/v1/classifier/predict", json=payload)

        # Should return 400 Bad Request for empty text
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_predict_endpoint_missing_text(self, client: TestClient):
        """
        Tests the /predict endpoint with missing text field.
        Verifies that proper validation errors are returned.
        """
        payload = {"model_type": "ensemble"}  # Missing text field
        response = client.post("/api/v1/classifier/predict", json=payload)

        # Should return 422 Unprocessable Entity for missing required field
        assert response.status_code == 422

    def test_predict_endpoint_invalid_model_type(self, client: TestClient, mock_model_loading):
        """
        Tests the /predict endpoint with an invalid model type.
        """
        with patch('app.services.classifier_service.ClassifierService.predict') as mock_predict:
            mock_predict.side_effect = ValueError("Model 'invalid_model' not found")

            payload = {
                "text": "Test text",
                "model_type": "invalid_model"
            }
            response = client.post("/api/v1/classifier/predict", json=payload)

            assert response.status_code == 400
            data = response.json()
            assert "detail" in data

    def test_batch_predict_endpoint_success(self, client: TestClient, mock_model_loading):
        """
        Tests the batch prediction endpoint with valid input.
        """
        with patch('app.services.classifier_service.ClassifierService.predict_batch') as mock_batch:
            mock_batch.return_value = [
                {
                    "text": "First text",
                    "prediction": "real",
                    "confidence": 0.8,
                    "probabilities": {"real": 0.8, "fake": 0.2},
                    "batch_index": 0
                },
                {
                    "text": "Second text",
                    "prediction": "fake",
                    "confidence": 0.9,
                    "probabilities": {"real": 0.1, "fake": 0.9},
                    "batch_index": 1
                }
            ]

            payload = {
                "texts": ["First text", "Second text"],
                "model_type": "ensemble"
            }
            response = client.post("/api/v1/classifier/predict/batch", json=payload)

            assert response.status_code == 200
            data = response.json()

            assert "results" in data
            assert "total_processed" in data
            assert "model_used" in data
            assert len(data["results"]) == 2
            assert data["total_processed"] == 2

    def test_batch_predict_empty_list(self, client: TestClient):
        """
        Tests the batch prediction endpoint with empty text list.
        """
        payload = {"texts": []}
        response = client.post("/api/v1/classifier/predict/batch", json=payload)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_train_endpoint_success(self, client: TestClient, mock_model_loading):
        """
        Tests the /train endpoint with valid training data.
        Verifies that a training job is successfully created.
        """
        payload = {
            "model_type": "logistic_regression",
            "training_data": {
                "X_train": [["feature1"], ["feature2"]],
                "y_train": [0, 1],
                "X_val": [["feature3"]],
                "y_val": [0]
            }
        }
        response = client.post("/api/v1/classifier/train", json=payload)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "training_id" in data
        assert "status" in data
        assert "message" in data
        assert "model_type" in data

        # Verify response values
        assert data["status"] == "started"
        assert data["model_type"] == "logistic_regression"

    def test_get_metrics_endpoint_success(self, client: TestClient, mock_model_loading):
        """
        Tests the /metrics endpoint for getting model performance metrics.
        """
        with patch('app.services.classifier_service.ClassifierService.get_model_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "model_type": "ensemble",
                "metrics_available": True,
                "accuracy": 0.95,
                "precision": 0.93,
                "recall": 0.97,
                "f1_score": 0.95
            }

            response = client.get("/api/v1/classifier/metrics?model_type=ensemble")

            assert response.status_code == 200
            data = response.json()

            assert "model_type" in data
            assert "metrics" in data