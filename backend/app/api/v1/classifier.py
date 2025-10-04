# backend/app/api/v1/classifier.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from typing import Dict, Any, List, Optional
import logging
import uuid

from app.services.classifier_service import ClassifierService
from app.models.classifier import (
    TextClassificationRequest,
    TextClassificationResponse,
    BatchClassificationRequest,
    BatchClassificationResponse,
    TrainingRequest,
    TrainingResponse,
    ModelMetricsResponse,
    ModelInfoResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get classifier service instance from app state
def get_classifier_service(request: Request) -> ClassifierService:
    """Get classifier service instance from app state."""
    return request.app.state.classifier_service

@router.post("/predict", response_model=TextClassificationResponse)
async def predict_fake_news(
    request: TextClassificationRequest,
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """
    Classify a single piece of text as real or fake news.

    Args:
        request: Text classification request containing text and optional model type

    Returns:
        Classification results with prediction, confidence, and probabilities
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text field cannot be empty.")

        prediction_result = await classifier_service.predict(
            text=request.text,
            model_type=request.model_type
        )

        return TextClassificationResponse(**prediction_result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@router.post("/predict/batch", response_model=BatchClassificationResponse)
async def predict_batch(
    request: BatchClassificationRequest,
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """
    Classify multiple pieces of text as real or fake news.

    Args:
        request: Batch classification request containing list of texts

    Returns:
        Batch classification results
    """
    try:
        if not request.texts or len(request.texts) == 0:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty.")

        if len(request.texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 texts.")

        batch_results = await classifier_service.predict_batch(
            texts=request.texts,
            model_type=request.model_type
        )

        return BatchClassificationResponse(
            results=batch_results,
            total_processed=len(batch_results),
            model_used=request.model_type
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")

@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """
    Start training a machine learning model. This is a long-running task
    that will be executed in the background.

    Args:
        request: Training configuration and data
        background_tasks: FastAPI background tasks

    Returns:
        Training job ID and status
    """
    try:
        # Generate unique training ID
        training_id = str(uuid.uuid4())

        # Add training task to background
        background_tasks.add_task(
            _background_training_task,
            classifier_service,
            request.model_type,
            request.training_data,
            training_id
        )

        logger.info(f"Training job {training_id} started for model {request.model_type}")

        return TrainingResponse(
            training_id=training_id,
            status="started",
            message=f"Training started for model {request.model_type}",
            model_type=request.model_type
        )

    except Exception as e:
        logger.error(f"Training initialization error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start training")

async def _background_training_task(
    classifier_service: ClassifierService,
    model_type: str,
    training_data: Dict[str, Any],
    training_id: str
):
    """Background task for model training."""
    try:
        logger.info(f"Starting background training for job {training_id}")

        result = await classifier_service.train_model(
            model_type=model_type,
            training_data=training_data,
            save_model=True
        )

        logger.info(f"Training job {training_id} completed successfully")

    except Exception as e:
        logger.error(f"Training job {training_id} failed: {e}")

@router.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(
    model_type: Optional[str] = "ensemble",
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """
    Get evaluation metrics for a specific model.

    Args:
        model_type: Type of model to get metrics for

    Returns:
        Model performance metrics
    """
    try:
        metrics = await classifier_service.get_model_metrics(model_type)

        return ModelMetricsResponse(
            model_type=model_type,
            metrics=metrics
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model metrics")

@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(
    model_type: Optional[str] = None,
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """
    Get information about loaded models.

    Args:
        model_type: Specific model to get info for, or None for all models

    Returns:
        Model information including training status and capabilities
    """
    try:
        info = await classifier_service.get_model_info(model_type)

        return ModelInfoResponse(
            model_type=model_type,
            info=info,
            available_models=classifier_service.get_available_models()
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

@router.post("/reload/{model_type}")
async def reload_model(
    model_type: str,
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """
    Reload a specific model from disk.

    Args:
        model_type: Type of model to reload

    Returns:
        Success status
    """
    try:
        success = await classifier_service.reload_model(model_type)

        if success:
            return {"message": f"Model {model_type} reloaded successfully", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to reload model {model_type}")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload model")

@router.get("/models/available")
async def get_available_models(
    classifier_service: ClassifierService = Depends(get_classifier_service)
):
    """
    Get list of available model types.

    Returns:
        List of available models and their readiness status
    """
    try:
        available_models = classifier_service.get_available_models()
        model_status = {}

        for model_type in available_models:
            model_status[model_type] = {
                "ready": classifier_service.is_model_ready(model_type),
                "available": True
            }

        return {
            "available_models": available_models,
            "model_status": model_status,
            "total_models": len(available_models)
        }

    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available models")