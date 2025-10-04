# backend/app/models/classifier.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from app.models.common import CommonBaseModel, ResponseMetadata


# ---------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------
class TextClassificationRequest(CommonBaseModel):
    text: str = Field(..., description="Input text to classify")
    model_type: str = Field(
        "ensemble", description="Model to use: ['ensemble', 'bert', 'lstm', 'random_forest', 'gradient_boosting', 'logistic_regression', 'naive_bayes']"
    )
    return_confidence: bool = Field(True, description="Return probability/confidence scores")
    explain_prediction: bool = Field(False, description="Include explanation (e.g., feature importance)")


class TrainingRequest(CommonBaseModel):
    training_data: List[Dict[str, Any]] = Field(
        ..., description="Labeled dataset [{'text': str, 'label': int}]"
    )
    model_hyperparameters: Optional[Dict[str, Any]] = Field(
        None, description="Optional model hyperparameters"
    )
    validation_split: float = Field(
        0.2, ge=0.05, le=0.5, description="Fraction of data reserved for validation"
    )


# ---------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------
class TextClassificationResponse(CommonBaseModel):
    text: str
    prediction: str = Field(..., description="Predicted label: fake or real")
    confidence: Optional[float] = Field(None, description="Confidence score of prediction")
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Class probability distribution {label: prob}"
    )
    explanation: Optional[Dict[str, Any]] = Field(
        None, description="Explanation of prediction (e.g., feature attributions)"
    )
    model_used: str
    processing_time: float
    metadata: Optional[ResponseMetadata] = None


class BatchClassificationRequest(CommonBaseModel):
    texts: List[str] = Field(..., description="List of texts to classify")
    model_type: str = Field(
        "ensemble", description="Model to use: ['ensemble', 'bert', 'lstm', 'random_forest', 'gradient_boosting', 'logistic_regression', 'naive_bayes']"
    )
    return_confidence: bool = Field(True, description="Return probability/confidence scores")


class BatchClassificationResponse(CommonBaseModel):
    results: List[Dict[str, Any]] = Field(..., description="List of classification results")
    total_processed: int = Field(..., description="Total number of texts processed")
    model_used: str = Field(..., description="Model used for classification")
    metadata: Optional[ResponseMetadata] = None


class TrainingResponse(CommonBaseModel):
    training_id: str = Field(..., description="Unique identifier for training job")
    status: str = Field(..., description="Training status: started, running, completed, failed")
    message: str = Field(..., description="Status message")
    model_type: str = Field(..., description="Type of model being trained")
    metadata: Optional[ResponseMetadata] = None


class ModelMetricsResponse(CommonBaseModel):
    model_type: str = Field(..., description="Type of model")
    metrics: Dict[str, Any] = Field(..., description="Model performance metrics")
    metadata: Optional[ResponseMetadata] = None


class ModelInfoResponse(CommonBaseModel):
    model_type: Optional[str] = Field(None, description="Specific model type or None for all")
    info: Dict[str, Any] = Field(..., description="Model information")
    available_models: List[str] = Field(..., description="List of available model types")
    metadata: Optional[ResponseMetadata] = None