# backend/app/models/data.py

from typing import List, Dict, Optional, Any
from pydantic import Field
from app.models.common import CommonBaseModel, ResponseMetadata


# ---------------------------------------------------------
# Dataset Upload Requests/Responses
# ---------------------------------------------------------
class DatasetUploadResponse(CommonBaseModel):
    dataset_id: str = Field(..., description="Unique identifier for the uploaded dataset")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="File size in bytes")
    dataset_type: str = Field(..., description="Type of dataset (csv, json)")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Synthetic Dataset Generation
# ---------------------------------------------------------
class SyntheticDatasetRequest(CommonBaseModel):
    num_samples: int = Field(1000, gt=0, le=100000, description="Number of samples to generate")
    fake_ratio: float = Field(0.3, ge=0, le=1, description="Proportion of fake news samples")
    random_seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    topics: Optional[List[str]] = Field(None, description="Custom topics for text generation")
    complexity: str = Field("medium", description="Text complexity: simple, medium, complex")


# ---------------------------------------------------------
# Dataset Information Responses
# ---------------------------------------------------------
class DatasetResponse(CommonBaseModel):
    dataset_id: str = Field(..., description="Dataset identifier")
    status: str = Field(..., description="Dataset status")
    message: str = Field(..., description="Status message")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Dataset statistics")
    metadata: Optional[ResponseMetadata] = None


class DatasetStatisticsResponse(CommonBaseModel):
    dataset_id: str = Field(..., description="Dataset identifier")
    statistics: Dict[str, Any] = Field(..., description="Comprehensive dataset statistics")
    analysis_successful: bool = Field(..., description="Whether analysis completed successfully")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Data Processing Requests/Responses
# ---------------------------------------------------------
class DataProcessingRequest(CommonBaseModel):
    dataset_id: str = Field(..., description="Dataset to process")
    test_size: float = Field(0.2, gt=0, lt=1, description="Proportion of data for testing")
    validation_size: float = Field(0.1, ge=0, lt=1, description="Proportion of data for validation")
    random_state: int = Field(42, description="Random seed for reproducibility")
    stratify: bool = Field(True, description="Whether to stratify splits by label")
    feature_extraction: Dict[str, Any] = Field(
        default_factory=dict, description="Feature extraction configuration"
    )


class DataProcessingResponse(CommonBaseModel):
    dataset_id: str = Field(..., description="Dataset identifier")
    processing_info: Dict[str, Any] = Field(..., description="Processing results and metadata")
    processing_successful: bool = Field(..., description="Whether processing completed successfully")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Cross Validation Requests/Responses
# ---------------------------------------------------------
class CrossValidationRequest(CommonBaseModel):
    dataset_id: str = Field(..., description="Dataset to create CV splits for")
    n_splits: int = Field(5, gt=1, le=20, description="Number of CV folds")
    random_state: int = Field(42, description="Random seed for reproducibility")
    stratify: bool = Field(True, description="Whether to stratify splits by label")


class CrossValidationResponse(CommonBaseModel):
    dataset_id: str = Field(..., description="Dataset identifier")
    cv_splits: Dict[str, Any] = Field(..., description="Cross-validation splits information")
    splits_successful: bool = Field(..., description="Whether splits were created successfully")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Dataset List Responses
# ---------------------------------------------------------
class DatasetSummary(CommonBaseModel):
    dataset_id: str = Field(..., description="Dataset identifier")
    num_samples: Any = Field(..., description="Number of samples (int or 'unknown')")
    cached: bool = Field(..., description="Whether dataset is currently cached")
    loaded_at: Optional[str] = Field(None, description="When dataset was loaded")


class DatasetListResponse(CommonBaseModel):
    datasets: List[DatasetSummary] = Field(..., description="List of available datasets")
    total_count: int = Field(..., description="Total number of datasets")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Export Requests/Responses
# ---------------------------------------------------------
class ExportRequest(CommonBaseModel):
    target_id: str = Field(..., description="ID of simulation or dataset to export")
    format: str = Field("csv", description="Export format: csv, json, xlsx")
    include_metadata: bool = Field(True, description="Whether to include metadata")
    compression: bool = Field(False, description="Whether to compress the output")


class ExportResponse(CommonBaseModel):
    target_id: str = Field(..., description="ID of exported item")
    format: str = Field(..., description="Export format used")
    file_size_bytes: int = Field(..., description="Size of exported file")
    download_url: str = Field(..., description="URL for downloading the file")
    expires_at: str = Field(..., description="When the download link expires")
    metadata: Optional[ResponseMetadata] = None