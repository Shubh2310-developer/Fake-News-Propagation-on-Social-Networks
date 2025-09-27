# backend/app/api/v1/data.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, FileResponse
from typing import Dict, Any, List, Optional
import logging
import tempfile
import csv
import io
import json
from pathlib import Path

from app.services.data_service import DataService
from app.models.data import (
    DatasetUploadResponse,
    SyntheticDatasetRequest,
    DatasetResponse,
    DatasetStatisticsResponse,
    DataProcessingRequest,
    DataProcessingResponse,
    CrossValidationRequest,
    CrossValidationResponse,
    DatasetListResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get data service instance
def get_data_service():
    """Get data service instance."""
    return DataService()

@router.post("/upload/dataset", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = Query("csv", description="Dataset type: csv, json"),
    data_service: DataService = Depends(get_data_service)
):
    """
    Upload a new dataset (e.g., in CSV or JSON format) for model training.

    Args:
        file: Uploaded dataset file
        dataset_type: Type of the dataset file

    Returns:
        Upload confirmation with dataset ID
    """
    try:
        # Validate file type
        if dataset_type not in ["csv", "json"]:
            raise HTTPException(status_code=400, detail="Unsupported dataset type")

        # Validate file size (limit to 100MB)
        if file.size and file.size > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large (max 100MB)")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{dataset_type}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()

            # Load the dataset
            dataset_id = await data_service.load_dataset(
                source=tmp_file.name,
                dataset_type=dataset_type
            )

        # Clean up temporary file
        Path(tmp_file.name).unlink()

        logger.info(f"Dataset uploaded successfully: {dataset_id}")

        return DatasetUploadResponse(
            dataset_id=dataset_id,
            filename=file.filename,
            size_bytes=file.size or 0,
            dataset_type=dataset_type,
            status="uploaded",
            message="Dataset uploaded and processed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload dataset")

@router.post("/generate/synthetic", response_model=DatasetResponse)
async def generate_synthetic_dataset(
    request: SyntheticDatasetRequest,
    data_service: DataService = Depends(get_data_service)
):
    """
    Generate a synthetic dataset for testing and experimentation.

    Args:
        request: Configuration for synthetic data generation

    Returns:
        Generated dataset information
    """
    try:
        # Convert Pydantic model to dict
        config = request.dict()

        # Generate synthetic dataset
        dataset_id = await data_service.load_dataset(
            source=config,
            dataset_type='synthetic'
        )

        # Get dataset statistics
        stats = await data_service.get_dataset_statistics(dataset_id)

        return DatasetResponse(
            dataset_id=dataset_id,
            status="generated",
            message="Synthetic dataset generated successfully",
            statistics=stats
        )

    except Exception as e:
        logger.error(f"Synthetic dataset generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate synthetic dataset")

@router.get("/dataset/{dataset_id}", response_model=DatasetResponse)
async def get_dataset_info(
    dataset_id: str,
    data_service: DataService = Depends(get_data_service)
):
    """
    Get information and statistics about a specific dataset.

    Args:
        dataset_id: Identifier of the dataset

    Returns:
        Dataset information including statistics and metadata
    """
    try:
        stats = await data_service.get_dataset_statistics(dataset_id)

        return DatasetResponse(
            dataset_id=dataset_id,
            status="available",
            message="Dataset information retrieved successfully",
            statistics=stats
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dataset information")

@router.get("/dataset/{dataset_id}/statistics", response_model=DatasetStatisticsResponse)
async def get_dataset_statistics(
    dataset_id: str,
    data_service: DataService = Depends(get_data_service)
):
    """
    Get detailed statistics for a specific dataset.

    Args:
        dataset_id: Identifier of the dataset

    Returns:
        Comprehensive dataset statistics
    """
    try:
        stats = await data_service.get_dataset_statistics(dataset_id)

        return DatasetStatisticsResponse(
            dataset_id=dataset_id,
            statistics=stats,
            analysis_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting dataset statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dataset statistics")

@router.post("/process/training", response_model=DataProcessingResponse)
async def process_data_for_training(
    request: DataProcessingRequest,
    data_service: DataService = Depends(get_data_service)
):
    """
    Process a dataset for machine learning training.
    This includes text cleaning, feature extraction, and train/test splitting.

    Args:
        request: Data processing configuration

    Returns:
        Processed data information and split statistics
    """
    try:
        processed_data = await data_service.get_processed_data_for_training(
            dataset_id=request.dataset_id,
            test_size=request.test_size,
            validation_size=request.validation_size,
            random_state=request.random_state,
            stratify=request.stratify
        )

        # Remove actual data arrays from response (too large for API)
        response_data = processed_data.copy()
        for split_name in ['train', 'test', 'validation']:
            if split_name in response_data['data_splits']:
                split_data = response_data['data_splits'][split_name]
                # Keep only metadata, remove actual arrays
                response_data['data_splits'][split_name] = {
                    'size': split_data['size']
                }

        return DataProcessingResponse(
            dataset_id=request.dataset_id,
            processing_info=response_data,
            processing_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Data processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process data for training")

@router.post("/cross-validation/splits", response_model=CrossValidationResponse)
async def create_cross_validation_splits(
    request: CrossValidationRequest,
    data_service: DataService = Depends(get_data_service)
):
    """
    Create cross-validation splits for a dataset.

    Args:
        request: Cross-validation configuration

    Returns:
        Cross-validation splits information
    """
    try:
        cv_splits = await data_service.create_cross_validation_splits(
            dataset_id=request.dataset_id,
            n_splits=request.n_splits,
            random_state=request.random_state,
            stratify=request.stratify
        )

        return CrossValidationResponse(
            dataset_id=request.dataset_id,
            cv_splits=cv_splits,
            splits_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Cross-validation splits error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create cross-validation splits")

@router.get("/datasets/list", response_model=DatasetListResponse)
async def list_datasets(
    data_service: DataService = Depends(get_data_service)
):
    """
    List all available datasets with metadata.

    Returns:
        List of datasets with basic information
    """
    try:
        datasets_info = await data_service.list_datasets()

        return DatasetListResponse(
            datasets=datasets_info["datasets"],
            total_count=datasets_info["total_count"]
        )

    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail="Failed to list datasets")

@router.delete("/dataset/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    data_service: DataService = Depends(get_data_service)
):
    """
    Delete a dataset from storage and cache.

    Args:
        dataset_id: Identifier of the dataset to delete

    Returns:
        Deletion confirmation
    """
    try:
        success = await data_service.delete_dataset(dataset_id)

        if success:
            return {
                "message": f"Dataset {dataset_id} deleted successfully",
                "dataset_id": dataset_id,
                "status": "deleted"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to delete dataset")

    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete dataset")

@router.get("/export/results/{simulation_id}")
async def export_simulation_results(
    simulation_id: str,
    format: str = Query("csv", description="Export format: csv, json"),
    include_details: bool = Query(False, description="Include detailed results")
):
    """
    Export simulation results in various formats for download.

    Args:
        simulation_id: Identifier of the simulation
        format: Export format (csv or json)
        include_details: Whether to include detailed round-by-round data

    Returns:
        File download response
    """
    try:
        # This would need to be integrated with SimulationService
        # For now, return a placeholder response

        if format == "csv":
            # Create CSV response
            output = io.StringIO()
            writer = csv.writer(output)

            # Write headers
            writer.writerow(["simulation_id", "status", "message"])
            writer.writerow([simulation_id, "placeholder", "CSV export not fully implemented"])

            output.seek(0)

            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=simulation_{simulation_id}_results.csv"}
            )

        elif format == "json":
            # Create JSON response
            results = {
                "simulation_id": simulation_id,
                "status": "placeholder",
                "message": "JSON export not fully implemented",
                "include_details": include_details
            }

            json_content = json.dumps(results, indent=2)

            return StreamingResponse(
                io.BytesIO(json_content.encode()),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=simulation_{simulation_id}_results.json"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export simulation results")

@router.get("/export/dataset/{dataset_id}")
async def export_dataset(
    dataset_id: str,
    format: str = Query("csv", description="Export format: csv, json"),
    data_service: DataService = Depends(get_data_service)
):
    """
    Export a dataset for download.

    Args:
        dataset_id: Identifier of the dataset
        format: Export format

    Returns:
        File download response
    """
    try:
        # Get dataset statistics to verify it exists
        await data_service.get_dataset_statistics(dataset_id)

        # Create placeholder export (actual implementation would fetch real data)
        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)

            writer.writerow(["text", "label"])
            writer.writerow(["Sample text for dataset export", "0"])
            writer.writerow([f"Dataset {dataset_id} export placeholder", "1"])

            output.seek(0)

            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=dataset_{dataset_id}.csv"}
            )

        elif format == "json":
            data = {
                "dataset_id": dataset_id,
                "data": [
                    {"text": "Sample text for dataset export", "label": 0},
                    {"text": f"Dataset {dataset_id} export placeholder", "label": 1}
                ]
            }

            json_content = json.dumps(data, indent=2)

            return StreamingResponse(
                io.BytesIO(json_content.encode()),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=dataset_{dataset_id}.json"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export dataset")