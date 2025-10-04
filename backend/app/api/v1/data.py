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
import pandas as pd

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
        # Import SimulationService to get results
        from app.services.simulation_service import SimulationService

        simulation_service = SimulationService()

        # Get simulation results
        try:
            results = await simulation_service.get_simulation_results(
                simulation_id=simulation_id,
                include_details=include_details
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        if format == "csv":
            # Create CSV response with simulation data
            output = io.StringIO()
            writer = csv.writer(output)

            # Write metadata section
            writer.writerow(["Simulation Results Export"])
            writer.writerow(["Simulation ID", simulation_id])
            writer.writerow(["Timestamp", results.get('timestamp', '')])
            writer.writerow(["Total Rounds", results.get('total_rounds', 0)])
            writer.writerow([])

            # Write network metrics section
            writer.writerow(["Network Metrics"])
            network_metrics = results.get('network_metrics', {})
            writer.writerow(["Metric", "Value"])
            for key, value in network_metrics.items():
                writer.writerow([key, value])
            writer.writerow([])

            # Write final payoffs section
            writer.writerow(["Final Payoffs"])
            final_payoffs = results.get('final_metrics', {}).get('final_payoffs', {})
            writer.writerow(["Player Type", "Final Payoff"])
            for player_type, payoff in final_payoffs.items():
                writer.writerow([player_type, payoff])
            writer.writerow([])

            # Write payoff trends section
            writer.writerow(["Payoff Trends Over Time"])
            payoff_trends = results.get('payoff_trends', {})
            if payoff_trends:
                # Write header with player types
                header = ["Round"] + list(payoff_trends.keys())
                writer.writerow(header)

                # Find the maximum number of rounds
                max_rounds = max(len(trends) for trends in payoff_trends.values())

                # Write payoff data for each round
                for round_num in range(max_rounds):
                    row = [round_num + 1]
                    for player_type in payoff_trends.keys():
                        if round_num < len(payoff_trends[player_type]):
                            row.append(payoff_trends[player_type][round_num])
                        else:
                            row.append('')
                    writer.writerow(row)
            writer.writerow([])

            # Write convergence analysis
            writer.writerow(["Convergence Analysis"])
            convergence = results.get('convergence_analysis', {})
            writer.writerow(["Status", convergence.get('status', 'unknown')])
            writer.writerow(["Converged", convergence.get('converged', False)])
            writer.writerow([])

            # Write detailed results if requested
            if include_details and 'raw_results' in results:
                writer.writerow(["Detailed Round Results"])
                writer.writerow(["Round", "Player Type", "Payoff", "Action", "Reputation"])

                for idx, round_result in enumerate(results['raw_results']):
                    round_num = idx + 1
                    for player_type, payoff in round_result.get('payoffs', {}).items():
                        reputation = round_result.get('player_states', {}).get(player_type, {}).get('reputation', 'N/A')
                        action = round_result.get('actions', {}).get(player_type, 'N/A')
                        writer.writerow([round_num, player_type, payoff, action, reputation])

            output.seek(0)

            return StreamingResponse(
                io.BytesIO(output.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=simulation_{simulation_id}_results.csv"}
            )

        elif format == "json":
            # Create JSON response with full simulation data
            export_data = {
                "simulation_id": simulation_id,
                "timestamp": results.get('timestamp', ''),
                "parameters": results.get('parameters', {}),
                "total_rounds": results.get('total_rounds', 0),
                "network_metrics": results.get('network_metrics', {}),
                "final_metrics": results.get('final_metrics', {}),
                "payoff_trends": results.get('payoff_trends', {}),
                "convergence_analysis": results.get('convergence_analysis', {})
            }

            # Include detailed results if requested
            if include_details and 'raw_results' in results:
                export_data["raw_results"] = results['raw_results']

            json_content = json.dumps(export_data, indent=2, default=str)

            return StreamingResponse(
                io.BytesIO(json_content.encode('utf-8')),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=simulation_{simulation_id}_results.json"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export simulation results: {str(e)}")

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
        stats = await data_service.get_dataset_statistics(dataset_id)

        # Retrieve the actual dataset from the cache
        dataset = data_service.dataset_cache.get(dataset_id)

        if dataset is None:
            # Try to load from storage if not in cache
            dataset_file = data_service.data_storage_path / f"{dataset_id}_data.csv"
            if dataset_file.exists():
                import pandas as pd
                dataset = pd.read_csv(str(dataset_file))
                data_service._add_to_cache(dataset_id, dataset)
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset {dataset_id} data not found in cache or storage"
                )

        if format == "csv":
            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(["text", "label"])

            # Write all dataset rows
            for _, row in dataset.iterrows():
                writer.writerow([row.get('text', ''), row.get('label', '')])

            output.seek(0)

            return StreamingResponse(
                io.BytesIO(output.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=dataset_{dataset_id}.csv"}
            )

        elif format == "json":
            # Convert dataset to JSON format
            data_list = []
            for _, row in dataset.iterrows():
                data_list.append({
                    "text": row.get('text', ''),
                    "label": int(row.get('label', 0)) if pd.notna(row.get('label')) else None
                })

            export_data = {
                "dataset_id": dataset_id,
                "num_samples": len(dataset),
                "statistics": stats,
                "data": data_list
            }

            json_content = json.dumps(export_data, indent=2, default=str)

            return StreamingResponse(
                io.BytesIO(json_content.encode('utf-8')),
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
        raise HTTPException(status_code=500, detail=f"Failed to export dataset: {str(e)}")