# backend/app/api/v1/simulation.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, Optional
import logging

from app.services.simulation_service import SimulationService
from app.models.simulation import (
    SimulationRequest,
    SimulationResponse,
    SimulationStatusResponse,
    SimulationResultsResponse,
    SimulationListResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get simulation service instance
def get_simulation_service():
    """Get simulation service instance."""
    return SimulationService()

@router.post("/run", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """
    Start a new game theory simulation. This is a long-running task
    that will be executed in the background.

    Args:
        request: Simulation configuration including network and game parameters
        background_tasks: FastAPI background tasks

    Returns:
        Simulation ID and initial status
    """
    try:
        # Convert Pydantic model to dict
        params = request.dict()

        # Start simulation
        simulation_id = await simulation_service.start_simulation(params)

        logger.info(f"Simulation {simulation_id} started")

        return SimulationResponse(
            simulation_id=simulation_id,
            status="started",
            message="Simulation started successfully",
            created_at=None  # Will be filled by the service
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to start simulation")

@router.get("/status/{simulation_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(
    simulation_id: str,
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """
    Get the current status of a running or completed simulation.

    Args:
        simulation_id: Unique identifier for the simulation

    Returns:
        Current simulation status, progress, and metadata
    """
    try:
        status_info = await simulation_service.get_simulation_status(simulation_id)

        return SimulationStatusResponse(**status_info)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get simulation status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve simulation status")

@router.get("/results/{simulation_id}", response_model=SimulationResultsResponse)
async def get_simulation_results(
    simulation_id: str,
    include_details: bool = Query(True, description="Include detailed round-by-round results"),
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """
    Retrieve the results of a completed simulation.

    Args:
        simulation_id: Unique identifier for the simulation
        include_details: Whether to include detailed results (affects response size)

    Returns:
        Complete simulation results including metrics and analysis
    """
    try:
        results = await simulation_service.get_simulation_results(
            simulation_id, include_details=include_details
        )

        return SimulationResultsResponse(
            simulation_id=simulation_id,
            results=results,
            include_details=include_details
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get simulation results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve simulation results")

@router.get("/list", response_model=SimulationListResponse)
async def list_simulations(
    status: Optional[str] = Query(None, description="Filter by status: pending, running, completed, failed"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of simulations to return"),
    offset: int = Query(0, ge=0, description="Number of simulations to skip"),
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """
    List simulations with optional filtering and pagination.

    Args:
        status: Optional status filter
        limit: Maximum number of results to return
        offset: Number of results to skip for pagination

    Returns:
        List of simulations with metadata and pagination info
    """
    try:
        # Validate status if provided
        valid_statuses = ['pending', 'running', 'completed', 'failed', 'cancelled']
        if status and status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )

        simulation_list = await simulation_service.list_simulations(
            status=status, limit=limit, offset=offset
        )

        return SimulationListResponse(**simulation_list)

    except Exception as e:
        logger.error(f"Failed to list simulations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve simulation list")

@router.post("/cancel/{simulation_id}")
async def cancel_simulation(
    simulation_id: str,
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """
    Cancel a running simulation.

    Args:
        simulation_id: Unique identifier for the simulation

    Returns:
        Cancellation confirmation
    """
    try:
        success = await simulation_service.cancel_simulation(simulation_id)

        if success:
            return {
                "message": f"Simulation {simulation_id} cancelled successfully",
                "simulation_id": simulation_id,
                "status": "cancelled"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel simulation")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel simulation")

@router.delete("/delete/{simulation_id}")
async def delete_simulation(
    simulation_id: str,
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """
    Delete simulation data and results.

    Args:
        simulation_id: Unique identifier for the simulation

    Returns:
        Deletion confirmation
    """
    try:
        success = await simulation_service.delete_simulation(simulation_id)

        if success:
            return {
                "message": f"Simulation {simulation_id} deleted successfully",
                "simulation_id": simulation_id,
                "status": "deleted"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to delete simulation")

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete simulation")

@router.get("/statistics")
async def get_simulation_statistics(
    simulation_service: SimulationService = Depends(get_simulation_service)
):
    """
    Get overall simulation statistics and system info.

    Returns:
        Statistics about simulations and system status
    """
    try:
        stats = simulation_service.get_simulation_statistics()
        return {
            "statistics": stats,
            "status": "active"
        }

    except Exception as e:
        logger.error(f"Failed to get simulation statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve simulation statistics")