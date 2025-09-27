# backend/app/api/v1/analysis.py

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging

from app.services.network_service import NetworkService
from app.models.analysis import (
    NetworkGenerationRequest,
    NetworkGenerationResponse,
    NetworkMetricsResponse,
    NetworkVisualizationResponse,
    PropagationSimulationRequest,
    PropagationSimulationResponse,
    NetworkComparisonResponse,
    NetworkListResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get network service instance
def get_network_service():
    """Get network service instance."""
    return NetworkService()

@router.post("/network/generate", response_model=NetworkGenerationResponse)
async def generate_network(
    request: NetworkGenerationRequest,
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Generate a new social network with specified parameters.

    Args:
        request: Network generation configuration

    Returns:
        Network ID and generation confirmation
    """
    try:
        # Convert Pydantic model to dict
        network_config = request.dict()

        # Generate network
        network_id = await network_service.generate_network(network_config)

        return NetworkGenerationResponse(
            network_id=network_id,
            status="generated",
            message=f"Network generated successfully with {request.num_nodes} nodes",
            configuration=network_config
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Network generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate network")

@router.get("/network/metrics", response_model=NetworkMetricsResponse)
async def get_network_metrics(
    graph_id: str = Query("default", description="Network identifier"),
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Get detailed structural metrics for a specified network graph.

    Args:
        graph_id: Identifier of the network to analyze

    Returns:
        Comprehensive network metrics including centrality and community structure
    """
    try:
        metrics = await network_service.get_network_metrics(graph_id)

        return NetworkMetricsResponse(
            network_id=graph_id,
            metrics=metrics,
            analysis_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting network metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute network metrics")

@router.get("/network/visualize", response_model=NetworkVisualizationResponse)
async def get_network_for_visualization(
    graph_id: str = Query("default", description="Network identifier"),
    layout_algorithm: str = Query("spring", description="Layout algorithm: spring, circular, random"),
    max_nodes: int = Query(1000, ge=10, le=5000, description="Maximum nodes to include"),
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Get network data formatted for frontend visualization libraries.

    Args:
        graph_id: Identifier of the network
        layout_algorithm: Algorithm for node positioning
        max_nodes: Maximum number of nodes to include

    Returns:
        Network visualization data with nodes and edges
    """
    try:
        viz_data = await network_service.get_visualization_data(
            network_id=graph_id,
            layout_algorithm=layout_algorithm,
            max_nodes=max_nodes
        )

        return NetworkVisualizationResponse(
            network_id=graph_id,
            visualization_data=viz_data,
            layout_algorithm=layout_algorithm,
            max_nodes=max_nodes
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating network visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate network visualization")

@router.post("/propagation/simulate", response_model=PropagationSimulationResponse)
async def simulate_information_propagation(
    request: PropagationSimulationRequest,
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Simulate information propagation through a social network.

    Args:
        request: Propagation simulation configuration

    Returns:
        Simulation results including spread metrics and timeline
    """
    try:
        # Convert Pydantic model to dict
        propagation_config = request.dict()
        network_id = propagation_config.pop('network_id')

        # Run propagation simulation
        results = await network_service.simulate_propagation(
            network_id=network_id,
            propagation_config=propagation_config
        )

        return PropagationSimulationResponse(
            network_id=network_id,
            simulation_results=results,
            simulation_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Propagation simulation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to simulate propagation")

@router.get("/propagation/visualize")
async def get_propagation_visualization(
    network_id: str = Query(..., description="Network identifier"),
    simulation_id: str = Query(..., description="Simulation identifier"),
    layout_algorithm: str = Query("spring", description="Layout algorithm"),
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Get visualization data for propagation simulation results.

    Args:
        network_id: Identifier of the network
        simulation_id: Identifier of the propagation simulation
        layout_algorithm: Algorithm for node positioning

    Returns:
        Propagation visualization data
    """
    try:
        # Note: This would need to be extended to store and retrieve simulation results
        # For now, return placeholder response
        return {
            "message": "Propagation visualization endpoint",
            "network_id": network_id,
            "simulation_id": simulation_id,
            "status": "placeholder"
        }

    except Exception as e:
        logger.error(f"Error generating propagation visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate propagation visualization")

@router.get("/network/community/visualize")
async def get_community_visualization(
    graph_id: str = Query("default", description="Network identifier"),
    community_method: str = Query("louvain", description="Community detection method"),
    layout_algorithm: str = Query("spring", description="Layout algorithm"),
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Get visualization data highlighting community structure in the network.

    Args:
        graph_id: Identifier of the network
        community_method: Community detection algorithm to use
        layout_algorithm: Algorithm for node positioning

    Returns:
        Community visualization data with color-coded communities
    """
    try:
        viz_data = await network_service.get_community_visualization(
            network_id=graph_id,
            community_method=community_method,
            layout_algorithm=layout_algorithm
        )

        return {
            "network_id": graph_id,
            "visualization_data": viz_data,
            "community_method": community_method,
            "layout_algorithm": layout_algorithm
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating community visualization: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate community visualization")

@router.post("/network/compare", response_model=NetworkComparisonResponse)
async def compare_networks(
    network_ids: List[str] = Query(..., description="List of network IDs to compare"),
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Compare structural properties across multiple networks.

    Args:
        network_ids: List of network identifiers to compare

    Returns:
        Comparison results highlighting similarities and differences
    """
    try:
        if len(network_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 networks required for comparison")

        if len(network_ids) > 10:
            raise HTTPException(status_code=400, detail="Cannot compare more than 10 networks at once")

        comparison_results = await network_service.compare_networks(network_ids)

        return NetworkComparisonResponse(
            network_ids=network_ids,
            comparison_results=comparison_results,
            comparison_successful=True
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Network comparison error: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare networks")

@router.get("/network/list", response_model=NetworkListResponse)
async def list_available_networks(
    network_service: NetworkService = Depends(get_network_service)
):
    """
    List all available networks with basic metadata.

    Returns:
        List of networks with node/edge counts and availability status
    """
    try:
        networks_data = await network_service.list_networks()

        return NetworkListResponse(
            networks=networks_data["networks"],
            total_count=networks_data["total_count"],
            cached_count=networks_data["cached_count"]
        )

    except Exception as e:
        logger.error(f"Error listing networks: {e}")
        raise HTTPException(status_code=500, detail="Failed to list networks")

@router.delete("/network/{network_id}")
async def delete_network(
    network_id: str,
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Delete a network from storage and cache.

    Args:
        network_id: Identifier of the network to delete

    Returns:
        Deletion confirmation
    """
    try:
        success = await network_service.delete_network(network_id)

        if success:
            return {
                "message": f"Network {network_id} deleted successfully",
                "network_id": network_id,
                "status": "deleted"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to delete network")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting network: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete network")

@router.get("/network/statistics")
async def get_network_statistics(
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Get overall statistics about available networks and system status.

    Returns:
        System statistics and network overview
    """
    try:
        networks_data = await network_service.list_networks()

        return {
            "total_networks": networks_data["total_count"],
            "cached_networks": networks_data["cached_count"],
            "storage_networks": networks_data["total_count"] - networks_data["cached_count"],
            "system_status": "active"
        }

    except Exception as e:
        logger.error(f"Error getting network statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve network statistics")