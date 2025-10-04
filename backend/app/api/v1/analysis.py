# backend/app/api/v1/analysis.py

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
import json

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

        # Generate a unique propagation ID for this simulation
        from datetime import datetime
        propagation_id = f"{network_id}_propagation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Save propagation results for later visualization
        propagation_storage_path = network_service.networks_storage_path / "propagations"
        propagation_storage_path.mkdir(exist_ok=True)

        propagation_file = propagation_storage_path / f"{propagation_id}_results.json"
        with open(propagation_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved propagation results to {propagation_file}")

        # Add propagation_id to results
        results['propagation_id'] = propagation_id

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
    propagation_id: str = Query(..., description="Propagation simulation identifier"),
    layout_algorithm: str = Query("spring", description="Layout algorithm"),
    time_step: Optional[int] = Query(None, description="Specific time step to visualize (None for all)"),
    network_service: NetworkService = Depends(get_network_service)
):
    """
    Get visualization data for propagation simulation results.

    Args:
        propagation_id: Identifier of the propagation simulation (format: network_id:timestamp)
        layout_algorithm: Algorithm for node positioning
        time_step: Optional specific time step to visualize

    Returns:
        Propagation visualization data with node states, edges, and timeline
    """
    try:
        # Parse propagation_id to extract network_id
        # Format is expected to be "network_id" or stored propagation results
        # For now, we'll look for stored propagation results in the network service

        # Check if propagation results are stored
        propagation_storage_path = network_service.networks_storage_path / "propagations"
        propagation_storage_path.mkdir(exist_ok=True)

        propagation_file = propagation_storage_path / f"{propagation_id}_results.json"

        if not propagation_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Propagation simulation {propagation_id} not found. Please run a propagation simulation first."
            )

        # Load propagation results
        with open(propagation_file, 'r') as f:
            propagation_data = json.load(f)

        network_id = propagation_data.get('network_id')
        propagation_results = propagation_data.get('results', {})

        # Get the network
        network = await network_service._get_network(network_id)
        if not network:
            raise HTTPException(
                status_code=404,
                detail=f"Network {network_id} not found"
            )

        # Generate visualization data
        visualization_data = await network_service.get_propagation_visualization(
            network_id=network_id,
            propagation_results=propagation_results,
            layout_algorithm=layout_algorithm
        )

        # Extract timeline data from propagation history
        propagation_history = propagation_results.get('propagation_history', [])
        timeline = []

        for step_data in propagation_history:
            timeline.append({
                'step': step_data.get('step', 0),
                'newly_infected': step_data.get('newly_infected', []),
                'total_infected': step_data.get('total_infected', 0)
            })

        # Build node states over time
        node_states = {}
        infected_nodes = set(propagation_results.get('final_infected_set', []))

        # Initialize all nodes as susceptible
        for node in network.nodes():
            node_states[str(node)] = {
                'state': 'susceptible',
                'infection_step': None,
                'attributes': {
                    'influence_score': network.nodes[node].get('influence_score', 0.5),
                    'credibility_score': network.nodes[node].get('credibility_score', 0.5),
                    'user_type': network.nodes[node].get('user_type', 'regular')
                }
            }

        # Update infection states based on propagation history
        for step_data in propagation_history:
            step_num = step_data.get('step', 0)
            for node in step_data.get('newly_infected', []):
                if str(node) in node_states:
                    node_states[str(node)]['state'] = 'infected'
                    node_states[str(node)]['infection_step'] = step_num

        # If time_step is specified, filter to that specific step
        if time_step is not None:
            filtered_timeline = [t for t in timeline if t['step'] == time_step]
            if not filtered_timeline:
                raise HTTPException(
                    status_code=400,
                    detail=f"Time step {time_step} not found in propagation history"
                )
            timeline = filtered_timeline

        # Build edge data with transmission information
        edges = []
        for edge in network.edges():
            source, target = edge
            edges.append({
                'source': str(source),
                'target': str(target),
                'trust': network.edges[edge].get('trust', 0.5),
                'interaction_strength': network.edges[edge].get('interaction_strength', 0.5),
                'transmitted': (
                    str(source) in node_states and
                    str(target) in node_states and
                    node_states[str(source)]['state'] == 'infected' and
                    node_states[str(target)]['state'] == 'infected' and
                    node_states[str(target)]['infection_step'] is not None and
                    node_states[str(source)]['infection_step'] is not None and
                    node_states[str(target)]['infection_step'] > node_states[str(source)]['infection_step']
                )
            })

        # Compile final response
        response = {
            "propagation_id": propagation_id,
            "network_id": network_id,
            "layout_algorithm": layout_algorithm,
            "visualization_data": visualization_data,
            "node_states": node_states,
            "edges": edges,
            "timeline": timeline,
            "summary": {
                "total_nodes": network.number_of_nodes(),
                "total_infected": len(infected_nodes),
                "infection_rate": len(infected_nodes) / network.number_of_nodes() if network.number_of_nodes() > 0 else 0,
                "total_steps": len(propagation_history),
                "model": propagation_results.get('model', 'unknown'),
                "content_type": propagation_results.get('content_properties', {}).get('content_type', 'unknown')
            },
            "metadata": {
                "simulation_timestamp": propagation_data.get('simulation_timestamp', ''),
                "configuration": propagation_data.get('configuration', {})
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating propagation visualization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate propagation visualization: {str(e)}")

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