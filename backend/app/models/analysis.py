# backend/app/models/analysis.py

from typing import List, Dict, Optional, Any
from pydantic import Field
from app.models.common import CommonBaseModel, ResponseMetadata


# ---------------------------------------------------------
# Network Generation Requests/Responses
# ---------------------------------------------------------
class NetworkGenerationRequest(CommonBaseModel):
    network_type: str = Field(
        "barabasi_albert",
        description="Type of network: barabasi_albert, watts_strogatz, erdos_renyi"
    )
    num_nodes: int = Field(1000, gt=0, le=10000, description="Number of nodes in the network")
    attachment_preference: int = Field(5, gt=0, description="Attachment preference for BA networks")
    rewiring_probability: float = Field(0.1, ge=0, le=1, description="Rewiring probability for WS networks")
    edge_probability: float = Field(0.01, ge=0, le=1, description="Edge probability for ER networks")
    k_neighbors: int = Field(10, gt=0, description="Number of neighbors for WS networks")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class NetworkGenerationResponse(CommonBaseModel):
    network_id: str = Field(..., description="Unique identifier for the generated network")
    status: str = Field(..., description="Generation status")
    message: str = Field(..., description="Success message")
    configuration: Dict[str, Any] = Field(..., description="Configuration used for generation")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Network Metrics Requests/Responses
# ---------------------------------------------------------
class NetworkMetricsResponse(CommonBaseModel):
    network_id: str = Field(..., description="Network identifier")
    metrics: Dict[str, Any] = Field(..., description="Comprehensive network metrics")
    analysis_successful: bool = Field(..., description="Whether analysis completed successfully")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Network Visualization Requests/Responses
# ---------------------------------------------------------
class NetworkVisualizationResponse(CommonBaseModel):
    network_id: str = Field(..., description="Network identifier")
    visualization_data: Dict[str, Any] = Field(..., description="Visualization data for frontend")
    layout_algorithm: str = Field(..., description="Layout algorithm used")
    max_nodes: int = Field(..., description="Maximum nodes included")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Information Propagation Requests/Responses
# ---------------------------------------------------------
class PropagationSimulationRequest(CommonBaseModel):
    network_id: str = Field(..., description="Network to simulate propagation on")
    base_transmission_rate: float = Field(0.1, ge=0, le=1, description="Base probability of transmission")
    time_steps: int = Field(50, gt=0, le=1000, description="Number of simulation time steps")
    recovery_rate: float = Field(0.05, ge=0, le=1, description="Rate at which nodes recover/become immune")
    content_type: str = Field("neutral", description="Type of content: fake, real, neutral")
    content_quality: float = Field(0.5, ge=0, le=1, description="Quality/credibility of content")
    virality_factor: float = Field(1.0, ge=0, le=5, description="Virality multiplier")
    emotional_appeal: float = Field(0.5, ge=0, le=1, description="Emotional appeal of content")
    model: str = Field("independent_cascade", description="Propagation model to use")
    num_initial_spreaders: int = Field(5, gt=0, description="Number of initial spreaders")
    initial_spreaders: List[str] = Field(default_factory=list, description="Specific initial spreader nodes")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class PropagationSimulationResponse(CommonBaseModel):
    network_id: str = Field(..., description="Network identifier")
    simulation_results: Dict[str, Any] = Field(..., description="Complete simulation results")
    simulation_successful: bool = Field(..., description="Whether simulation completed successfully")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Network Comparison Requests/Responses
# ---------------------------------------------------------
class NetworkComparisonResponse(CommonBaseModel):
    network_ids: List[str] = Field(..., description="Networks compared")
    comparison_results: Dict[str, Any] = Field(..., description="Comparison analysis results")
    comparison_successful: bool = Field(..., description="Whether comparison completed successfully")
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Network List Responses
# ---------------------------------------------------------
class NetworkSummary(CommonBaseModel):
    network_id: str = Field(..., description="Network identifier")
    num_nodes: Any = Field(..., description="Number of nodes (int or 'unknown')")
    num_edges: Any = Field(..., description="Number of edges (int or 'unknown')")
    cached: bool = Field(..., description="Whether network is currently cached in memory")


class NetworkListResponse(CommonBaseModel):
    networks: List[NetworkSummary] = Field(..., description="List of available networks")
    total_count: int = Field(..., description="Total number of networks")
    cached_count: int = Field(..., description="Number of networks currently cached")
    metadata: Optional[ResponseMetadata] = None