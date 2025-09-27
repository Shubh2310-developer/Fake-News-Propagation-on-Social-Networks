# backend/app/models/simulation.py

from typing import List, Dict, Optional, Any
from pydantic import Field
from app.models.common import CommonBaseModel, ResponseMetadata


# ---------------------------------------------------------
# Simulation Requests
# ---------------------------------------------------------
class NetworkConfigRequest(CommonBaseModel):
    num_nodes: int = Field(1000, gt=0, le=10000, description="Number of nodes in network")
    network_type: str = Field("barabasi_albert", description="Type of network topology")
    attachment_preference: int = Field(5, gt=0, description="Attachment preference for BA networks")
    rewiring_probability: float = Field(0.1, ge=0, le=1, description="Rewiring probability for WS networks")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GameConfigRequest(CommonBaseModel):
    num_rounds: int = Field(100, gt=0, le=1000, description="Number of simulation rounds")
    num_spreaders: int = Field(10, gt=0, description="Number of fake news spreaders")
    num_fact_checkers: int = Field(5, gt=0, description="Number of fact checkers")
    num_platforms: int = Field(1, gt=0, description="Number of social media platforms")
    learning_rate: float = Field(0.1, gt=0, le=1, description="Learning rate for strategy updates")
    exploration_rate: float = Field(0.1, ge=0, le=1, description="Exploration rate for strategy selection")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SimulationRequest(CommonBaseModel):
    network_config: NetworkConfigRequest = Field(..., description="Network configuration parameters")
    game_config: GameConfigRequest = Field(..., description="Game theory configuration parameters")
    description: str = Field("", description="Optional description of the simulation")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing simulation")
    save_detailed_history: bool = Field(True, description="Save detailed round-by-round results")
    save_network: bool = Field(True, description="Save network structure")


class NetworkGenerationRequest(CommonBaseModel):
    network_type: str = Field(
        "barabasi_albert",
        description="Type of synthetic network: barabasi_albert, watts_strogatz, erdos_renyi, block_model"
    )
    num_nodes: int = Field(..., gt=0, le=10000)
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Extra parameters for network generation (e.g., attachment preference)"
    )


# ---------------------------------------------------------
# Simulation Responses
# ---------------------------------------------------------
class SimulationStatus(CommonBaseModel):
    simulation_id: str
    status: str = Field(..., description="running, completed, error, cancelled")
    progress: Optional[float] = Field(None, ge=0, le=1, description="Progress percentage")
    estimated_completion_time: Optional[float] = Field(None, description="Estimated seconds until completion")
    metadata: Optional[ResponseMetadata] = None


class SimulationStep(CommonBaseModel):
    time_step: int
    total_infected: int
    newly_infected: int
    infection_rate: float


class SimulationResponse(CommonBaseModel):
    simulation_id: str
    final_states: Dict[str, str] = Field(..., description="Node → state mapping (susceptible/infected)")
    propagation_log: List[SimulationStep]
    total_reach: int
    cascade_depth: int
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Network Response
# ---------------------------------------------------------
class NetworkNode(CommonBaseModel):
    id: str
    influence_score: float
    credibility_score: float
    activity_level: float
    user_type: str
    verified: bool


class NetworkEdge(CommonBaseModel):
    source: str
    target: str
    trust: float
    interaction_strength: float
    connection_type: str


class NetworkResponse(CommonBaseModel):
    network_id: str
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    properties: Dict[str, Any]
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Additional Simulation Response Models
# ---------------------------------------------------------
class SimulationResponse(CommonBaseModel):
    simulation_id: str = Field(..., description="Unique identifier for the simulation")
    status: str = Field(..., description="Initial status: started, pending")
    message: str = Field(..., description="Status message")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    metadata: Optional[ResponseMetadata] = None


class SimulationStatusResponse(CommonBaseModel):
    simulation_id: str = Field(..., description="Unique identifier for the simulation")
    status: str = Field(..., description="Current status: pending, running, completed, failed, cancelled")
    progress: float = Field(0.0, ge=0, le=1, description="Completion progress (0.0 to 1.0)")
    created_at: str = Field(..., description="Creation timestamp")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary information")
    metadata: Optional[ResponseMetadata] = None


class SimulationResultsResponse(CommonBaseModel):
    simulation_id: str = Field(..., description="Unique identifier for the simulation")
    results: Dict[str, Any] = Field(..., description="Complete simulation results")
    include_details: bool = Field(True, description="Whether detailed results are included")
    metadata: Optional[ResponseMetadata] = None


class SimulationSummary(CommonBaseModel):
    simulation_id: str = Field(..., description="Unique identifier")
    status: str = Field(..., description="Current status")
    created_at: str = Field(..., description="Creation timestamp")
    description: str = Field("", description="Simulation description")
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary data")


class SimulationListResponse(CommonBaseModel):
    simulations: List[SimulationSummary] = Field(..., description="List of simulations")
    total: int = Field(..., description="Total number of simulations matching filter")
    limit: int = Field(..., description="Maximum results per page")
    offset: int = Field(..., description="Number of results skipped")
    has_more: bool = Field(..., description="Whether more results are available")
    metadata: Optional[ResponseMetadata] = None