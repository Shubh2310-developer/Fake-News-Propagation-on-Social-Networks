# backend/app/models/game_theory.py

from typing import List, Dict, Optional, Tuple, Any
from pydantic import Field
from app.models.common import CommonBaseModel, ResponseMetadata


# ---------------------------------------------------------
# Strategy & Payoff Schemas
# ---------------------------------------------------------
class Strategy(CommonBaseModel):
    player: str = Field(..., description="Player type: spreader, fact_checker, platform")
    name: str = Field(..., description="Strategy name (e.g., post_fake_news, quick_verification)")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Optional parameterization of strategy"
    )


class Payoff(CommonBaseModel):
    player: str
    value: float


class PayoffMatrix(CommonBaseModel):
    players: List[str]
    strategies: Dict[str, List[str]]
    payoffs: List[List[Dict[str, float]]] = Field(
        ..., description="Nested payoff matrix, indexed by strategy combinations"
    )


# ---------------------------------------------------------
# Equilibrium Schemas
# ---------------------------------------------------------
class EquilibriumResult(CommonBaseModel):
    equilibrium_type: str = Field(..., description="Type: pure, mixed, bayesian, ESS")
    strategies: Dict[str, str] = Field(..., description="Chosen strategy per player")
    payoffs: Dict[str, float] = Field(..., description="Equilibrium payoffs per player")
    stability: Optional[str] = Field(None, description="Stability notion (e.g., ESS, sequential)")


class MixedStrategyEquilibrium(CommonBaseModel):
    player: str
    probabilities: Dict[str, float] = Field(
        ..., description="Strategy � probability mapping"
    )


# ---------------------------------------------------------
# Game & Simulation Schemas
# ---------------------------------------------------------
class GameDefinition(CommonBaseModel):
    players: List[str]
    strategy_spaces: Dict[str, List[str]]
    utility_functions: Optional[Dict[str, str]] = Field(
        None, description="Symbolic or formula representation of utility functions"
    )


class NashEquilibriumResponse(CommonBaseModel):
    equilibria: List[EquilibriumResult]
    mixed_strategies: Optional[List[MixedStrategyEquilibrium]] = None
    metadata: Optional[ResponseMetadata] = None


class RepeatedGameRequest(CommonBaseModel):
    base_game: GameDefinition
    num_rounds: int = Field(100, ge=1, description="Number of rounds in repeated game")
    learning_rate: float = Field(0.01, gt=0, description="Learning rate for adaptation")


class RepeatedGameResult(CommonBaseModel):
    history: List[Dict[str, Any]]
    reputation_scores: Dict[str, float]
    final_equilibria: List[EquilibriumResult]
    metadata: Optional[ResponseMetadata] = None


# ---------------------------------------------------------
# Equilibrium API Request/Response Models
# ---------------------------------------------------------
class EquilibriumRequest(CommonBaseModel):
    request_id: Optional[str] = Field(None, description="Optional request identifier")
    game_parameters: Dict[str, Any] = Field(..., description="Game configuration parameters")
    include_mixed: bool = Field(True, description="Whether to compute mixed strategy equilibria")
    include_stability_analysis: bool = Field(True, description="Whether to perform stability analysis")
    max_strategies_per_player: int = Field(10, gt=0, le=20, description="Limit strategy space size")


class EquilibriumResponse(CommonBaseModel):
    request_id: Optional[str] = Field(None, description="Request identifier")
    game_parameters: Dict[str, Any] = Field(..., description="Game parameters used")
    equilibria: Dict[str, Any] = Field(..., description="Calculated equilibria results")
    computation_successful: bool = Field(..., description="Whether computation succeeded")
    metadata: Optional[ResponseMetadata] = None


class SensitivityAnalysisRequest(CommonBaseModel):
    request_id: Optional[str] = Field(None, description="Optional request identifier")
    base_parameters: Dict[str, Any] = Field(..., description="Base game parameters")
    parameter_variations: Dict[str, List[float]] = Field(
        ..., description="Parameter names mapped to lists of values to test"
    )


class SensitivityAnalysisResponse(CommonBaseModel):
    request_id: Optional[str] = Field(None, description="Request identifier")
    base_parameters: Dict[str, Any] = Field(..., description="Base parameters used")
    sensitivity_analysis: Dict[str, Any] = Field(..., description="Sensitivity analysis results")
    analysis_successful: bool = Field(..., description="Whether analysis succeeded")
    metadata: Optional[ResponseMetadata] = None


class ScenarioComparisonRequest(CommonBaseModel):
    request_id: Optional[str] = Field(None, description="Optional request identifier")
    scenarios: List[Dict[str, Any]] = Field(
        ..., min_items=2, max_items=10, description="List of scenarios to compare"
    )


class ScenarioComparisonResponse(CommonBaseModel):
    request_id: Optional[str] = Field(None, description="Request identifier")
    scenarios: List[Dict[str, Any]] = Field(..., description="Scenarios compared")
    comparison_results: Dict[str, Any] = Field(..., description="Comparison analysis results")
    comparison_successful: bool = Field(..., description="Whether comparison succeeded")
    metadata: Optional[ResponseMetadata] = None


class StrategySpaceInfoResponse(CommonBaseModel):
    strategy_spaces: Dict[str, Dict[str, Any]] = Field(..., description="Strategy space information")
    payoff_weights: Dict[str, Dict[str, float]] = Field(..., description="Current payoff weights")
    total_combinations: int = Field(0, description="Total strategy combinations possible")
    metadata: Optional[ResponseMetadata] = None