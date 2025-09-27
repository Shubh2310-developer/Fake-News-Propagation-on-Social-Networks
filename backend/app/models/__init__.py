"""
Models package for Fake News Game Theory backend.
Contains SQLAlchemy ORM models and Pydantic schemas.
"""

from app.core.database import Base

# Import common schemas
from app.models.common import (
    CommonBaseModel,
    ResponseMetadata,
    ErrorResponse,
    SuccessResponse,
    Pagination,
    IDResponse,
)

# Import individual ORM models
from app.models.user import User
from app.models.news import NewsArticle, Classification
from app.models.social import SocialNode, SocialEdge
# Note: Simulation ORM models would be defined separately from Pydantic schemas
# from app.models.simulation import Simulation, SimulationResult, Payoff

# Import classifier schemas
from app.models.classifier import (
    TextClassificationRequest,
    TrainingRequest,
    TextClassificationResponse,
    ModelMetricsResponse,
)

# Import game theory schemas
from app.models.game_theory import (
    Strategy,
    Payoff,
    PayoffMatrix,
    EquilibriumResult,
    MixedStrategyEquilibrium,
    GameDefinition,
    NashEquilibriumResponse,
    RepeatedGameRequest,
    RepeatedGameResult,
)

# Import simulation schemas
from app.models.simulation import (
    SimulationRequest,
    NetworkGenerationRequest,
    SimulationStatus,
    SimulationStep,
    SimulationResponse,
    NetworkNode,
    NetworkEdge,
    NetworkResponse,
)

__all__ = [
    # SQLAlchemy Base
    "Base",
    # Common schemas
    "CommonBaseModel",
    "ResponseMetadata",
    "ErrorResponse",
    "SuccessResponse",
    "Pagination",
    "IDResponse",
    # Classifier schemas
    "TextClassificationRequest",
    "TrainingRequest",
    "TextClassificationResponse",
    "ModelMetricsResponse",
    # Game Theory schemas
    "Strategy",
    "Payoff",
    "PayoffMatrix",
    "EquilibriumResult",
    "MixedStrategyEquilibrium",
    "GameDefinition",
    "NashEquilibriumResponse",
    "RepeatedGameRequest",
    "RepeatedGameResult",
    # Simulation schemas
    "SimulationRequest",
    "NetworkGenerationRequest",
    "SimulationStatus",
    "SimulationStep",
    "SimulationResponse",
    "NetworkNode",
    "NetworkEdge",
    "NetworkResponse",
    # ORM Models
    "User",
    "NewsArticle",
    "Classification",
    "SocialNode",
    "SocialEdge",
    # Simulation ORM models would be added here when implemented
]