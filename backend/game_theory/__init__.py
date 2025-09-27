# backend/game_theory/__init__.py

from .players import Player, Spreader, FactChecker, Platform
from .strategies import SpreaderStrategy, FactCheckerStrategy, PlatformStrategy
from .payoffs import calculate_payoffs, get_default_payoff_weights
from .equilibrium import NashEquilibriumSolver, GameMatrixBuilder
from .simulation import RepeatedGameSimulation, SimulationConfig, RoundResult
from .analysis import EquilibriumAnalyzer, SimulationAnalyzer, StabilityMetrics

__all__ = [
    "Player",
    "Spreader",
    "FactChecker",
    "Platform",
    "SpreaderStrategy",
    "FactCheckerStrategy",
    "PlatformStrategy",
    "calculate_payoffs",
    "get_default_payoff_weights",
    "NashEquilibriumSolver",
    "GameMatrixBuilder",
    "RepeatedGameSimulation",
    "SimulationConfig",
    "RoundResult",
    "EquilibriumAnalyzer",
    "SimulationAnalyzer",
    "StabilityMetrics",
]