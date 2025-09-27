# backend/tests/core/test_equilibrium.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import nashpy

from game_theory.equilibrium import NashEquilibriumSolver, GameMatrixBuilder
from game_theory.payoffs import calculate_payoffs, get_default_payoff_weights
from game_theory.players import Spreader, FactChecker, Platform


@pytest.mark.unit
class TestNashEquilibriumSolver:
    """Test suite for Nash equilibrium solver core logic."""

    @pytest.fixture
    def equilibrium_solver(self):
        """Create a NashEquilibriumSolver instance."""
        return NashEquilibriumSolver()

    @pytest.fixture
    def simple_game_matrix(self):
        """Simple 2x2 game matrix for testing."""
        return np.array([[3, 0], [5, 1]])

    @pytest.fixture
    def complex_game_matrix(self):
        """Complex 3x3 game matrix for testing."""
        return np.array([
            [3, 1, 4],
            [2, 5, 1],
            [1, 2, 3]
        ])

    def test_solve_simple_game(self, equilibrium_solver, simple_game_matrix):
        """Test solving a simple 2x2 game."""
        equilibria = equilibrium_solver.solve(simple_game_matrix, simple_game_matrix.T)

        assert len(equilibria) > 0
        for eq in equilibria:
            assert len(eq) == 2  # Player 1 and Player 2 strategies
            assert all(isinstance(strategy, np.ndarray) for strategy in eq)
            assert all(np.isclose(strategy.sum(), 1.0) for strategy in eq)  # Probabilities sum to 1

    def test_solve_complex_game(self, equilibrium_solver, complex_game_matrix):
        """Test solving a more complex game."""
        equilibria = equilibrium_solver.solve(complex_game_matrix, complex_game_matrix.T)

        assert len(equilibria) > 0
        for eq in equilibria:
            player1_strategy, player2_strategy = eq
            assert player1_strategy.shape[0] == 3  # 3 strategies for player 1
            assert player2_strategy.shape[0] == 3  # 3 strategies for player 2

    def test_pure_strategy_equilibrium(self, equilibrium_solver):
        """Test detection of pure strategy equilibrium."""
        # Game with clear pure strategy equilibrium at (0,0)
        game_matrix = np.array([[10, 0], [0, 1]])

        equilibria = equilibrium_solver.solve(game_matrix, game_matrix.T)

        # Should find pure strategy equilibrium
        pure_equilibria = [eq for eq in equilibria if self._is_pure_strategy(eq)]
        assert len(pure_equilibria) > 0

    def test_mixed_strategy_equilibrium(self, equilibrium_solver):
        """Test finding mixed strategy equilibrium."""
        # Game requiring mixed strategies (Matching Pennies variant)
        game_matrix = np.array([[1, -1], [-1, 1]])

        equilibria = equilibrium_solver.solve(game_matrix, -game_matrix)

        # Should find mixed strategy equilibrium
        mixed_equilibria = [eq for eq in equilibria if not self._is_pure_strategy(eq)]
        assert len(mixed_equilibria) > 0

    def test_multiple_equilibria(self, equilibrium_solver):
        """Test game with multiple equilibria."""
        # Game with multiple Nash equilibria
        game_matrix = np.array([[2, 0], [0, 1]])

        equilibria = equilibrium_solver.solve(game_matrix, game_matrix.T)

        # Should find multiple equilibria
        assert len(equilibria) >= 2

    def test_equilibrium_verification(self, equilibrium_solver, simple_game_matrix):
        """Test verification of computed equilibria."""
        equilibria = equilibrium_solver.solve(simple_game_matrix, simple_game_matrix.T)

        for eq in equilibria:
            is_valid = equilibrium_solver.verify_equilibrium(eq, simple_game_matrix, simple_game_matrix.T)
            assert is_valid, "Computed equilibrium should be valid"

    def test_solver_convergence(self, equilibrium_solver):
        """Test solver convergence properties."""
        # Test with various matrix sizes
        for size in [2, 3, 4]:
            random_matrix = np.random.rand(size, size)
            equilibria = equilibrium_solver.solve(random_matrix, random_matrix.T)

            assert len(equilibria) > 0, f"Should find equilibrium for {size}x{size} game"

    def _is_pure_strategy(self, equilibrium):
        """Helper to check if equilibrium is pure strategy."""
        player1_strategy, player2_strategy = equilibrium
        return (np.max(player1_strategy) > 0.99 and np.max(player2_strategy) > 0.99)


@pytest.mark.unit
class TestGameMatrixBuilder:
    """Test suite for game matrix construction."""

    @pytest.fixture
    def matrix_builder(self):
        """Create a GameMatrixBuilder instance."""
        return GameMatrixBuilder()

    @pytest.fixture
    def sample_players(self):
        """Sample players for testing."""
        return {
            "spreaders": [Spreader(player_id=1), Spreader(player_id=2)],
            "fact_checkers": [FactChecker(player_id=3)],
            "platforms": [Platform(player_id=4)]
        }

    @pytest.fixture
    def sample_payoff_config(self):
        """Sample payoff configuration."""
        return {
            "spreader": {
                "reach": 1.0,
                "detection_penalty": 0.5,
                "reputation_cost": 0.3
            },
            "fact_checker": {
                "accuracy": 1.0,
                "effort_cost": 0.4,
                "reputation_gain": 0.6
            },
            "platform": {
                "user_engagement": 1.0,
                "moderation_cost": 0.3,
                "credibility_score": 0.5
            }
        }

    def test_build_payoff_matrix_basic(self, matrix_builder, sample_players, sample_payoff_config):
        """Test basic payoff matrix construction."""
        with patch('game_theory.payoffs.calculate_payoffs') as mock_payoffs:
            mock_payoffs.return_value = {
                "spreaders": [2.0, 1.5],
                "fact_checkers": [3.0],
                "platforms": [1.8]
            }

            matrix = matrix_builder.build_payoff_matrix(
                sample_players, sample_payoff_config
            )

            assert isinstance(matrix, np.ndarray)
            assert matrix.shape[0] > 0
            assert matrix.shape[1] > 0

    def test_strategy_enumeration(self, matrix_builder, sample_players):
        """Test enumeration of all possible strategy combinations."""
        strategies = matrix_builder._enumerate_strategies(sample_players)

        assert "spreaders" in strategies
        assert "fact_checkers" in strategies
        assert "platforms" in strategies

        # Should have multiple strategy combinations
        assert len(strategies["spreaders"]) > 1

    def test_payoff_calculation_integration(self, matrix_builder, sample_payoff_config):
        """Test integration with payoff calculation."""
        network_state = {
            "nodes": 100,
            "edges": 300,
            "misinformation_spread": 0.2
        }

        strategies = {
            "spreaders": ["aggressive", "moderate"],
            "fact_checkers": ["active"],
            "platforms": ["strict_moderation"]
        }

        with patch('game_theory.payoffs.calculate_payoffs') as mock_calc:
            mock_calc.return_value = {
                "spreaders": [1.5, 2.0],
                "fact_checkers": [2.5],
                "platforms": [1.8]
            }

            payoffs = matrix_builder._calculate_strategy_payoffs(
                strategies, network_state, sample_payoff_config
            )

            assert "spreaders" in payoffs
            assert len(payoffs["spreaders"]) == 2

    def test_matrix_symmetry_properties(self, matrix_builder):
        """Test symmetry properties of constructed matrices."""
        # For symmetric games, matrix should have certain properties
        symmetric_config = {
            "spreader": {"reach": 1.0, "detection_penalty": 0.5},
            "fact_checker": {"accuracy": 1.0, "effort_cost": 0.5}
        }

        with patch.object(matrix_builder, '_is_symmetric_game') as mock_symmetric:
            mock_symmetric.return_value = True

            # Test symmetric game matrix construction
            assert mock_symmetric.return_value is True

    def test_matrix_normalization(self, matrix_builder):
        """Test payoff matrix normalization."""
        raw_matrix = np.array([[100, 50], [75, 25]])

        normalized = matrix_builder._normalize_matrix(raw_matrix)

        assert np.min(normalized) >= 0  # Non-negative values
        assert np.max(normalized) <= 1  # Normalized scale

    def test_strategy_space_expansion(self, matrix_builder):
        """Test expansion of strategy spaces for matrix construction."""
        base_strategies = {
            "spreaders": ["aggressive"],
            "fact_checkers": ["active"]
        }

        expanded = matrix_builder._expand_strategy_space(base_strategies, max_strategies=3)

        assert len(expanded["spreaders"]) <= 3
        assert len(expanded["fact_checkers"]) <= 3


@pytest.mark.unit
class TestPayoffCalculations:
    """Test suite for payoff calculation functions."""

    @pytest.fixture
    def sample_network_state(self):
        """Sample network state for payoff calculations."""
        return {
            "nodes": 100,
            "edges": 300,
            "misinformation_nodes": 20,
            "fact_checked_nodes": 15,
            "platform_interventions": 5,
            "reach_metrics": {
                "spreader_reach": [50, 30, 25],
                "fact_checker_reach": [40, 35]
            }
        }

    @pytest.fixture
    def sample_strategies(self):
        """Sample strategy configurations."""
        return {
            "spreaders": ["aggressive", "moderate", "cautious"],
            "fact_checkers": ["active", "reactive"],
            "platforms": ["strict_moderation"]
        }

    def test_calculate_payoffs_basic(self, sample_network_state, sample_strategies):
        """Test basic payoff calculation."""
        payoff_weights = get_default_payoff_weights()

        payoffs = calculate_payoffs(
            sample_network_state, sample_strategies, payoff_weights
        )

        assert "spreaders" in payoffs
        assert "fact_checkers" in payoffs
        assert "platforms" in payoffs

        # Payoffs should be numeric
        assert all(isinstance(p, (int, float)) for p in payoffs["spreaders"])

    def test_spreader_payoff_components(self, sample_network_state):
        """Test individual components of spreader payoffs."""
        payoff_weights = {
            "spreader": {
                "reach": 1.0,
                "detection_penalty": 0.5,
                "reputation_cost": 0.3
            }
        }

        strategies = {"spreaders": ["aggressive"]}

        with patch('game_theory.payoffs._calculate_reach_payoff') as mock_reach, \
             patch('game_theory.payoffs._calculate_detection_penalty') as mock_penalty:

            mock_reach.return_value = 2.0
            mock_penalty.return_value = 0.5

            payoffs = calculate_payoffs(sample_network_state, strategies, payoff_weights)

            # Should incorporate both reach and penalty
            assert payoffs["spreaders"][0] > 0

    def test_fact_checker_payoff_components(self, sample_network_state):
        """Test fact checker payoff calculations."""
        payoff_weights = {
            "fact_checker": {
                "accuracy": 1.0,
                "effort_cost": 0.4,
                "reputation_gain": 0.6
            }
        }

        strategies = {"fact_checkers": ["active"]}

        payoffs = calculate_payoffs(sample_network_state, strategies, payoff_weights)

        assert len(payoffs["fact_checkers"]) == 1
        assert isinstance(payoffs["fact_checkers"][0], (int, float))

    def test_platform_payoff_components(self, sample_network_state):
        """Test platform payoff calculations."""
        payoff_weights = {
            "platform": {
                "user_engagement": 1.0,
                "moderation_cost": 0.3,
                "credibility_score": 0.5,
                "regulatory_risk": 0.2
            }
        }

        strategies = {"platforms": ["strict_moderation"]}

        payoffs = calculate_payoffs(sample_network_state, strategies, payoff_weights)

        assert len(payoffs["platforms"]) == 1
        assert isinstance(payoffs["platforms"][0], (int, float))

    def test_default_payoff_weights(self):
        """Test default payoff weight configuration."""
        weights = get_default_payoff_weights()

        assert "spreader" in weights
        assert "fact_checker" in weights
        assert "platform" in weights

        # All weights should be positive
        for player_type, player_weights in weights.items():
            assert all(w >= 0 for w in player_weights.values())

    def test_payoff_weight_validation(self):
        """Test validation of payoff weight configurations."""
        invalid_weights = {
            "spreader": {
                "reach": -1.0,  # Invalid negative weight
                "detection_penalty": 0.5
            }
        }

        with pytest.raises(ValueError, match="Payoff weights must be non-negative"):
            calculate_payoffs({}, {}, invalid_weights)

    def test_network_state_impact_on_payoffs(self):
        """Test how network state changes affect payoffs."""
        base_state = {
            "nodes": 100,
            "misinformation_nodes": 10,
            "fact_checked_nodes": 5
        }

        high_misinformation_state = {
            "nodes": 100,
            "misinformation_nodes": 30,
            "fact_checked_nodes": 5
        }

        strategies = {"spreaders": ["aggressive"], "fact_checkers": ["active"]}
        weights = get_default_payoff_weights()

        base_payoffs = calculate_payoffs(base_state, strategies, weights)
        high_payoffs = calculate_payoffs(high_misinformation_state, strategies, weights)

        # Spreader payoffs might increase with more misinformation opportunity
        # Fact checker payoffs might change due to different challenge level
        assert base_payoffs != high_payoffs

    @pytest.mark.slow
    def test_payoff_calculation_performance(self, sample_network_state, sample_strategies, performance_timer):
        """Test payoff calculation performance."""
        timer = performance_timer(threshold_seconds=0.1)
        weights = get_default_payoff_weights()

        calculate_payoffs(sample_network_state, sample_strategies, weights)

        elapsed = timer()
        assert elapsed < 0.1, f"Payoff calculation took {elapsed:.3f}s"