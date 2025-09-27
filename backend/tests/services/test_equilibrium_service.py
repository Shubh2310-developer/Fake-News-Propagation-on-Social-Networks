# backend/tests/services/test_equilibrium_service.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from app.services.equilibrium_service import EquilibriumService
from game_theory.equilibrium import NashEquilibriumSolver, GameMatrixBuilder


@pytest.mark.unit
class TestEquilibriumService:
    """Test suite for EquilibriumService business logic."""

    @pytest.fixture
    def equilibrium_service(self):
        """Create an EquilibriumService instance for testing."""
        return EquilibriumService()

    @pytest.fixture
    def sample_equilibrium_config(self):
        """Sample equilibrium calculation configuration."""
        return {
            "network_size": 100,
            "detection_capability": 1.0,
            "max_strategies_per_player": 3,
            "payoff_weights": {
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
                    "credibility_score": 0.5,
                    "regulatory_risk": 0.2
                }
            }
        }

    def test_calculate_nash_equilibrium_success(self, equilibrium_service, sample_equilibrium_config):
        """Test successful Nash equilibrium calculation."""
        with patch.object(equilibrium_service, '_build_game_matrix') as mock_matrix, \
             patch.object(equilibrium_service, '_solve_equilibrium') as mock_solve:

            mock_matrix.return_value = np.array([[3, 1], [0, 2]])
            mock_solve.return_value = {
                "equilibrium_strategies": {
                    "spreaders": [0.6, 0.4],
                    "fact_checkers": [0.7, 0.3],
                    "platforms": [0.8, 0.2]
                },
                "expected_payoffs": {
                    "spreaders": 1.8,
                    "fact_checkers": 2.1,
                    "platforms": 1.5
                },
                "stability_metrics": {
                    "convergence_iterations": 15,
                    "stability_score": 0.95
                }
            }

            result = equilibrium_service.calculate_nash_equilibrium(sample_equilibrium_config)

            assert "equilibrium_strategies" in result
            assert "expected_payoffs" in result
            assert "stability_metrics" in result
            assert "spreaders" in result["equilibrium_strategies"]

    def test_calculate_equilibrium_invalid_config(self, equilibrium_service):
        """Test equilibrium calculation with invalid configuration."""
        invalid_config = {
            "network_size": -10,  # Invalid
            "detection_capability": 1.0
        }

        with pytest.raises(ValueError, match="Invalid equilibrium configuration"):
            equilibrium_service.calculate_nash_equilibrium(invalid_config)

    def test_analyze_strategy_stability(self, equilibrium_service, sample_equilibrium_config):
        """Test strategy stability analysis."""
        equilibrium_strategies = {
            "spreaders": [0.6, 0.4],
            "fact_checkers": [0.7, 0.3],
            "platforms": [0.8, 0.2]
        }

        with patch.object(equilibrium_service, '_calculate_stability_metrics') as mock_stability:
            mock_stability.return_value = {
                "stability_score": 0.92,
                "deviation_sensitivity": 0.05,
                "convergence_rate": 0.85,
                "robustness_index": 0.88
            }

            result = equilibrium_service.analyze_strategy_stability(
                equilibrium_strategies, sample_equilibrium_config
            )

            assert "stability_score" in result
            assert "deviation_sensitivity" in result
            assert 0 <= result["stability_score"] <= 1

    def test_compare_equilibria(self, equilibrium_service):
        """Test comparison between different equilibria."""
        equilibrium1 = {
            "equilibrium_strategies": {
                "spreaders": [0.6, 0.4],
                "fact_checkers": [0.7, 0.3]
            },
            "expected_payoffs": {
                "spreaders": 1.8,
                "fact_checkers": 2.1
            }
        }

        equilibrium2 = {
            "equilibrium_strategies": {
                "spreaders": [0.5, 0.5],
                "fact_checkers": [0.8, 0.2]
            },
            "expected_payoffs": {
                "spreaders": 1.6,
                "fact_checkers": 2.3
            }
        }

        comparison = equilibrium_service.compare_equilibria([equilibrium1, equilibrium2])

        assert "strategy_differences" in comparison
        assert "payoff_differences" in comparison
        assert "dominant_equilibrium" in comparison

    def test_game_matrix_construction(self, equilibrium_service, sample_equilibrium_config):
        """Test game matrix construction."""
        with patch('game_theory.equilibrium.GameMatrixBuilder') as mock_builder:
            mock_matrix_builder = MagicMock()
            mock_matrix_builder.build_payoff_matrix.return_value = np.array([[3, 1], [0, 2]])
            mock_builder.return_value = mock_matrix_builder

            matrix = equilibrium_service._build_game_matrix(sample_equilibrium_config)

            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (2, 2)

    def test_equilibrium_solver_integration(self, equilibrium_service):
        """Test integration with Nash equilibrium solver."""
        game_matrix = np.array([[3, 1], [0, 2]])

        with patch('game_theory.equilibrium.NashEquilibriumSolver') as mock_solver:
            mock_solver_instance = MagicMock()
            mock_solver_instance.solve.return_value = ([0.6, 0.4], [0.7, 0.3])
            mock_solver.return_value = mock_solver_instance

            strategies = equilibrium_service._solve_equilibrium(game_matrix)

            assert len(strategies) == 2  # Two strategy profiles
            mock_solver_instance.solve.assert_called_once()

    def test_payoff_calculation(self, equilibrium_service, sample_equilibrium_config):
        """Test payoff calculation for given strategies."""
        strategies = {
            "spreaders": [0.6, 0.4],
            "fact_checkers": [0.7, 0.3],
            "platforms": [0.8, 0.2]
        }

        with patch.object(equilibrium_service, '_calculate_expected_payoffs') as mock_payoffs:
            mock_payoffs.return_value = {
                "spreaders": 1.8,
                "fact_checkers": 2.1,
                "platforms": 1.5
            }

            payoffs = equilibrium_service.calculate_expected_payoffs(
                strategies, sample_equilibrium_config
            )

            assert "spreaders" in payoffs
            assert "fact_checkers" in payoffs
            assert "platforms" in payoffs
            assert all(isinstance(p, (int, float)) for p in payoffs.values())

    def test_equilibrium_convergence_analysis(self, equilibrium_service):
        """Test analysis of equilibrium convergence properties."""
        equilibrium_data = {
            "equilibrium_strategies": {
                "spreaders": [0.6, 0.4],
                "fact_checkers": [0.7, 0.3]
            },
            "convergence_history": [
                {"iteration": 1, "strategies": {"spreaders": [0.5, 0.5]}},
                {"iteration": 2, "strategies": {"spreaders": [0.55, 0.45]}},
                {"iteration": 3, "strategies": {"spreaders": [0.58, 0.42]}}
            ]
        }

        analysis = equilibrium_service.analyze_convergence(equilibrium_data)

        assert "convergence_rate" in analysis
        assert "stability_index" in analysis
        assert "iterations_to_convergence" in analysis

    def test_strategy_profile_validation(self, equilibrium_service):
        """Test validation of strategy profiles."""
        valid_strategies = {
            "spreaders": [0.6, 0.4],  # Sum = 1.0
            "fact_checkers": [0.7, 0.3]  # Sum = 1.0
        }

        # Should not raise exception
        equilibrium_service._validate_strategy_profile(valid_strategies)

        invalid_strategies = {
            "spreaders": [0.6, 0.5],  # Sum = 1.1 > 1.0
            "fact_checkers": [0.7, 0.3]
        }

        with pytest.raises(ValueError, match="Strategy probabilities must sum to 1"):
            equilibrium_service._validate_strategy_profile(invalid_strategies)

    def test_multiple_equilibria_handling(self, equilibrium_service, sample_equilibrium_config):
        """Test handling of multiple Nash equilibria."""
        with patch.object(equilibrium_service, '_solve_equilibrium') as mock_solve:
            mock_solve.return_value = [
                {"strategies": {"spreaders": [0.6, 0.4]}, "payoffs": {"spreaders": 1.8}},
                {"strategies": {"spreaders": [0.7, 0.3]}, "payoffs": {"spreaders": 1.9}}
            ]

            equilibria = equilibrium_service.find_all_equilibria(sample_equilibrium_config)

            assert len(equilibria) == 2
            assert all("strategies" in eq for eq in equilibria)

    @pytest.mark.slow
    def test_equilibrium_calculation_performance(self, equilibrium_service, sample_equilibrium_config, performance_timer):
        """Test equilibrium calculation performance."""
        timer = performance_timer(threshold_seconds=1.0)

        with patch.object(equilibrium_service, '_solve_equilibrium') as mock_solve:
            mock_solve.return_value = {
                "equilibrium_strategies": {"spreaders": [0.6, 0.4]},
                "expected_payoffs": {"spreaders": 1.8}
            }

            equilibrium_service.calculate_nash_equilibrium(sample_equilibrium_config)

        elapsed = timer()
        assert elapsed < 1.0, f"Equilibrium calculation took {elapsed:.3f}s"

    def test_equilibrium_perturbation_analysis(self, equilibrium_service):
        """Test analysis of equilibrium under parameter perturbations."""
        base_config = {
            "network_size": 100,
            "detection_capability": 1.0,
            "payoff_weights": {"spreader": {"reach": 1.0}}
        }

        perturbations = [
            {"detection_capability": 0.9},
            {"detection_capability": 1.1},
            {"payoff_weights": {"spreader": {"reach": 0.8}}}
        ]

        with patch.object(equilibrium_service, 'calculate_nash_equilibrium') as mock_calc:
            mock_calc.return_value = {
                "equilibrium_strategies": {"spreaders": [0.6, 0.4]},
                "expected_payoffs": {"spreaders": 1.8}
            }

            analysis = equilibrium_service.analyze_parameter_sensitivity(
                base_config, perturbations
            )

            assert "sensitivity_metrics" in analysis
            assert "parameter_effects" in analysis

    def test_mixed_strategy_interpretation(self, equilibrium_service):
        """Test interpretation of mixed strategy equilibria."""
        mixed_strategies = {
            "spreaders": [0.3, 0.4, 0.3],  # Three strategies
            "fact_checkers": [0.6, 0.4],
            "platforms": [0.8, 0.2]
        }

        interpretation = equilibrium_service.interpret_mixed_strategies(mixed_strategies)

        assert "strategy_meanings" in interpretation
        assert "dominant_strategies" in interpretation
        assert "behavioral_insights" in interpretation

    def test_equilibrium_robustness_testing(self, equilibrium_service, sample_equilibrium_config):
        """Test robustness of equilibrium solutions."""
        equilibrium = {
            "equilibrium_strategies": {
                "spreaders": [0.6, 0.4],
                "fact_checkers": [0.7, 0.3]
            }
        }

        with patch.object(equilibrium_service, '_test_robustness') as mock_robust:
            mock_robust.return_value = {
                "robustness_score": 0.85,
                "critical_perturbations": [],
                "stability_radius": 0.12
            }

            robustness = equilibrium_service.test_equilibrium_robustness(
                equilibrium, sample_equilibrium_config
            )

            assert "robustness_score" in robustness
            assert 0 <= robustness["robustness_score"] <= 1