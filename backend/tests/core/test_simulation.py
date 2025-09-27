# backend/tests/core/test_simulation.py

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import networkx as nx

from game_theory.simulation import RepeatedGameSimulation, SimulationConfig, RoundResult
from game_theory.players import Spreader, FactChecker, Platform
from game_theory.strategies import SpreaderStrategy, FactCheckerStrategy, PlatformStrategy


@pytest.mark.unit
class TestRepeatedGameSimulation:
    """Test suite for repeated game simulation core logic."""

    @pytest.fixture
    def simulation_config(self):
        """Sample simulation configuration."""
        return SimulationConfig(
            num_rounds=10,
            num_spreaders=3,
            num_fact_checkers=2,
            num_platforms=1,
            learning_rate=0.1,
            exploration_rate=0.1,
            random_seed=42
        )

    @pytest.fixture
    def sample_network(self):
        """Sample network for testing."""
        G = nx.barabasi_albert_graph(50, 3, seed=42)
        return G

    @pytest.fixture
    def simulation(self, simulation_config, sample_network):
        """Create a RepeatedGameSimulation instance."""
        return RepeatedGameSimulation(simulation_config, sample_network)

    def test_simulation_initialization(self, simulation, simulation_config):
        """Test simulation initialization."""
        assert simulation.config == simulation_config
        assert simulation.current_round == 0
        assert len(simulation.players["spreaders"]) == 3
        assert len(simulation.players["fact_checkers"]) == 2
        assert len(simulation.players["platforms"]) == 1

    def test_player_creation(self, simulation_config, sample_network):
        """Test creation of players with correct types."""
        sim = RepeatedGameSimulation(simulation_config, sample_network)

        # Check player types
        assert all(isinstance(p, Spreader) for p in sim.players["spreaders"])
        assert all(isinstance(p, FactChecker) for p in sim.players["fact_checkers"])
        assert all(isinstance(p, Platform) for p in sim.players["platforms"])

        # Check unique IDs
        all_ids = []
        for player_list in sim.players.values():
            all_ids.extend([p.player_id for p in player_list])
        assert len(all_ids) == len(set(all_ids))  # All IDs unique

    def test_run_single_round(self, simulation):
        """Test execution of a single game round."""
        with patch.object(simulation, '_collect_actions') as mock_actions, \
             patch.object(simulation, '_update_network_state') as mock_update, \
             patch.object(simulation, '_calculate_round_payoffs') as mock_payoffs:

            mock_actions.return_value = {
                "spreaders": [1, 0, 1],  # Actions for 3 spreaders
                "fact_checkers": [1, 1],  # Actions for 2 fact checkers
                "platforms": [0]  # Action for 1 platform
            }
            mock_payoffs.return_value = {
                "spreaders": [2.0, 1.5, 2.5],
                "fact_checkers": [3.0, 2.8],
                "platforms": [1.8]
            }

            round_result = simulation.run_single_round()

            assert isinstance(round_result, RoundResult)
            assert round_result.round_number == 1
            assert "spreaders" in round_result.actions
            assert "spreaders" in round_result.payoffs

    def test_run_full_simulation(self, simulation):
        """Test running complete simulation."""
        with patch.object(simulation, 'run_single_round') as mock_round:
            mock_round.return_value = RoundResult(
                round_number=1,
                actions={"spreaders": [1, 0, 1]},
                payoffs={"spreaders": [2.0, 1.5, 2.5]},
                network_state={"misinformation_spread": 0.2}
            )

            results = simulation.run_simulation()

            assert len(results) == 10  # Should run for 10 rounds
            assert mock_round.call_count == 10

    def test_action_collection(self, simulation):
        """Test collection of player actions."""
        # Mock player strategies
        for spreader in simulation.players["spreaders"]:
            spreader.strategy = MagicMock()
            spreader.strategy.choose_action.return_value = 1

        for fact_checker in simulation.players["fact_checkers"]:
            fact_checker.strategy = MagicMock()
            fact_checker.strategy.choose_action.return_value = 1

        for platform in simulation.players["platforms"]:
            platform.strategy = MagicMock()
            platform.strategy.choose_action.return_value = 0

        actions = simulation._collect_actions()

        assert "spreaders" in actions
        assert "fact_checkers" in actions
        assert "platforms" in actions
        assert len(actions["spreaders"]) == 3
        assert len(actions["fact_checkers"]) == 2
        assert len(actions["platforms"]) == 1

    def test_network_state_updates(self, simulation):
        """Test network state updates after actions."""
        actions = {
            "spreaders": [1, 0, 1],  # 2 spreaders take action
            "fact_checkers": [1, 1],  # 2 fact checkers active
            "platforms": [1]  # 1 platform intervenes
        }

        initial_state = simulation._get_network_state()
        simulation._update_network_state(actions)
        updated_state = simulation._get_network_state()

        # Network state should change
        assert updated_state != initial_state

    def test_payoff_calculations(self, simulation):
        """Test payoff calculations for a round."""
        actions = {
            "spreaders": [1, 0, 1],
            "fact_checkers": [1, 1],
            "platforms": [0]
        }

        network_state = {
            "misinformation_spread": 0.3,
            "fact_checked_content": 0.2,
            "platform_interventions": 0
        }

        with patch('game_theory.payoffs.calculate_payoffs') as mock_calc:
            mock_calc.return_value = {
                "spreaders": [2.0, 1.0, 2.5],
                "fact_checkers": [3.0, 3.2],
                "platforms": [1.5]
            }

            payoffs = simulation._calculate_round_payoffs(actions, network_state)

            assert "spreaders" in payoffs
            assert len(payoffs["spreaders"]) == 3

    def test_strategy_learning(self, simulation):
        """Test player strategy learning and adaptation."""
        # Set up players with learning-capable strategies
        for spreader in simulation.players["spreaders"]:
            spreader.strategy = MagicMock()
            spreader.strategy.update_strategy = MagicMock()

        round_result = RoundResult(
            round_number=1,
            actions={"spreaders": [1, 0, 1]},
            payoffs={"spreaders": [2.0, 1.0, 2.5]},
            network_state={}
        )

        simulation._update_player_strategies(round_result)

        # Check that strategy updates were called
        for spreader in simulation.players["spreaders"]:
            spreader.strategy.update_strategy.assert_called_once()

    def test_convergence_detection(self, simulation):
        """Test detection of strategy convergence."""
        # Mock stable strategies over multiple rounds
        stable_history = [
            {"spreaders": [1, 1, 0], "fact_checkers": [1, 1]},
            {"spreaders": [1, 1, 0], "fact_checkers": [1, 1]},
            {"spreaders": [1, 1, 0], "fact_checkers": [1, 1]},
            {"spreaders": [1, 1, 0], "fact_checkers": [1, 1]}
        ]

        simulation.action_history = stable_history

        convergence = simulation._check_convergence(window_size=3, threshold=0.1)

        assert convergence is True

    def test_random_seed_reproducibility(self, simulation_config, sample_network):
        """Test that simulations are reproducible with same seed."""
        sim1 = RepeatedGameSimulation(simulation_config, sample_network)
        sim2 = RepeatedGameSimulation(simulation_config, sample_network)

        with patch.object(sim1, '_get_random_action') as mock1, \
             patch.object(sim2, '_get_random_action') as mock2:

            mock1.return_value = 1
            mock2.return_value = 1

            # Both simulations should produce identical results
            np.random.seed(42)
            result1 = sim1._get_random_action()

            np.random.seed(42)
            result2 = sim2._get_random_action()

            assert result1 == result2

    def test_simulation_statistics(self, simulation):
        """Test calculation of simulation statistics."""
        # Run mock simulation
        mock_results = [
            RoundResult(1, {"spreaders": [1, 0, 1]}, {"spreaders": [2.0, 1.0, 2.5]}, {}),
            RoundResult(2, {"spreaders": [0, 1, 1]}, {"spreaders": [1.5, 2.0, 2.0]}, {}),
            RoundResult(3, {"spreaders": [1, 1, 0]}, {"spreaders": [2.2, 2.1, 1.0]}, {})
        ]

        simulation.simulation_results = mock_results

        stats = simulation.calculate_statistics()

        assert "average_payoffs" in stats
        assert "strategy_frequencies" in stats
        assert "convergence_metrics" in stats

    @pytest.mark.slow
    def test_simulation_performance(self, simulation_config, sample_network, performance_timer):
        """Test simulation performance requirements."""
        timer = performance_timer(threshold_seconds=2.0)

        sim = RepeatedGameSimulation(simulation_config, sample_network)

        with patch.object(sim, '_collect_actions') as mock_actions:
            mock_actions.return_value = {"spreaders": [1], "fact_checkers": [1], "platforms": [0]}

            sim.run_simulation()

        elapsed = timer()
        assert elapsed < 2.0, f"Simulation took {elapsed:.3f}s, should be < 2.0s"

    def test_memory_management(self, simulation):
        """Test that simulation manages memory efficiently."""
        # Run simulation and check memory doesn't grow excessively
        initial_memory = simulation._get_memory_usage()

        with patch.object(simulation, 'run_single_round') as mock_round:
            mock_round.return_value = RoundResult(1, {}, {}, {})
            simulation.run_simulation()

        final_memory = simulation._get_memory_usage()

        # Memory shouldn't grow excessively (allow some overhead)
        memory_growth = final_memory - initial_memory
        assert memory_growth < 100  # Less than 100MB growth

    def test_error_handling_invalid_actions(self, simulation):
        """Test error handling for invalid player actions."""
        # Mock invalid action
        invalid_actions = {
            "spreaders": [2, 0, 1],  # Invalid action value
            "fact_checkers": [1, 1],
            "platforms": [0]
        }

        with pytest.raises(ValueError, match="Invalid action"):
            simulation._validate_actions(invalid_actions)

    def test_network_metrics_tracking(self, simulation):
        """Test tracking of network metrics throughout simulation."""
        simulation.run_single_round()

        network_state = simulation._get_network_state()

        assert "misinformation_spread" in network_state
        assert "clustering_coefficient" in network_state
        assert "average_path_length" in network_state

    def test_strategy_diversity_metrics(self, simulation):
        """Test calculation of strategy diversity metrics."""
        action_history = [
            {"spreaders": [1, 0, 1, 0, 1]},
            {"spreaders": [0, 1, 1, 1, 0]},
            {"spreaders": [1, 1, 0, 0, 1]}
        ]

        diversity = simulation._calculate_strategy_diversity(action_history)

        assert "entropy" in diversity
        assert "variation_coefficient" in diversity
        assert 0 <= diversity["entropy"] <= 1


@pytest.mark.unit
class TestSimulationConfig:
    """Test suite for simulation configuration."""

    def test_config_validation_valid(self):
        """Test validation of valid configuration."""
        config = SimulationConfig(
            num_rounds=10,
            num_spreaders=3,
            num_fact_checkers=2,
            num_platforms=1,
            learning_rate=0.1,
            exploration_rate=0.1,
            random_seed=42
        )

        # Should not raise exception
        config.validate()

    def test_config_validation_invalid_rounds(self):
        """Test validation with invalid number of rounds."""
        with pytest.raises(ValueError, match="num_rounds must be positive"):
            SimulationConfig(
                num_rounds=0,  # Invalid
                num_spreaders=3,
                num_fact_checkers=2,
                num_platforms=1,
                learning_rate=0.1,
                exploration_rate=0.1,
                random_seed=42
            )

    def test_config_validation_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        with pytest.raises(ValueError, match="learning_rate must be between 0 and 1"):
            SimulationConfig(
                num_rounds=10,
                num_spreaders=3,
                num_fact_checkers=2,
                num_platforms=1,
                learning_rate=1.5,  # Invalid
                exploration_rate=0.1,
                random_seed=42
            )

    def test_config_serialization(self):
        """Test configuration serialization/deserialization."""
        config = SimulationConfig(
            num_rounds=10,
            num_spreaders=3,
            num_fact_checkers=2,
            num_platforms=1,
            learning_rate=0.1,
            exploration_rate=0.1,
            random_seed=42
        )

        # Test to_dict
        config_dict = config.to_dict()
        assert "num_rounds" in config_dict
        assert config_dict["num_rounds"] == 10

        # Test from_dict
        restored_config = SimulationConfig.from_dict(config_dict)
        assert restored_config.num_rounds == config.num_rounds