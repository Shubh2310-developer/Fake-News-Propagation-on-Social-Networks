# backend/game_theory/equilibrium.py

import numpy as np
import nashpy as nash
from typing import List, Tuple, Dict, Any, Optional
import logging
from .strategies import SpreaderStrategy, FactCheckerStrategy, PlatformStrategy
from .payoffs import calculate_payoffs, get_default_payoff_weights

logger = logging.getLogger(__name__)


class NashEquilibriumSolver:
    """
    Calculates pure and mixed strategy Nash Equilibria for a given game.
    """

    def __init__(self, payoff_matrices: Dict[str, np.ndarray]):
        """
        Initializes the solver with payoff matrices for each player.

        Args:
            payoff_matrices: Dictionary mapping player names to their payoff matrices
                           Example: {'player1': matrix1, 'player2': matrix2}
        """
        self.payoff_matrices = payoff_matrices
        self.player_names = list(payoff_matrices.keys())

        # For two-player games, use nashpy
        if len(self.player_names) == 2:
            player1_name, player2_name = self.player_names
            self.player1_payoffs = payoff_matrices[player1_name]
            self.player2_payoffs = payoff_matrices[player2_name]
            self.game = nash.Game(self.player1_payoffs, self.player2_payoffs)
        else:
            self.game = None

        logger.info(f"Initialized Nash Equilibrium Solver for {len(self.player_names)} players")

    def find_pure_strategy_equilibria(self) -> List[Dict[str, Any]]:
        """
        Finds all pure strategy Nash Equilibria by iterating through outcomes.

        Returns:
            List of equilibria, each containing strategy indices and payoffs
        """
        if len(self.player_names) != 2:
            return self._find_pure_strategy_equilibria_n_player()

        equilibria = []
        rows, cols = self.player1_payoffs.shape

        for r in range(rows):
            for c in range(cols):
                # Check if player1 has a better move
                player1_can_improve = self.player1_payoffs[r, c] < np.max(self.player1_payoffs[:, c])
                # Check if player2 has a better move
                player2_can_improve = self.player2_payoffs[r, c] < np.max(self.player2_payoffs[r, :])

                if not player1_can_improve and not player2_can_improve:
                    equilibrium = {
                        'type': 'pure_strategy',
                        'strategies': {
                            self.player_names[0]: r,
                            self.player_names[1]: c
                        },
                        'payoffs': {
                            self.player_names[0]: float(self.player1_payoffs[r, c]),
                            self.player_names[1]: float(self.player2_payoffs[r, c])
                        }
                    }
                    equilibria.append(equilibrium)

        logger.info(f"Found {len(equilibria)} pure strategy equilibria")
        return equilibria

    def find_mixed_strategy_equilibria(self) -> List[Dict[str, Any]]:
        """
        Uses the nashpy library to find mixed strategy Nash Equilibria.

        Returns:
            List of mixed strategy equilibria with strategy probabilities and expected payoffs
        """
        if len(self.player_names) != 2 or self.game is None:
            logger.warning("Mixed strategy equilibria calculation only supports 2-player games")
            return []

        try:
            # Use support enumeration algorithm to find all mixed strategy equilibria
            equilibria = []
            for eq in self.game.support_enumeration():
                if len(eq) == 2:  # Valid equilibrium
                    strategy1, strategy2 = eq

                    # Calculate expected payoffs
                    expected_payoff1 = float(strategy1 @ self.player1_payoffs @ strategy2)
                    expected_payoff2 = float(strategy1 @ self.player2_payoffs @ strategy2)

                    equilibrium = {
                        'type': 'mixed_strategy',
                        'strategies': {
                            self.player_names[0]: strategy1.tolist(),
                            self.player_names[1]: strategy2.tolist()
                        },
                        'payoffs': {
                            self.player_names[0]: expected_payoff1,
                            self.player_names[1]: expected_payoff2
                        }
                    }
                    equilibria.append(equilibrium)

            logger.info(f"Found {len(equilibria)} mixed strategy equilibria")
            return equilibria

        except Exception as e:
            logger.error(f"Error finding mixed strategy equilibria: {e}")
            return []

    def _find_pure_strategy_equilibria_n_player(self) -> List[Dict[str, Any]]:
        """
        Find pure strategy equilibria for n-player games (n > 2).

        Returns:
            List of pure strategy equilibria
        """
        # This is a simplified implementation for demonstration
        # In practice, this would require more sophisticated algorithms
        logger.warning("N-player pure strategy equilibrium finding is simplified")
        return []

    def _is_best_response(self, player_strategy: int, other_strategies: Dict[str, int],
                         player_name: str) -> bool:
        """
        Check if a player's strategy is the best response to other players' strategies.

        Args:
            player_strategy: The strategy index for the player
            other_strategies: Dictionary of other players' strategy indices
            player_name: Name of the player whose strategy we're checking

        Returns:
            True if the strategy is a best response, False otherwise
        """
        if player_name not in self.payoff_matrices:
            return False

        payoff_matrix = self.payoff_matrices[player_name]

        # For 2-player games
        if len(self.player_names) == 2:
            other_player = [p for p in self.player_names if p != player_name][0]
            other_strategy = other_strategies[other_player]

            if player_name == self.player_names[0]:
                # Player 1's payoffs are indexed by (own_strategy, other_strategy)
                current_payoff = payoff_matrix[player_strategy, other_strategy]
                max_payoff = np.max(payoff_matrix[:, other_strategy])
            else:
                # Player 2's payoffs are indexed by (other_strategy, own_strategy)
                current_payoff = payoff_matrix[other_strategy, player_strategy]
                max_payoff = np.max(payoff_matrix[other_strategy, :])

            return np.isclose(current_payoff, max_payoff)

        # For n-player games, this would be more complex
        return False

    def analyze_equilibrium_stability(self, equilibrium: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the stability properties of a given equilibrium.

        Args:
            equilibrium: An equilibrium dictionary from find_pure_strategy_equilibria

        Returns:
            Dictionary containing stability analysis results
        """
        stability_analysis = {
            'is_strict': False,
            'is_symmetric': False,
            'dominance_properties': {},
            'robustness_score': 0.0
        }

        try:
            if equilibrium['type'] == 'pure_strategy':
                stability_analysis.update(self._analyze_pure_strategy_stability(equilibrium))
            elif equilibrium['type'] == 'mixed_strategy':
                stability_analysis.update(self._analyze_mixed_strategy_stability(equilibrium))

        except Exception as e:
            logger.error(f"Error analyzing equilibrium stability: {e}")

        return stability_analysis

    def _analyze_pure_strategy_stability(self, equilibrium: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stability properties of a pure strategy equilibrium."""
        analysis = {}

        # Check if equilibrium is strict (unique best response)
        if len(self.player_names) == 2:
            strategies = equilibrium['strategies']
            player1_strategy = strategies[self.player_names[0]]
            player2_strategy = strategies[self.player_names[1]]

            # Check strictness for player 1
            player1_payoffs_at_strategy = self.player1_payoffs[:, player2_strategy]
            max_payoff = np.max(player1_payoffs_at_strategy)
            num_best_responses = np.sum(np.isclose(player1_payoffs_at_strategy, max_payoff))

            # Check strictness for player 2
            player2_payoffs_at_strategy = self.player2_payoffs[player1_strategy, :]
            max_payoff2 = np.max(player2_payoffs_at_strategy)
            num_best_responses2 = np.sum(np.isclose(player2_payoffs_at_strategy, max_payoff2))

            analysis['is_strict'] = (num_best_responses == 1) and (num_best_responses2 == 1)

        return analysis

    def _analyze_mixed_strategy_stability(self, equilibrium: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stability properties of a mixed strategy equilibrium."""
        analysis = {}

        # Mixed strategy equilibria are generally less strict than pure strategy ones
        analysis['is_strict'] = False

        # Calculate support size (number of strategies played with positive probability)
        strategies = equilibrium['strategies']
        for player_name, strategy_probs in strategies.items():
            support_size = np.sum(np.array(strategy_probs) > 1e-6)
            analysis[f'{player_name}_support_size'] = int(support_size)

        return analysis


class GameMatrixBuilder:
    """
    Builds payoff matrices for game theory analysis from strategy spaces and payoff functions.
    """

    def __init__(self, strategy_spaces: Dict[str, List], payoff_function, payoff_weights: Optional[Dict] = None):
        """
        Initialize the matrix builder.

        Args:
            strategy_spaces: Dictionary mapping player names to lists of their possible strategies
            payoff_function: Function that calculates payoffs given strategies and outcomes
            payoff_weights: Optional weights for payoff calculations
        """
        self.strategy_spaces = strategy_spaces
        self.payoff_function = payoff_function
        self.payoff_weights = payoff_weights or get_default_payoff_weights()

    def build_payoff_matrices(self, outcome_scenarios: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Build payoff matrices for all players across all strategy combinations.

        Args:
            outcome_scenarios: List of possible game outcomes to average over

        Returns:
            Dictionary mapping player names to their payoff matrices
        """
        player_names = list(self.strategy_spaces.keys())
        matrices = {}

        # For 2-player games
        if len(player_names) == 2:
            player1, player2 = player_names
            strategies1 = self.strategy_spaces[player1]
            strategies2 = self.strategy_spaces[player2]

            # Initialize matrices
            matrix1 = np.zeros((len(strategies1), len(strategies2)))
            matrix2 = np.zeros((len(strategies1), len(strategies2)))

            # Fill matrices
            for i, strategy1 in enumerate(strategies1):
                for j, strategy2 in enumerate(strategies2):
                    # Average payoffs across outcome scenarios
                    avg_payoffs = self._calculate_average_payoffs(
                        {player1: strategy1, player2: strategy2},
                        outcome_scenarios
                    )

                    matrix1[i, j] = avg_payoffs.get(player1, 0.0)
                    matrix2[i, j] = avg_payoffs.get(player2, 0.0)

            matrices[player1] = matrix1
            matrices[player2] = matrix2

        logger.info(f"Built payoff matrices for {len(player_names)} players")
        return matrices

    def _calculate_average_payoffs(self, strategies: Dict[str, Any],
                                 outcome_scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average payoffs across multiple outcome scenarios.

        Args:
            strategies: Dictionary of player strategies
            outcome_scenarios: List of possible outcomes

        Returns:
            Dictionary mapping player names to average payoffs
        """
        total_payoffs = {}

        for outcome in outcome_scenarios:
            payoffs = self.payoff_function(strategies, outcome, self.payoff_weights)

            for player, payoff in payoffs.items():
                if player not in total_payoffs:
                    total_payoffs[player] = 0.0
                total_payoffs[player] += payoff

        # Calculate averages
        num_scenarios = len(outcome_scenarios)
        if num_scenarios > 0:
            return {player: total / num_scenarios for player, total in total_payoffs.items()}

        return total_payoffs