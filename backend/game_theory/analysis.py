# backend/game_theory/analysis.py

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from .simulation import RoundResult

logger = logging.getLogger(__name__)


@dataclass
class StabilityMetrics:
    """Container for equilibrium stability analysis results."""
    is_evolutionary_stable: bool
    is_trembling_hand_perfect: bool
    is_sequential_equilibrium: bool
    robustness_score: float
    convergence_rate: float
    deviation_resistance: float


class EquilibriumAnalyzer:
    """
    Provides advanced analysis of game-theoretic equilibria.
    """

    def __init__(self, payoff_matrix_p1: np.ndarray, payoff_matrix_p2: np.ndarray):
        """
        Initialize the equilibrium analyzer.

        Args:
            payoff_matrix_p1: Payoff matrix for player 1
            payoff_matrix_p2: Payoff matrix for player 2
        """
        self.u1 = payoff_matrix_p1
        self.u2 = payoff_matrix_p2
        self.n_strategies_p1, self.n_strategies_p2 = payoff_matrix_p1.shape

        logger.info(f"Initialized EquilibriumAnalyzer with {self.n_strategies_p1}x{self.n_strategies_p2} game")

    def is_evolutionary_stable_strategy(self, eq_strategy: Tuple[int, int],
                                      epsilon: float = 0.01) -> bool:
        """
        Checks if a symmetric pure strategy is an Evolutionary Stable Strategy (ESS).

        For a strategy to be ESS, it must satisfy:
        1. (Nash) u(eq, eq) >= u(s, eq) for all other strategies s
        2. If u(eq, eq) == u(s, eq), then u(eq, s) > u(s, s)

        Args:
            eq_strategy: The equilibrium strategy indices (row, col)
            epsilon: Tolerance for equality comparisons

        Returns:
            True if the strategy is an ESS, False otherwise
        """
        try:
            # Check if game is symmetric (required for ESS analysis)
            if not self._is_symmetric_game():
                logger.warning("ESS analysis requires a symmetric game")
                return False

            s1, s2 = eq_strategy

            # For ESS, we typically analyze symmetric strategies where s1 == s2
            if s1 != s2:
                logger.warning("ESS analysis typically applies to symmetric strategy profiles")

            eq_payoff = self.u1[s1, s2]

            # Check ESS conditions against all alternative strategies
            for alt_strategy in range(self.n_strategies_p1):
                if alt_strategy == s1:
                    continue

                # Condition 1: Nash equilibrium condition
                alt_vs_eq_payoff = self.u1[alt_strategy, s2]
                if alt_vs_eq_payoff > eq_payoff + epsilon:
                    return False

                # Condition 2: Stability condition
                if abs(alt_vs_eq_payoff - eq_payoff) <= epsilon:
                    # Payoffs are equal, check second condition
                    eq_vs_alt_payoff = self.u1[s1, alt_strategy]
                    alt_vs_alt_payoff = self.u1[alt_strategy, alt_strategy]

                    if eq_vs_alt_payoff <= alt_vs_alt_payoff + epsilon:
                        return False

            return True

        except Exception as e:
            logger.error(f"Error in ESS analysis: {e}")
            return False

    def check_trembling_hand_perfection(self, equilibrium: Tuple[np.ndarray, np.ndarray],
                                      epsilon: float = 0.001) -> bool:
        """
        Checks for Trembling Hand Perfect Equilibrium.

        A strategy is trembling hand perfect if it remains optimal even when
        opponents make small mistakes (trembles) with probability epsilon.

        Args:
            equilibrium: Tuple of mixed strategy probability distributions
            epsilon: Probability of trembling (making mistakes)

        Returns:
            True if the equilibrium is trembling hand perfect
        """
        try:
            strategy1, strategy2 = equilibrium

            # Ensure strategies are properly normalized
            strategy1 = strategy1 / np.sum(strategy1)
            strategy2 = strategy2 / np.sum(strategy2)

            # Create perturbed strategies (trembling)
            perturbed_strategy1 = (1 - epsilon) * strategy1 + epsilon / self.n_strategies_p1
            perturbed_strategy2 = (1 - epsilon) * strategy2 + epsilon / self.n_strategies_p2

            # Check if original strategies are still best responses to perturbed strategies
            best_response1 = self._find_best_response(perturbed_strategy2, player=1)
            best_response2 = self._find_best_response(perturbed_strategy1, player=2)

            # Check if the original strategies are close to the best responses
            tolerance = 0.1
            is_perfect1 = self._strategies_close(strategy1, best_response1, tolerance)
            is_perfect2 = self._strategies_close(strategy2, best_response2, tolerance)

            return is_perfect1 and is_perfect2

        except Exception as e:
            logger.error(f"Error in trembling hand perfection analysis: {e}")
            return False

    def check_sequential_equilibrium(self, game_tree: Optional[Dict] = None) -> bool:
        """
        Placeholder for Sequential Equilibrium analysis.

        Sequential equilibrium analysis requires the extensive form (game tree)
        representation, which is more complex than the normal form used here.

        Args:
            game_tree: Optional game tree representation

        Returns:
            True if sequential equilibrium conditions are met
        """
        # This is a placeholder - sequential equilibrium analysis requires
        # extensive form games and belief systems
        logger.warning("Sequential equilibrium analysis not implemented for normal form games")
        return False

    def analyze_stability(self, equilibria: List[Dict[str, Any]]) -> Dict[str, StabilityMetrics]:
        """
        Runs a comprehensive suite of stability checks on found equilibria.

        Args:
            equilibria: List of equilibrium dictionaries

        Returns:
            Dictionary mapping equilibrium identifiers to stability metrics
        """
        analysis_results = {}

        for i, eq in enumerate(equilibria):
            eq_id = f"equilibrium_{i}"

            try:
                # Initialize metrics
                metrics = StabilityMetrics(
                    is_evolutionary_stable=False,
                    is_trembling_hand_perfect=False,
                    is_sequential_equilibrium=False,
                    robustness_score=0.0,
                    convergence_rate=0.0,
                    deviation_resistance=0.0
                )

                if eq['type'] == 'pure_strategy':
                    # Analyze pure strategy equilibrium
                    strategies = eq['strategies']
                    if len(strategies) == 2:
                        player_names = list(strategies.keys())
                        strategy_indices = (strategies[player_names[0]], strategies[player_names[1]])

                        metrics.is_evolutionary_stable = self.is_evolutionary_stable_strategy(strategy_indices)
                        metrics.robustness_score = self._calculate_robustness_score(strategy_indices)
                        metrics.deviation_resistance = self._calculate_deviation_resistance(strategy_indices)

                elif eq['type'] == 'mixed_strategy':
                    # Analyze mixed strategy equilibrium
                    strategies = eq['strategies']
                    if len(strategies) == 2:
                        player_names = list(strategies.keys())
                        strategy_tuple = (
                            np.array(strategies[player_names[0]]),
                            np.array(strategies[player_names[1]])
                        )

                        metrics.is_trembling_hand_perfect = self.check_trembling_hand_perfection(strategy_tuple)
                        metrics.convergence_rate = self._estimate_convergence_rate(strategy_tuple)

                analysis_results[eq_id] = metrics

            except Exception as e:
                logger.error(f"Error analyzing equilibrium {eq_id}: {e}")

        return analysis_results

    def _is_symmetric_game(self) -> bool:
        """Check if the game is symmetric (u1 = u2.T)."""
        try:
            return np.allclose(self.u1, self.u2.T, rtol=1e-5)
        except Exception:
            return False

    def _find_best_response(self, opponent_strategy: np.ndarray, player: int) -> np.ndarray:
        """
        Find the best response strategy for a player given opponent's strategy.

        Args:
            opponent_strategy: Opponent's mixed strategy
            player: Player number (1 or 2)

        Returns:
            Best response strategy as probability distribution
        """
        if player == 1:
            expected_payoffs = self.u1 @ opponent_strategy
        else:
            expected_payoffs = self.u2.T @ opponent_strategy

        # Pure strategy best response
        best_action = np.argmax(expected_payoffs)
        best_response = np.zeros(len(expected_payoffs))
        best_response[best_action] = 1.0

        return best_response

    def _strategies_close(self, strategy1: np.ndarray, strategy2: np.ndarray,
                         tolerance: float) -> bool:
        """Check if two strategies are close within tolerance."""
        return np.allclose(strategy1, strategy2, atol=tolerance)

    def _calculate_robustness_score(self, strategy_indices: Tuple[int, int]) -> float:
        """
        Calculate a robustness score for a pure strategy equilibrium.

        Args:
            strategy_indices: Indices of the equilibrium strategies

        Returns:
            Robustness score between 0 and 1
        """
        try:
            s1, s2 = strategy_indices
            eq_payoff1 = self.u1[s1, s2]
            eq_payoff2 = self.u2[s1, s2]

            # Calculate how much better this equilibrium is compared to deviations
            max_deviation_payoff1 = np.max(self.u1[:, s2])
            max_deviation_payoff2 = np.max(self.u2[s1, :])

            # Robustness is based on the payoff advantage of staying in equilibrium
            advantage1 = eq_payoff1 - np.mean(self.u1[:, s2])
            advantage2 = eq_payoff2 - np.mean(self.u2[s1, :])

            # Normalize to [0, 1]
            max_possible_advantage1 = np.max(self.u1) - np.min(self.u1)
            max_possible_advantage2 = np.max(self.u2) - np.min(self.u2)

            if max_possible_advantage1 > 0 and max_possible_advantage2 > 0:
                robustness1 = max(0, advantage1 / max_possible_advantage1)
                robustness2 = max(0, advantage2 / max_possible_advantage2)
                return (robustness1 + robustness2) / 2
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Error calculating robustness score: {e}")
            return 0.0

    def _calculate_deviation_resistance(self, strategy_indices: Tuple[int, int]) -> float:
        """
        Calculate how resistant the equilibrium is to unilateral deviations.

        Args:
            strategy_indices: Indices of the equilibrium strategies

        Returns:
            Deviation resistance score between 0 and 1
        """
        try:
            s1, s2 = strategy_indices

            # Calculate the cost of deviating for each player
            eq_payoff1 = self.u1[s1, s2]
            eq_payoff2 = self.u2[s1, s2]

            # Find the best alternative strategies
            alt_payoffs1 = np.delete(self.u1[:, s2], s1)
            alt_payoffs2 = np.delete(self.u2[s1, :], s2)

            if len(alt_payoffs1) > 0 and len(alt_payoffs2) > 0:
                best_alt_payoff1 = np.max(alt_payoffs1)
                best_alt_payoff2 = np.max(alt_payoffs2)

                # Resistance is based on the payoff loss from deviation
                resistance1 = max(0, eq_payoff1 - best_alt_payoff1)
                resistance2 = max(0, eq_payoff2 - best_alt_payoff2)

                # Normalize
                max_payoff_range1 = np.max(self.u1) - np.min(self.u1)
                max_payoff_range2 = np.max(self.u2) - np.min(self.u2)

                if max_payoff_range1 > 0 and max_payoff_range2 > 0:
                    norm_resistance1 = resistance1 / max_payoff_range1
                    norm_resistance2 = resistance2 / max_payoff_range2
                    return (norm_resistance1 + norm_resistance2) / 2

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating deviation resistance: {e}")
            return 0.0

    def _estimate_convergence_rate(self, strategy_tuple: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Estimate the convergence rate for a mixed strategy equilibrium.

        This is a simplified estimation - in practice, this would require
        dynamic analysis of the learning process.

        Args:
            strategy_tuple: Tuple of mixed strategies

        Returns:
            Estimated convergence rate
        """
        try:
            strategy1, strategy2 = strategy_tuple

            # Simple heuristic: more concentrated strategies converge faster
            concentration1 = np.sum(strategy1 ** 2)  # Gini coefficient-like measure
            concentration2 = np.sum(strategy2 ** 2)

            # Average concentration as a proxy for convergence rate
            avg_concentration = (concentration1 + concentration2) / 2

            return min(1.0, avg_concentration)

        except Exception as e:
            logger.error(f"Error estimating convergence rate: {e}")
            return 0.0


class SimulationAnalyzer:
    """
    Analyzes the results of repeated game simulations.
    """

    def __init__(self, simulation_history: List[RoundResult]):
        """
        Initialize with simulation history.

        Args:
            simulation_history: List of RoundResult objects from simulation
        """
        self.history = simulation_history
        self.num_rounds = len(simulation_history)

        logger.info(f"Initialized SimulationAnalyzer with {self.num_rounds} rounds of data")

    def analyze_convergence(self, window_size: int = 20) -> Dict[str, Any]:
        """
        Analyze whether and how strategies converged over time.

        Args:
            window_size: Size of the moving window for convergence analysis

        Returns:
            Dictionary containing convergence analysis results
        """
        if self.num_rounds < window_size * 2:
            return {"error": "Insufficient data for convergence analysis"}

        convergence_analysis = {
            "window_size": window_size,
            "convergence_detected": False,
            "convergence_round": None,
            "strategy_stability": {},
            "payoff_stability": {}
        }

        # Analyze strategy convergence
        strategy_variance_over_time = self._calculate_strategy_variance_over_time(window_size)
        convergence_analysis["strategy_variance"] = strategy_variance_over_time

        # Detect convergence point
        convergence_round = self._detect_convergence_point(strategy_variance_over_time)
        if convergence_round is not None:
            convergence_analysis["convergence_detected"] = True
            convergence_analysis["convergence_round"] = convergence_round

        # Analyze payoff stability
        payoff_stability = self._analyze_payoff_stability(window_size)
        convergence_analysis["payoff_stability"] = payoff_stability

        return convergence_analysis

    def analyze_learning_dynamics(self) -> Dict[str, Any]:
        """
        Analyze how players' strategies evolved through learning.

        Returns:
            Dictionary containing learning dynamics analysis
        """
        learning_analysis = {
            "reputation_evolution": self._analyze_reputation_evolution(),
            "strategy_adaptation": self._analyze_strategy_adaptation(),
            "performance_trends": self._analyze_performance_trends()
        }

        return learning_analysis

    def calculate_efficiency_metrics(self) -> Dict[str, Any]:
        """
        Calculate various efficiency metrics for the game outcomes.

        Returns:
            Dictionary containing efficiency metrics
        """
        if not self.history:
            return {"error": "No data available"}

        # Calculate social welfare over time
        social_welfare = []
        pareto_efficiency = []

        for result in self.history:
            total_payoff = sum(result.payoffs.values())
            social_welfare.append(total_payoff)

            # Simplified Pareto efficiency check
            is_pareto_efficient = self._check_pareto_efficiency(result.payoffs)
            pareto_efficiency.append(is_pareto_efficient)

        return {
            "social_welfare": {
                "mean": np.mean(social_welfare),
                "std": np.std(social_welfare),
                "trend": self._calculate_trend(social_welfare)
            },
            "pareto_efficiency_rate": np.mean(pareto_efficiency),
            "efficiency_over_time": social_welfare
        }

    def _calculate_strategy_variance_over_time(self, window_size: int) -> List[float]:
        """Calculate the variance in strategy choices over moving windows."""
        # This is a simplified implementation
        # In practice, this would track actual strategy distributions
        variance_over_time = []

        for i in range(window_size, self.num_rounds - window_size):
            window_results = self.history[i-window_size:i+window_size]
            # Simplified variance calculation based on payoff variance
            payoffs = [sum(result.payoffs.values()) for result in window_results]
            variance = np.var(payoffs)
            variance_over_time.append(variance)

        return variance_over_time

    def _detect_convergence_point(self, variance_series: List[float],
                                threshold: float = 0.1) -> Optional[int]:
        """Detect the point where variance drops below threshold (convergence)."""
        for i, variance in enumerate(variance_series):
            if variance < threshold:
                # Check if it stays below threshold for some time
                if all(v < threshold for v in variance_series[i:i+10]):
                    return i
        return None

    def _analyze_payoff_stability(self, window_size: int) -> Dict[str, float]:
        """Analyze the stability of payoffs over time."""
        stability_metrics = {}

        # Calculate payoff stability for each player type
        player_types = set()
        for result in self.history:
            player_types.update(result.payoffs.keys())

        for player_type in player_types:
            payoffs = [result.payoffs.get(player_type, 0) for result in self.history]

            # Calculate stability as inverse of coefficient of variation
            if len(payoffs) > 0:
                mean_payoff = np.mean(payoffs)
                std_payoff = np.std(payoffs)
                cv = std_payoff / abs(mean_payoff) if mean_payoff != 0 else float('inf')
                stability = 1 / (1 + cv)  # Stability between 0 and 1
                stability_metrics[player_type] = stability

        return stability_metrics

    def _analyze_reputation_evolution(self) -> Dict[str, List[float]]:
        """Analyze how player reputations evolved over time."""
        reputation_evolution = {}

        for result in self.history:
            for player_id, state in result.player_states.items():
                if player_id not in reputation_evolution:
                    reputation_evolution[player_id] = []
                reputation_evolution[player_id].append(state.get('reputation', 0.5))

        return reputation_evolution

    def _analyze_strategy_adaptation(self) -> Dict[str, Any]:
        """Analyze how strategies adapted over time."""
        # This would track actual strategy choices in a more complete implementation
        return {"placeholder": "Strategy adaptation analysis not fully implemented"}

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends for each player type."""
        performance_trends = {}

        # Calculate performance trends for each player type
        player_types = set()
        for result in self.history:
            player_types.update(result.payoffs.keys())

        for player_type in player_types:
            payoffs = [result.payoffs.get(player_type, 0) for result in self.history]
            trend = self._calculate_trend(payoffs)
            performance_trends[player_type] = {
                "trend": trend,
                "final_average": np.mean(payoffs[-10:]) if len(payoffs) >= 10 else np.mean(payoffs)
            }

        return performance_trends

    def _check_pareto_efficiency(self, payoffs: Dict[str, float]) -> bool:
        """
        Simplified Pareto efficiency check.

        In practice, this would require comparing to all possible alternative outcomes.
        """
        # This is a placeholder - real Pareto efficiency requires complete information
        # about all possible outcomes
        return True  # Simplified assumption

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate the trend (slope) of a series of values."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)