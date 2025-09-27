# backend/game_theory/simulation.py

import numpy as np
import random
from typing import List, Dict, Any, Optional, Callable
import logging
from dataclasses import dataclass, field
from .players import Player, Spreader, FactChecker, Platform
from .strategies import (
    SpreaderStrategy, FactCheckerStrategy, PlatformStrategy,
    create_default_spreader_strategy, create_default_fact_checker_strategy,
    create_default_platform_strategy
)
from .payoffs import calculate_payoffs, get_default_payoff_weights

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration parameters for the repeated game simulation."""
    num_rounds: int = 100
    learning_rate: float = 0.1
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    reputation_update_rate: float = 0.05
    network_effects: bool = True
    random_seed: Optional[int] = None


@dataclass
class RoundResult:
    """Results from a single round of the game."""
    round_number: int
    strategies: Dict[str, Any]
    outcome: Dict[str, Any]
    payoffs: Dict[str, float]
    player_states: Dict[str, Dict[str, Any]]
    reputation_changes: Dict[str, float] = field(default_factory=dict)


class LearningModule:
    """Handles strategy adaptation and learning for players."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.q_tables = {}  # Q-learning tables for each player
        self.strategy_counts = {}  # Track strategy usage frequency

    def initialize_player(self, player_id: str, strategy_space_size: int):
        """Initialize learning structures for a player."""
        self.q_tables[player_id] = np.zeros(strategy_space_size)
        self.strategy_counts[player_id] = np.zeros(strategy_space_size)

    def update_q_values(self, player_id: str, strategy_index: int, reward: float):
        """Update Q-values using Q-learning algorithm."""
        if player_id in self.q_tables:
            current_q = self.q_tables[player_id][strategy_index]
            self.q_tables[player_id][strategy_index] = current_q + self.config.learning_rate * (reward - current_q)

    def select_strategy(self, player_id: str, available_strategies: List[Any],
                       exploration_rate: Optional[float] = None) -> int:
        """
        Select a strategy using epsilon-greedy exploration.

        Args:
            player_id: ID of the player selecting strategy
            available_strategies: List of available strategies
            exploration_rate: Override exploration rate

        Returns:
            Index of selected strategy
        """
        if player_id not in self.q_tables:
            return random.randint(0, len(available_strategies) - 1)

        exploration = exploration_rate or self.config.exploration_rate

        if random.random() < exploration:
            # Explore: random strategy
            strategy_index = random.randint(0, len(available_strategies) - 1)
        else:
            # Exploit: best known strategy
            strategy_index = np.argmax(self.q_tables[player_id])
            if strategy_index >= len(available_strategies):
                strategy_index = 0

        # Update strategy count
        self.strategy_counts[player_id][strategy_index] += 1
        return strategy_index

    def get_strategy_distribution(self, player_id: str) -> Optional[np.ndarray]:
        """Get the probability distribution over strategies for a player."""
        if player_id not in self.strategy_counts:
            return None

        counts = self.strategy_counts[player_id]
        total = np.sum(counts)
        if total == 0:
            return np.ones(len(counts)) / len(counts)

        return counts / total


class NetworkSimulator:
    """Simulates network effects and information propagation."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.base_reach = 1000
        self.viral_multiplier = 2.5
        self.detection_probability = 0.3

    def simulate_propagation(self, strategies: Dict[str, Any],
                           player_reputations: Dict[str, float]) -> Dict[str, Any]:
        """
        Simulate information propagation based on strategies and network effects.

        Args:
            strategies: Current strategies of all players
            player_reputations: Current reputation scores

        Returns:
            Dictionary containing propagation outcomes
        """
        outcome = {
            'reach': 0,
            'detection_penalty': 0,
            'reputation_cost': 0,
            'accuracy': 0.5,
            'effort_cost': 0,
            'reputation_gain': 0,
            'user_engagement': 0,
            'moderation_cost': 0,
            'credibility_score': 0.5,
            'regulatory_risk': 0
        }

        # Extract strategies
        spreader_strategy = strategies.get('spreader')
        fact_checker_strategy = strategies.get('fact_checker')
        platform_strategy = strategies.get('platform')

        if spreader_strategy:
            # Calculate content reach
            base_reach = self.base_reach
            reach_multiplier = 1.0

            # Content type affects reach
            if spreader_strategy.content_type.value == 'fake_emotional':
                reach_multiplier *= self.viral_multiplier
            elif spreader_strategy.content_type.value == 'fake_neutral':
                reach_multiplier *= 1.5

            # Timing affects reach
            if spreader_strategy.timing.value == 'peak_hours':
                reach_multiplier *= 1.3

            # Reputation affects reach
            spreader_reputation = player_reputations.get('spreader', 0.5)
            reach_multiplier *= (0.5 + spreader_reputation)

            outcome['reach'] = int(base_reach * reach_multiplier)

        if fact_checker_strategy and spreader_strategy:
            # Calculate detection probability
            detection_prob = self.detection_probability

            # Fact checker intensity affects detection
            if fact_checker_strategy.intensity.value == 'comprehensive':
                detection_prob *= 1.8
            elif fact_checker_strategy.intensity.value == 'thorough':
                detection_prob *= 1.4

            # Content type affects detection difficulty
            if spreader_strategy.content_type.value == 'fake_emotional':
                detection_prob *= 1.2  # Easier to detect
            elif spreader_strategy.content_type.value == 'true':
                detection_prob *= 0.1  # Harder to "detect" true content as false

            # Calculate outcomes
            is_detected = random.random() < detection_prob

            if is_detected and spreader_strategy.content_type.value != 'true':
                outcome['detection_penalty'] = 100
                outcome['reputation_cost'] = 50
                outcome['accuracy'] = 0.9
                outcome['reputation_gain'] = 20
            elif not is_detected and spreader_strategy.content_type.value != 'true':
                outcome['accuracy'] = 0.1
                outcome['reputation_gain'] = -10
            else:
                outcome['accuracy'] = 0.8

            # Effort cost based on intensity
            effort_costs = {'basic': 10, 'thorough': 25, 'comprehensive': 50}
            outcome['effort_cost'] = effort_costs.get(fact_checker_strategy.intensity.value, 10)

        if platform_strategy:
            # Calculate platform metrics
            base_engagement = 500
            engagement_multiplier = 1.0

            # Policy affects engagement and credibility
            if platform_strategy.policy.value == 'permissive':
                engagement_multiplier *= 1.4
                outcome['credibility_score'] = 0.3
                outcome['regulatory_risk'] = 50
            elif platform_strategy.policy.value == 'strict':
                engagement_multiplier *= 0.7
                outcome['credibility_score'] = 0.8
                outcome['regulatory_risk'] = 10
            else:  # moderate
                engagement_multiplier *= 1.0
                outcome['credibility_score'] = 0.6
                outcome['regulatory_risk'] = 25

            outcome['user_engagement'] = base_engagement * engagement_multiplier

            # Moderation cost based on enforcement
            moderation_costs = {'lenient': 20, 'standard': 40, 'aggressive': 80}
            outcome['moderation_cost'] = moderation_costs.get(
                platform_strategy.enforcement_level.value, 40
            )

        return outcome


class RepeatedGameSimulation:
    """
    Manages the execution of the game over multiple rounds with player adaptation.
    """

    def __init__(self, players: List[Player], config: Optional[SimulationConfig] = None):
        """
        Initialize the repeated game simulation.

        Args:
            players: List of Player instances
            config: Simulation configuration parameters
        """
        self.players = {player.player_id: player for player in players}
        self.config = config or SimulationConfig()
        self.history: List[RoundResult] = []

        # Initialize components
        self.learning_module = LearningModule(self.config)
        self.network_simulator = NetworkSimulator(self.config)
        self.payoff_weights = get_default_payoff_weights()

        # Set random seed if provided
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Initialize learning for each player
        self._initialize_learning()

        logger.info(f"Initialized simulation with {len(self.players)} players for {self.config.num_rounds} rounds")

    def _initialize_learning(self):
        """Initialize learning structures for all players."""
        strategy_space_sizes = {
            'spreader': 6,  # 3 content types × 2 timings
            'fact_checker': 12,  # 3 intensities × 4 speeds × 1 (simplified)
            'platform': 9  # 3 policies × 3 enforcement levels
        }

        for player_id, player in self.players.items():
            player_type = player.player_type.lower()
            if player_type in strategy_space_sizes:
                self.learning_module.initialize_player(player_id, strategy_space_sizes[player_type])

    def _select_strategies(self, round_number: int) -> Dict[str, Any]:
        """
        Select strategies for all players for the current round.

        Args:
            round_number: Current round number

        Returns:
            Dictionary mapping player types to selected strategies
        """
        strategies = {}
        current_exploration = self.config.exploration_rate * (self.config.exploration_decay ** round_number)

        for player_id, player in self.players.items():
            player_type = player.player_type.lower()

            if player_type == 'spreader':
                # Simplified strategy selection - would be more sophisticated in practice
                strategy_index = self.learning_module.select_strategy(
                    player_id, list(range(6)), current_exploration
                )
                strategies['spreader'] = self._index_to_spreader_strategy(strategy_index)

            elif player_type == 'factchecker':
                strategy_index = self.learning_module.select_strategy(
                    player_id, list(range(12)), current_exploration
                )
                strategies['fact_checker'] = self._index_to_fact_checker_strategy(strategy_index)

            elif player_type == 'platform':
                strategy_index = self.learning_module.select_strategy(
                    player_id, list(range(9)), current_exploration
                )
                strategies['platform'] = self._index_to_platform_strategy(strategy_index)

        return strategies

    def _index_to_spreader_strategy(self, index: int) -> SpreaderStrategy:
        """Convert strategy index to SpreaderStrategy object."""
        from .strategies import SpreaderContentType, SpreaderTiming, SpreaderTargeting

        content_types = list(SpreaderContentType)
        timings = list(SpreaderTiming)

        content_index = index // 2
        timing_index = index % 2

        return SpreaderStrategy(
            content_type=content_types[content_index % len(content_types)],
            timing=timings[timing_index],
            targeting=SpreaderTargeting.BROAD
        )

    def _index_to_fact_checker_strategy(self, index: int) -> FactCheckerStrategy:
        """Convert strategy index to FactCheckerStrategy object."""
        from .strategies import VerificationIntensity, ResponseSpeed, ResourceAllocation

        intensities = list(VerificationIntensity)
        speeds = list(ResponseSpeed)

        intensity_index = index // 4
        speed_index = index % 4

        return FactCheckerStrategy(
            intensity=intensities[intensity_index % len(intensities)],
            response_speed=speeds[speed_index % len(speeds)],
            resource_allocation=ResourceAllocation.BALANCED
        )

    def _index_to_platform_strategy(self, index: int) -> PlatformStrategy:
        """Convert strategy index to PlatformStrategy object."""
        from .strategies import ContentPolicy, EnforcementLevel, TransparencyLevel

        policies = list(ContentPolicy)
        enforcement_levels = list(EnforcementLevel)

        policy_index = index // 3
        enforcement_index = index % 3

        return PlatformStrategy(
            policy=policies[policy_index % len(policies)],
            enforcement_level=enforcement_levels[enforcement_index % len(enforcement_levels)],
            transparency=TransparencyLevel.PARTIAL
        )

    def _simulate_round(self, round_number: int, strategies: Dict[str, Any]) -> RoundResult:
        """
        Simulate a single round of the game.

        Args:
            round_number: Current round number
            strategies: Selected strategies for this round

        Returns:
            RoundResult containing all round outcomes
        """
        # Get current player reputations
        player_reputations = {
            player_id: player.get_reputation_score()
            for player_id, player in self.players.items()
        }

        # Simulate network effects and outcomes
        game_outcome = self.network_simulator.simulate_propagation(strategies, player_reputations)

        # Calculate payoffs
        payoffs = calculate_payoffs(strategies, game_outcome, self.payoff_weights)

        # Update player states
        reputation_changes = self._update_player_states(game_outcome, payoffs)

        # Capture player states
        player_states = {
            player_id: {
                'reputation': player.get_reputation_score(),
                'type': player.player_type
            }
            for player_id, player in self.players.items()
        }

        return RoundResult(
            round_number=round_number,
            strategies=strategies,
            outcome=game_outcome,
            payoffs=payoffs,
            player_states=player_states,
            reputation_changes=reputation_changes
        )

    def _update_player_states(self, game_outcome: Dict[str, Any],
                            payoffs: Dict[str, float]) -> Dict[str, float]:
        """
        Update player reputations and states based on round outcomes.

        Args:
            game_outcome: Outcomes from the round
            payoffs: Calculated payoffs for each player

        Returns:
            Dictionary of reputation changes for each player
        """
        reputation_changes = {}

        for player_id, player in self.players.items():
            player_type = player.player_type.lower()
            reputation_change = 0.0

            if player_type == 'spreader':
                # Spreader reputation decreases if detected spreading fake news
                if game_outcome.get('detection_penalty', 0) > 0:
                    reputation_change = -self.config.reputation_update_rate
                else:
                    # Small positive change for not being detected
                    reputation_change = self.config.reputation_update_rate * 0.1

            elif player_type == 'factchecker':
                # Fact checker reputation based on accuracy
                accuracy = game_outcome.get('accuracy', 0.5)
                if accuracy > 0.7:
                    reputation_change = self.config.reputation_update_rate
                elif accuracy < 0.3:
                    reputation_change = -self.config.reputation_update_rate

            elif player_type == 'platform':
                # Platform reputation based on credibility score
                credibility = game_outcome.get('credibility_score', 0.5)
                if credibility > 0.6:
                    reputation_change = self.config.reputation_update_rate * 0.5
                elif credibility < 0.4:
                    reputation_change = -self.config.reputation_update_rate * 0.5

            # Apply reputation change
            player.update_reputation(reputation_change)
            reputation_changes[player_id] = reputation_change

            # Update learning based on payoffs
            player_payoff = payoffs.get(player_type, 0.0)
            # This would need strategy index tracking for proper Q-learning update
            # For now, we'll use a simplified approach

        return reputation_changes

    def run_simulation(self) -> List[RoundResult]:
        """
        Execute the full multi-round simulation.

        Returns:
            List of RoundResult objects containing complete simulation history
        """
        logger.info(f"Starting simulation of {self.config.num_rounds} rounds")

        for round_num in range(self.config.num_rounds):
            # Select strategies for all players
            strategies = self._select_strategies(round_num)

            # Simulate the round
            round_result = self._simulate_round(round_num, strategies)

            # Store result
            self.history.append(round_result)

            # Log progress
            if (round_num + 1) % 20 == 0:
                logger.info(f"Completed round {round_num + 1}/{self.config.num_rounds}")

        logger.info(f"Simulation of {self.config.num_rounds} rounds complete")
        return self.history

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the simulation results.

        Returns:
            Dictionary containing simulation statistics and analysis
        """
        if not self.history:
            return {"error": "No simulation data available"}

        # Calculate average payoffs over time
        player_payoffs = {}
        for result in self.history:
            for player_type, payoff in result.payoffs.items():
                if player_type not in player_payoffs:
                    player_payoffs[player_type] = []
                player_payoffs[player_type].append(payoff)

        # Calculate final reputations
        final_reputations = {}
        for player_id, player in self.players.items():
            final_reputations[player_id] = player.get_reputation_score()

        # Strategy distributions
        strategy_distributions = {}
        for player_id in self.players.keys():
            distribution = self.learning_module.get_strategy_distribution(player_id)
            if distribution is not None:
                strategy_distributions[player_id] = distribution.tolist()

        return {
            "total_rounds": len(self.history),
            "average_payoffs": {
                player: np.mean(payoffs) for player, payoffs in player_payoffs.items()
            },
            "final_reputations": final_reputations,
            "strategy_distributions": strategy_distributions,
            "convergence_analysis": self._analyze_convergence()
        }

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze whether strategies have converged over time."""
        if len(self.history) < 10:
            return {"status": "insufficient_data"}

        # Simple convergence analysis based on strategy variance in recent rounds
        recent_rounds = self.history[-20:]  # Last 20 rounds

        # This would be more sophisticated in practice
        return {
            "status": "analysis_placeholder",
            "recent_rounds_analyzed": len(recent_rounds)
        }