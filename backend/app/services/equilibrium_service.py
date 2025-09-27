# backend/app/services/equilibrium_service.py

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import itertools

from game_theory import (
    NashEquilibriumSolver,
    GameMatrixBuilder,
    EquilibriumAnalyzer,
    SpreaderStrategy,
    FactCheckerStrategy,
    PlatformStrategy,
    calculate_payoffs,
    get_default_payoff_weights
)
from game_theory.strategies import (
    SpreaderContentType,
    SpreaderTiming,
    SpreaderTargeting,
    VerificationIntensity,
    ResponseSpeed,
    ResourceAllocation,
    ContentPolicy,
    EnforcementLevel,
    TransparencyLevel
)

logger = logging.getLogger(__name__)


class EquilibriumService:
    """Handles the calculation and analysis of Nash equilibria."""

    def __init__(self):
        """Initialize the equilibrium service."""
        self.payoff_weights = get_default_payoff_weights()
        self.strategy_spaces = self._define_strategy_spaces()

        logger.info("EquilibriumService initialized")

    def _define_strategy_spaces(self) -> Dict[str, List[Any]]:
        """Define the strategy spaces for each player type."""
        spreader_strategies = []
        for content_type in SpreaderContentType:
            for timing in SpreaderTiming:
                for targeting in SpreaderTargeting:
                    spreader_strategies.append(SpreaderStrategy(
                        content_type=content_type,
                        timing=timing,
                        targeting=targeting
                    ))

        fact_checker_strategies = []
        for intensity in VerificationIntensity:
            for speed in ResponseSpeed:
                for allocation in ResourceAllocation:
                    fact_checker_strategies.append(FactCheckerStrategy(
                        intensity=intensity,
                        response_speed=speed,
                        resource_allocation=allocation
                    ))

        platform_strategies = []
        for policy in ContentPolicy:
            for enforcement in EnforcementLevel:
                for transparency in TransparencyLevel:
                    platform_strategies.append(PlatformStrategy(
                        policy=policy,
                        enforcement_level=enforcement,
                        transparency=transparency
                    ))

        return {
            'spreader': spreader_strategies,
            'fact_checker': fact_checker_strategies,
            'platform': platform_strategies
        }

    async def calculate_equilibria(self,
                                 game_params: Dict[str, Any],
                                 include_mixed: bool = True,
                                 include_stability_analysis: bool = True) -> Dict[str, Any]:
        """
        Find and analyze equilibria for the given game parameters.

        Args:
            game_params: Game configuration parameters
            include_mixed: Whether to compute mixed strategy equilibria
            include_stability_analysis: Whether to perform stability analysis

        Returns:
            Dictionary containing equilibrium analysis results
        """
        try:
            logger.info("Starting equilibrium calculation")

            # Update payoff weights if provided
            if 'payoff_weights' in game_params:
                custom_weights = game_params['payoff_weights']
                self.payoff_weights.update(custom_weights)

            # Generate outcome scenarios for payoff matrix construction
            outcome_scenarios = self._generate_outcome_scenarios(game_params)

            # Build payoff matrices
            logger.info("Constructing payoff matrices")
            payoff_matrices = await self._construct_payoff_matrices(
                game_params, outcome_scenarios
            )

            # Initialize solver
            solver = NashEquilibriumSolver(payoff_matrices)

            # Find pure strategy equilibria
            logger.info("Finding pure strategy equilibria")
            pure_equilibria = solver.find_pure_strategy_equilibria()

            results = {
                "game_parameters": game_params,
                "timestamp": datetime.utcnow().isoformat(),
                "strategy_space_sizes": {
                    player: len(strategies)
                    for player, strategies in self.strategy_spaces.items()
                },
                "pure_strategy_equilibria": pure_equilibria,
                "equilibrium_count": {
                    "pure": len(pure_equilibria)
                }
            }

            # Find mixed strategy equilibria if requested
            if include_mixed:
                logger.info("Finding mixed strategy equilibria")
                try:
                    mixed_equilibria = solver.find_mixed_strategy_equilibria()
                    results["mixed_strategy_equilibria"] = mixed_equilibria
                    results["equilibrium_count"]["mixed"] = len(mixed_equilibria)
                except Exception as e:
                    logger.warning(f"Mixed strategy equilibrium calculation failed: {e}")
                    results["mixed_strategy_equilibria"] = []
                    results["equilibrium_count"]["mixed"] = 0

            # Stability analysis if requested
            if include_stability_analysis and pure_equilibria:
                logger.info("Performing stability analysis")
                try:
                    analyzer = EquilibriumAnalyzer(
                        payoff_matrices.get('spreader', np.array([[0]])),
                        payoff_matrices.get('fact_checker', np.array([[0]]))
                    )
                    stability_results = analyzer.analyze_stability(pure_equilibria)
                    results["stability_analysis"] = stability_results
                except Exception as e:
                    logger.warning(f"Stability analysis failed: {e}")
                    results["stability_analysis"] = {}

            # Add interpretation
            results["interpretation"] = self._interpret_equilibria(results)

            logger.info(f"Equilibrium calculation completed. Found {len(pure_equilibria)} pure equilibria")
            return results

        except Exception as e:
            logger.error(f"Failed to calculate equilibria: {e}")
            raise

    def _generate_outcome_scenarios(self, game_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate representative outcome scenarios for payoff matrix construction.

        Args:
            game_params: Game configuration parameters

        Returns:
            List of outcome scenario dictionaries
        """
        base_scenarios = [
            # Scenario 1: High detection, low reach
            {
                'reach': 100,
                'detection_penalty': 80,
                'reputation_cost': 60,
                'accuracy': 0.9,
                'effort_cost': 50,
                'reputation_gain': 30,
                'user_engagement': 200,
                'moderation_cost': 40,
                'credibility_score': 0.8,
                'regulatory_risk': 20
            },
            # Scenario 2: Low detection, high reach
            {
                'reach': 1000,
                'detection_penalty': 10,
                'reputation_cost': 5,
                'accuracy': 0.2,
                'effort_cost': 20,
                'reputation_gain': -10,
                'user_engagement': 800,
                'moderation_cost': 15,
                'credibility_score': 0.3,
                'regulatory_risk': 60
            },
            # Scenario 3: Moderate case
            {
                'reach': 500,
                'detection_penalty': 40,
                'reputation_cost': 30,
                'accuracy': 0.6,
                'effort_cost': 35,
                'reputation_gain': 10,
                'user_engagement': 450,
                'moderation_cost': 25,
                'credibility_score': 0.6,
                'regulatory_risk': 35
            }
        ]

        # Add parameter-specific variations
        scenarios = []
        for base_scenario in base_scenarios:
            # Apply parameter modifiers
            scenario = base_scenario.copy()

            # Network effects
            network_size = game_params.get('network_size', 1000)
            if network_size > 5000:
                scenario['reach'] *= 2
                scenario['user_engagement'] *= 1.5

            # Detection capability adjustments
            detection_capability = game_params.get('detection_capability', 1.0)
            scenario['accuracy'] *= detection_capability
            scenario['detection_penalty'] *= detection_capability

            scenarios.append(scenario)

        return scenarios

    async def _construct_payoff_matrices(self,
                                       game_params: Dict[str, Any],
                                       outcome_scenarios: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Construct payoff matrices by evaluating outcomes for all strategy combinations.

        Args:
            game_params: Game configuration parameters
            outcome_scenarios: List of outcome scenarios

        Returns:
            Dictionary mapping player names to their payoff matrices
        """
        try:
            # For computational efficiency, we'll create a simplified 2-player game
            # between spreader and fact_checker, with platform as fixed
            spreader_strategies = self.strategy_spaces['spreader']
            fact_checker_strategies = self.strategy_spaces['fact_checker']

            # Limit strategy spaces for computational feasibility
            max_strategies = game_params.get('max_strategies_per_player', 5)
            spreader_strategies = spreader_strategies[:max_strategies]
            fact_checker_strategies = fact_checker_strategies[:max_strategies]

            # Initialize payoff matrices
            n_spreader = len(spreader_strategies)
            n_fact_checker = len(fact_checker_strategies)

            spreader_payoffs = np.zeros((n_spreader, n_fact_checker))
            fact_checker_payoffs = np.zeros((n_spreader, n_fact_checker))

            # Use a default platform strategy
            default_platform_strategy = self.strategy_spaces['platform'][0]

            logger.info(f"Computing payoffs for {n_spreader}x{n_fact_checker} strategy combinations")

            # Compute payoffs for each strategy combination
            for i, spreader_strategy in enumerate(spreader_strategies):
                for j, fact_checker_strategy in enumerate(fact_checker_strategies):
                    # Average payoffs across all outcome scenarios
                    total_spreader_payoff = 0.0
                    total_fact_checker_payoff = 0.0

                    for scenario in outcome_scenarios:
                        strategies = {
                            'spreader': spreader_strategy,
                            'fact_checker': fact_checker_strategy,
                            'platform': default_platform_strategy
                        }

                        payoffs = calculate_payoffs(strategies, scenario, self.payoff_weights)

                        total_spreader_payoff += payoffs.get('spreader', 0.0)
                        total_fact_checker_payoff += payoffs.get('fact_checker', 0.0)

                    # Average across scenarios
                    num_scenarios = len(outcome_scenarios)
                    spreader_payoffs[i, j] = total_spreader_payoff / num_scenarios
                    fact_checker_payoffs[i, j] = total_fact_checker_payoff / num_scenarios

            return {
                'spreader': spreader_payoffs,
                'fact_checker': fact_checker_payoffs
            }

        except Exception as e:
            logger.error(f"Failed to construct payoff matrices: {e}")
            raise

    def _interpret_equilibria(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide interpretation and insights for the calculated equilibria.

        Args:
            results: Equilibrium calculation results

        Returns:
            Dictionary containing interpretation insights
        """
        interpretation = {
            "summary": {},
            "strategic_insights": [],
            "policy_implications": []
        }

        pure_equilibria = results.get('pure_strategy_equilibria', [])
        mixed_equilibria = results.get('mixed_strategy_equilibria', [])

        # Summary statistics
        interpretation["summary"] = {
            "total_pure_equilibria": len(pure_equilibria),
            "total_mixed_equilibria": len(mixed_equilibria),
            "equilibrium_existence": len(pure_equilibria) > 0 or len(mixed_equilibria) > 0
        }

        # Analyze pure strategy equilibria
        if pure_equilibria:
            # Extract strategy patterns
            spreader_strategies = []
            fact_checker_strategies = []

            for eq in pure_equilibria:
                strategies = eq.get('strategies', {})
                if 'spreader' in strategies:
                    spreader_strategies.append(strategies['spreader'])
                if 'fact_checker' in strategies:
                    fact_checker_strategies.append(strategies['fact_checker'])

            # Strategic insights
            if spreader_strategies:
                # Analyze dominant spreader strategies
                strategy_counts = {}
                for strategy_idx in spreader_strategies:
                    strategy = self.strategy_spaces['spreader'][strategy_idx]
                    content_type = strategy.content_type.value
                    strategy_counts[content_type] = strategy_counts.get(content_type, 0) + 1

                if strategy_counts:
                    dominant_content_type = max(strategy_counts, key=strategy_counts.get)
                    interpretation["strategic_insights"].append(
                        f"Spreaders most frequently use {dominant_content_type} content in equilibrium"
                    )

            # Policy implications
            if fact_checker_strategies:
                intensity_counts = {}
                for strategy_idx in fact_checker_strategies:
                    strategy = self.strategy_spaces['fact_checker'][strategy_idx]
                    intensity = strategy.intensity.value
                    intensity_counts[intensity] = intensity_counts.get(intensity, 0) + 1

                if intensity_counts:
                    dominant_intensity = max(intensity_counts, key=intensity_counts.get)
                    interpretation["policy_implications"].append(
                        f"Fact-checkers optimally use {dominant_intensity} verification intensity"
                    )

        # Stability insights
        stability_analysis = results.get('stability_analysis', {})
        if stability_analysis:
            stable_count = sum(
                1 for eq_analysis in stability_analysis.values()
                if eq_analysis.get('is_evolutionary_stable', False)
            )
            if stable_count > 0:
                interpretation["strategic_insights"].append(
                    f"{stable_count} equilibria show evolutionary stability"
                )

        return interpretation

    async def analyze_parameter_sensitivity(self,
                                          base_params: Dict[str, Any],
                                          parameter_variations: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze how equilibria change with parameter variations.

        Args:
            base_params: Base game parameters
            parameter_variations: Dictionary mapping parameter names to lists of values

        Returns:
            Dictionary containing sensitivity analysis results
        """
        try:
            logger.info("Starting parameter sensitivity analysis")

            sensitivity_results = {
                "base_parameters": base_params,
                "variations": parameter_variations,
                "results": {},
                "summary": {}
            }

            for param_name, param_values in parameter_variations.items():
                logger.info(f"Analyzing sensitivity to {param_name}")

                param_results = []
                for param_value in param_values:
                    # Create modified parameters
                    modified_params = base_params.copy()
                    modified_params[param_name] = param_value

                    try:
                        # Calculate equilibria for this parameter value
                        equilibria_result = await self.calculate_equilibria(
                            modified_params,
                            include_mixed=False,  # Skip mixed for efficiency
                            include_stability_analysis=False
                        )

                        param_results.append({
                            "parameter_value": param_value,
                            "num_equilibria": len(equilibria_result.get('pure_strategy_equilibria', [])),
                            "equilibria": equilibria_result.get('pure_strategy_equilibria', [])
                        })

                    except Exception as e:
                        logger.warning(f"Failed to calculate equilibria for {param_name}={param_value}: {e}")
                        param_results.append({
                            "parameter_value": param_value,
                            "num_equilibria": 0,
                            "error": str(e)
                        })

                sensitivity_results["results"][param_name] = param_results

                # Calculate summary statistics
                equilibria_counts = [r.get('num_equilibria', 0) for r in param_results]
                sensitivity_results["summary"][param_name] = {
                    "min_equilibria": min(equilibria_counts),
                    "max_equilibria": max(equilibria_counts),
                    "mean_equilibria": sum(equilibria_counts) / len(equilibria_counts),
                    "varies_with_parameter": len(set(equilibria_counts)) > 1
                }

            logger.info("Parameter sensitivity analysis completed")
            return sensitivity_results

        except Exception as e:
            logger.error(f"Parameter sensitivity analysis failed: {e}")
            raise

    async def compare_equilibria_across_scenarios(self,
                                                scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare equilibria across different game scenarios.

        Args:
            scenarios: List of game scenario configurations

        Returns:
            Dictionary containing scenario comparison results
        """
        try:
            logger.info(f"Comparing equilibria across {len(scenarios)} scenarios")

            comparison_results = {
                "scenarios": scenarios,
                "timestamp": datetime.utcnow().isoformat(),
                "results": [],
                "comparison": {}
            }

            all_equilibria = []

            for i, scenario in enumerate(scenarios):
                try:
                    scenario_result = await self.calculate_equilibria(
                        scenario,
                        include_mixed=False,
                        include_stability_analysis=True
                    )

                    scenario_summary = {
                        "scenario_index": i,
                        "scenario_name": scenario.get('name', f'Scenario {i+1}'),
                        "num_equilibria": len(scenario_result.get('pure_strategy_equilibria', [])),
                        "equilibria": scenario_result.get('pure_strategy_equilibria', []),
                        "stability": scenario_result.get('stability_analysis', {})
                    }

                    comparison_results["results"].append(scenario_summary)
                    all_equilibria.extend(scenario_summary["equilibria"])

                except Exception as e:
                    logger.warning(f"Failed to analyze scenario {i}: {e}")
                    comparison_results["results"].append({
                        "scenario_index": i,
                        "scenario_name": scenario.get('name', f'Scenario {i+1}'),
                        "error": str(e)
                    })

            # Cross-scenario analysis
            if all_equilibria:
                comparison_results["comparison"] = self._analyze_equilibria_patterns(all_equilibria)

            logger.info("Scenario comparison completed")
            return comparison_results

        except Exception as e:
            logger.error(f"Scenario comparison failed: {e}")
            raise

    def _analyze_equilibria_patterns(self, equilibria: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across multiple equilibria."""
        if not equilibria:
            return {}

        # Extract strategy indices
        spreader_strategies = []
        fact_checker_strategies = []

        for eq in equilibria:
            strategies = eq.get('strategies', {})
            if 'spreader' in strategies:
                spreader_strategies.append(strategies['spreader'])
            if 'fact_checker' in strategies:
                fact_checker_strategies.append(strategies['fact_checker'])

        # Analyze frequency of strategies
        spreader_frequency = {}
        for strategy_idx in spreader_strategies:
            spreader_frequency[strategy_idx] = spreader_frequency.get(strategy_idx, 0) + 1

        fact_checker_frequency = {}
        for strategy_idx in fact_checker_strategies:
            fact_checker_frequency[strategy_idx] = fact_checker_frequency.get(strategy_idx, 0) + 1

        return {
            "total_equilibria_analyzed": len(equilibria),
            "unique_spreader_strategies": len(set(spreader_strategies)),
            "unique_fact_checker_strategies": len(set(fact_checker_strategies)),
            "most_frequent_spreader_strategy": max(spreader_frequency, key=spreader_frequency.get) if spreader_frequency else None,
            "most_frequent_fact_checker_strategy": max(fact_checker_frequency, key=fact_checker_frequency.get) if fact_checker_frequency else None,
            "strategy_diversity": {
                "spreader": len(set(spreader_strategies)) / len(spreader_strategies) if spreader_strategies else 0,
                "fact_checker": len(set(fact_checker_strategies)) / len(fact_checker_strategies) if fact_checker_strategies else 0
            }
        }

    def get_strategy_space_info(self) -> Dict[str, Any]:
        """Get information about the defined strategy spaces."""
        return {
            "strategy_spaces": {
                player_type: {
                    "count": len(strategies),
                    "examples": [str(strategies[i]) for i in range(min(3, len(strategies)))]
                }
                for player_type, strategies in self.strategy_spaces.items()
            },
            "payoff_weights": self.payoff_weights
        }

    def update_payoff_weights(self, new_weights: Dict[str, Dict[str, float]]) -> bool:
        """
        Update payoff weights for future calculations.

        Args:
            new_weights: New payoff weights

        Returns:
            True if update was successful
        """
        try:
            # Validate weights structure
            from game_theory.payoffs import validate_payoff_weights
            if validate_payoff_weights(new_weights):
                self.payoff_weights = new_weights
                logger.info("Payoff weights updated successfully")
                return True
            else:
                logger.error("Invalid payoff weights structure")
                return False

        except Exception as e:
            logger.error(f"Failed to update payoff weights: {e}")
            return False