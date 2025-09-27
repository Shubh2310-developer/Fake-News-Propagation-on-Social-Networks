# backend/game_theory/payoffs.py

from typing import Dict, Any, Union
import logging
from .strategies import SpreaderStrategy, FactCheckerStrategy, PlatformStrategy

logger = logging.getLogger(__name__)


def calculate_payoffs(
    strategies: Dict[str, Union[SpreaderStrategy, FactCheckerStrategy, PlatformStrategy]],
    game_outcome: Dict[str, Any],
    payoff_weights: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Calculates the utility for each player based on strategies and outcomes.

    Args:
        strategies: Dictionary containing the chosen strategy for each player type
        game_outcome: Dictionary containing the results of the round's simulation
        payoff_weights: Dictionary containing the weights for each player's utility function

    Returns:
        Dictionary mapping player types to their calculated payoffs
    """
    try:
        payoffs = {}

        if 'spreader' in strategies:
            payoffs['spreader'] = _calculate_spreader_payoff(
                strategies['spreader'], game_outcome, payoff_weights.get('spreader', {})
            )

        if 'fact_checker' in strategies:
            payoffs['fact_checker'] = _calculate_fact_checker_payoff(
                strategies['fact_checker'], game_outcome, payoff_weights.get('fact_checker', {})
            )

        if 'platform' in strategies:
            payoffs['platform'] = _calculate_platform_payoff(
                strategies['platform'], game_outcome, payoff_weights.get('platform', {})
            )

        logger.debug(f"Calculated payoffs: {payoffs}")
        return payoffs

    except Exception as e:
        logger.error(f"Error calculating payoffs: {e}")
        raise


def _calculate_spreader_payoff(
    strategy: SpreaderStrategy,
    outcome: Dict[str, Any],
    weights: Dict[str, float]
) -> float:
    """
    Implements the Spreader's utility function:
    U_spreader = ± * Reach - ² * Detection_Penalty - ³ * Reputation_Cost

    Args:
        strategy: The spreader's chosen strategy
        outcome: Game outcome metrics
        weights: Weights for utility function components

    Returns:
        Calculated payoff for the spreader
    """
    # Default weights if not provided
    alpha = weights.get('reach', 1.0)
    beta = weights.get('detection', 0.5)
    gamma = weights.get('reputation', 0.3)

    # Extract outcome metrics
    reach = outcome.get('reach', 0)
    detection_penalty = outcome.get('detection_penalty', 0)
    reputation_cost = outcome.get('reputation_cost', 0)

    # Strategy-based modifiers
    strategy_modifier = _get_spreader_strategy_modifier(strategy)

    # Calculate utility
    utility = (alpha * reach * strategy_modifier) - (beta * detection_penalty) - (gamma * reputation_cost)

    logger.debug(f"Spreader payoff: reach={reach}, detection_penalty={detection_penalty}, "
                f"reputation_cost={reputation_cost}, modifier={strategy_modifier}, utility={utility}")

    return utility


def _calculate_fact_checker_payoff(
    strategy: FactCheckerStrategy,
    outcome: Dict[str, Any],
    weights: Dict[str, float]
) -> float:
    """
    Implements the Fact Checker's utility function:
    U_fact_checker = ± * Accuracy - ² * Effort_Cost + ³ * Reputation_Gain

    Args:
        strategy: The fact checker's chosen strategy
        outcome: Game outcome metrics
        weights: Weights for utility function components

    Returns:
        Calculated payoff for the fact checker
    """
    # Default weights if not provided
    alpha = weights.get('accuracy', 1.0)
    beta = weights.get('effort_cost', 0.4)
    gamma = weights.get('reputation_gain', 0.2)

    # Extract outcome metrics
    accuracy = outcome.get('accuracy', 0.5)
    effort_cost = outcome.get('effort_cost', 0)
    reputation_gain = outcome.get('reputation_gain', 0)

    # Strategy-based modifiers
    strategy_modifier = _get_fact_checker_strategy_modifier(strategy)

    # Calculate utility
    utility = (alpha * accuracy * strategy_modifier) - (beta * effort_cost) + (gamma * reputation_gain)

    logger.debug(f"Fact checker payoff: accuracy={accuracy}, effort_cost={effort_cost}, "
                f"reputation_gain={reputation_gain}, modifier={strategy_modifier}, utility={utility}")

    return utility


def _calculate_platform_payoff(
    strategy: PlatformStrategy,
    outcome: Dict[str, Any],
    weights: Dict[str, float]
) -> float:
    """
    Implements the Platform's utility function:
    U_platform = ± * User_Engagement - ² * Moderation_Cost + ³ * Credibility_Score - ´ * Regulatory_Risk

    Args:
        strategy: The platform's chosen strategy
        outcome: Game outcome metrics
        weights: Weights for utility function components

    Returns:
        Calculated payoff for the platform
    """
    # Default weights if not provided
    alpha = weights.get('engagement', 1.0)
    beta = weights.get('moderation_cost', 0.3)
    gamma = weights.get('credibility', 0.5)
    delta = weights.get('regulatory_risk', 0.4)

    # Extract outcome metrics
    user_engagement = outcome.get('user_engagement', 0)
    moderation_cost = outcome.get('moderation_cost', 0)
    credibility_score = outcome.get('credibility_score', 0.5)
    regulatory_risk = outcome.get('regulatory_risk', 0)

    # Strategy-based modifiers
    strategy_modifier = _get_platform_strategy_modifier(strategy)

    # Calculate utility
    utility = (alpha * user_engagement) - (beta * moderation_cost) + (gamma * credibility_score * strategy_modifier) - (delta * regulatory_risk)

    logger.debug(f"Platform payoff: engagement={user_engagement}, moderation_cost={moderation_cost}, "
                f"credibility={credibility_score}, regulatory_risk={regulatory_risk}, modifier={strategy_modifier}, utility={utility}")

    return utility


def _get_spreader_strategy_modifier(strategy: SpreaderStrategy) -> float:
    """
    Calculate strategy-based modifier for spreader payoff.

    Args:
        strategy: The spreader's strategy

    Returns:
        Modifier value (1.0 = baseline)
    """
    modifier = 1.0

    # Content type affects reach potential
    if strategy.content_type.value == 'fake_emotional':
        modifier *= 1.5  # Emotional content spreads faster but risks detection
    elif strategy.content_type.value == 'fake_neutral':
        modifier *= 1.2  # Neutral fake content less detectable
    elif strategy.content_type.value == 'true':
        modifier *= 0.8  # True content spreads slower but safer

    # Timing affects audience size
    if strategy.timing.value == 'peak_hours':
        modifier *= 1.3

    # Targeting affects efficiency
    if strategy.targeting.value == 'targeted':
        modifier *= 1.1
    elif strategy.targeting.value == 'niche':
        modifier *= 0.9

    return modifier


def _get_fact_checker_strategy_modifier(strategy: FactCheckerStrategy) -> float:
    """
    Calculate strategy-based modifier for fact checker payoff.

    Args:
        strategy: The fact checker's strategy

    Returns:
        Modifier value (1.0 = baseline)
    """
    modifier = 1.0

    # Verification intensity affects accuracy
    if strategy.intensity.value == 'comprehensive':
        modifier *= 1.4
    elif strategy.intensity.value == 'thorough':
        modifier *= 1.2
    elif strategy.intensity.value == 'basic':
        modifier *= 0.9

    # Response speed affects effectiveness
    if strategy.response_speed.value == 'immediate':
        modifier *= 1.3
    elif strategy.response_speed.value == 'fast':
        modifier *= 1.1
    elif strategy.response_speed.value == 'delayed':
        modifier *= 0.8

    return modifier


def _get_platform_strategy_modifier(strategy: PlatformStrategy) -> float:
    """
    Calculate strategy-based modifier for platform payoff.

    Args:
        strategy: The platform's strategy

    Returns:
        Modifier value (1.0 = baseline)
    """
    modifier = 1.0

    # Content policy affects credibility
    if strategy.policy.value == 'strict':
        modifier *= 1.3  # Higher credibility but may reduce engagement
    elif strategy.policy.value == 'moderate':
        modifier *= 1.1
    elif strategy.policy.value == 'permissive':
        modifier *= 0.9  # Lower credibility but may increase engagement

    # Enforcement level affects effectiveness
    if strategy.enforcement_level.value == 'aggressive':
        modifier *= 1.2
    elif strategy.enforcement_level.value == 'lenient':
        modifier *= 0.9

    # Transparency affects user trust
    if strategy.transparency.value == 'transparent':
        modifier *= 1.1
    elif strategy.transparency.value == 'opaque':
        modifier *= 0.95

    return modifier


def get_default_payoff_weights() -> Dict[str, Dict[str, float]]:
    """
    Get default payoff weights for all player types.

    Returns:
        Dictionary containing default weights for each player type
    """
    return {
        'spreader': {
            'reach': 1.0,
            'detection': 0.5,
            'reputation': 0.3
        },
        'fact_checker': {
            'accuracy': 1.0,
            'effort_cost': 0.4,
            'reputation_gain': 0.2
        },
        'platform': {
            'engagement': 1.0,
            'moderation_cost': 0.3,
            'credibility': 0.5,
            'regulatory_risk': 0.4
        }
    }


def validate_payoff_weights(weights: Dict[str, Dict[str, float]]) -> bool:
    """
    Validate that payoff weights are properly formatted and within reasonable ranges.

    Args:
        weights: Payoff weights to validate

    Returns:
        True if weights are valid, False otherwise
    """
    required_players = ['spreader', 'fact_checker', 'platform']
    required_spreader_weights = ['reach', 'detection', 'reputation']
    required_fact_checker_weights = ['accuracy', 'effort_cost', 'reputation_gain']
    required_platform_weights = ['engagement', 'moderation_cost', 'credibility', 'regulatory_risk']

    try:
        for player in required_players:
            if player not in weights:
                return False

            player_weights = weights[player]

            if player == 'spreader':
                required_weights = required_spreader_weights
            elif player == 'fact_checker':
                required_weights = required_fact_checker_weights
            else:  # platform
                required_weights = required_platform_weights

            for weight_name in required_weights:
                if weight_name not in player_weights:
                    return False
                if not isinstance(player_weights[weight_name], (int, float)):
                    return False
                if player_weights[weight_name] < 0:
                    return False

        return True

    except Exception:
        return False