# backend/game_theory/strategies.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional


# --- Spreader Strategy Components ---
class SpreaderContentType(Enum):
    """Types of content a spreader can choose to distribute."""
    FAKE_EMOTIONAL = 'fake_emotional'
    FAKE_NEUTRAL = 'fake_neutral'
    TRUE = 'true'


class SpreaderTiming(Enum):
    """Timing strategies for content distribution."""
    PEAK_HOURS = 'peak_hours'
    OFF_HOURS = 'off_hours'


class SpreaderTargeting(Enum):
    """Targeting strategies for content distribution."""
    BROAD = 'broad'
    TARGETED = 'targeted'
    NICHE = 'niche'


@dataclass(frozen=True)
class SpreaderStrategy:
    """Complete strategy specification for a Spreader player."""
    content_type: SpreaderContentType
    timing: SpreaderTiming
    targeting: SpreaderTargeting = SpreaderTargeting.BROAD

    def __str__(self) -> str:
        return f"Spreader({self.content_type.value}, {self.timing.value}, {self.targeting.value})"


# --- Fact-Checker Strategy Components ---
class VerificationIntensity(Enum):
    """Intensity levels for fact-checking verification."""
    BASIC = 'basic'
    THOROUGH = 'thorough'
    COMPREHENSIVE = 'comprehensive'


class ResponseSpeed(Enum):
    """Speed of fact-checker response."""
    IMMEDIATE = 'immediate'
    FAST = 'fast'
    STANDARD = 'standard'
    DELAYED = 'delayed'


class ResourceAllocation(Enum):
    """How fact-checkers allocate their resources."""
    CONSERVATIVE = 'conservative'
    BALANCED = 'balanced'
    AGGRESSIVE = 'aggressive'


@dataclass(frozen=True)
class FactCheckerStrategy:
    """Complete strategy specification for a FactChecker player."""
    intensity: VerificationIntensity
    response_speed: ResponseSpeed = ResponseSpeed.STANDARD
    resource_allocation: ResourceAllocation = ResourceAllocation.BALANCED

    def __str__(self) -> str:
        return f"FactChecker({self.intensity.value}, {self.response_speed.value}, {self.resource_allocation.value})"


# --- Platform Strategy Components ---
class ContentPolicy(Enum):
    """Platform content moderation policies."""
    PERMISSIVE = 'permissive'
    MODERATE = 'moderate'
    STRICT = 'strict'


class EnforcementLevel(Enum):
    """How strictly platforms enforce their policies."""
    LENIENT = 'lenient'
    STANDARD = 'standard'
    AGGRESSIVE = 'aggressive'


class TransparencyLevel(Enum):
    """Platform transparency in moderation decisions."""
    OPAQUE = 'opaque'
    PARTIAL = 'partial'
    TRANSPARENT = 'transparent'


@dataclass(frozen=True)
class PlatformStrategy:
    """Complete strategy specification for a Platform player."""
    policy: ContentPolicy
    enforcement_level: EnforcementLevel = EnforcementLevel.STANDARD
    transparency: TransparencyLevel = TransparencyLevel.PARTIAL

    def __str__(self) -> str:
        return f"Platform({self.policy.value}, {self.enforcement_level.value}, {self.transparency.value})"


# --- Strategy Factory Functions ---
def create_default_spreader_strategy() -> SpreaderStrategy:
    """Create a default spreader strategy."""
    return SpreaderStrategy(
        content_type=SpreaderContentType.TRUE,
        timing=SpreaderTiming.PEAK_HOURS,
        targeting=SpreaderTargeting.BROAD
    )


def create_default_fact_checker_strategy() -> FactCheckerStrategy:
    """Create a default fact-checker strategy."""
    return FactCheckerStrategy(
        intensity=VerificationIntensity.BASIC,
        response_speed=ResponseSpeed.STANDARD,
        resource_allocation=ResourceAllocation.BALANCED
    )


def create_default_platform_strategy() -> PlatformStrategy:
    """Create a default platform strategy."""
    return PlatformStrategy(
        policy=ContentPolicy.MODERATE,
        enforcement_level=EnforcementLevel.STANDARD,
        transparency=TransparencyLevel.PARTIAL
    )


# --- Strategy Validation Functions ---
def validate_strategy_combination(
    spreader_strategy: SpreaderStrategy,
    fact_checker_strategy: FactCheckerStrategy,
    platform_strategy: PlatformStrategy
) -> bool:
    """
    Validate that a combination of strategies is valid for simulation.

    Args:
        spreader_strategy: The spreader's chosen strategy
        fact_checker_strategy: The fact-checker's chosen strategy
        platform_strategy: The platform's chosen strategy

    Returns:
        True if the strategy combination is valid, False otherwise
    """
    # Basic validation - could be extended with more complex rules
    return all([
        isinstance(spreader_strategy, SpreaderStrategy),
        isinstance(fact_checker_strategy, FactCheckerStrategy),
        isinstance(platform_strategy, PlatformStrategy)
    ])


def get_strategy_effectiveness_modifier(
    strategy: SpreaderStrategy
) -> float:
    """
    Calculate a modifier for strategy effectiveness based on strategy components.

    Args:
        strategy: The spreader strategy to evaluate

    Returns:
        A multiplier for strategy effectiveness (1.0 = baseline)
    """
    effectiveness = 1.0

    # Content type affects reach potential
    if strategy.content_type == SpreaderContentType.FAKE_EMOTIONAL:
        effectiveness *= 1.5  # Emotional content spreads faster
    elif strategy.content_type == SpreaderContentType.FAKE_NEUTRAL:
        effectiveness *= 1.2  # Neutral fake content less detectable

    # Timing affects audience size
    if strategy.timing == SpreaderTiming.PEAK_HOURS:
        effectiveness *= 1.3  # More people online during peak hours

    # Targeting affects efficiency
    if strategy.targeting == SpreaderTargeting.TARGETED:
        effectiveness *= 1.1  # Targeted content more effective
    elif strategy.targeting == SpreaderTargeting.NICHE:
        effectiveness *= 0.8  # Niche targeting limits reach

    return effectiveness