# backend/game_theory/players.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class Player(ABC):
    """Abstract base class for a player in the game."""

    def __init__(self, player_id: str, player_type: str):
        self.player_id = player_id
        self.player_type = player_type
        self.reputation_score: float = 0.5  # Initial reputation
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    def get_objective(self) -> str:
        """Return the player's objective."""
        raise NotImplementedError

    def update_reputation(self, score_change: float) -> None:
        """Update the player's reputation score."""
        self.reputation_score = max(0.0, min(1.0, self.reputation_score + score_change))

    def add_to_history(self, action: Dict[str, Any]) -> None:
        """Add an action to the player's history."""
        self.history.append(action)

    def get_reputation_score(self) -> float:
        """Get the current reputation score."""
        return self.reputation_score


class Spreader(Player):
    """Represents an Information Source player."""

    def __init__(self, player_id: str, spreader_type: str = 'profit-driven'):
        super().__init__(player_id, 'Spreader')
        self.spreader_type = spreader_type  # e.g., 'malicious', 'misinformed'
        self.content_reach: int = 0
        self.engagement_score: float = 0.0
        self.detection_count: int = 0

    def get_objective(self) -> str:
        return "Maximize content reach and engagement"

    def update_metrics(self, reach: int, engagement: float, detected: bool = False) -> None:
        """Update spreader's performance metrics."""
        self.content_reach += reach
        self.engagement_score += engagement
        if detected:
            self.detection_count += 1

    def get_detection_rate(self) -> float:
        """Calculate the detection rate for this spreader."""
        if not self.history:
            return 0.0
        return self.detection_count / len(self.history)


class FactChecker(Player):
    """Represents an Information Validator player."""

    def __init__(self, player_id: str, checker_type: str = 'professional'):
        super().__init__(player_id, 'FactChecker')
        self.checker_type = checker_type  # e.g., 'automated', 'crowd-sourced'
        self.accuracy_score: float = 0.5
        self.verification_count: int = 0
        self.resource_budget: float = 100.0
        self.resource_used: float = 0.0

    def get_objective(self) -> str:
        return "Maximize detection accuracy while minimizing effort costs"

    def update_accuracy(self, correct_verifications: int, total_verifications: int) -> None:
        """Update the fact checker's accuracy score."""
        if total_verifications > 0:
            self.accuracy_score = correct_verifications / total_verifications
            self.verification_count += total_verifications

    def use_resources(self, amount: float) -> bool:
        """Use resources for verification. Returns True if successful."""
        if self.resource_used + amount <= self.resource_budget:
            self.resource_used += amount
            return True
        return False

    def get_remaining_resources(self) -> float:
        """Get the remaining resource budget."""
        return self.resource_budget - self.resource_used

    def reset_resources(self) -> None:
        """Reset resource usage for a new round."""
        self.resource_used = 0.0


class Platform(Player):
    """Represents an Information Mediator player."""

    def __init__(self, player_id: str = 'main_platform'):
        super().__init__(player_id, 'Platform')
        self.user_engagement: float = 0.0
        self.credibility_score: float = 0.5
        self.moderation_cost: float = 0.0
        self.regulatory_compliance: float = 1.0

    def get_objective(self) -> str:
        return "Balance user engagement with platform credibility"

    def update_engagement(self, engagement_change: float) -> None:
        """Update platform user engagement."""
        self.user_engagement = max(0.0, self.user_engagement + engagement_change)

    def update_credibility(self, credibility_change: float) -> None:
        """Update platform credibility score."""
        self.credibility_score = max(0.0, min(1.0, self.credibility_score + credibility_change))

    def add_moderation_cost(self, cost: float) -> None:
        """Add to the platform's moderation costs."""
        self.moderation_cost += cost

    def update_compliance(self, compliance_score: float) -> None:
        """Update regulatory compliance score."""
        self.regulatory_compliance = max(0.0, min(1.0, compliance_score))

    def get_total_utility(self) -> float:
        """Calculate overall platform utility."""
        # Simple utility function balancing engagement and credibility
        return (self.user_engagement * self.credibility_score) - (self.moderation_cost * 0.1)