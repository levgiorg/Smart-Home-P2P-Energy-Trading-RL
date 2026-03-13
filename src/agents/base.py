from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseAgent(ABC):
    """Abstract base for all RL agents. The training loop calls only these methods."""

    @abstractmethod
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return actions of shape [num_houses, action_dim]."""

    @abstractmethod
    def update(self) -> dict[str, float]:
        """Perform one gradient step. Returns loss dict for logging."""

    @abstractmethod
    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: bool,
    ) -> None:
        """Store one transition. Off-policy: add to replay buffer; on-policy: add to rollout."""

    @abstractmethod
    def ready_to_update(self) -> bool:
        """Whether enough data has been collected for an update step."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist all agent state to path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore agent state from path."""

    @property
    @abstractmethod
    def is_on_policy(self) -> bool:
        """True for PPO (clears buffer after update), False for off-policy agents."""
