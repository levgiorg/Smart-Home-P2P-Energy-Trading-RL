from __future__ import annotations

"""Training callbacks — placeholder for BookKeeper integration and early stopping."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import TrainingResult

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """Stop training if reward does not improve for `patience` episodes."""

    def __init__(self, patience: int = 100, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("-inf")
        self._wait = 0

    def should_stop(self, current_reward: float) -> bool:
        if current_reward > self._best + self.min_delta:
            self._best = current_reward
            self._wait = 0
        else:
            self._wait += 1
        return self._wait >= self.patience
