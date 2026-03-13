from __future__ import annotations

import numpy as np
import torch


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay (PER).

    Based on Schaul et al. 2016. Priorities stored as raw |TD errors|.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device(device)
        self._pos = 0
        self.size = 0

        self._storage: list = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float | None = None,
    ) -> None:
        priority = self._max_priority if td_error is None else (abs(td_error) + 1e-6) ** self.alpha
        self._storage[self._pos] = (state, action, reward, next_state, done)
        self._priorities[self._pos] = priority
        self._pos = (self._pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        priorities = self._priorities[: self.size]
        probs = priorities / priorities.sum()

        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        samples = [self._storage[i] for i in indices]

        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        def _t(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor(np.array(arr), dtype=dtype, device=self.device)

        return (
            _t(states, torch.float32),
            _t(actions, torch.long),
            _t(rewards, torch.float32),
            _t(next_states, torch.float32),
            _t(dones, torch.bool),
            torch.tensor(weights, dtype=torch.float32, device=self.device),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            p = (abs(float(err)) + 1e-6) ** self.alpha
            self._priorities[idx] = p
            self._max_priority = max(self._max_priority, p)

    def __len__(self) -> int:
        return self.size
