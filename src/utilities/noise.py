from __future__ import annotations

import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process noise for temporal correlation (DDPG)."""

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2) -> None:
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(size) * mu

    def reset(self) -> None:
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state = self.state + dx
        return self.state.copy()


class GaussianNoise:
    """Independent Gaussian noise for TD3 exploration."""

    def __init__(self, size: int, sigma: float = 0.1) -> None:
        self.size = size
        self.sigma = sigma

    def sample(self) -> np.ndarray:
        return np.random.normal(0.0, self.sigma, self.size)
