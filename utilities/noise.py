import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating noise.

    Attributes:
        mu (float): Mean of the process.
        theta (float): Speed of mean reversion.
        sigma (float): Volatility parameter.
        state (np.ndarray): Current state of the process.
    """

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state
