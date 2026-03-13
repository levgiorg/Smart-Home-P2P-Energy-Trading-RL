from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20
LOG_STD_MAX = 2


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """Tanh-bounded deterministic actor for DDPG and TD3."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.net = _mlp(state_dim, hidden_dims, action_dim)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))


class StochasticActor(nn.Module):
    """Gaussian stochastic actor for SAC and PPO.

    For SAC: squashed Gaussian (tanh after sampling).
    For PPO: clamped raw Gaussian (no squash).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int] = (256, 256),
        squash: bool = True,
    ) -> None:
        super().__init__()
        self.squash = squash
        self.action_dim = action_dim
        self.base = _mlp(state_dim, hidden_dims, hidden_dims[-1])
        # Replace last ReLU output with a shared trunk
        self.trunk = nn.Sequential(*list(self.base.children())[:-1])
        trunk_out = hidden_dims[-1]
        self.mean_layer = nn.Linear(trunk_out, action_dim)
        self.log_std_layer = nn.Linear(trunk_out, action_dim)

        for layer in [self.mean_layer, self.log_std_layer]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) without sampling."""
        x = self.trunk(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob).

        SAC path: tanh squash applied, log_prob corrected for squash.
        PPO path: raw Gaussian clamped to [-1, 1].
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()

        if self.squash:
            action = torch.tanh(raw)
            # Numerically stable log_prob correction for tanh squash
            log_prob = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        else:
            action = raw.clamp(-1.0, 1.0)
            log_prob = dist.log_prob(raw)

        return action, log_prob.sum(-1, keepdim=True)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_prob and entropy for given actions (used in PPO update)."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy
