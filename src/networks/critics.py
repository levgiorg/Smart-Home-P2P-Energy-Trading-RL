from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class Critic(nn.Module):
    """Single Q(s, a) network for DDPG."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int, int] = (400, 300, 300),
    ) -> None:
        super().__init__()
        self.net = _mlp(state_dim + action_dim, hidden_dims, 1)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class TwinCritic(nn.Module):
    """Twin Q-networks (Q1, Q2) for SAC and TD3.

    Returns min(Q1, Q2) for target computation to reduce overestimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.q1 = _mlp(state_dim + action_dim, hidden_dims, 1)
        self.q2 = _mlp(state_dim + action_dim, hidden_dims, 1)
        for net in [self.q1, self.q2]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Q1, Q2) values."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
