from __future__ import annotations

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """State-value V(s) network for PPO."""

    def __init__(self, state_dim: int, hidden_dims: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
