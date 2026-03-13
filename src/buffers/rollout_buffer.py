from __future__ import annotations

import torch


class RolloutBuffer:
    """On-policy rollout buffer with GAE advantage estimation for PPO."""

    def __init__(
        self,
        rollout_length: int,
        state_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.rollout_length = rollout_length
        self.device = torch.device(device)
        self._pos = 0
        self._full = False

        self.states = torch.zeros((rollout_length, state_dim), device=self.device)
        self.actions = torch.zeros((rollout_length, action_dim), device=self.device)
        self.log_probs = torch.zeros((rollout_length, 1), device=self.device)
        self.rewards = torch.zeros((rollout_length, 1), device=self.device)
        self.values = torch.zeros((rollout_length, 1), device=self.device)
        self.dones = torch.zeros((rollout_length, 1), device=self.device)
        self.advantages = torch.zeros((rollout_length, 1), device=self.device)
        self.returns = torch.zeros((rollout_length, 1), device=self.device)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        self.states[self._pos] = state.flatten()
        self.actions[self._pos] = action.flatten()
        self.log_probs[self._pos] = log_prob.reshape(1)
        self.rewards[self._pos] = float(reward)
        self.values[self._pos] = value.reshape(1)
        self.dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self.rollout_length
        if self._pos == 0:
            self._full = True

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns in-place."""
        last_val = last_value.reshape(1)
        gae = torch.zeros(1, device=self.device)
        length = self.rollout_length if self._full else self._pos

        for t in reversed(range(length)):
            next_val = last_val if t == length - 1 else self.values[t + 1]
            next_done = torch.zeros(1, device=self.device) if t == length - 1 else self.dones[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get(self) -> dict[str, torch.Tensor]:
        length = self.rollout_length if self._full else self._pos
        adv = self.advantages[:length]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "states": self.states[:length],
            "actions": self.actions[:length],
            "log_probs": self.log_probs[:length],
            "returns": self.returns[:length],
            "advantages": adv,
        }

    def clear(self) -> None:
        self._pos = 0
        self._full = False

    def ready(self) -> bool:
        return self._full or self._pos >= self.rollout_length

    def __len__(self) -> int:
        return self.rollout_length if self._full else self._pos
