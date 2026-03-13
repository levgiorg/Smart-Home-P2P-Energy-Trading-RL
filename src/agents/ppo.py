from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffers.rollout_buffer import RolloutBuffer
from ..config.agent_configs import PPOConfig
from ..networks.actors import StochasticActor
from ..networks.value import ValueNetwork
from .base import BaseAgent


class _SingleHousePPO:
    """Per-house PPO agent (internal helper)."""

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig) -> None:
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.epochs = config.epochs_per_update
        self.batch_size = config.batch_size
        self.value_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.rollout_length = config.rollout_length

        self.actor = StochasticActor(state_dim, action_dim, config.hidden_dims, squash=False).to(self.device)
        self.value = ValueNetwork(state_dim, config.hidden_dims).to(self.device)
        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()), lr=config.lr
        )

        self.buffer = RolloutBuffer(config.rollout_length, state_dim, action_dim, device=self.device)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = mean.clamp(-1.0, 1.0).squeeze(0)
                log_prob = torch.zeros(1, device=self.device)
            else:
                action, log_prob = self.actor.sample(state)
                action = action.squeeze(0)
            value = self.value(state).squeeze(0)
        return action.cpu(), log_prob.cpu(), value.cpu()

    def update(self) -> dict[str, float] | None:
        if not self.buffer.ready():
            return None

        # Compute GAE with last value = 0 (episode end)
        self.buffer.compute_gae(torch.zeros(1, device=self.device), self.gamma, self.gae_lambda)
        data = self.buffer.get()
        self.buffer.clear()

        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]
        returns = data["returns"]
        advantages = data["advantages"]

        n = states.shape[0]
        all_losses: dict[str, list[float]] = {"policy": [], "value": [], "entropy": [], "total": []}

        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                sb = states[idx]
                ab = actions[idx]
                old_lp_b = old_log_probs[idx]
                ret_b = returns[idx]
                adv_b = advantages[idx]

                new_lp, entropy = self.actor.evaluate(sb, ab)
                ratio = (new_lp - old_lp_b).exp()

                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.value(sb)
                value_loss = F.mse_loss(value_pred, ret_b)

                entropy_loss = -entropy.mean()
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.opt.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm,
                )
                self.opt.step()

                all_losses["policy"].append(policy_loss.item())
                all_losses["value"].append(value_loss.item())
                all_losses["entropy"].append(entropy_loss.item())
                all_losses["total"].append(total_loss.item())

        return {k: float(np.mean(v)) for k, v in all_losses.items()}


class PPOAgent(BaseAgent):
    """Multi-agent PPO — one independent PPO per house."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: PPOConfig,
        num_houses: int = 10,
        state_dim_per_house: int = 17,
        action_dim_per_house: int = 3,
    ) -> None:
        self.num_houses = num_houses
        self.sdph = state_dim_per_house
        self.adph = action_dim_per_house
        self.device = torch.device(config.device)

        self.agents = [
            _SingleHousePPO(state_dim_per_house, action_dim_per_house, config)
            for _ in range(num_houses)
        ]

        self._last_log_probs: list[torch.Tensor] = [torch.zeros(1)] * num_houses
        self._last_values: list[torch.Tensor] = [torch.zeros(1)] * num_houses

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        state = state.flatten()
        actions = []
        for i, agent in enumerate(self.agents):
            s = state[i * self.sdph : (i + 1) * self.sdph]
            a, lp, v = agent.select_action(s, deterministic)
            actions.append(a)
            self._last_log_probs[i] = lp
            self._last_values[i] = v
        return torch.stack(actions)

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: bool,
    ) -> None:
        state = state.flatten()
        for i, agent in enumerate(self.agents):
            s = state[i * self.sdph : (i + 1) * self.sdph]
            a = action[i] if action.dim() > 1 else action[i * self.adph : (i + 1) * self.adph]
            r = float(reward[i]) if reward.numel() > 1 else float(reward)
            agent.buffer.push(s, a, self._last_log_probs[i], r, self._last_values[i], done)

    def update(self) -> dict[str, float]:
        agg: dict[str, list[float]] = {"policy": [], "value": [], "entropy": [], "total": []}
        for agent in self.agents:
            result = agent.update()
            if result:
                for k in agg:
                    if k in result:
                        agg[k].append(result[k])
        if not agg["total"]:
            return {}
        return {k: float(np.mean(v)) for k, v in agg.items() if v}

    def ready_to_update(self) -> bool:
        return all(a.buffer.ready() for a in self.agents)

    def save(self, path: str) -> None:
        ckpt: dict = {}
        for i, agent in enumerate(self.agents):
            ckpt[f"agent_{i}_actor"] = agent.actor.state_dict()
            ckpt[f"agent_{i}_value"] = agent.value.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(ckpt[f"agent_{i}_actor"])
            agent.value.load_state_dict(ckpt[f"agent_{i}_value"])

    @property
    def is_on_policy(self) -> bool:
        return True
