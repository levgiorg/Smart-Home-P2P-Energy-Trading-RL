from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffers.replay_buffer import ReplayBuffer
from ..config.agent_configs import SACConfig
from ..networks.actors import StochasticActor
from ..networks.critics import TwinCritic
from .base import BaseAgent


class _SingleHouseSAC:
    """Per-house SAC agent (internal helper)."""

    def __init__(self, state_dim: int, action_dim: int, config: SACConfig) -> None:
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.auto_entropy = config.auto_entropy_tuning

        self.actor = StochasticActor(state_dim, action_dim, config.hidden_dims, squash=True).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        if self.auto_entropy:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(
                np.log(config.initial_alpha), requires_grad=True, device=self.device
            )
            self.alpha_opt = optim.Adam([self.log_alpha], lr=config.lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.initial_alpha

        self.memory = ReplayBuffer(config.memory_size, device=self.device)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        state = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean).squeeze(0)
            else:
                action, _ = self.actor.sample(state)
                action = action.squeeze(0)
        return action.cpu()

    def update(self) -> dict[str, float] | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        s = batch["states"]
        a = batch["actions"]
        r = batch["rewards"]
        ns = batch["next_states"]
        d = batch["dones"]

        alpha = self.log_alpha.exp() if self.auto_entropy else self.alpha

        # Critic update
        with torch.no_grad():
            next_a, next_lp = self.actor.sample(ns)
            q1_t, q2_t = self.target_critic(ns, next_a)
            min_q_t = torch.min(q1_t, q2_t) - alpha * next_lp
            target_q = r + self.gamma * (1.0 - d) * min_q_t

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        new_a, new_lp = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        actor_loss = (alpha * new_lp - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha update
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss_t = -(self.log_alpha * (new_lp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss_t.backward()
            self.alpha_opt.step()
            alpha_loss = alpha_loss_t.item()

        # Soft update target critics
        for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": float(alpha_loss),
            "alpha": float(alpha) if not self.auto_entropy else float(self.log_alpha.exp().item()),
        }


class SACAgent(BaseAgent):
    """Multi-agent SAC — one independent SAC per house."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: SACConfig,
        num_houses: int = 10,
        state_dim_per_house: int = 17,
        action_dim_per_house: int = 3,
    ) -> None:
        self.num_houses = num_houses
        self.sdph = state_dim_per_house
        self.adph = action_dim_per_house
        self.device = torch.device(config.device)

        self.agents = [
            _SingleHouseSAC(state_dim_per_house, action_dim_per_house, config)
            for _ in range(num_houses)
        ]

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        state = state.flatten()
        actions = [
            self.agents[i].select_action(state[i * self.sdph : (i + 1) * self.sdph], deterministic)
            for i in range(self.num_houses)
        ]
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
        next_state = next_state.flatten()
        for i, agent in enumerate(self.agents):
            s = state[i * self.sdph : (i + 1) * self.sdph]
            ns = next_state[i * self.sdph : (i + 1) * self.sdph]
            a = action[i] if action.dim() > 1 else action[i * self.adph : (i + 1) * self.adph]
            r = reward[i] if reward.numel() > 1 else reward
            agent.memory.push(s, a, ns, r, done)

    def update(self) -> dict[str, float]:
        keys = ["actor_loss", "critic_loss", "alpha_loss", "alpha"]
        agg: dict[str, list[float]] = {k: [] for k in keys}
        for agent in self.agents:
            result = agent.update()
            if result:
                for k in keys:
                    if k in result:
                        agg[k].append(result[k])
        if not agg["actor_loss"]:
            return {}
        return {k: float(np.mean(v)) for k, v in agg.items() if v}

    def ready_to_update(self) -> bool:
        return all(len(a.memory) >= a.batch_size for a in self.agents)

    def save(self, path: str) -> None:
        ckpt: dict = {}
        for i, agent in enumerate(self.agents):
            ckpt[f"agent_{i}_actor"] = agent.actor.state_dict()
            ckpt[f"agent_{i}_critic"] = agent.critic.state_dict()
            if agent.auto_entropy:
                ckpt[f"agent_{i}_log_alpha"] = agent.log_alpha.detach()
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(ckpt[f"agent_{i}_actor"])
            agent.critic.load_state_dict(ckpt[f"agent_{i}_critic"])
            if agent.auto_entropy and f"agent_{i}_log_alpha" in ckpt:
                agent.log_alpha.data.copy_(ckpt[f"agent_{i}_log_alpha"])

    @property
    def is_on_policy(self) -> bool:
        return False
