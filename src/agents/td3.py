from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffers.replay_buffer import ReplayBuffer
from ..config.agent_configs import TD3Config
from ..networks.actors import DeterministicActor
from ..networks.critics import TwinCritic
from ..utilities.noise import GaussianNoise
from .base import BaseAgent


class _SingleHouseTD3:
    """Per-house TD3 agent (internal helper)."""

    def __init__(self, state_dim: int, action_dim: int, config: TD3Config) -> None:
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_delay = config.policy_delay
        self._update_step = 0

        self.actor = DeterministicActor(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = TwinCritic(state_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        self.memory = ReplayBuffer(config.memory_size, device=self.device)
        self.exploration_noise = GaussianNoise(action_dim, sigma=config.exploration_noise)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        state = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0)
        if not deterministic:
            noise = torch.tensor(self.exploration_noise.sample(), dtype=torch.float32)
            action = (action.cpu() + noise).clamp(-1.0, 1.0)
        else:
            action = action.cpu()
        return action

    def update(self) -> dict[str, float] | None:
        if len(self.memory) < self.batch_size:
            return None

        self._update_step += 1
        batch = self.memory.sample(self.batch_size)
        s = batch["states"]
        a = batch["actions"]
        r = batch["rewards"]
        ns = batch["next_states"]
        d = batch["dones"]

        # Target policy smoothing
        with torch.no_grad():
            noise = torch.randn_like(a) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_a = (self.target_actor(ns) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = self.target_critic(ns, next_a)
            target_q = r + self.gamma * (1.0 - d) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = 0.0
        if self._update_step % self.policy_delay == 0:
            actor_loss_t = -self.critic.q1(torch.cat([s, self.actor(s)], dim=-1)).mean()
            self.actor_opt.zero_grad()
            actor_loss_t.backward()
            self.actor_opt.step()
            actor_loss = actor_loss_t.item()

            for tp, sp in zip(self.target_actor.parameters(), self.actor.parameters()):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)
            for tp, sp in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

        return {"actor_loss": float(actor_loss), "critic_loss": critic_loss.item()}


class TD3Agent(BaseAgent):
    """Multi-agent TD3 — one independent TD3 per house."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: TD3Config,
        num_houses: int = 10,
        state_dim_per_house: int = 17,
        action_dim_per_house: int = 3,
        **_,
    ) -> None:
        self.num_houses = num_houses
        self.sdph = state_dim_per_house
        self.adph = action_dim_per_house
        self.device = torch.device(config.device)

        self.agents = [
            _SingleHouseTD3(state_dim_per_house, action_dim_per_house, config)
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
        agg: dict[str, list[float]] = {"actor_loss": [], "critic_loss": []}
        for agent in self.agents:
            result = agent.update()
            if result:
                for k in agg:
                    agg[k].append(result[k])
        if not agg["actor_loss"]:
            return {}
        return {k: float(np.mean(v)) for k, v in agg.items()}

    def ready_to_update(self) -> bool:
        return all(len(a.memory) >= a.batch_size for a in self.agents)

    def save(self, path: str) -> None:
        ckpt: dict = {}
        for i, agent in enumerate(self.agents):
            ckpt[f"agent_{i}_actor"] = agent.actor.state_dict()
            ckpt[f"agent_{i}_critic"] = agent.critic.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(ckpt[f"agent_{i}_actor"])
            agent.critic.load_state_dict(ckpt[f"agent_{i}_critic"])
            agent.target_actor = copy.deepcopy(agent.actor)
            agent.target_critic = copy.deepcopy(agent.critic)

    @property
    def is_on_policy(self) -> bool:
        return False
