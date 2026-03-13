from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffers.replay_buffer import ReplayBuffer
from ..config.agent_configs import DDPGConfig
from ..networks.actors import DeterministicActor
from ..networks.critics import Critic
from ..utilities.noise import OUNoise
from .base import BaseAgent


class _SingleHouseDDPG:
    """Per-house DDPG agent (internal helper)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DDPGConfig,
        house_id: int,
    ) -> None:
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size

        self.actor = DeterministicActor(state_dim, action_dim, config.actor_hidden).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, config.critic_hidden).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=config.lr_critic, weight_decay=1e-2
        )

        self.memory = ReplayBuffer(config.memory_size, device=self.device)
        # Noise only on HVAC + battery actions (2 dims); price handled separately
        self.noise = OUNoise(action_dim - 1, theta=config.noise_theta, sigma=config.noise_sigma)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state.to(self.device)).squeeze(0)
        self.actor.train()

        if not deterministic:
            noise = torch.tensor(self.noise.sample(), dtype=torch.float32)
            action[: len(noise)] += noise
            action[2] = action[2].clamp(0.0, 1.0)

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

        with torch.no_grad():
            next_a = self.target_actor(ns)
            target_q = r + self.gamma * self.target_critic(ns, next_a) * (1.0 - d)

        critic_loss = F.mse_loss(self.critic(s, a), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)


class DDPGAgent(BaseAgent):
    """Multi-agent DDPG — one independent DDPG per house."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DDPGConfig,
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
            _SingleHouseDDPG(state_dim_per_house, action_dim_per_house, config, i)
            for i in range(num_houses)
        ]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

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
        metrics: dict[str, list[float]] = {"actor_loss": [], "critic_loss": []}
        for agent in self.agents:
            result = agent.update()
            if result:
                metrics["actor_loss"].append(result["actor_loss"])
                metrics["critic_loss"].append(result["critic_loss"])
        if not metrics["actor_loss"]:
            return {}
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def ready_to_update(self) -> bool:
        return all(len(a.memory) >= a.batch_size for a in self.agents)

    def save(self, path: str) -> None:
        checkpoint: dict = {"num_houses": self.num_houses}
        for i, agent in enumerate(self.agents):
            checkpoint[f"agent_{i}_actor"] = agent.actor.state_dict()
            checkpoint[f"agent_{i}_critic"] = agent.critic.state_dict()
            checkpoint[f"agent_{i}_actor_opt"] = agent.actor_opt.state_dict()
            checkpoint[f"agent_{i}_critic_opt"] = agent.critic_opt.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(ckpt[f"agent_{i}_actor"])
            agent.critic.load_state_dict(ckpt[f"agent_{i}_critic"])
            agent.actor.load_state_dict(
                ckpt.get(f"agent_{i}_actor_opt", agent.actor_opt.state_dict())
            )
            agent.critic.load_state_dict(
                ckpt.get(f"agent_{i}_critic_opt", agent.critic_opt.state_dict())
            )

    @property
    def is_on_policy(self) -> bool:
        return False
