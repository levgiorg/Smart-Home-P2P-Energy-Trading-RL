from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..config.agent_configs import DQNConfig
from .base import BaseAgent

# Reuse existing DQN networks and action discretizer from the legacy modules
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models_code.dqn_networks import DQNNetwork, DoubleDQNNetwork  # noqa: E402
from utilities.action_discretizer import ActionDiscretizer  # noqa: E402


class _SingleHouseDQN:
    """Per-house DQN agent wrapping the legacy DQN network."""

    def __init__(self, state_dim: int, action_bounds: dict, config: DQNConfig) -> None:
        self.device = torch.device(config.device)
        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.use_double = config.use_double_dqn
        self._learn_steps = 0

        self.discretizer = ActionDiscretizer(action_bounds)
        num_actions = self.discretizer.total_actions

        net_cls = DoubleDQNNetwork if self.use_double else DQNNetwork
        self.q_net = net_cls(
            state_dim, num_actions, config.fc1_dims, config.fc2_dims, config.fc3_dims, device=self.device
        )
        self.target_net = net_cls(
            state_dim, num_actions, config.fc1_dims, config.fc2_dims, config.fc3_dims, device=self.device
        )
        self._hard_update()

        self.opt = optim.Adam(
            self._params(),
            lr=config.lr,
        )
        self.memory: deque = deque(maxlen=config.memory_size)

    def _params(self):
        if self.use_double:
            return list(self.q_net.q_network_1.parameters()) + list(self.q_net.q_network_2.parameters())
        return list(self.q_net.parameters())

    def _hard_update(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return self.discretizer.sample_random_action()
        state = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net.forward(state, network="main") if self.use_double else self.q_net.forward(state)
        return int(torch.argmax(q, dim=1).item())

    def update(self) -> dict[str, float] | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        r = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        ns = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        d = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)

        if self.use_double:
            cur_q = self.q_net.forward(s, network="main").gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = torch.argmax(self.q_net.forward(ns, network="main"), dim=1)
                next_q = self.target_net.forward(ns, network="main").gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            cur_q = self.q_net.get_action_values(s, a)
            with torch.no_grad():
                next_q = self.target_net.forward(ns).max(1)[0]

        target_q = r + self.gamma * next_q * (~d)
        loss = F.smooth_l1_loss(cur_q, target_q)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params(), 10.0)
        self.opt.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_freq == 0:
            self._hard_update()

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return {"q_loss": loss.item(), "epsilon": self.epsilon}


class DQNAgent(BaseAgent):
    """Multi-agent DQN — one independent DQN per house."""

    def __init__(
        self,
        state_dim: int,
        action_bounds: dict,
        config: DQNConfig,
        num_houses: int = 10,
        state_dim_per_house: int = 17,
        action_dim_per_house: int = 3,
    ) -> None:
        self.num_houses = num_houses
        self.sdph = state_dim_per_house
        self.adph = action_dim_per_house
        self.device = torch.device(config.device)
        self._action_bounds = action_bounds

        self.agents = [
            _SingleHouseDQN(state_dim_per_house, action_bounds, config)
            for _ in range(num_houses)
        ]

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        state = state.flatten()
        continuous = []
        for i, agent in enumerate(self.agents):
            s = state[i * self.sdph : (i + 1) * self.sdph]
            discrete = agent.select_action(s, deterministic)
            cont = agent.discretizer.discrete_to_continuous(discrete)
            continuous.append(torch.tensor(cont, dtype=torch.float32))
        return torch.stack(continuous)

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: bool,
    ) -> None:
        state = state.flatten().numpy()
        next_state = next_state.flatten().numpy()
        for i, agent in enumerate(self.agents):
            s = state[i * self.sdph : (i + 1) * self.sdph]
            ns = next_state[i * self.sdph : (i + 1) * self.sdph]
            # Re-discretize continuous action for storage
            a_cont = action[i].numpy() if action.dim() > 1 else action[i * self.adph : (i + 1) * self.adph].numpy()
            a_disc = agent.discretizer.continuous_to_discrete(a_cont) if hasattr(agent.discretizer, "continuous_to_discrete") else 0
            r = float(reward[i]) if reward.numel() > 1 else float(reward)
            agent.memory.append((s, a_disc, r, ns, done))

    def update(self) -> dict[str, float]:
        agg: dict[str, list[float]] = {"q_loss": [], "epsilon": []}
        for agent in self.agents:
            result = agent.update()
            if result:
                for k in agg:
                    if k in result:
                        agg[k].append(result[k])
        if not agg["q_loss"]:
            return {}
        return {k: float(np.mean(v)) for k, v in agg.items() if v}

    def ready_to_update(self) -> bool:
        return all(len(a.memory) >= a.batch_size for a in self.agents)

    def save(self, path: str) -> None:
        ckpt: dict = {}
        for i, agent in enumerate(self.agents):
            ckpt[f"agent_{i}_q_net"] = agent.q_net.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        for i, agent in enumerate(self.agents):
            agent.q_net.load_state_dict(ckpt[f"agent_{i}_q_net"])
            agent._hard_update()

    @property
    def is_on_policy(self) -> bool:
        return False
