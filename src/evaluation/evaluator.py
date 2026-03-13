from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..agents.base import BaseAgent
    from ..environment.energy_env import EnergyEnv
    from ..environment.scenarios import StressScenario

logger = logging.getLogger(__name__)


class Evaluator:
    """Agent-agnostic evaluation runner.

    Supports:
    - Single-episode deterministic rollout
    - Multi-seed aggregation
    - Stress-scenario evaluation
    """

    def __init__(self, agent: "BaseAgent", env: "EnergyEnv") -> None:
        self.agent = agent
        self.env = env

    def evaluate_episode(self) -> dict[str, float]:
        state = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = self.agent.select_action(state, deterministic=True)
            state, reward, done, info = self.env.step(action)
            total_reward += float(reward.sum() if reward.numel() > 1 else reward)
            steps += 1

        return {"reward": total_reward, "steps": steps}

    def evaluate_multi_seed(
        self,
        checkpoint_pattern: str,
        seeds: list[int],
        episodes_per_seed: int = 5,
    ) -> dict[str, float]:
        """Evaluate across multiple seeds, loading checkpoints per seed."""
        rewards: list[float] = []
        for seed in seeds:
            ckpt = checkpoint_pattern.format(seed=seed)
            if Path(ckpt).exists():
                self.agent.load(ckpt)
            for _ in range(episodes_per_seed):
                result = self.evaluate_episode()
                rewards.append(result["reward"])

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "n": len(rewards),
        }

    def evaluate_scenarios(
        self,
        scenarios: dict[str, "StressScenario"],
        episodes: int = 3,
    ) -> dict[str, dict[str, float]]:
        """Evaluate agent under each stress scenario."""
        results: dict[str, dict[str, float]] = {}
        original_scenario = self.env._scenario

        for name, scenario in scenarios.items():
            self.env._scenario = scenario
            ep_rewards: list[float] = []
            for _ in range(episodes):
                result = self.evaluate_episode()
                ep_rewards.append(result["reward"])
            results[name] = {
                "mean": float(np.mean(ep_rewards)),
                "std": float(np.std(ep_rewards)),
            }
            logger.info("Scenario %s: mean=%.2f ± %.2f", name, results[name]["mean"], results[name]["std"])

        self.env._scenario = original_scenario
        return results
