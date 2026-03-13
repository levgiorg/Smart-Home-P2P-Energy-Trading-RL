from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.base import BaseAgent
    from ..config.experiment import ExperimentConfig
    from ..environment.energy_env import EnergyEnv

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    episode_rewards: list[float] = field(default_factory=list)
    episode_metrics: list[dict] = field(default_factory=list)
    best_reward: float = float("-inf")
    total_episodes: int = 0

    @property
    def mean_reward(self) -> float:
        if not self.episode_rewards:
            return 0.0
        return float(sum(self.episode_rewards) / len(self.episode_rewards))

    @property
    def final_mean_reward(self) -> float:
        """Mean of last 100 episodes."""
        tail = self.episode_rewards[-100:]
        return float(sum(tail) / len(tail)) if tail else 0.0


class Trainer:
    """Agent-agnostic training loop.

    Calls only BaseAgent methods — no agent-specific logic leaks here.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: EnergyEnv,
        config: ExperimentConfig,
        run_dir: Path | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self.run_dir = run_dir
        self._metrics_writer: csv.DictWriter | None = None
        self._metrics_file: TextIOWrapper | None = None
        self._log_file: TextIOWrapper | None = None

        if run_dir:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._open_logfiles(run_dir)

    def _open_logfiles(self, run_dir: Path) -> None:
        self._metrics_file = open(run_dir / "metrics.csv", "w", newline="")
        fieldnames = ["episode", "reward", "mean_100"]
        self._metrics_writer = csv.DictWriter(self._metrics_file, fieldnames=fieldnames)
        self._metrics_writer.writeheader()
        self._log_file = open(run_dir / "training_log.jsonl", "w")

    def _close_logfiles(self) -> None:
        if self._metrics_file:
            self._metrics_file.close()
        if self._log_file:
            self._log_file.close()

    def train(self, num_episodes: int) -> TrainingResult:
        result = TrainingResult()
        best_reward = float("-inf")
        best_path = self.run_dir / "model_best.pt" if self.run_dir else None

        try:
            for episode in range(num_episodes):
                ep_reward = self._run_episode(result, episode)
                result.episode_rewards.append(ep_reward)
                result.total_episodes += 1

                mean_100 = result.final_mean_reward
                if ep_reward > best_reward and self.run_dir:
                    best_reward = ep_reward
                    result.best_reward = best_reward
                    self.agent.save(str(best_path))

                if self._metrics_writer:
                    self._metrics_writer.writerow(
                        {"episode": episode, "reward": ep_reward, "mean_100": mean_100}
                    )
                    self._metrics_file.flush()

                if episode % 50 == 0:
                    logger.info(
                        "Episode %d | reward=%.2f | mean_100=%.2f", episode, ep_reward, mean_100
                    )

        finally:
            if self.run_dir:
                self.agent.save(str(self.run_dir / "model_final.pt"))
            self._close_logfiles()

        return result

    def _run_episode(self, result: TrainingResult, episode: int) -> float:
        state = self.env.reset()
        done = False
        ep_reward = 0.0
        step = 0

        while not done:
            deterministic = False
            action = self.agent.select_action(state, deterministic=deterministic)
            next_state, reward, done, info = self.env.step(action)

            self.agent.store_transition(state, action, next_state, reward, done)

            if self.agent.ready_to_update():
                metrics = self.agent.update()
                if metrics and self._log_file:
                    entry = {"episode": episode, "step": step, **metrics}
                    self._log_file.write(json.dumps(entry) + "\n")

            ep_reward += float(reward.sum() if reward.numel() > 1 else reward)
            state = next_state
            step += 1

        return ep_reward
