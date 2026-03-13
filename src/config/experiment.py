from __future__ import annotations

from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .agent_configs import DDPGConfig, DQNConfig, PPOConfig, SACConfig, TD3Config
from .base import EnvConfig, MarketConfig, RewardConfig

AgentConfig = DDPGConfig | SACConfig | TD3Config | PPOConfig | DQNConfig

_AGENT_CONFIG_MAP: dict[str, type] = {
    "ddpg": DDPGConfig,
    "sac": SACConfig,
    "td3": TD3Config,
    "ppo": PPOConfig,
    "dqn": DQNConfig,
}


class ExperimentConfig(BaseModel):
    agent: Literal["ddpg", "sac", "td3", "ppo", "dqn"] = "ddpg"
    seed: int = 42
    num_episodes: int = 1000
    device: str = "cpu"
    env: EnvConfig = Field(default_factory=EnvConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    agent_params: dict[str, Any] = Field(default_factory=dict)
    results_dir: str = "results"
    run_name: str | None = None

    model_config = {"frozen": False}

    def get_agent_config(self) -> AgentConfig:
        cls = _AGENT_CONFIG_MAP[self.agent]
        params = {"device": self.device}
        params.update(self.agent_params)
        return cls(**params)

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        import json

        with open(path, "w") as f:
            yaml.dump(json.loads(self.model_dump_json()), f, default_flow_style=False)
