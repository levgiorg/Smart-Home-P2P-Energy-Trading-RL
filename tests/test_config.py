"""Tests for Pydantic config system."""

import pytest

from src.config.agent_configs import DDPGConfig, PPOConfig, SACConfig, TD3Config
from src.config.base import EnvConfig, MarketConfig
from src.config.experiment import ExperimentConfig


def test_env_config_defaults():
    cfg = EnvConfig()
    assert cfg.num_houses == 10
    assert cfg.state_dim_per_house == 17
    assert cfg.action_dim_per_house == 3


def test_market_config_frozen():
    cfg = MarketConfig()
    with pytest.raises(Exception):
        cfg.mechanism_type = "ceiling"  # type: ignore[misc]


def test_ddpg_config():
    cfg = DDPGConfig()
    assert cfg.gamma == 0.99
    assert cfg.actor_hidden == (400, 300)


def test_sac_config():
    cfg = SACConfig()
    assert cfg.auto_entropy_tuning is True
    assert cfg.hidden_dims == (256, 256)


def test_td3_config():
    cfg = TD3Config()
    assert cfg.policy_delay == 2


def test_ppo_config():
    cfg = PPOConfig()
    assert cfg.clip_epsilon == 0.2


def test_experiment_config_agent_dispatch():
    for agent_name in ["ddpg", "sac", "td3", "ppo", "dqn"]:
        cfg = ExperimentConfig(agent=agent_name)
        agent_cfg = cfg.get_agent_config()
        assert agent_cfg is not None


def test_experiment_config_yaml_roundtrip(tmp_path):
    cfg = ExperimentConfig(agent="sac", seed=7)
    out = str(tmp_path / "config.yaml")
    cfg.to_yaml(out)
    loaded = ExperimentConfig.from_yaml(out)
    assert loaded.agent == "sac"
    assert loaded.seed == 7
