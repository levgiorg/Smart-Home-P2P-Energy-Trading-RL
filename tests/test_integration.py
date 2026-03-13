"""10-episode smoke training runs for SAC and DDPG agents.

All tests are marked `integration` and require the CSV data files under data/.
They are skipped automatically when data is absent.

Each test follows the same pattern:
1. Build ExperimentConfig for the target agent.
2. Instantiate EnergyEnv.
3. Create the agent via create_agent().
4. Run Trainer.train(10).
5. Assert result.total_episodes == 10 and sanity-check reward types.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.trainer import TrainingResult

# ---------------------------------------------------------------------------
# Module-level skip guard — no data directory → skip the whole module
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
pytestmark = pytest.mark.integration


def _data_available() -> bool:
    required = [
        DATA_DIR / "ninja_weather_55.6838_12.5354_uncorrected.csv",
        DATA_DIR / "2014_DK2_spot_prices.csv",
    ]
    return DATA_DIR.exists() and all(p.exists() for p in required)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NUM_HOUSES = 10
STATE_DIM_PER_HOUSE = 17
ACTION_DIM_PER_HOUSE = 3


def _agent_kwargs(env) -> dict:
    """Build the keyword arguments required by create_agent()."""
    return dict(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_houses=env.num_houses,
        state_dim_per_house=STATE_DIM_PER_HOUSE,
        action_dim_per_house=ACTION_DIM_PER_HOUSE,
    )


def _run_smoke(agent_name: str, agent_params: dict) -> TrainingResult:
    """Create env + agent, train 10 episodes, and return the result."""
    from src.agents import create_agent
    from src.config.base import EnvConfig, MarketConfig
    from src.config.experiment import ExperimentConfig
    from src.environment.energy_env import EnergyEnv
    from src.training.trainer import Trainer

    cfg = ExperimentConfig(
        agent=agent_name,
        seed=0,
        num_episodes=10,
        device="cpu",
        agent_params=agent_params,
    )
    env = EnergyEnv(EnvConfig(), MarketConfig())
    agent = create_agent(
        agent_name,
        config=cfg.get_agent_config(),
        **_agent_kwargs(env),
    )
    trainer = Trainer(agent=agent, env=env, config=cfg)
    return trainer.train(10)


# ---------------------------------------------------------------------------
# SAC smoke run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_sac_10_episode_smoke():
    result = _run_smoke("sac", {"batch_size": 32, "memory_size": 500})
    assert result.total_episodes == 10


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_sac_episode_rewards_recorded():
    result = _run_smoke("sac", {"batch_size": 32, "memory_size": 500})
    assert len(result.episode_rewards) == 10


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_sac_episode_rewards_are_finite():
    result = _run_smoke("sac", {"batch_size": 32, "memory_size": 500})
    for ep_reward in result.episode_rewards:
        assert isinstance(ep_reward, float)
        assert ep_reward == ep_reward  # not NaN
        assert abs(ep_reward) < 1e9, f"Reward magnitude suspiciously large: {ep_reward}"


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_sac_mean_reward_property():
    result = _run_smoke("sac", {"batch_size": 32, "memory_size": 500})
    assert isinstance(result.mean_reward, float)


# ---------------------------------------------------------------------------
# DDPG smoke run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_ddpg_10_episode_smoke():
    result = _run_smoke("ddpg", {"batch_size": 32, "memory_size": 500})
    assert result.total_episodes == 10


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_ddpg_episode_rewards_recorded():
    result = _run_smoke("ddpg", {"batch_size": 32, "memory_size": 500})
    assert len(result.episode_rewards) == 10


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_ddpg_episode_rewards_are_finite():
    result = _run_smoke("ddpg", {"batch_size": 32, "memory_size": 500})
    for ep_reward in result.episode_rewards:
        assert isinstance(ep_reward, float)
        assert ep_reward == ep_reward  # not NaN


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_ddpg_mean_reward_property():
    result = _run_smoke("ddpg", {"batch_size": 32, "memory_size": 500})
    assert isinstance(result.mean_reward, float)


# ---------------------------------------------------------------------------
# Cross-agent consistency checks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_trainer_result_total_episodes_matches_requested():
    """Regardless of agent, total_episodes must equal what was requested."""
    for agent_name in ("sac", "ddpg"):
        result = _run_smoke(agent_name, {"batch_size": 32, "memory_size": 500})
        assert result.total_episodes == 10, (
            f"{agent_name}: expected total_episodes=10, got {result.total_episodes}"
        )


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_trainer_result_episode_rewards_list_length():
    """episode_rewards list length must match total_episodes."""
    for agent_name in ("sac", "ddpg"):
        result = _run_smoke(agent_name, {"batch_size": 32, "memory_size": 500})
        assert len(result.episode_rewards) == result.total_episodes


@pytest.mark.skipif(not _data_available(), reason="data/ CSV files not found")
def test_training_result_final_mean_reward_within_last_100():
    """final_mean_reward is mean of the last 100 episodes (or all if fewer)."""
    result = _run_smoke("sac", {"batch_size": 32, "memory_size": 500})
    tail = result.episode_rewards[-100:]
    expected = sum(tail) / len(tail)
    assert abs(result.final_mean_reward - expected) < 1e-6
