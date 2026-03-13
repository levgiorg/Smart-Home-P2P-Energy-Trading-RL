"""Verify that same seed + config produces identical agent outputs."""
import torch
import pytest

from src.utilities.seeding import set_all_seeds
from src.config.agent_configs import SACConfig, DDPGConfig
from src.agents.sac import SACAgent
from src.agents.ddpg import DDPGAgent


NUM_HOUSES = 2
SDPH, ADPH = 17, 3
KWARGS = dict(
    state_dim=NUM_HOUSES * SDPH,
    action_dim=NUM_HOUSES * ADPH,
    num_houses=NUM_HOUSES,
    state_dim_per_house=SDPH,
    action_dim_per_house=ADPH,
)


def _actions_for_seed(agent_cls, cfg, seed: int) -> torch.Tensor:
    set_all_seeds(seed)
    agent = agent_cls(config=cfg, **KWARGS)
    state = torch.randn(NUM_HOUSES * SDPH, generator=torch.Generator().manual_seed(seed))
    return agent.select_action(state, deterministic=True)


def test_sac_deterministic_same_seed():
    cfg = SACConfig(batch_size=4, memory_size=50)
    a1 = _actions_for_seed(SACAgent, cfg, seed=42)
    a2 = _actions_for_seed(SACAgent, cfg, seed=42)
    assert torch.allclose(a1, a2), "SAC with same seed should produce identical actions"


def test_sac_different_seeds_differ():
    cfg = SACConfig(batch_size=4, memory_size=50)
    a1 = _actions_for_seed(SACAgent, cfg, seed=0)
    a2 = _actions_for_seed(SACAgent, cfg, seed=1)
    assert not torch.allclose(a1, a2), "Different seeds should produce different networks"


def test_ddpg_deterministic_same_seed():
    cfg = DDPGConfig(batch_size=4, memory_size=50)
    a1 = _actions_for_seed(DDPGAgent, cfg, seed=99)
    a2 = _actions_for_seed(DDPGAgent, cfg, seed=99)
    assert torch.allclose(a1, a2)
