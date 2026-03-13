"""Smoke tests: construct each agent, forward pass, 10-transition update cycle."""

import torch

from src.config.agent_configs import DDPGConfig, PPOConfig, SACConfig, TD3Config

NUM_HOUSES = 2
SDPH = 17
ADPH = 3
STATE_DIM = NUM_HOUSES * SDPH
ACTION_DIM = NUM_HOUSES * ADPH
ACTION_BOUNDS = {
    "hvac_energy": [-1.0, 1.0],
    "battery_action": [-1.0, 1.0],
    "selling_price": [0.0, 1.0],
}
KWARGS = dict(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    num_houses=NUM_HOUSES,
    state_dim_per_house=SDPH,
    action_dim_per_house=ADPH,
)


def _dummy_transition():
    s = torch.randn(STATE_DIM)
    a = torch.randn(NUM_HOUSES, ADPH)
    ns = torch.randn(STATE_DIM)
    r = torch.randn(NUM_HOUSES)
    return s, a, ns, r


def _run_smoke(agent):
    s, a, ns, r = _dummy_transition()
    # select_action
    actions = agent.select_action(s)
    assert actions.shape == (NUM_HOUSES, ADPH)

    # store 10 transitions (enough for any agent to at least call update)
    for _ in range(10):
        agent.store_transition(s, actions, ns, r, False)

    # update (may return {} if buffer not ready — that's fine)
    result = agent.update()
    assert isinstance(result, dict)


def test_ddpg_smoke():
    from src.agents.ddpg import DDPGAgent

    cfg = DDPGConfig(batch_size=4, memory_size=50)
    agent = DDPGAgent(config=cfg, **KWARGS)
    _run_smoke(agent)
    assert not agent.is_on_policy


def test_sac_smoke():
    from src.agents.sac import SACAgent

    cfg = SACConfig(batch_size=4, memory_size=50)
    agent = SACAgent(config=cfg, **KWARGS)
    _run_smoke(agent)
    assert not agent.is_on_policy


def test_td3_smoke():
    from src.agents.td3 import TD3Agent

    cfg = TD3Config(batch_size=4, memory_size=50)
    agent = TD3Agent(config=cfg, **KWARGS)
    _run_smoke(agent)
    assert not agent.is_on_policy


def test_ppo_smoke():
    from src.agents.ppo import PPOAgent

    cfg = PPOConfig(rollout_length=8, batch_size=4, epochs_per_update=1)
    agent = PPOAgent(config=cfg, **KWARGS)
    s, a, ns, r = _dummy_transition()
    # PPO needs select_action first to record log_probs
    for _ in range(8):
        actions = agent.select_action(s)
        agent.store_transition(s, actions, ns, r, False)
    result = agent.update()
    assert isinstance(result, dict)
    assert agent.is_on_policy


def test_agent_save_load(tmp_path):
    from src.agents.sac import SACAgent

    cfg = SACConfig(batch_size=4, memory_size=50)
    agent = SACAgent(config=cfg, **KWARGS)
    path = str(tmp_path / "model.pt")
    agent.save(path)
    agent.load(path)
