"""Tests for replay and rollout buffers."""
import torch
import pytest

from src.buffers.replay_buffer import ReplayBuffer
from src.buffers.prioritized_buffer import PrioritizedReplayBuffer
from src.buffers.rollout_buffer import RolloutBuffer


S, A = 17, 3


def _push_n(buf, n):
    for _ in range(n):
        buf.push(
            torch.randn(S), torch.randn(A), torch.randn(S),
            torch.tensor([0.5]), False
        )


# ── ReplayBuffer ──────────────────────────────────────────────────────────────

def test_replay_push_sample():
    buf = ReplayBuffer(100)
    _push_n(buf, 50)
    assert len(buf) == 50
    batch = buf.sample(32)
    assert batch["states"].shape == (32, S)
    assert batch["actions"].shape == (32, A)
    assert batch["rewards"].shape == (32, 1)
    assert batch["dones"].shape == (32, 1)


def test_replay_overwrite():
    buf = ReplayBuffer(10)
    _push_n(buf, 25)
    assert len(buf) == 10


def test_replay_raises_if_too_few():
    buf = ReplayBuffer(100)
    _push_n(buf, 5)
    with pytest.raises(ValueError):
        buf.sample(32)


# ── PrioritizedReplayBuffer ───────────────────────────────────────────────────

import numpy as np

def test_prioritized_buffer():
    buf = PrioritizedReplayBuffer(200, alpha=0.6, beta=0.4)
    for _ in range(100):
        buf.add(np.random.randn(S), 0, 0.5, np.random.randn(S), False)
    assert len(buf) == 100
    states, actions, rewards, next_states, dones, weights, indices = buf.sample(32)
    assert states.shape == (32, S)
    assert len(indices) == 32


# ── RolloutBuffer ─────────────────────────────────────────────────────────────

def test_rollout_buffer_gae():
    buf = RolloutBuffer(rollout_length=64, state_dim=S, action_dim=A)
    for _ in range(64):
        buf.push(
            torch.randn(S), torch.randn(A),
            torch.tensor([-0.5]), 1.0, torch.tensor([0.3]), False
        )
    assert buf.ready()
    buf.compute_gae(torch.zeros(1), gamma=0.99, gae_lambda=0.95)
    data = buf.get()
    assert data["states"].shape == (64, S)
    assert data["advantages"].shape == (64, 1)
    # Advantages should be normalised
    assert abs(float(data["advantages"].mean())) < 0.1


def test_rollout_buffer_clear():
    buf = RolloutBuffer(rollout_length=16, state_dim=S, action_dim=A)
    for _ in range(16):
        buf.push(torch.randn(S), torch.randn(A), torch.tensor([0.0]), 0.0, torch.tensor([0.0]), False)
    buf.clear()
    assert not buf.ready()
    assert len(buf) == 0
