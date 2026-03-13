"""Tests for shared network architectures."""
import torch
import pytest

from src.networks.actors import DeterministicActor, StochasticActor
from src.networks.critics import Critic, TwinCritic
from src.networks.value import ValueNetwork


STATE_DIM = 17
ACTION_DIM = 3
BATCH = 4


def test_deterministic_actor_shape():
    net = DeterministicActor(STATE_DIM, ACTION_DIM)
    x = torch.randn(BATCH, STATE_DIM)
    out = net(x)
    assert out.shape == (BATCH, ACTION_DIM)
    assert out.abs().max() <= 1.0 + 1e-5  # tanh bounded


def test_stochastic_actor_sample():
    net = StochasticActor(STATE_DIM, ACTION_DIM, squash=True)
    x = torch.randn(BATCH, STATE_DIM)
    action, log_prob = net.sample(x)
    assert action.shape == (BATCH, ACTION_DIM)
    assert log_prob.shape == (BATCH, 1)


def test_stochastic_actor_ppo_mode():
    net = StochasticActor(STATE_DIM, ACTION_DIM, squash=False)
    x = torch.randn(BATCH, STATE_DIM)
    action, log_prob = net.sample(x)
    # clamped to [-1, 1]
    assert action.abs().max() <= 1.0 + 1e-5


def test_critic_shape():
    net = Critic(STATE_DIM, ACTION_DIM)
    s = torch.randn(BATCH, STATE_DIM)
    a = torch.randn(BATCH, ACTION_DIM)
    q = net(s, a)
    assert q.shape == (BATCH, 1)


def test_twin_critic_shapes():
    net = TwinCritic(STATE_DIM, ACTION_DIM)
    s = torch.randn(BATCH, STATE_DIM)
    a = torch.randn(BATCH, ACTION_DIM)
    q1, q2 = net(s, a)
    assert q1.shape == (BATCH, 1)
    assert q2.shape == (BATCH, 1)
    min_q = net.min_q(s, a)
    assert min_q.shape == (BATCH, 1)


def test_value_network_shape():
    net = ValueNetwork(STATE_DIM)
    x = torch.randn(BATCH, STATE_DIM)
    v = net(x)
    assert v.shape == (BATCH, 1)
