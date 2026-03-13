"""Tests for seed management utilities."""

import random

import numpy as np
import torch

from src.utilities.seeding import set_all_seeds


def test_set_all_seeds_python():
    set_all_seeds(42)
    v1 = random.random()
    set_all_seeds(42)
    v2 = random.random()
    assert v1 == v2


def test_set_all_seeds_numpy():
    set_all_seeds(7)
    v1 = np.random.randn(5).tolist()
    set_all_seeds(7)
    v2 = np.random.randn(5).tolist()
    assert v1 == v2


def test_set_all_seeds_torch():
    set_all_seeds(123)
    t1 = torch.randn(10)
    set_all_seeds(123)
    t2 = torch.randn(10)
    assert torch.allclose(t1, t2)
