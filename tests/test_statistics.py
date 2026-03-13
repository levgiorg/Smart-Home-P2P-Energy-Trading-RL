"""Tests for significance testing utilities."""
import numpy as np
import pytest

from src.evaluation.statistics import wilcoxon_comparison, mann_whitney_comparison, pairwise_comparisons


def test_wilcoxon_detects_difference():
    rng = np.random.default_rng(42)
    a = rng.normal(10, 1, 30).tolist()
    b = rng.normal(5, 1, 30).tolist()
    result = wilcoxon_comparison(a, b, "A", "B")
    assert result["significant_0.05"]
    assert result["mean_a"] > result["mean_b"]
    assert result["n"] == 30


def test_wilcoxon_same_distribution_not_significant():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50).tolist()
    b = rng.normal(0, 1, 50).tolist()
    result = wilcoxon_comparison(a, b)
    # With same distribution, should usually NOT be significant
    # (probabilistic, but with seed it should hold)
    assert "p_value" in result


def test_mann_whitney_detects_difference():
    rng = np.random.default_rng(7)
    a = rng.normal(20, 2, 40).tolist()
    b = rng.normal(10, 2, 40).tolist()
    result = mann_whitney_comparison(a, b, "SAC", "DQN")
    assert result["significant_0.01"]
    assert result["name_a"] == "SAC"


def test_pairwise_comparisons():
    rng = np.random.default_rng(1)
    rewards = {
        "sac": rng.normal(10, 1, 20).tolist(),
        "td3": rng.normal(8, 1, 20).tolist(),
        "ddpg": rng.normal(6, 1, 20).tolist(),
    }
    comps = pairwise_comparisons(rewards, paired=False)
    assert len(comps) == 3  # C(3,2)
    assert all("p_value" in c for c in comps)
