from __future__ import annotations

"""Non-parametric significance tests for comparing RL algorithm performance."""

import numpy as np
from scipy import stats


def wilcoxon_comparison(
    rewards_a: list[float],
    rewards_b: list[float],
    name_a: str = "A",
    name_b: str = "B",
) -> dict:
    """Wilcoxon signed-rank test for paired samples.

    Use when both agents are evaluated on the same environment seeds.
    """
    a = np.array(rewards_a)
    b = np.array(rewards_b)

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    stat, p_value = stats.wilcoxon(a, b)
    diff = a - b
    effect_size = float(np.median(diff) / (np.std(diff) + 1e-9))

    return {
        "test": "wilcoxon",
        "name_a": name_a,
        "name_b": name_b,
        "n": n,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_0.05": bool(p_value < 0.05),
        "significant_0.01": bool(p_value < 0.01),
        "effect_size": effect_size,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a)),
        "std_b": float(np.std(b)),
    }


def mann_whitney_comparison(
    rewards_a: list[float],
    rewards_b: list[float],
    name_a: str = "A",
    name_b: str = "B",
) -> dict:
    """Mann-Whitney U test for independent samples (different seeds per agent)."""
    a = np.array(rewards_a)
    b = np.array(rewards_b)

    stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial correlation as effect size
    n_a, n_b = len(a), len(b)
    effect_size = 1.0 - (2.0 * float(stat)) / (n_a * n_b)

    return {
        "test": "mann_whitney",
        "name_a": name_a,
        "name_b": name_b,
        "n_a": n_a,
        "n_b": n_b,
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_0.05": bool(p_value < 0.05),
        "significant_0.01": bool(p_value < 0.01),
        "effect_size_r": effect_size,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a)),
        "std_b": float(np.std(b)),
    }


def pairwise_comparisons(
    results: dict[str, list[float]],
    paired: bool = False,
) -> list[dict]:
    """Run all pairwise comparisons across multiple agents."""
    agents = sorted(results.keys())
    comparisons: list[dict] = []

    for i, a in enumerate(agents):
        for b in agents[i + 1 :]:
            if paired:
                comp = wilcoxon_comparison(results[a], results[b], name_a=a, name_b=b)
            else:
                comp = mann_whitney_comparison(results[a], results[b], name_a=a, name_b=b)
            comparisons.append(comp)

    return comparisons
