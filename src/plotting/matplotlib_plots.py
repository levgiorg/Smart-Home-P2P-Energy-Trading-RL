from __future__ import annotations

"""Quick-look matplotlib plots using the same SupBlue/SupRed/SupGreen palette."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "lines.linewidth": 1.5,
    }
)

from .palette import AGENT_COLORS, AGENT_MARKERS, COLORS


def plot_convergence(
    data: dict[str, pd.DataFrame],
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    output_path: str | None = None,
    agent_colors: dict[str, str] | None = None,
) -> None:
    """Line plot with confidence bands for each agent."""
    ac = agent_colors or AGENT_COLORS
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    for agent, df in data.items():
        color = COLORS.get(ac.get(agent, "SupGray"), (0.5, 0.5, 0.5))
        marker = AGENT_MARKERS.get(agent, "o")
        ep = df["episode"].to_numpy()
        mean = df["mean"].to_numpy()
        std = df["std"].to_numpy() if "std" in df.columns else np.zeros_like(mean)

        ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.15)
        ax.plot(ep, mean, color=color, marker=marker, markevery=max(1, len(ep) // 20), label=agent.upper())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    plt.close(fig)


def plot_bar_comparison(
    data: dict[str, dict[str, float]],
    errors: dict[str, dict[str, float]] | None = None,
    xlabel: str = "Metric",
    ylabel: str = "Value",
    output_path: str | None = None,
    agent_colors: dict[str, str] | None = None,
) -> None:
    """Grouped bar chart with optional error bars."""
    ac = agent_colors or AGENT_COLORS
    agents = list(data.keys())
    metrics = sorted({m for d in data.values() for m in d})

    x = np.arange(len(metrics))
    width = 0.8 / len(agents)
    offsets = np.linspace(-(len(agents) - 1) / 2, (len(agents) - 1) / 2, len(agents)) * width

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for agent, offset in zip(agents, offsets):
        color = COLORS.get(ac.get(agent, "SupGray"), (0.5, 0.5, 0.5))
        vals = [data[agent].get(m, 0.0) for m in metrics]
        errs = [errors[agent].get(m, 0.0) for m in metrics] if errors and agent in errors else None
        ax.bar(x + offset, vals, width, label=agent.upper(), color=color, yerr=errs, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    plt.close(fig)


def plot_heatmap(
    data: pd.DataFrame,
    xlabel: str = "Parameter A",
    ylabel: str = "Parameter B",
    output_path: str | None = None,
) -> None:
    """2D heatmap for sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    im = ax.imshow(data.values, aspect="auto", origin="lower", cmap="hot")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in data.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in data.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    plt.close(fig)
