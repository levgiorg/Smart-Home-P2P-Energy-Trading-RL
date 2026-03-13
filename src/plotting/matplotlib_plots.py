from __future__ import annotations

"""Quick-look matplotlib plots using the same SupBlue/SupRed/SupGreen palette."""

import json
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
        ax.plot(
            ep,
            mean,
            color=color,
            marker=marker,
            markevery=max(1, len(ep) // 20),
            label=agent.upper(),
        )

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


def plot_run_reward_curve(
    metrics_csv: str | Path,
    output_path: str | Path,
    agent: str = "",
    mechanism: str = "",
) -> None:
    """Plot episode reward and rolling mean from a single run's metrics.csv.

    Saves reward_curve.png to output_path.
    """
    df = pd.read_csv(metrics_csv)
    episodes = df["episode"].to_numpy()
    rewards = df["reward"].to_numpy()
    mean_100 = df["mean_100"].to_numpy()

    color_raw = COLORS.get(AGENT_COLORS.get(agent.lower(), "SupBlue"), (0.24, 0.46, 0.71))
    color_mean = COLORS.get("SupOrange", (0.93, 0.55, 0.14))

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(episodes, rewards, color=color_raw, alpha=0.45, linewidth=1.0, label="Episode reward")
    ax.plot(episodes, mean_100, color=color_mean, linewidth=2.0, label="Mean (last 100)")

    title_parts = [p for p in [agent.upper(), mechanism] if p]
    if title_parts:
        ax.set_title(" | ".join(title_parts), fontsize=10)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_run_loss_curve(
    training_log_jsonl: str | Path,
    output_path: str | Path,
    agent: str = "",
) -> None:
    """Plot actor/critic/entropy losses from a single run's training_log.jsonl.

    Only generates the file if the log contains loss keys. Silently skips if empty.
    """
    records = []
    try:
        with open(training_log_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        return

    if not records:
        return

    df = pd.DataFrame(records)
    loss_cols = [c for c in df.columns if c.endswith("_loss") or c == "alpha"]
    if not loss_cols:
        return

    palette_keys = ["SupBlue", "SupRed", "SupGreen", "SupOrange", "SupGray"]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for i, col in enumerate(loss_cols):
        color = COLORS.get(palette_keys[i % len(palette_keys)], (0.5, 0.5, 0.5))
        ax.plot(
            df.index.to_numpy(),
            df[col].to_numpy(),
            color=color,
            linewidth=1.0,
            alpha=0.8,
            label=col.replace("_", " ").title(),
        )

    if agent:
        ax.set_title(f"{agent.upper()} — Training Losses", fontsize=10)

    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
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
