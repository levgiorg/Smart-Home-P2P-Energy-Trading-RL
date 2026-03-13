"""Cross-algorithm comparison CLI with statistics and publication plots.

Usage:
    python compare.py --agents ddpg,sac,td3,ppo --seeds 5 --mechanism detection --episodes 1000
    python compare.py --results-dir results/ --agents sac,td3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agents", default="ddpg,sac,td3,ppo", help="Comma-separated agent names")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--mechanism", default="detection")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _train_agent(agent_name: str, seed: int, mechanism: str, episodes: int) -> tuple[list[float], float]:
    """Train a single agent and return episode rewards + final mean."""
    from src.config.experiment import ExperimentConfig
    from src.config.base import MarketConfig
    from src.utilities.seeding import set_all_seeds
    from src.environment.energy_env import EnergyEnv
    from src.agents import create_agent
    from src.training.trainer import Trainer

    set_all_seeds(seed)
    config = ExperimentConfig(
        agent=agent_name,
        seed=seed,
        num_episodes=episodes,
        market=MarketConfig(mechanism_type=mechanism if mechanism != "none" else None),
    )
    env = EnergyEnv(config.env, config.market)
    agent_cfg = config.get_agent_config()
    agent = create_agent(
        agent_name,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=agent_cfg,
        num_houses=env.num_houses,
        state_dim_per_house=config.env.state_dim_per_house,
        action_dim_per_house=config.env.action_dim_per_house,
    )
    trainer = Trainer(agent, env, config)
    result = trainer.train(episodes)
    return result.episode_rewards, result.final_mean_reward


def main() -> None:
    args = parse_args()
    agents = args.agents.split(",")
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"{args.results_dir}/comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rewards: dict[str, list[float]] = {a: [] for a in agents}
    all_curves: dict[str, list[list[float]]] = {a: [] for a in agents}

    for agent_name in agents:
        for seed in range(args.seeds):
            logger.info("Training %s seed=%d", agent_name, seed)
            ep_rewards, _ = _train_agent(agent_name, seed, args.mechanism, args.episodes)
            all_rewards[agent_name].extend(ep_rewards[-50:])  # last 50 ep for final metric
            all_curves[agent_name].append(ep_rewards)

    # Build convergence DataFrames
    convergence_data: dict[str, pd.DataFrame] = {}
    for agent, curves in all_curves.items():
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        convergence_data[agent] = pd.DataFrame({
            "episode": np.arange(min_len),
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
        })

    # Save raw data
    raw: dict[str, list[float]] = {}
    for a in agents:
        raw[a] = all_rewards[a]
    with open(output_dir / "raw_data.json", "w") as f:
        json.dump(raw, f)

    # Significance tests
    from src.evaluation.statistics import pairwise_comparisons
    comparisons = pairwise_comparisons(all_rewards, paired=False)
    with open(output_dir / "significance.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    # Publication plots
    from src.plotting.matplotlib_plots import plot_convergence, plot_bar_comparison
    from src.plotting.tikz_export import TikZExporter

    plot_convergence(convergence_data, output_path=str(output_dir / "convergence.png"))
    logger.info("Saved convergence.png")

    final_means = {a: {"final_reward": float(np.mean(all_rewards[a]))} for a in agents}
    final_stds = {a: {"final_reward": float(np.std(all_rewards[a]))} for a in agents}
    plot_bar_comparison(final_means, errors=final_stds, ylabel="Final Reward",
                        output_path=str(output_dir / "final_reward_bars.png"))
    logger.info("Saved final_reward_bars.png")

    exporter = TikZExporter()
    exporter.convergence_plot(
        convergence_data, caption="Training convergence comparison.",
        label="convergence", output_path=str(output_dir / "convergence.tex")
    )
    exporter.bar_chart(
        final_means, errors=final_stds, ylabel="Final Reward",
        caption="Final reward comparison.", label="final_reward",
        output_path=str(output_dir / "final_reward_bars.tex")
    )
    exporter.significance_table(
        comparisons, caption="Pairwise significance tests (Mann-Whitney U).",
        label="significance", output_path=str(output_dir / "significance_table.tex")
    )

    # Summary table
    print("\n=== Algorithm Comparison ===")
    print(f"{'Agent':<10} {'Mean ± Std':>15}")
    for a in agents:
        m = np.mean(all_rewards[a])
        s = np.std(all_rewards[a])
        print(f"  {a:<8}   {m:6.2f} ± {s:.2f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
