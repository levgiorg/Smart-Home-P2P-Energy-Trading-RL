"""Sensitivity sweep CLI.

Usage:
    python sweep.py --agent sac --param market.k_p --range 0.5,2.0,0.25 --seeds 3 --episodes 10
    python sweep.py --agent sac --params market.k_p,market.k_v --grid --range 0.5,1.5,0.5 --episodes 10
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="sac", choices=["ddpg", "sac", "td3", "ppo", "dqn"])
    p.add_argument("--param", default=None, help="Single parameter to sweep (e.g. market.k_p)")
    p.add_argument("--params", default=None, help="Two params for 2D sweep, comma-separated")
    p.add_argument("--range", dest="param_range", default="0.5,2.0,0.5",
                   help="start,stop,step for the parameter range")
    p.add_argument("--grid", action="store_true", help="Use grid sweep (2D)")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--mechanism", default="detection")
    p.add_argument("--output-dir", default="results/sensitivity")
    return p.parse_args()


def _make_values(range_str: str) -> list[float]:
    parts = [float(x) for x in range_str.split(",")]
    if len(parts) == 3:
        start, stop, step = parts
        return list(np.arange(start, stop + step * 0.5, step))
    return parts


def _build_train_fn(agent_name: str, mechanism: str, episodes: int):
    """Returns a callable that trains for given params and returns mean reward."""

    def train_fn(params: dict) -> float:
        from src.config.experiment import ExperimentConfig
        from src.config.base import MarketConfig
        from src.utilities.seeding import set_all_seeds
        from src.environment.energy_env import EnergyEnv
        from src.agents import create_agent
        from src.training.trainer import Trainer

        seed = int(params.get("seed", 0))
        set_all_seeds(seed)

        market_overrides = {k.replace("market.", ""): v for k, v in params.items() if k.startswith("market.")}
        config = ExperimentConfig(
            agent=agent_name,
            seed=seed,
            num_episodes=episodes,
            market=MarketConfig(mechanism_type=mechanism if mechanism != "none" else None, **market_overrides),
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
        return result.final_mean_reward

    return train_fn


def main() -> None:
    args = parse_args()

    from src.evaluation.sensitivity import ParameterSweep
    from src.plotting.matplotlib_plots import plot_heatmap

    values = _make_values(args.param_range)
    train_fn = _build_train_fn(args.agent, args.mechanism, args.episodes)
    sweeper = ParameterSweep(train_fn, output_dir=args.output_dir)

    if args.grid and args.params:
        param_a, param_b = args.params.split(",")
        grid = sweeper.sweep_2d(param_a, values, param_b, values, {}, seeds=args.seeds)

        import pandas as pd
        df = pd.DataFrame(grid, index=[f"{v:.3f}" for v in values], columns=[f"{v:.3f}" for v in values])
        plot_heatmap(df, xlabel=param_b, ylabel=param_a,
                     output_path=f"{args.output_dir}/heatmap_{param_a}_{param_b}.png")
        logger.info("2D sweep done. Grid shape: %s", grid.shape)

    elif args.param:
        results = sweeper.sweep_1d(args.param, values, {}, seeds=args.seeds)
        import json
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        with open(f"{args.output_dir}/sweep_{args.param}.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("1D sweep done for %s", args.param)
    else:
        print("Specify --param for 1D sweep or --params A,B --grid for 2D.")


if __name__ == "__main__":
    main()
