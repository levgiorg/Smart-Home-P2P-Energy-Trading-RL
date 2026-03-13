"""Post-training evaluation CLI.

Usage:
    python evaluate.py --checkpoint results/sac_detection_seed42_*/model_final.pt --scenarios all
    python evaluate.py --checkpoint results/run/model_final.pt --scenarios normal,price_spike
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scenarios", default="normal", help="Comma-separated list or 'all'")
    p.add_argument("--agent", default="sac", choices=["ddpg", "sac", "td3", "ppo", "dqn"])
    p.add_argument("--mechanism", default="detection")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.agents import create_agent
    from src.config.base import MarketConfig
    from src.config.experiment import ExperimentConfig
    from src.environment.energy_env import EnergyEnv
    from src.environment.scenarios import SCENARIOS
    from src.evaluation.evaluator import Evaluator
    from src.utilities.seeding import set_all_seeds

    set_all_seeds(args.seed)

    config = ExperimentConfig(
        agent=args.agent,
        seed=args.seed,
        market=MarketConfig(mechanism_type=args.mechanism if args.mechanism != "none" else None),
    )

    env = EnergyEnv(config.env, config.market)
    agent_cfg = config.get_agent_config()
    agent = create_agent(
        args.agent,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=agent_cfg,
        num_houses=env.num_houses,
        state_dim_per_house=config.env.state_dim_per_house,
        action_dim_per_house=config.env.action_dim_per_house,
    )

    # Resolve checkpoint glob
    ckpt_paths = sorted(glob.glob(args.checkpoint))
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoint found matching: {args.checkpoint}")
    agent.load(ckpt_paths[0])
    logger.info("Loaded checkpoint: %s", ckpt_paths[0])

    evaluator = Evaluator(agent, env)

    scenario_names = (
        list(SCENARIOS.keys()) if args.scenarios == "all" else args.scenarios.split(",")
    )
    scenarios = {name: SCENARIOS[name] for name in scenario_names if name in SCENARIOS}

    results = evaluator.evaluate_scenarios(scenarios, episodes=args.episodes)
    print("\n=== Evaluation Results ===")
    for scenario, metrics in results.items():
        print(f"  {scenario:20s}: mean={metrics['mean']:.2f} ± {metrics['std']:.2f}")

    output = Path(ckpt_paths[0]).parent / "evaluation_results.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
