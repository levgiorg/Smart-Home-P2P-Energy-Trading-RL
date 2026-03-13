"""Single-run training CLI.

Usage:
    python train.py --agent sac --mechanism detection --seed 42 --episodes 1000
    python train.py --config experiments/sac_adaptive.yaml
"""

from __future__ import annotations

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an RL agent for P2P energy trading.")
    p.add_argument("--agent", default="ddpg", choices=["ddpg", "sac", "td3", "ppo", "dqn"])
    p.add_argument(
        "--mechanism", default="detection", choices=["detection", "ceiling", "adaptive", "none"]
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--config", default=None, help="Path to a YAML ExperimentConfig file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.config.base import MarketConfig
    from src.config.experiment import ExperimentConfig
    from src.utilities.seeding import set_all_seeds

    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        mechanism = None if args.mechanism == "none" else args.mechanism
        config = ExperimentConfig(
            agent=args.agent,
            seed=args.seed,
            num_episodes=args.episodes,
            device=args.device,
            results_dir=args.results_dir,
            market=MarketConfig(mechanism_type=mechanism),
        )

    set_all_seeds(config.seed)

    from src.agents import create_agent
    from src.environment.energy_env import EnergyEnv
    from src.experiment.registry import ExperimentRegistry
    from src.experiment.versioning import RunVersioner

    versioner = RunVersioner(config.results_dir)
    registry = ExperimentRegistry(f"{config.results_dir}/experiments.db")

    run_dir = versioner.create_run_dir(config)
    versioner.snapshot_config(config, run_dir)
    git_info = versioner.capture_git_info(run_dir)
    versioner.capture_environment_info(run_dir)
    run_id = registry.register_run(config, git_info)

    logger.info("Run ID: %s | Dir: %s", run_id, run_dir)

    env = EnergyEnv(config.env, config.market)
    agent_cfg = config.get_agent_config()

    agent = create_agent(
        config.agent,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=agent_cfg,
        num_houses=env.num_houses,
        state_dim_per_house=config.env.state_dim_per_house,
        action_dim_per_house=config.env.action_dim_per_house,
    )

    from src.training.trainer import Trainer

    t0 = time.time()
    try:
        trainer = Trainer(agent, env, config, run_dir=run_dir)
        result = trainer.train(config.num_episodes)
        elapsed = time.time() - t0
        final_metrics = {
            "mean_reward": result.mean_reward,
            "best_reward": result.best_reward,
            "total_episodes": result.total_episodes,
            "training_time_sec": elapsed,
        }
        versioner.save_summary(run_dir, final_metrics)
        registry.update_status(run_id, "completed", final_metrics, str(run_dir))
        logger.info(
            "Training complete. mean_reward=%.2f  best=%.2f", result.mean_reward, result.best_reward
        )
    except Exception:
        registry.update_status(run_id, "failed")
        raise


if __name__ == "__main__":
    main()
