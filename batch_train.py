"""Batch/parallel training CLI.

Usage:
    python batch_train.py --batch experiments/full_comparison.yaml --gpus 0,1 --max-concurrent 4
    python batch_train.py --status
    python batch_train.py --batch experiments/full_comparison.yaml --resume-failed
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", default=None, help="Path to batch YAML config")
    p.add_argument("--gpus", default="0", help="Comma-separated GPU device IDs")
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument("--status", action="store_true", help="Show live status of all runs")
    p.add_argument("--resume-failed", action="store_true")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.experiment.batch import BatchOrchestrator

    if args.status:
        from src.experiment.registry import ExperimentRegistry

        registry = ExperimentRegistry(f"{args.results_dir}/experiments.db")
        runs = registry.list_runs()
        print(
            f"{'run_id':<14} {'agent':<8} {'mechanism':<12} {'seed':<6} {'status':<12} {'mean_reward'}"
        )
        for r in runs:
            print(
                f"  {r['run_id']:<12} {r['agent'] or '?':<8} {r['mechanism'] or '?':<12} "
                f"{r['seed'] or '?':<6} {r['status']:<12} {r['mean_reward'] or '-'}"
            )
        return

    if not args.batch:
        print("Provide --batch <yaml> or --status")
        sys.exit(1)

    gpus = [int(g) for g in args.gpus.split(",")]
    orchestrator = BatchOrchestrator(args.batch, results_dir=args.results_dir)

    if args.resume_failed:
        orchestrator.resume_failed()
    else:
        runs = orchestrator.generate_run_matrix()
        logger.info("Generated %d runs from batch config.", len(runs))
        orchestrator.launch(gpus=gpus, max_concurrent=args.max_concurrent)
        logger.info("All runs launched.")


if __name__ == "__main__":
    main()
