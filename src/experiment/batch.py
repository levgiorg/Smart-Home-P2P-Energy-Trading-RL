from __future__ import annotations

"""Batch/parallel training orchestrator.

Reads a YAML experiment matrix, expands to individual run configs,
and launches each as a separate subprocess with GPU assignment.
"""

import logging
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import yaml

from .registry import ExperimentRegistry

logger = logging.getLogger(__name__)


class BatchOrchestrator:
    """Launches and monitors parallel training runs."""

    def __init__(
        self,
        batch_config_path: str,
        results_dir: str = "results",
        registry_path: str = "results/experiments.db",
    ) -> None:
        with open(batch_config_path) as f:
            self._batch = yaml.safe_load(f)
        self.results_dir = results_dir
        self.registry = ExperimentRegistry(registry_path)
        self._processes: list[dict] = []

    def generate_run_matrix(self) -> list[dict]:
        """Expand matrix section into individual run configs (cartesian product)."""
        matrix = self._batch.get("matrix", {})
        shared = self._batch.get("shared", {})
        overrides = self._batch.get("overrides", {})

        keys = list(matrix.keys())
        values = [matrix[k] if isinstance(matrix[k], list) else [matrix[k]] for k in keys]

        runs: list[dict] = []
        for combo in product(*values):
            run = dict(shared)
            run.update(dict(zip(keys, combo)))
            # Apply per-agent overrides
            agent = run.get("agent")
            if agent and agent in overrides:
                run.update(overrides[agent])
            runs.append(run)
        return runs

    def launch(
        self,
        gpus: list[int] | None = None,
        max_concurrent: int = 4,
    ) -> None:
        """Launch all runs, assigning GPUs round-robin."""
        runs = self.generate_run_matrix()
        gpus = gpus or [0]
        active: list[subprocess.Popen] = []

        for i, run_params in enumerate(runs):
            while len(active) >= max_concurrent:
                active = [p for p in active if p.poll() is None]
                time.sleep(2)

            gpu_id = gpus[i % len(gpus)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

            args = [sys.executable, "train.py"]
            for k, v in run_params.items():
                if k == "env":
                    for ek, ev in v.items():
                        args += [f"--env.{ek}", str(ev)]
                else:
                    args += [f"--{k}", str(v)]

            proc = subprocess.Popen(args, env=env, cwd=str(Path.cwd()))
            active.append(proc)
            logger.info("Launched run %d/%d: %s", i + 1, len(runs), run_params)

        # Wait for remaining
        for p in active:
            p.wait()

    def status(self) -> list[dict]:
        return self.registry.list_runs()

    def resume_failed(self) -> None:
        failed = self.registry.list_runs(status="failed")
        logger.info("%d failed runs to resume.", len(failed))
        for run in failed:
            config_path = None
            if run.get("results_dir"):
                config_path = Path(run["results_dir"]) / "config.yaml"
            if config_path and config_path.exists():
                subprocess.Popen([sys.executable, "train.py", "--config", str(config_path)])
