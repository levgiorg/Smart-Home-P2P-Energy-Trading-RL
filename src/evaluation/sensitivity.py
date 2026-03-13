from __future__ import annotations

"""Sensitivity analysis: 1D and 2D parameter sweeps."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ParameterSweep:
    """Run 1D or 2D parameter sweeps and collect metrics."""

    def __init__(
        self,
        train_fn: Callable[[dict[str, Any]], float],
        output_dir: str = "results/sensitivity",
    ) -> None:
        """
        Args:
            train_fn: Callable that takes a param dict and returns a scalar metric
                      (e.g., mean final reward).
            output_dir: Directory to save heatmap data.
        """
        self.train_fn = train_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sweep_1d(
        self,
        param: str,
        values: list[float],
        base_params: dict[str, Any],
        seeds: int = 3,
    ) -> dict[str, list[float]]:
        """Sweep a single parameter over given values.

        Returns:
            {param_value: [metrics across seeds]}
        """
        results: dict[str, list[float]] = {}
        for v in values:
            params = dict(base_params)
            params[param] = v
            seed_results: list[float] = []
            for seed in range(seeds):
                params["seed"] = seed
                metric = self.train_fn(params)
                seed_results.append(metric)
            results[str(v)] = seed_results
            logger.info(
                "%s=%.4f → mean=%.2f ± %.2f", param, v, np.mean(seed_results), np.std(seed_results)
            )

        return results

    def sweep_2d(
        self,
        param_a: str,
        values_a: list[float],
        param_b: str,
        values_b: list[float],
        base_params: dict[str, Any],
        seeds: int = 1,
    ) -> np.ndarray:
        """Grid sweep over two parameters.

        Returns:
            2D array of shape [len(values_a), len(values_b)] with mean metrics.
        """
        grid = np.zeros((len(values_a), len(values_b)))

        for i, va in enumerate(values_a):
            for j, vb in enumerate(values_b):
                params = dict(base_params)
                params[param_a] = va
                params[param_b] = vb
                seed_results: list[float] = []
                for seed in range(seeds):
                    params["seed"] = seed
                    metric = self.train_fn(params)
                    seed_results.append(metric)
                grid[i, j] = float(np.mean(seed_results))
                logger.info("%s=%.3f, %s=%.3f → %.2f", param_a, va, param_b, vb, grid[i, j])

        np.save(str(self.output_dir / f"sweep_{param_a}_{param_b}.npy"), grid)
        return grid
