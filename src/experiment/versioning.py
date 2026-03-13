from __future__ import annotations

"""Run directory management, config snapshots, git info, and environment capture."""

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RunVersioner:
    """Creates self-contained, reproducible run directories."""

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = Path(results_dir)

    def create_run_dir(self, config: Any) -> Path:
        """Create sequentially numbered run directory.

        Format: run_NNN_<agent>_<mechanism>_seed<seed>
        The counter is derived from existing run_NNN_* directories in results_dir,
        ensuring a monotonically increasing, human-readable name.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        agent = getattr(config, "agent", "unknown")
        mechanism = getattr(getattr(config, "market", None), "mechanism_type", "none") or "none"
        seed = getattr(config, "seed", 0)

        # Determine next run number from existing run_NNN_* directories
        existing = [
            d.name
            for d in self.results_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("run_")
            and len(d.name) > 4
            and d.name[4:7].isdigit()
        ]
        if existing:
            last = max(int(n[4:7]) for n in existing)
            run_num = last + 1
        else:
            run_num = 1

        name = f"run_{run_num:03d}_{agent}_{mechanism}_seed{seed}"
        run_dir = self.results_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def snapshot_config(self, config: Any, run_dir: Path) -> None:
        """Freeze config to YAML — exact reproduction guaranteed."""
        try:
            import yaml

            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(json.loads(config.model_dump_json()), f, default_flow_style=False)
        except Exception as exc:
            logger.warning("Failed to write config.yaml, falling back to empty JSON: %s", exc)
            with open(run_dir / "config.json", "w") as f:
                json.dump({}, f)

    def capture_git_info(self, run_dir: Path) -> dict:
        """Capture git commit, branch, dirty status."""
        info: dict[str, Any] = {}
        try:
            info["commit"] = _git("rev-parse", "HEAD")
            info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
            info["dirty"] = bool(_git("status", "--porcelain"))
            info["diff_stat"] = _git("diff", "--stat")
        except Exception as exc:
            logger.warning("Failed to capture git info: %s", exc)
            info = {"error": "git unavailable"}
        with open(run_dir / "git_info.json", "w") as f:
            json.dump(info, f, indent=2)
        return info

    def capture_environment_info(self, run_dir: Path) -> dict:
        """Capture Python, torch, CUDA versions and pip freeze."""
        import sys

        import torch

        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
        }
        with open(run_dir / "env_info.json", "w") as f:
            json.dump(info, f, indent=2)

        try:
            pip_out = subprocess.check_output(["pip", "freeze"], text=True)
            (run_dir / "pip_freeze.txt").write_text(pip_out)
        except Exception as exc:
            logger.warning("Failed to capture pip freeze: %s", exc)

        return info

    def save_summary(self, run_dir: Path, metrics: dict) -> None:
        with open(run_dir / "summary.json", "w") as f:
            json.dump(metrics, f, indent=2)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()
