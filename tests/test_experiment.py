"""Tests for experiment registry, versioning, and batch matrix expansion."""
import json
from pathlib import Path

import pytest

from src.config.experiment import ExperimentConfig
from src.config.base import MarketConfig
from src.experiment.registry import ExperimentRegistry
from src.experiment.versioning import RunVersioner


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_register_and_query(tmp_path):
    reg = ExperimentRegistry(str(tmp_path / "exp.db"))
    cfg = ExperimentConfig(agent="sac", seed=0, market=MarketConfig(mechanism_type="detection"))
    run_id = reg.register_run(cfg)
    assert len(run_id) == 12

    runs = reg.list_runs(agent="sac")
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id


def test_registry_update_status(tmp_path):
    reg = ExperimentRegistry(str(tmp_path / "exp.db"))
    cfg = ExperimentConfig(agent="td3", seed=1)
    run_id = reg.register_run(cfg)
    reg.update_status(run_id, "completed", {"mean_reward": 42.0, "best_reward": 55.0, "total_episodes": 100})

    runs = reg.list_runs(status="completed")
    assert any(r["run_id"] == run_id for r in runs)
    run = next(r for r in runs if r["run_id"] == run_id)
    assert run["mean_reward"] == pytest.approx(42.0)


def test_registry_best_run(tmp_path):
    reg = ExperimentRegistry(str(tmp_path / "exp.db"))
    for seed in range(3):
        cfg = ExperimentConfig(agent="sac", seed=seed)
        run_id = reg.register_run(cfg)
        reg.update_status(run_id, "completed", {"mean_reward": float(seed * 10), "best_reward": float(seed * 12)})
    best = reg.best_run("sac")
    assert best is not None
    assert best["mean_reward"] == pytest.approx(20.0)


# ── Versioner ─────────────────────────────────────────────────────────────────

def test_versioner_creates_dir(tmp_path):
    versioner = RunVersioner(results_dir=str(tmp_path))
    cfg = ExperimentConfig(agent="ppo", seed=3)
    run_dir = versioner.create_run_dir(cfg)
    assert run_dir.exists()
    assert "ppo" in run_dir.name


def test_versioner_config_snapshot(tmp_path):
    versioner = RunVersioner(results_dir=str(tmp_path))
    cfg = ExperimentConfig(agent="ddpg", seed=5)
    run_dir = versioner.create_run_dir(cfg)
    versioner.snapshot_config(cfg, run_dir)
    yaml_path = run_dir / "config.yaml"
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "ddpg" in content


def test_versioner_summary(tmp_path):
    versioner = RunVersioner(results_dir=str(tmp_path))
    cfg = ExperimentConfig(agent="sac", seed=0)
    run_dir = versioner.create_run_dir(cfg)
    metrics = {"mean_reward": 10.5, "total_episodes": 100}
    versioner.save_summary(run_dir, metrics)
    data = json.loads((run_dir / "summary.json").read_text())
    assert data["mean_reward"] == pytest.approx(10.5)


# ── BatchOrchestrator matrix expansion ───────────────────────────────────────

def test_batch_matrix_expansion(tmp_path):
    import yaml
    config = {
        "matrix": {"agent": ["sac", "td3"], "mechanism": ["detection"], "seed": [0, 1]},
        "shared": {"episodes": 10},
    }
    cfg_path = tmp_path / "batch.yaml"
    cfg_path.write_text(yaml.dump(config))

    from src.experiment.batch import BatchOrchestrator
    orch = BatchOrchestrator(str(cfg_path), results_dir=str(tmp_path))
    runs = orch.generate_run_matrix()
    assert len(runs) == 4  # 2 agents × 1 mechanism × 2 seeds
    agents = {r["agent"] for r in runs}
    assert agents == {"sac", "td3"}
