# Training Codemap

**Last Updated:** 2026-03-13
**Entry Points:** `train.py`, `batch_train.py`, `sweep.py`

## Architecture

```
train.py (CLI)                     batch_train.py (CLI)
     │                                     │
     ▼                                     ▼
ExperimentConfig ◄── YAML / args    BatchOrchestrator
     │                                 ├── generate_run_matrix()   (cartesian product)
     ├── EnvConfig                     ├── launch()                (subprocess per run)
     ├── MarketConfig                  └── resume_failed()
     ├── RewardConfig                         │
     └── AgentConfig                   spawns train.py subprocesses
          │
          ▼
     RunVersioner ──► run_dir/
          │               ├── config.yaml
          │               ├── git_info.json
          │               ├── environment_info.json
          │               ├── metrics.csv
          │               ├── training_log.jsonl
          │               ├── model_best.pt
          │               └── model_final.pt
          │
     ExperimentRegistry (SQLite) ──► results/experiments.db
          │
          ▼
     Trainer.train(num_episodes)
          │
          └── _run_episode() loop
                  ├── env.reset()
                  ├── agent.select_action(state)
                  ├── env.step(action)
                  ├── agent.store_transition(...)
                  ├── agent.ready_to_update() → agent.update()
                  └── log metrics to CSV / JSONL
```

## Key Modules

| Module | Purpose | Key Exports | Dependencies |
|--------|---------|-------------|--------------|
| `train.py` | Single-run CLI entry point | `main()` | `src.training.trainer`, `src.agents`, `src.environment.energy_env`, `src.experiment.*` |
| `batch_train.py` | Parallel batch training CLI | `main()` | `src.experiment.batch`, `src.experiment.registry` |
| `sweep.py` | Hyperparameter sweep CLI | `main()` | `src.evaluation.sensitivity` |
| `src/training/trainer.py` | Agent-agnostic training loop | `Trainer`, `TrainingResult` | `BaseAgent`, `EnergyEnv`, `ExperimentConfig` |
| `src/training/callbacks.py` | Optional training callbacks | — | — |
| `src/experiment/registry.py` | SQLite experiment tracker | `ExperimentRegistry` | `sqlite3` |
| `src/experiment/versioning.py` | Run directory creation and snapshotting | `RunVersioner` | `pathlib`, `subprocess` (git) |
| `src/experiment/batch.py` | Matrix expansion + subprocess launcher | `BatchOrchestrator` | `yaml`, `subprocess`, `ExperimentRegistry` |
| `src/config/experiment.py` | Top-level experiment config model | `ExperimentConfig` | `pydantic`, `yaml` |
| `src/config/base.py` | Sub-configs for env, market, reward | `EnvConfig`, `MarketConfig`, `RewardConfig` | `pydantic` |
| `src/config/agent_configs.py` | Per-agent hyperparameter models | `DDPGConfig`, `SACConfig`, `TD3Config`, `PPOConfig`, `DQNConfig` | `pydantic` |

## Data Flow: Single Training Run

```
1. parse_args() / ExperimentConfig.from_yaml()
2. set_all_seeds(config.seed)
3. RunVersioner.create_run_dir()      → results/<agent>_<mechanism>_<ts>/
4. RunVersioner.snapshot_config()     → config.yaml
5. RunVersioner.capture_git_info()    → git_info.json
6. ExperimentRegistry.register_run()  → INSERT into experiments.db
7. EnergyEnv(env_config, market_config)
8. create_agent(config.agent, ...)
9. Trainer(agent, env, config, run_dir).train(num_episodes)
   └── episode loop:
       a. env.reset() → state_t
       b. while not done:
            action = agent.select_action(state_t)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(...)
            if agent.ready_to_update(): agent.update()
            ep_reward += reward.sum()
       c. log episode to metrics.csv
       d. if ep_reward > best: agent.save(model_best.pt)
10. agent.save(model_final.pt)
11. ExperimentRegistry.update_status("completed", final_metrics)
```

## Batch Training

`batch_train.py` reads a YAML experiment matrix:

```yaml
shared:
  episodes: 2000
  results_dir: results/batch_run

matrix:
  agent: [ddpg, sac, td3]
  mechanism: [detection, ceiling, none]
  seed: [0, 1, 2]

overrides:
  sac:
    agent_params:
      batch_size: 256
```

`BatchOrchestrator.generate_run_matrix()` takes the cartesian product of `matrix` keys, merges `shared`, and applies `overrides` per-agent. Each run is launched as a `subprocess` calling `train.py` with the expanded args. GPU assignment is round-robin across the provided `--gpus` list.

## Training Outputs (per run)

| File | Content |
|------|---------|
| `config.yaml` | Full `ExperimentConfig` snapshot |
| `git_info.json` | Commit hash, branch, dirty flag |
| `environment_info.json` | Python / package versions |
| `metrics.csv` | `episode, reward, mean_100` per episode |
| `training_log.jsonl` | Per-update loss metrics (actor_loss, critic_loss, …) |
| `model_best.pt` | Checkpoint at best episode reward |
| `model_final.pt` | Final checkpoint after all episodes |
| `summary.json` | `{mean_reward, best_reward, total_episodes, training_time_sec}` |

## CLI Reference

### `train.py`
```
python train.py --agent sac --mechanism detection --seed 42 --episodes 1000
python train.py --config experiments/sac_adaptive.yaml
```
Arguments: `--agent {ddpg,sac,td3,ppo,dqn}`, `--mechanism {detection,ceiling,adaptive,none}`, `--seed`, `--episodes`, `--device`, `--results-dir`, `--config`

### `batch_train.py`
```
python batch_train.py --batch experiments/full_comparison.yaml --gpus 0,1 --max-concurrent 4
python batch_train.py --status
python batch_train.py --batch ... --resume-failed
```

### `sweep.py`
Delegates to `src.evaluation.sensitivity.ParameterSweep` for 1D or 2D parameter sweeps.

## External Dependencies

- `pydantic` — Typed, serializable config models
- `yaml` — Batch config parsing
- `sqlite3` (stdlib) — Experiment registry persistence
- `subprocess` (stdlib) — Parallel run launching
- `csv`, `json` (stdlib) — Metrics/log writing

## Related Areas

- [models.md](models.md) — Agents trained by `Trainer`
- [environment.md](environment.md) — `EnergyEnv` stepped during training
- [analysis.md](analysis.md) — Post-training evaluation and plotting
