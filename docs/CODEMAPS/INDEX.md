# Codebase Index

**Last Updated:** 2026-03-13
**Project:** Smart Home P2P Energy Trading — RL Framework

## Overview

Multi-agent deep reinforcement learning system for peer-to-peer energy trading among smart homes. Each home autonomously manages HVAC, battery storage, and P2P selling price via an actor-critic agent (DDPG primary; SAC, TD3, PPO, DQN also available). Three anti-cartel market regulation mechanisms prevent coordinated price manipulation.

## Repository Layout

```
Smart-Home-P2P-Energy-Trading-RL/
├── train.py               # Single-run training CLI
├── batch_train.py         # Parallel training orchestrator
├── evaluate.py            # Evaluation CLI
├── compare.py             # Cross-run comparison
├── sweep.py               # Hyperparameter sweep CLI
│
├── environment/           # Legacy physics environment (in-use)
│   ├── environment.py     # Core simulation (843 lines)
│   └── anti_cartel.py     # Detection/Ceiling mechanisms (286 lines)
│
├── src/                   # Modern refactored package
│   ├── agents/            # DDPG, SAC, TD3, PPO, DQN + BaseAgent
│   ├── networks/          # Actors, Critics, ValueNetwork
│   ├── buffers/           # ReplayBuffer, PrioritizedBuffer, RolloutBuffer
│   ├── training/          # Trainer, Callbacks
│   ├── environment/       # EnergyEnv wrapper, Scenarios, Wrappers
│   ├── market/            # MarketRegulator protocol + Static/Adaptive
│   ├── config/            # Pydantic configs (EnvConfig, MarketConfig, ExperimentConfig)
│   ├── evaluation/        # Evaluator, ParameterSweep, Statistics
│   ├── experiment/        # BatchOrchestrator, ExperimentRegistry, RunVersioner
│   ├── plotting/          # Matplotlib plots, TikZ export, palette
│   └── utilities/         # ActionUtils, OUNoise, GaussianNoise, Seeding
│
├── analysis/              # Standalone analysis scripts
├── utilities/             # Legacy action_discretizer
├── models_code/           # Legacy DQN networks (dqn_networks.py)
├── hyperparameters/       # Legacy Config class
├── data/                  # CSV data (weather, prices, LSTM predictions)
├── results/               # Training outputs (metrics.csv, model_*.pt, experiments.db)
└── experiments/           # YAML batch experiment definitions
```

## Codemap Files

| File | Covers |
|------|--------|
| [environment.md](environment.md) | RL environment, physics simulation, anti-cartel mechanisms |
| [models.md](models.md) | Neural network architectures, RL agents |
| [training.md](training.md) | Training loop, experiment management, batch orchestration |
| [utilities.md](utilities.md) | Buffers, noise, seeding, config, market regulators |
| [analysis.md](analysis.md) | Evaluation, plotting, sensitivity analysis, standalone scripts |

## Key Architectural Insights

1. **Dual-layer architecture**: `environment/` holds the legacy physics simulation; `src/` is a clean refactor that wraps it. Production code calls `src.environment.energy_env.EnergyEnv` which delegates to `environment.environment.Environment`.

2. **Independent multi-agent**: Each house gets its own agent instance (e.g., `_SingleHouseDDPG`). There is no shared policy or centralized critic; agents operate independently on per-house state slices.

3. **Market regulator as a protocol**: `src/market/base.py` defines `MarketRegulator` as a `Protocol`, allowing static, detection, ceiling, and adaptive implementations to be swapped without changing training code.

4. **Pydantic-frozen configs**: All configs (`EnvConfig`, `MarketConfig`, `ExperimentConfig`) are Pydantic `BaseModel` objects, making serialization to/from YAML trivial and preventing accidental mutation.

5. **Agent-agnostic training loop**: `Trainer` calls only `BaseAgent` abstract methods — no agent-specific logic leaks into the training loop.

## External Dependencies

- torch — Neural networks, GPU tensors
- numpy — Numerical operations
- pandas — CSV data loading
- pydantic — Typed configuration models
- matplotlib — Plotting
- yaml — Batch config parsing
- sqlite3 (stdlib) — Experiment registry
