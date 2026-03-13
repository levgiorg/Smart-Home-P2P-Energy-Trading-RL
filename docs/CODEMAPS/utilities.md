# Utilities Codemap

**Last Updated:** 2026-03-13
**Entry Points:** `src/utilities/`, `src/buffers/`, `src/market/`, `src/config/`

## Architecture

```
src/utilities/
  ├── noise.py          OUNoise, GaussianNoise     ← exploration for off-policy agents
  ├── seeding.py        set_all_seeds()             ← reproducibility
  └── action_utils.py   action scaling helpers

src/buffers/
  ├── replay_buffer.py       off-policy uniform buffer (pre-allocated tensors)
  ├── prioritized_buffer.py  PER (priority-based sampling)
  └── rollout_buffer.py      on-policy rollout storage (PPO)

src/market/
  ├── base.py           MarketRegulator (Protocol)
  ├── static.py         DetectionRegulator, CeilingRegulator
  ├── adaptive.py       AdaptiveRegulator
  └── __init__.py       create_regulator() factory

src/config/
  ├── base.py           EnvConfig, MarketConfig, RewardConfig
  ├── agent_configs.py  DDPGConfig, SACConfig, TD3Config, PPOConfig, DQNConfig
  └── experiment.py     ExperimentConfig (top-level)

utilities/             ← legacy
  └── action_discretizer.py
```

## Key Modules

| Module | Purpose | Key Exports | Dependencies |
|--------|---------|-------------|--------------|
| `src/utilities/noise.py` | Temporal (OU) and independent (Gaussian) exploration noise | `OUNoise`, `GaussianNoise` | `numpy` |
| `src/utilities/seeding.py` | Seed all RNGs for reproducibility | `set_all_seeds` | `random`, `numpy`, `torch` |
| `src/utilities/action_utils.py` | Action scaling / clipping helpers | — | `numpy` |
| `src/buffers/replay_buffer.py` | GPU-optional replay buffer with lazy tensor allocation | `ReplayBuffer` | `torch` |
| `src/buffers/prioritized_buffer.py` | Prioritized experience replay | `PrioritizedReplayBuffer` | `torch`, `numpy` |
| `src/buffers/rollout_buffer.py` | Fixed-length rollout storage for PPO | `RolloutBuffer` | `torch` |
| `src/market/base.py` | `MarketRegulator` protocol (structural typing) | `MarketRegulator` | `typing.Protocol` |
| `src/market/static.py` | Detection-based and ceiling-based regulators (static thresholds) | `DetectionRegulator`, `CeilingRegulator` | `numpy`, `MarketConfig` |
| `src/market/adaptive.py` | Adaptive regulator with rolling-window thresholds | `AdaptiveRegulator` | `numpy`, `MarketConfig` |
| `src/market/__init__.py` | Regulator factory | `create_regulator` | All regulator classes |
| `src/config/base.py` | Core environment, market, reward configs | `EnvConfig`, `MarketConfig`, `RewardConfig` | `pydantic` |
| `src/config/agent_configs.py` | Per-algorithm hyperparameter configs | `DDPGConfig`, `SACConfig`, `TD3Config`, `PPOConfig`, `DQNConfig` | `pydantic` |
| `src/config/experiment.py` | Root config; YAML load/save | `ExperimentConfig` | `pydantic`, `yaml` |

## Noise Processes

### OUNoise (`src/utilities/noise.py`)
Ornstein-Uhlenbeck process for temporally correlated exploration (used by DDPG).
```
dx = θ(μ − x) + σ·N(0,1)
```
Defaults: `mu=0.0`, `theta=0.15`, `sigma=0.2`.  
Applied only to HVAC + battery action dims (not selling price).

### GaussianNoise
Independent Gaussian noise (used by TD3 for target policy smoothing).  
Default `sigma=0.1`.

## Replay Buffer (`src/buffers/replay_buffer.py`)

- Pre-allocates 5 tensors of shape `(capacity, dim)` on first `push()` call.
- Circular buffer: `_pos` wraps at `capacity`.
- `sample(batch_size)` returns random indices as a dict of tensors.
- GPU-optional: pass `device="cuda:0"` to store transitions on GPU.

```
push(state, action, next_state, reward, done)
sample(batch_size) → {states, actions, next_states, rewards, dones}
```

## Market Regulator Protocol

All market regulators must implement:
```python
class MarketRegulator(Protocol):
    def calculate_penalties(selling_prices, grid_price) -> list[float]
    def update(selling_prices, episode_done=False) -> None
    def get_price_ceiling(grid_price) -> float   # inf = no ceiling
```

### `create_regulator(config: MarketConfig) → MarketRegulator`
| `mechanism_type` | Returns |
|-----------------|---------|
| `"detection"` | `DetectionRegulator` (static thresholds from config) |
| `"ceiling"` | `CeilingRegulator` |
| `"adaptive"` | `AdaptiveRegulator` (rolling-window dynamic thresholds) |
| `None` | `NullRegulator` (zero penalties) |

### AdaptiveRegulator Threshold Computation
Thresholds `(δ_p, δ_v, δ_c)` are derived from rolling price statistics each step:
- `δ_p = mean(pairwise spreads) + k_p × std(spreads)` — price proximity threshold
- `δ_v = max(δ_v_min, mean(variances) − k_v × std(variances))` — variance floor
- `δ_c = min(δ_c_max, mean(correlations) + k_c × std(correlations))` — correlation ceiling

## Configuration Models (Pydantic)

### `EnvConfig` (frozen)
| Key | Default | Description |
|-----|---------|-------------|
| `num_houses` | 10 | Number of smart homes |
| `battery_capacity_min/max` | 0.6 / 10.0 kWh | Battery bounds |
| `battery_charging_efficiency` | 0.98 | Charge efficiency |
| `temperature_min/max` | 20 / 22 °C | Comfort zone |
| `hvac_efficiency` | 1.1 | HVAC COP |
| `grid_fee` | 0.018 | Transaction fee |
| `num_hours` | 24 | Episode length |

### `MarketConfig` (frozen)
| Key | Default | Description |
|-----|---------|-------------|
| `mechanism_type` | `"detection"` | Anti-cartel mechanism |
| `monitoring_window` | 100 | Episode history length |
| `penalty_factor` | 1.5 | Base penalty multiplier |
| `markup_limit` | 0.2 | Ceiling: `price ≤ grid × (1−0.2)` |
| `similarity_threshold` | 0.85 | Correlation detection threshold |

### `RewardConfig` (frozen)
| Key | Default | Description |
|-----|---------|-------------|
| `beta` | 1.2 | Cost weight in reward |
| `depreciation_coeff` | 1.0 | Battery wear cost factor |

### `ExperimentConfig`
Top-level config composed of `EnvConfig`, `MarketConfig`, `RewardConfig`, and one `AgentConfig`. Supports `from_yaml()` / `to_yaml()` for reproducible experiment management.

## External Dependencies

- `pydantic` — Typed config models with validation
- `numpy` — Noise processes, buffer index sampling
- `torch` — Buffer tensor storage

## Related Areas

- [models.md](models.md) — Agents that use buffers and noise
- [training.md](training.md) — Config consumed by Trainer; regulators plugged into EnergyEnv
- [environment.md](environment.md) — Legacy `anti_cartel.py` (parallel to `src/market/`)
