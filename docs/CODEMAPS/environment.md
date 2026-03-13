# Environment Codemap

**Last Updated:** 2026-03-13
**Entry Points:** `environment/environment.py`, `src/environment/energy_env.py`

## Architecture

```
ExperimentConfig ──► EnergyEnv (src/environment/energy_env.py)
                         │
                         ├── Environment (environment/environment.py)   ← physics core
                         │       ├── AntiCartelMechanism                ← reward shaping
                         │       ├── Utilities (legacy)                 ← action unscaling
                         │       └── Config (legacy hyperparameters)
                         │
                         ├── MarketRegulator (src/market/)              ← pluggable regulator
                         └── StressScenario (src/environment/scenarios.py)
```

## Key Modules

| Module | Purpose | Key Exports | Dependencies |
|--------|---------|-------------|--------------|
| `environment/environment.py` | Core physics simulation — HVAC, battery, P2P trading | `Environment` | `numpy`, `pandas`, `torch`, `AntiCartelMechanism`, `Utilities`, `Config` |
| `environment/anti_cartel.py` | Detection and ceiling anti-cartel mechanisms | `AntiCartelMechanism` | `numpy`, `Config` |
| `src/environment/energy_env.py` | Thin wrapper with clean interface; MarketRegulator integration | `EnergyEnv` | `Environment`, `MarketRegulator`, `StressScenario` |
| `src/environment/scenarios.py` | Predefined stress scenarios for evaluation | `StressScenario`, `SCENARIOS` | — |
| `src/environment/wrappers.py` | Additional environment wrappers | — | `EnergyEnv` |

## State Space

Per-house state vector (7 base components + N house selling prices):

| Component | Description |
|-----------|-------------|
| `inside_temperature` | Current indoor temperature (°C) |
| `ambient_temperature` | Outdoor temperature from weather CSV |
| `sun_power` | Solar generation for this house (kW) |
| `price` | Current grid spot price (DKK/kWh) |
| `battery` | Current battery state of charge (kWh) |
| `power_demand` | Base load consumption (kW) |
| `hour_of_day` | Hour within the day (0–23) |
| `selling_prices[0..N-1]` | All houses' current selling prices |

**Total per-house:** `7 + num_houses` → default `17` (with 10 houses).  
**Total:** `17 × num_houses` → default `170`.

## Action Space

Per-house actions (3 continuous dimensions):

| Dim | Action | Bounds |
|-----|--------|--------|
| 0 | `hvac_energy` | `[-1.0, 1.0]` normalized |
| 1 | `battery_action` | `[-1.0, 1.0]` (+ charge, − discharge) |
| 2 | `selling_price` | `[0.0, 1.0]` ratio × grid price |

## Data Flow (one step)

```
agent.select_action(state)
        │
        ▼
Environment.step(actions)
  1. unscale actions (Utilities.unscaler)
  2. _set_selling_prices()          → apply ceiling if mechanism == "ceiling"
  3. _calculate_energy_balances()   → excess / deficit per house
  4. _execute_p2p_trading()         → match sellers (cheapest first) to buyers
  5. _update_houses_and_calculate_rewards()
       ├── _update_temperature()    → exponential mixing with ambient
       ├── _update_battery_state()  → clamp to [min, max] with efficiency
       ├── _calculate_energy_metrics()
       └── _calculate_reward()      → -β·cost + trading_profit
  6. _apply_anti_cartel_mechanism() → subtract penalties from rewards
  7. advance time; optionally _update_dynamic_variables()
        │
        ▼
  return (state, rewards, done, infos)
```

## Reward Function

```
reward_i = -β × (hvac_cost + battery_depreciation + temp_penalty) + trading_profit_i
```

After anti-cartel:
```
reward_i -= penalty_i
```

Config keys: `reward.beta` (default 1.2), `cost_model.depreciation_coeff` (default 1.0), `environment.temperature_max/min` (22°C / 20°C).

## Anti-Cartel Mechanisms (`environment/anti_cartel.py`)

| Mechanism | Key Logic | Penalty Trigger |
|-----------|-----------|-----------------|
| `detection` | Tracks episode-averaged prices in a deque; detects pairwise price matching, low variance, and high correlation | Pattern thresholds exceeded |
| `ceiling` | Enforces `price ≤ grid × (1 − markup_limit)` | Price exceeds ceiling |
| `None` | Free market — no penalties | Never |

Detection patterns checked per house pair:
- **Price matching:** `|mean_i − mean_j| / max < 0.05` → `0.5 × penalty_factor × price`
- **Low variance:** `std < 1e-4` for both → `0.3 × penalty_factor × price`
- **Correlation:** `corr(i,j) > similarity_threshold` → `0.4 × penalty_factor × price`

## Stress Scenarios (`src/environment/scenarios.py`)

| Name | Modifier |
|------|----------|
| `normal` | No change |
| `price_spike` | `price × 3.0` |
| `solar_dropout` | Solar reduced 80% |
| `demand_surge` | Demand × 1.5 |
| `combined_crisis` | All three above |

## Data Sources

| Dataset | Path | Description |
|---------|------|-------------|
| Weather | `data/ninja_weather_55.6838_12.5354_uncorrected.csv` | Hourly ambient temperature (Copenhagen) |
| Spot prices | `data/2014_DK2_spot_prices.csv` | Danish DK2 electricity market |
| Consumption | `data/lstms_predictions/Consumption_prediction_house_{3,4,6}.csv` | LSTM-predicted + actual hourly load |
| Generation | `data/lstms_predictions/Generation_prediction_house_{3,4,6}.csv` | LSTM-predicted + actual solar output |

## External Dependencies

- `numpy` — Array operations, energy balance calculations
- `pandas` — CSV loading and slicing
- `torch` — Action tensor handling

## Related Areas

- [models.md](models.md) — Agents that consume environment observations
- [training.md](training.md) — Training loop that calls `env.step()` / `env.reset()`
- [utilities.md](utilities.md) — `MarketRegulator` protocol and implementations
