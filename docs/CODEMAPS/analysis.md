# Analysis Codemap

**Last Updated:** 2026-03-13
**Entry Points:** `evaluate.py`, `compare.py`, `analysis/analyze_main.py`, `src/evaluation/`, `src/plotting/`

## Architecture

```
evaluate.py (CLI)
    └── src/evaluation/evaluator.py
            ├── evaluate_episode()           single deterministic rollout
            ├── evaluate_multi_seed()        aggregate across seeds/checkpoints
            └── evaluate_scenarios()         stress-scenario suite

compare.py (CLI)
    └── src/experiment/registry.py
            └── ExperimentRegistry.compare()  SQL query → list[dict]

sweep.py (CLI)
    └── src/evaluation/sensitivity.py
            ├── ParameterSweep.sweep_1d()
            └── ParameterSweep.sweep_2d() → npy grid → src/plotting/matplotlib_plots.plot_heatmap()

src/plotting/
    ├── palette.py             shared color / marker defs
    ├── matplotlib_plots.py    plot_convergence, plot_bar_comparison, plot_heatmap
    ├── templates.py           figure templates / layout helpers
    └── tikz_export.py         TikZ/pgfplots export for LaTeX papers

analysis/ (standalone scripts)
    ├── analyze_main.py
    ├── cartel_analyzer.py
    ├── enhanced_analyzer.py
    ├── improved_cartel_analyzer.py
    ├── run_analyzer.py
    ├── single_run_price_analyzer.py
    └── debug_runs.py
```

## Key Modules

| Module | Purpose | Key Exports | Dependencies |
|--------|---------|-------------|--------------|
| `src/evaluation/evaluator.py` | Agent-agnostic evaluation runner | `Evaluator` | `numpy`, `BaseAgent`, `EnergyEnv` |
| `src/evaluation/sensitivity.py` | 1D/2D parameter sweep framework | `ParameterSweep` | `numpy` |
| `src/evaluation/statistics.py` | Statistical aggregation utilities | — | `numpy`, `scipy` |
| `src/plotting/palette.py` | Shared color / marker definitions | `COLORS`, `TIKZ_COLORS`, `AGENT_COLORS`, `AGENT_MARKERS` | — |
| `src/plotting/matplotlib_plots.py` | Convergence, bar, heatmap plots | `plot_convergence`, `plot_bar_comparison`, `plot_heatmap` | `matplotlib`, `numpy`, `pandas` |
| `src/plotting/templates.py` | Figure layout templates | — | `matplotlib` |
| `src/plotting/tikz_export.py` | Export figures for LaTeX | — | — |
| `analysis/cartel_analyzer.py` | P2P price pattern analysis | — | `pandas`, `matplotlib` |
| `analysis/enhanced_analyzer.py` | Multi-metric enhanced analysis | — | `pandas`, `matplotlib` |
| `analysis/single_run_price_analyzer.py` | Price dynamics for a single run | — | `pandas`, `matplotlib` |

## Evaluator API

```python
evaluator = Evaluator(agent, env)

# Single episode
result = evaluator.evaluate_episode()
# → {"reward": float, "steps": int}

# Multi-seed aggregation (loads checkpoints per seed)
result = evaluator.evaluate_multi_seed(
    checkpoint_pattern="results/{seed}/model_best.pt",
    seeds=[0, 1, 2, 3, 4],
    episodes_per_seed=5,
)
# → {"mean_reward", "std_reward", "min_reward", "max_reward", "n"}

# Stress scenario suite
scenarios = {"price_spike": StressScenario(...), ...}
result = evaluator.evaluate_scenarios(scenarios, episodes=3)
# → {"price_spike": {"mean": float, "std": float}, ...}
```

## Sensitivity Analysis (`src/evaluation/sensitivity.py`)

```python
sweep = ParameterSweep(train_fn=my_train_fn, output_dir="results/sensitivity")

# 1D sweep
results = sweep.sweep_1d("penalty_factor", [0.5, 1.0, 1.5, 2.0], base_params, seeds=3)

# 2D sweep → saves npy grid
grid = sweep.sweep_2d("beta", [0.8, 1.0, 1.2], "markup_limit", [0.1, 0.2, 0.3], base_params)
```
Output grid saved to `results/sensitivity/sweep_{param_a}_{param_b}.npy`.

## Plotting API (`src/plotting/matplotlib_plots.py`)

### `plot_convergence(data, xlabel, ylabel, output_path, agent_colors)`
Line plot with ±1 std confidence bands. `data` is `{agent_name: DataFrame(episode, mean, std)}`.

### `plot_bar_comparison(data, errors, xlabel, ylabel, output_path, agent_colors)`
Grouped bar chart. `data` is `{agent_name: {metric_name: value}}`.

### `plot_heatmap(data, xlabel, ylabel, output_path)`
2D heatmap from a DataFrame (for sensitivity sweep results).

## Color Palette (`src/plotting/palette.py`)

| Name | RGB | Default assignment |
|------|-----|--------------------|
| SupBlue | (65, 105, 225) | SAC (best method) |
| SupRed | (220, 60, 60) | TD3 |
| SupGreen | (60, 160, 60) | DDPG |
| SupOrange | (255, 140, 0) | — |
| SupGray | (128, 128, 128) | DQN |

`AGENT_COLORS` maps agent names to color names; can be overridden per-plot to reflect actual performance ranking.

## Standalone Analysis Scripts

All scripts in `analysis/` operate directly on `results/` CSV files and produce matplotlib figures.

| Script | What it analyzes |
|--------|-----------------|
| `analyze_main.py` | Main multi-mechanism comparison entry point |
| `cartel_analyzer.py` | P2P price distribution and cartel detection metrics |
| `enhanced_analyzer.py` | Extended metrics: efficiency, profit, temperature |
| `improved_cartel_analyzer.py` | Refined price pattern detection visualizations |
| `run_analyzer.py` | Across-run aggregation from `experiments.db` |
| `single_run_price_analyzer.py` | Time-series of selling prices for one run |
| `debug_runs.py` | Debugging utility for problematic training runs |

## Predefined Stress Scenarios (`src/environment/scenarios.py`)

| Name | Price mult. | Solar reduction | Demand mult. |
|------|-------------|-----------------|--------------|
| `normal` | 1.0 | 0% | 1.0 |
| `price_spike` | 3.0 | 0% | 1.0 |
| `solar_dropout` | 1.0 | 80% | 1.0 |
| `demand_surge` | 1.0 | 0% | 1.5 |
| `combined_crisis` | 3.0 | 80% | 1.5 |

## External Dependencies

- `matplotlib` — All plot generation
- `numpy` — Statistical aggregation, sweep grids
- `pandas` — CSV results loading
- `scipy` — Statistical tests (in `statistics.py`)

## Related Areas

- [training.md](training.md) — Produces `results/` artifacts consumed here
- [models.md](models.md) — Agents loaded for evaluation
- [environment.md](environment.md) — `EnergyEnv` and `StressScenario` used by `Evaluator`
