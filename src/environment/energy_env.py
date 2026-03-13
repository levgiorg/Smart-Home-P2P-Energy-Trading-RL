from __future__ import annotations

"""Thin wrapper around the legacy Environment that:
- Accepts a MarketConfig and builds the appropriate MarketRegulator
- Accepts an optional StressScenario for evaluation
- Exposes a clean interface: reset() → state, step(action) → (next_state, reward, done, info)

The heavy physics / data-loading logic lives in environment/environment.py unchanged.
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from ..config.base import EnvConfig, MarketConfig
from ..market import create_regulator
from .scenarios import StressScenario


class EnergyEnv:
    """Wraps the legacy Environment and plugs in a MarketRegulator + StressScenario."""

    def __init__(
        self,
        env_config: EnvConfig,
        market_config: MarketConfig,
        scenario: StressScenario | None = None,
        eval_mode: bool = False,
        dynamic: bool = False,
    ) -> None:
        from environment.environment import Environment  # noqa: PLC0415

        self._env = Environment(dynamic=dynamic, eval_mode=eval_mode)
        self._regulator = create_regulator(market_config)
        self._scenario = scenario
        self._config = env_config
        self._market_config = market_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> torch.Tensor:
        state = self._env.reset()
        return torch.tensor(state, dtype=torch.float32)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, dict]:
        """Execute one time step.

        Args:
            actions: Tensor of shape [num_houses, 3] (normalized, pre-unscaling).

        Returns:
            next_state, rewards, done, info
        """
        # Apply scenario modifiers to environment data before stepping
        if self._scenario is not None:
            self._apply_scenario(self._scenario)

        next_state, rewards, done, info = self._env.step(actions)

        # Update regulator price history
        selling_prices = actions[:, 2].tolist() if isinstance(actions, torch.Tensor) else [a[2] for a in actions]
        self._regulator.update(selling_prices, episode_done=done)

        next_state_t = torch.tensor(next_state, dtype=torch.float32)
        reward_t = torch.tensor(rewards, dtype=torch.float32)
        return next_state_t, reward_t, done, info

    def _apply_scenario(self, scenario: StressScenario) -> None:
        """Apply stress scenario multipliers to the live environment data."""
        env = self._env
        if scenario.price_multiplier != 1.0 and hasattr(env, "price"):
            env.price = env.price * scenario.price_multiplier  # type: ignore[operator]
        if scenario.solar_reduction > 0.0 and hasattr(env, "sun_power"):
            env.sun_power = [p * (1.0 - scenario.solar_reduction) for p in env.sun_power]  # type: ignore[operator]
        if scenario.demand_multiplier != 1.0 and hasattr(env, "power_demand"):
            env.power_demand = [p * scenario.demand_multiplier for p in env.power_demand]  # type: ignore[operator]

    @property
    def state_dim(self) -> int:
        return self._env.state_dim

    @property
    def action_dim(self) -> int:
        return self._env.action_dim

    @property
    def action_bounds(self) -> dict:
        return self._env.ACTION_BOUNDS

    @property
    def num_houses(self) -> int:
        return self._env.num_houses

    @property
    def regulator(self):
        return self._regulator
