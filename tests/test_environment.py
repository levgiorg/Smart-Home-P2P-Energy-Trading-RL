"""Tests for EnergyEnv wrapper: reset, step, shapes, reward sign, scenario injection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.config.base import EnvConfig, MarketConfig
from src.environment.scenarios import SCENARIOS, StressScenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_HOUSES = 10
STATE_DIM = 170  # NUM_HOUSES * 17 (7 base + 10 selling prices)
ACTION_DIM = 30  # NUM_HOUSES * 3


def _make_mock_legacy_env(
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    num_houses: int = NUM_HOUSES,
    done: bool = False,
) -> MagicMock:
    """Return a MagicMock that mimics the legacy Environment interface."""
    env = MagicMock()
    env.state_dim = state_dim
    env.action_dim = action_dim
    env.num_houses = num_houses
    env.ACTION_BOUNDS = {
        "hvac_energy": [-1.0, 1.0],
        "battery_action": [-1.0, 1.0],
        "selling_price": [0.5, 0.95],
    }
    env.reset.return_value = [0.0] * state_dim
    env.step.return_value = (
        [0.0] * state_dim,
        [-1.0] * num_houses,
        done,
        {},
    )
    return env


def _make_env_with_mock(
    env_config: EnvConfig | None = None,
    market_config: MarketConfig | None = None,
    scenario: StressScenario | None = None,
    legacy_mock: MagicMock | None = None,
) -> tuple:
    """Create an EnergyEnv whose legacy Environment is replaced by a mock.

    Returns (energy_env, legacy_mock).
    """
    from src.environment.energy_env import EnergyEnv

    cfg = env_config or EnvConfig()
    mkt = market_config or MarketConfig()
    mock = legacy_mock or _make_mock_legacy_env()

    with patch("src.environment.energy_env.EnergyEnv.__init__", return_value=None):
        env = EnergyEnv.__new__(EnergyEnv)

    from src.market import create_regulator

    env._env = mock
    env._regulator = create_regulator(mkt)
    env._scenario = scenario
    env._config = cfg
    env._market_config = mkt

    return env, mock


# ---------------------------------------------------------------------------
# Unit tests — using mocks (no CSV data required)
# ---------------------------------------------------------------------------


class TestResetInterface:
    def test_reset_returns_tensor(self):
        env, _ = _make_env_with_mock()
        state = env.reset()
        assert isinstance(state, torch.Tensor)

    def test_reset_shape_matches_state_dim(self):
        env, _ = _make_env_with_mock()
        state = env.reset()
        assert state.shape == (STATE_DIM,)

    def test_reset_dtype_float32(self):
        env, _ = _make_env_with_mock()
        state = env.reset()
        assert state.dtype == torch.float32

    def test_reset_calls_legacy_reset(self):
        env, mock = _make_env_with_mock()
        env.reset()
        mock.reset.assert_called_once()

    def test_reset_returns_finite_values(self):
        env, _ = _make_env_with_mock()
        state = env.reset()
        assert torch.isfinite(state).all()


class TestStepInterface:
    def test_step_returns_four_tuple(self):
        env, _ = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        result = env.step(actions)
        assert len(result) == 4

    def test_step_next_state_shape(self):
        env, _ = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        next_state, _, _, _ = env.step(actions)
        assert next_state.shape == (STATE_DIM,)

    def test_step_rewards_shape(self):
        env, _ = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        _, rewards, _, _ = env.step(actions)
        assert rewards.shape == (NUM_HOUSES,)

    def test_step_next_state_dtype_float32(self):
        env, _ = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        next_state, _, _, _ = env.step(actions)
        assert next_state.dtype == torch.float32

    def test_step_rewards_dtype_float32(self):
        env, _ = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        _, rewards, _, _ = env.step(actions)
        assert rewards.dtype == torch.float32

    def test_step_done_is_bool(self):
        env, _ = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        _, _, done, _ = env.step(actions)
        assert isinstance(done, bool)

    def test_step_info_is_dict(self):
        env, _ = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        _, _, _, info = env.step(actions)
        assert isinstance(info, dict)

    def test_step_done_true_when_episode_ends(self):
        mock = _make_mock_legacy_env(done=True)
        env, _ = _make_env_with_mock(legacy_mock=mock)
        actions = torch.zeros(NUM_HOUSES, 3)
        _, _, done, _ = env.step(actions)
        assert done is True

    def test_step_calls_legacy_step(self):
        env, mock = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        env.step(actions)
        mock.step.assert_called_once()

    def test_step_negative_reward_sign(self):
        """Legacy environment returns costs — rewards should be negative."""
        env, _ = _make_env_with_mock()
        actions = torch.zeros(NUM_HOUSES, 3)
        _, rewards, _, _ = env.step(actions)
        assert (rewards <= 0).all(), "Expected all rewards to be non-positive"


class TestProperties:
    def test_state_dim_property(self):
        env, _ = _make_env_with_mock()
        assert env.state_dim == STATE_DIM

    def test_action_dim_property(self):
        env, _ = _make_env_with_mock()
        assert env.action_dim == ACTION_DIM

    def test_num_houses_property(self):
        env, _ = _make_env_with_mock()
        assert env.num_houses == NUM_HOUSES

    def test_action_bounds_is_dict(self):
        env, _ = _make_env_with_mock()
        bounds = env.action_bounds
        assert isinstance(bounds, dict)

    def test_action_bounds_keys(self):
        env, _ = _make_env_with_mock()
        bounds = env.action_bounds
        assert "hvac_energy" in bounds
        assert "battery_action" in bounds
        assert "selling_price" in bounds

    def test_regulator_is_not_none(self):
        env, _ = _make_env_with_mock()
        assert env.regulator is not None


class TestScenarioInjection:
    def _step(self, env: object) -> tuple:
        actions = torch.zeros(NUM_HOUSES, 3)
        return env.step(actions)

    def test_no_scenario_does_not_modify_env(self):
        """Without a scenario the legacy env step is called without any attribute mutation.

        We verify this by tracking that env.price is never reassigned: use a spec'd
        mock that records attribute writes, and confirm no scenario method ran.
        """
        env, mock = _make_env_with_mock(scenario=None)
        original_price = 0.5
        mock.price = original_price
        self._step(env)
        # Without a scenario, _apply_scenario is never invoked, so price stays put.
        assert mock.price == original_price

    def test_price_spike_scenario_applied(self):
        """price_multiplier != 1 triggers env.price assignment."""
        mock = _make_mock_legacy_env()
        mock.price = 0.5
        scenario = SCENARIOS["price_spike"]
        env, mock_env = _make_env_with_mock(scenario=scenario, legacy_mock=mock)
        self._step(env)
        # After applying price_multiplier=3.0, env.price should be updated
        assert mock_env.price == pytest.approx(0.5 * 3.0)

    def test_solar_dropout_scenario_applied(self):
        """solar_reduction > 0 triggers env.sun_power assignment."""
        mock = _make_mock_legacy_env()
        mock.sun_power = [1.0] * NUM_HOUSES
        scenario = SCENARIOS["solar_dropout"]
        env, mock_env = _make_env_with_mock(scenario=scenario, legacy_mock=mock)
        self._step(env)
        expected = [1.0 * (1.0 - 0.8)] * NUM_HOUSES
        assert mock_env.sun_power == pytest.approx(expected)

    def test_demand_surge_scenario_applied(self):
        """demand_multiplier != 1 triggers env.power_demand assignment."""
        mock = _make_mock_legacy_env()
        mock.power_demand = [2.0] * NUM_HOUSES
        scenario = SCENARIOS["demand_surge"]
        env, mock_env = _make_env_with_mock(scenario=scenario, legacy_mock=mock)
        self._step(env)
        expected = [2.0 * 1.5] * NUM_HOUSES
        assert mock_env.power_demand == pytest.approx(expected)

    def test_normal_scenario_no_price_change(self):
        """Normal scenario (all multipliers == 1.0) must not overwrite env.price."""
        mock = _make_mock_legacy_env()
        original_price = 0.4
        mock.price = original_price
        scenario = SCENARIOS["normal"]
        env, mock_env = _make_env_with_mock(scenario=scenario, legacy_mock=mock)
        self._step(env)
        assert mock_env.price == pytest.approx(original_price)

    def test_combined_crisis_all_applied(self):
        """Combined crisis should modify price, solar, and demand."""
        mock = _make_mock_legacy_env()
        mock.price = 0.3
        mock.sun_power = [1.0] * NUM_HOUSES
        mock.power_demand = [1.0] * NUM_HOUSES
        scenario = SCENARIOS["combined_crisis"]
        env, mock_env = _make_env_with_mock(scenario=scenario, legacy_mock=mock)
        self._step(env)
        assert mock_env.price == pytest.approx(0.3 * 3.0)
        assert mock_env.sun_power == pytest.approx([1.0 * (1.0 - 0.8)] * NUM_HOUSES)
        assert mock_env.power_demand == pytest.approx([1.0 * 1.5] * NUM_HOUSES)


class TestRegulatorIntegration:
    def test_regulator_update_called_on_step(self):
        """EnergyEnv calls regulator.update after each step."""
        env, _ = _make_env_with_mock()
        regulator_mock = MagicMock()
        env._regulator = regulator_mock
        actions = torch.zeros(NUM_HOUSES, 3)
        env.step(actions)
        regulator_mock.update.assert_called_once()

    def test_regulator_receives_selling_prices(self):
        """Regulator update receives the selling-price column (index 2) of actions."""
        env, _ = _make_env_with_mock()
        regulator_mock = MagicMock()
        env._regulator = regulator_mock
        actions = torch.full((NUM_HOUSES, 3), 0.7)
        env.step(actions)
        call_args = regulator_mock.update.call_args
        selling_prices = call_args[0][0]
        assert len(selling_prices) == NUM_HOUSES
        assert all(abs(p - 0.7) < 1e-5 for p in selling_prices)


# ---------------------------------------------------------------------------
# Integration tests — require real CSV data files
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEnergyEnvIntegration:
    """Tests that exercise the real legacy Environment with actual CSV data."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        from pathlib import Path

        data_dir = Path(__file__).parent.parent / "data"
        if not data_dir.exists():
            pytest.skip("data/ directory not found — skipping integration tests")

    @pytest.fixture
    def env(self):
        from src.environment.energy_env import EnergyEnv

        return EnergyEnv(EnvConfig(), MarketConfig())

    def test_real_reset_returns_tensor(self, env):
        state = env.reset()
        assert isinstance(state, torch.Tensor)

    def test_real_reset_shape(self, env):
        state = env.reset()
        assert state.shape == (env.state_dim,)

    def test_real_step_shapes(self, env):
        env.reset()
        actions = torch.zeros(env.num_houses, 3)
        ns, rewards, done, info = env.step(actions)
        assert ns.shape == (env.state_dim,)
        assert rewards.shape == (env.num_houses,)

    def test_real_full_episode_terminates(self, env):
        """A full episode must reach done=True within a finite number of steps."""
        env.reset()
        done = False
        steps = 0
        max_steps = 200
        while not done and steps < max_steps:
            actions = torch.zeros(env.num_houses, 3)
            _, _, done, _ = env.step(actions)
            steps += 1
        assert done, "Episode did not terminate within 200 steps"

    def test_real_state_dim_matches_config(self, env):
        """state_dim exposed by EnergyEnv must equal num_houses * 17."""
        assert env.state_dim == env.num_houses * 17

    def test_real_action_dim_matches_config(self, env):
        """action_dim exposed by EnergyEnv must equal num_houses * 3."""
        assert env.action_dim == env.num_houses * 3

    def test_real_scenario_price_spike(self):
        """Price spike scenario should not crash and episode completes."""
        from src.environment.energy_env import EnergyEnv

        env = EnergyEnv(EnvConfig(), MarketConfig(), scenario=SCENARIOS["price_spike"])
        env.reset()
        actions = torch.zeros(env.num_houses, 3)
        ns, rewards, done, info = env.step(actions)
        assert ns.shape == (env.state_dim,)

    def test_real_scenario_solar_dropout(self):
        """Solar dropout scenario should not crash on step."""
        from src.environment.energy_env import EnergyEnv

        env = EnergyEnv(EnvConfig(), MarketConfig(), scenario=SCENARIOS["solar_dropout"])
        env.reset()
        actions = torch.zeros(env.num_houses, 3)
        ns, _, _, _ = env.step(actions)
        assert ns.shape == (env.state_dim,)
