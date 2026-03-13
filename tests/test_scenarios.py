"""Tests for stress scenarios."""
import pytest
from src.environment.scenarios import StressScenario, SCENARIOS


def test_scenarios_dict_keys():
    expected = {"normal", "price_spike", "solar_dropout", "demand_surge", "combined_crisis"}
    assert set(SCENARIOS.keys()) == expected


def test_normal_scenario_no_modifiers():
    s = SCENARIOS["normal"]
    assert s.price_multiplier == 1.0
    assert s.solar_reduction == 0.0
    assert s.demand_multiplier == 1.0


def test_combined_crisis_all_active():
    s = SCENARIOS["combined_crisis"]
    assert s.price_multiplier > 1.0
    assert s.solar_reduction > 0.0
    assert s.demand_multiplier > 1.0


def test_scenario_frozen():
    s = StressScenario(name="test", price_multiplier=2.0)
    with pytest.raises(Exception):
        s.price_multiplier = 3.0  # type: ignore[misc]


def test_scenario_custom():
    s = StressScenario(name="custom", price_multiplier=1.5, solar_reduction=0.3)
    assert s.name == "custom"
    assert s.price_multiplier == 1.5
