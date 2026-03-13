from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StressScenario:
    """Multiplicative modifiers applied to environment data during evaluation."""

    name: str
    price_multiplier: float = 1.0
    solar_reduction: float = 0.0   # 0.0 = no reduction, 0.8 = 80% reduction
    demand_multiplier: float = 1.0


SCENARIOS: dict[str, StressScenario] = {
    "normal": StressScenario(name="normal"),
    "price_spike": StressScenario(name="price_spike", price_multiplier=3.0),
    "solar_dropout": StressScenario(name="solar_dropout", solar_reduction=0.8),
    "demand_surge": StressScenario(name="demand_surge", demand_multiplier=1.5),
    "combined_crisis": StressScenario(
        name="combined_crisis",
        price_multiplier=3.0,
        solar_reduction=0.8,
        demand_multiplier=1.5,
    ),
}
