from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    num_houses: int = 10
    state_dim_per_house: int = 17
    action_dim_per_house: int = 3
    battery_capacity_min: float = 0.6
    battery_capacity_max: float = 10.0
    battery_charging_max_rate: float = 3.0
    battery_discharging_max_rate: float = 3.0
    battery_charging_efficiency: float = 0.98
    battery_discharging_efficiency: float = 0.98
    temperature_min: float = 20.0
    temperature_max: float = 22.0
    hvac_efficiency: float = 1.1
    max_energy_consumption: float = 200.0
    initial_inside_temperature: float = 21.0
    comfort_penalty: float = 4.0
    grid_fee: float = 0.018
    initial_selling_price_ratio: float = 0.8
    min_selling_price: float = 0.55
    hvac_action_bounds: list[float] = Field(default_factory=lambda: [-1.0, 1.0])
    battery_action_bounds: list[float] = Field(default_factory=lambda: [-1.0, 1.0])
    selling_price_bounds: list[float] = Field(default_factory=lambda: [0.0, 1.0])
    weather_data_path: str = "data/ninja_weather_55.6838_12.5354_uncorrected.csv"
    price_data_path: str = "data/2014_DK2_spot_prices.csv"
    num_hours: int = 24
    random_seed: int | None = 42

    model_config = {"frozen": True}


class MarketConfig(BaseModel):
    mechanism_type: Literal["detection", "ceiling", "adaptive"] | None = "detection"
    monitoring_window: int = 100
    penalty_factor: float = 1.5
    markup_limit: float = 0.2
    market_elasticity: float = -0.5
    # Static thresholds (used when mechanism_type != "adaptive")
    price_band_threshold: float = 0.05
    min_price_variance: float = 1e-4
    similarity_threshold: float = 0.85
    # Adaptive params
    k_p: float = 1.0
    k_v: float = 1.0
    k_c: float = 1.0
    delta_v_min: float = 1e-5
    delta_c_max: float = 0.95
    rolling_window: int = 50

    model_config = {"frozen": True}


class RewardConfig(BaseModel):
    beta: float = 1.2
    price_penalty: float = 90.0
    delta: float = 0.9
    depreciation_coeff: float = 1.0

    model_config = {"frozen": True}
