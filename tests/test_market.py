"""Tests for market regulators."""

import numpy as np

from src.config.base import MarketConfig
from src.market import create_regulator
from src.market.adaptive import AdaptiveRegulator
from src.market.static import StaticRegulator

GRID_PRICE = 0.25
PRICES = [0.18, 0.18, 0.18, 0.19, 0.20, 0.18, 0.19, 0.18, 0.20, 0.19]


# ── StaticRegulator / detection ───────────────────────────────────────────────


def test_detection_no_penalty_without_history():
    cfg = MarketConfig(mechanism_type="detection")
    reg = StaticRegulator(cfg)
    penalties = reg.calculate_penalties(PRICES, GRID_PRICE)
    # Not enough history yet → all zeros
    assert all(p == 0.0 for p in penalties)


def test_detection_penalises_similar_prices():
    cfg = MarketConfig(mechanism_type="detection", monitoring_window=20, similarity_threshold=0.5)
    reg = StaticRegulator(cfg)
    # Build history with similar prices to trigger pattern detection
    for ep in range(20):
        # simulate end-of-episode update
        reg.update([0.18] * 10, episode_done=True)
    penalties = reg.calculate_penalties([0.18] * 10, GRID_PRICE)
    assert any(p > 0 for p in penalties)


# ── StaticRegulator / ceiling ─────────────────────────────────────────────────


def test_ceiling_price():
    cfg = MarketConfig(mechanism_type="ceiling", markup_limit=0.2)
    reg = StaticRegulator(cfg)
    ceiling = reg.get_price_ceiling(GRID_PRICE)
    assert ceiling < GRID_PRICE
    assert ceiling >= GRID_PRICE * 0.5


def test_ceiling_penalty_for_above():
    cfg = MarketConfig(mechanism_type="ceiling", markup_limit=0.2, penalty_factor=1.5)
    reg = StaticRegulator(cfg)
    ceiling = reg.get_price_ceiling(GRID_PRICE)
    above_ceiling = [ceiling + 0.05] * 10
    penalties = reg.calculate_penalties(above_ceiling, GRID_PRICE)
    assert all(p > 0 for p in penalties)


def test_no_penalty_below_ceiling():
    cfg = MarketConfig(mechanism_type="ceiling", markup_limit=0.2)
    reg = StaticRegulator(cfg)
    ceiling = reg.get_price_ceiling(GRID_PRICE)
    below = [ceiling - 0.01] * 10
    penalties = reg.calculate_penalties(below, GRID_PRICE)
    assert all(p == 0.0 for p in penalties)


# ── StaticRegulator / null ────────────────────────────────────────────────────


def test_null_regulator_no_penalties():
    cfg = MarketConfig(mechanism_type=None)
    reg = StaticRegulator(cfg)
    assert all(p == 0.0 for p in reg.calculate_penalties(PRICES, GRID_PRICE))
    assert reg.get_price_ceiling(GRID_PRICE) == float("inf")


# ── AdaptiveRegulator ─────────────────────────────────────────────────────────


def test_adaptive_threshold_convergence():
    cfg = MarketConfig(mechanism_type="adaptive", rolling_window=10, monitoring_window=30)
    reg = AdaptiveRegulator(cfg)
    # Inject varied price history to build rolling stats
    rng = np.random.default_rng(0)
    for _ in range(30):
        prices = rng.uniform(0.15, 0.25, 10).tolist()
        reg.update(prices, episode_done=True)
    # Should compute adaptive thresholds without error
    penalties = reg.calculate_penalties(PRICES, GRID_PRICE)
    assert len(penalties) == 10
    assert all(isinstance(p, float) for p in penalties)


# ── Factory ───────────────────────────────────────────────────────────────────


def test_create_regulator_detection():
    cfg = MarketConfig(mechanism_type="detection")
    reg = create_regulator(cfg)
    assert isinstance(reg, StaticRegulator)


def test_create_regulator_adaptive():
    cfg = MarketConfig(mechanism_type="adaptive")
    reg = create_regulator(cfg)
    assert isinstance(reg, AdaptiveRegulator)
