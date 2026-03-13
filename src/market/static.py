from __future__ import annotations

from collections import deque

import numpy as np

from ..config.base import MarketConfig


class StaticRegulator:
    """Fixed-threshold anti-cartel regulator (detection + ceiling mechanisms)."""

    def __init__(self, config: MarketConfig) -> None:
        self.mechanism_type = config.mechanism_type
        self.penalty_factor = config.penalty_factor
        self.markup_limit = config.markup_limit
        self.market_elasticity = config.market_elasticity

        if config.mechanism_type == "detection":
            self.monitoring_window = config.monitoring_window
            self.similarity_threshold = config.similarity_threshold
            self.min_price_variance = config.min_price_variance
            self.price_band_threshold = config.price_band_threshold
            self._sustained = int(config.monitoring_window * 0.3)
            self.price_history: deque = deque(maxlen=config.monitoring_window)
            self._current_episode: list = []

    # ------------------------------------------------------------------
    # MarketRegulator interface
    # ------------------------------------------------------------------

    def calculate_penalties(self, selling_prices: list[float], grid_price: float) -> list[float]:
        if self.mechanism_type is None:
            return [0.0] * len(selling_prices)
        if self.mechanism_type == "detection":
            return self._detection_penalties(selling_prices)
        if self.mechanism_type == "ceiling":
            return self._ceiling_penalties(selling_prices, grid_price)
        return [0.0] * len(selling_prices)

    def update(self, selling_prices: list[float], episode_done: bool = False) -> None:
        if self.mechanism_type != "detection":
            return
        self._current_episode.append(selling_prices)
        if episode_done and self._current_episode:
            episode_avg = np.mean(np.array(self._current_episode), axis=0)
            self.price_history.append(episode_avg)
            self._current_episode = []

    def get_price_ceiling(self, grid_price: float) -> float:
        if self.mechanism_type != "ceiling":
            return float("inf")
        ceiling = grid_price * (1.0 - self.markup_limit)
        return float(np.clip(ceiling, grid_price * 0.5, grid_price * 0.95))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detection_penalties(self, selling_prices: list[float]) -> list[float]:
        if len(self.price_history) < self._sustained:
            return [0.0] * len(selling_prices)

        penalties = [0.0] * len(selling_prices)
        hist = np.array(list(self.price_history))
        n = len(selling_prices)
        window = hist[-self._sustained :]

        for i in range(n):
            for j in range(i + 1, n):
                if i >= window.shape[1] or j >= window.shape[1]:
                    continue
                pi, pj = window[:, i], window[:, j]
                mi, mj = pi.mean(), pj.mean()
                si, sj = pi.std(), pj.std()

                if abs(mi - mj) / max(mi, mj, 1e-9) < self.price_band_threshold:
                    pen = self.penalty_factor * 0.5
                    penalties[i] += pen * selling_prices[i]
                    penalties[j] += pen * selling_prices[j]

                if si < self.min_price_variance and sj < self.min_price_variance:
                    pen = self.penalty_factor * 0.3
                    penalties[i] += pen * selling_prices[i]
                    penalties[j] += pen * selling_prices[j]

                if si > 0 and sj > 0:
                    corr = np.corrcoef(pi, pj)[0, 1]
                    if not np.isnan(corr) and corr > self.similarity_threshold:
                        pen = self.penalty_factor * 0.4
                        penalties[i] += pen * selling_prices[i]
                        penalties[j] += pen * selling_prices[j]

        return penalties

    def _ceiling_penalties(self, selling_prices: list[float], grid_price: float) -> list[float]:
        ceiling = self.get_price_ceiling(grid_price)
        penalties = []
        for price in selling_prices:
            excess = max(0.0, price - ceiling)
            penalties.append(self.penalty_factor * excess * 1.5)
        return penalties
