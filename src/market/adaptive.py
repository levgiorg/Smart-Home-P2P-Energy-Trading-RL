from __future__ import annotations

from collections import deque

import numpy as np

from ..config.base import MarketConfig


class AdaptiveRegulator:
    """Adaptive anti-cartel regulator with rolling window thresholds.

    Thresholds (delta_p, delta_v, delta_c) are updated from the rolling
    distribution of observed prices rather than being hardcoded.
    """

    def __init__(self, config: MarketConfig) -> None:
        self.penalty_factor = config.penalty_factor
        self.markup_limit = config.markup_limit
        self.k_p = config.k_p
        self.k_v = config.k_v
        self.k_c = config.k_c
        self.delta_v_min = config.delta_v_min
        self.delta_c_max = config.delta_c_max
        self.rolling_window = config.rolling_window

        self.price_history: deque = deque(maxlen=config.monitoring_window)
        self._current_episode: list = []

    # ------------------------------------------------------------------
    # MarketRegulator interface
    # ------------------------------------------------------------------

    def calculate_penalties(self, selling_prices: list[float], grid_price: float) -> list[float]:
        if len(self.price_history) < max(5, self.rolling_window // 4):
            return [0.0] * len(selling_prices)

        delta_p, delta_v, delta_c = self._compute_thresholds()
        return self._detection_penalties(selling_prices, delta_p, delta_v, delta_c)

    def update(self, selling_prices: list[float], episode_done: bool = False) -> None:
        self._current_episode.append(selling_prices)
        if episode_done and self._current_episode:
            episode_avg = np.mean(np.array(self._current_episode), axis=0)
            self.price_history.append(episode_avg)
            self._current_episode = []

    def get_price_ceiling(self, grid_price: float) -> float:
        return float("inf")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_thresholds(self) -> tuple[float, float, float]:
        window = np.array(list(self.price_history)[-self.rolling_window :])

        if window.shape[0] < 2:
            return 0.05, self.delta_v_min, 0.85

        # Pairwise spreads
        n = window.shape[1]
        spreads: list[float] = []
        variances: list[float] = []
        for i in range(n):
            variances.append(float(np.std(window[:, i])))
            for j in range(i + 1, n):
                mi, mj = np.mean(window[:, i]), np.mean(window[:, j])
                denom = max(max(mi, mj), 1e-9)
                spreads.append(abs(mi - mj) / denom)

        spreads_arr = np.array(spreads)
        delta_p = float(np.mean(spreads_arr) + self.k_p * np.std(spreads_arr))

        var_arr = np.array(variances)
        delta_v = float(max(self.delta_v_min, np.mean(var_arr) - self.k_v * np.std(var_arr)))

        # Cross-house correlations
        corrs: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.std(window[:, i]) > 0 and np.std(window[:, j]) > 0:
                    c = float(np.corrcoef(window[:, i], window[:, j])[0, 1])
                    if not np.isnan(c):
                        corrs.append(c)
        if corrs:
            corr_arr = np.array(corrs)
            delta_c = float(min(self.delta_c_max, np.mean(corr_arr) + self.k_c * np.std(corr_arr)))
        else:
            delta_c = self.delta_c_max

        return delta_p, delta_v, delta_c

    def _detection_penalties(
        self,
        selling_prices: list[float],
        delta_p: float,
        delta_v: float,
        delta_c: float,
    ) -> list[float]:
        window = np.array(list(self.price_history)[-self.rolling_window :])
        penalties = [0.0] * len(selling_prices)
        n = len(selling_prices)
        for i in range(n):
            for j in range(i + 1, n):
                if i >= window.shape[1] or j >= window.shape[1]:
                    continue
                pi, pj = window[:, i], window[:, j]
                mi, mj = pi.mean(), pj.mean()
                si, sj = pi.std(), pj.std()

                if abs(mi - mj) / max(max(mi, mj), 1e-9) < delta_p:
                    pen = self.penalty_factor * 0.5
                    penalties[i] += pen * selling_prices[i]
                    penalties[j] += pen * selling_prices[j]

                if si < delta_v and sj < delta_v:
                    pen = self.penalty_factor * 0.3
                    penalties[i] += pen * selling_prices[i]
                    penalties[j] += pen * selling_prices[j]

                if si > 0 and sj > 0:
                    corr = np.corrcoef(pi, pj)[0, 1]
                    if not np.isnan(corr) and corr > delta_c:
                        pen = self.penalty_factor * 0.4
                        penalties[i] += pen * selling_prices[i]
                        penalties[j] += pen * selling_prices[j]

        return penalties
