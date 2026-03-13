from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MarketRegulator(Protocol):
    """Interface all market regulators must satisfy."""

    def calculate_penalties(self, selling_prices: list[float], grid_price: float) -> list[float]:
        """Return penalty list (one per house) for the current time step."""
        ...

    def update(self, selling_prices: list[float], episode_done: bool = False) -> None:
        """Update internal price history / rolling statistics."""
        ...

    def get_price_ceiling(self, grid_price: float) -> float:
        """Return the maximum allowed selling price (inf = no ceiling)."""
        ...
