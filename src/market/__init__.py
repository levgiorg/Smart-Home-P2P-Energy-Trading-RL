from .base import MarketRegulator
from .static import StaticRegulator
from .adaptive import AdaptiveRegulator


def create_regulator(config) -> MarketRegulator:
    """Factory: return the appropriate regulator based on MarketConfig."""
    if config.mechanism_type == "adaptive":
        return AdaptiveRegulator(config)
    return StaticRegulator(config)


__all__ = ["MarketRegulator", "StaticRegulator", "AdaptiveRegulator", "create_regulator"]
