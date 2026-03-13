from __future__ import annotations

import torch

from ..config.base import EnvConfig


class ActionUnscaler:
    """Unscale tanh-bounded actions [-1, 1] to physical values.

    HVAC: [-1, 1] → [0, max_energy_consumption]  (via scaling from tanh output)
    Battery: [-1, 1] → [-discharge_rate, charge_rate]
    Price: [0, 1] (already normalized, passed through unchanged)
    """

    def __init__(self, config: EnvConfig) -> None:
        self.max_hvac = config.max_energy_consumption
        self.charge_rate = config.battery_charging_max_rate
        self.discharge_rate = config.battery_discharging_max_rate

    def unscale(self, actions: torch.Tensor) -> torch.Tensor:
        """Unscale actions of shape [num_houses, 3] → physical values.

        Args:
            actions: Tensor of shape [num_houses, 3] in normalized space.

        Returns:
            Tensor of same shape with physical values.
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        hvac = actions[:, 0] * self.max_hvac
        battery = -self.discharge_rate + (actions[:, 1] + 1.0) * (self.charge_rate + self.discharge_rate) / 2.0
        price = actions[:, 2]

        return torch.stack([hvac, battery, price], dim=1)
