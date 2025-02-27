import torch
import numbers

from hyperparameters import Config


class Utilities:
    def __init__(self, num_houses, centralized=False):
        """
        Initialize utilities helper class for action handling.
        
        Args:
            num_houses: Number of houses in the environment
            centralized: Whether to use centralized action handling
        """
        config = Config()
        self.max_energy_consumption = config.get('environment', 'max_energy_consumption')
        self.battery_discharging_max_rate = config.get('environment', 'battery_discharging_max_rate')
        self.battery_charging_max_rate = config.get('environment', 'battery_charging_max_rate')
        self.battery_capacity_min = config.get('environment', 'battery_capacity_min')
        self.battery_capacity_max = config.get('environment', 'battery_capacity_max')
        self.num_hours = config.get('simulation', 'num_hours')
        self.random_seed = config.get('simulation', 'random_seed')
        self.num_houses = num_houses
        self.centralized = centralized

    def unscaler(self, actions):
        """
        Unscales actions from normalized [-1,1] range to actual physical values.

        Args:
            actions: Tensor of actions.
                Expected shape is either [num_houses, 3] or [1, 3 * num_houses].
                The three actions are: HVAC power, battery action, and selling price.
                
        Returns:
            Unscaled actions with physical values.
        """
        if self.centralized:
            # Centralized mode logic remains unchanged
            if isinstance(actions, numbers.Number) or actions.numel() == 1:
                battery_action_unscaled = -self.battery_discharging_max_rate + (actions + 1) * (self.battery_charging_max_rate + self.battery_discharging_max_rate) / 2
                return battery_action_unscaled
            elif actions.dim() == 2 and actions.size(1) == 1:
                hvac_action = actions[:, 0] * self.max_energy_consumption
                return hvac_action.unsqueeze(1)
            elif actions.dim() == 2 and actions.size(0) == 1 and actions.size(1) == self.num_houses:
                actions = actions.view(self.num_houses, 1)
                hvac_action = actions[:, 0] * self.max_energy_consumption
                return hvac_action.unsqueeze(1)
            else:
                raise ValueError(f"Expected actions tensor of shape [num_houses, 1], [1, {self.num_houses}], or scalar, but got {actions.shape}")
        else:
            # Decentralized mode with three actions per house
            if actions.dim() == 2 and actions.size(1) == 3:
                # Actions are in shape [num_houses, 3]
                hvac_action = actions[:, 0] * self.max_energy_consumption  # Power for HVAC
                battery_action = -self.battery_discharging_max_rate + (actions[:, 1] + 1) * (self.battery_charging_max_rate + self.battery_discharging_max_rate) / 2  # Battery action
                price_action = actions[:, 2]  # Selling price (already normalized between 0 and 1)
                return torch.stack((hvac_action, battery_action, price_action), dim=1)
            elif actions.dim() == 2 and actions.size(0) == 1 and actions.size(1) == 3 * self.num_houses:
                # Actions are of shape [1, 3*num_houses], reshape to [num_houses, 3]
                actions = actions.view(self.num_houses, 3)
                hvac_action = actions[:, 0] * self.max_energy_consumption  # Power for HVAC
                battery_action = -self.battery_discharging_max_rate + (actions[:, 1] + 1) * (self.battery_charging_max_rate + self.battery_discharging_max_rate) / 2  # Battery action
                price_action = actions[:, 2]  # Selling price (already normalized between 0 and 1)
                return torch.stack((hvac_action, battery_action, price_action), dim=1)
            else:
                raise ValueError(f"Expected actions tensor of shape [num_houses, 3] or [1, {3*self.num_houses}], but got {actions.shape}")