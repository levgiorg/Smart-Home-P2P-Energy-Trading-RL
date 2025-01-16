import torch
from hyperparameters import Config
import numbers

class Utilities:
    def __init__(self, num_houses, centralized=False):
        config = Config()
        self.e_max = config.get('environment', 'e_max')
        self.d_max = config.get('environment', 'd_max')
        self.c_max = config.get('environment', 'c_max')
        self.battery_capacity_min = config.get('environment', 'battery_capacity_min')
        self.battery_capacity_max = config.get('environment', 'battery_capacity_max')
        self.num_hours = config.get('simulation', 'num_hours')
        self.random_seed = config.get('simulation', 'random_seed')
        self.num_houses = num_houses  # Set number of houses
        self.centralized = centralized  # Set centralized flag

    def unscaler(self, actions):
        """
        Unscales actions for all agents.

        Args:
            actions: Tensor of actions.
                - Decentralized mode: Expected shape is either [num_houses, 2] or [1, 2 * num_houses].
                - Centralized mode: Expected shape is either [num_houses, 1], [1, num_houses], or scalar for orchestrator action.

        Returns:
            Unscaled actions.
        """
        if self.centralized:
            if isinstance(actions, numbers.Number) or actions.numel() == 1:
                # Orchestrator action (scalar)
                action_unscaled = -self.d_max + (actions + 1) * (self.c_max + self.d_max) / 2
                return action_unscaled
            elif actions.dim() == 2 and actions.size(1) == 1:
                # House actions: shape [num_houses, 1]
                action0 = actions[:, 0] * self.e_max  # Power for HVAC
                return action0.unsqueeze(1)  # Return as [num_houses, 1]
            elif actions.dim() == 2 and actions.size(0) == 1 and actions.size(1) == self.num_houses:
                # Actions of shape [1, num_houses], reshape to [num_houses, 1]
                actions = actions.view(self.num_houses, 1)
                action0 = actions[:, 0] * self.e_max  # Power for HVAC
                return action0.unsqueeze(1)
            else:
                raise ValueError(f"Expected actions tensor of shape [num_houses, 1], [1, {self.num_houses}], or scalar, but got {actions.shape}")
        else:
            # Decentralized mode
            if actions.dim() == 2 and actions.size(1) == 2:
                # Actions are already in shape [num_houses, 2]
                action0 = actions[:, 0] * self.e_max  # Power for HVAC
                action1 = -self.d_max + (actions[:, 1] + 1) * (self.c_max + self.d_max) / 2  # Battery action
                return torch.stack((action0, action1), dim=1)
            elif actions.dim() == 2 and actions.size(0) == 1 and actions.size(1) == 2 * self.num_houses:
                # Actions are of shape [1, 2*num_houses], reshape to [num_houses, 2]
                actions = actions.view(self.num_houses, 2)
                action0 = actions[:, 0] * self.e_max  # Power for HVAC
                action1 = -self.d_max + (actions[:, 1] + 1) * (self.c_max + self.d_max) / 2  # Battery action
                return torch.stack((action0, action1), dim=1)
            else:
                raise ValueError(f"Expected actions tensor of shape [num_houses, 2] or [1, {2*self.num_houses}], but got {actions.shape}")
            