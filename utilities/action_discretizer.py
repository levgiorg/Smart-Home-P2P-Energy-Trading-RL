import numpy as np
import torch


class ActionDiscretizer:
    """
    Action space discretization utilities for DQN implementation.

    Converts between continuous DDPG action space and discrete DQN action space
    to enable fair comparison between algorithms. The discretization maintains
    physical meaningfulness while creating a large enough action space to
    demonstrate the curse of dimensionality in discrete RL.
    """

    def __init__(self, action_bounds: dict[str, list[float]]):
        """
        Initialize action discretizer with continuous action bounds.

        Args:
            action_bounds: Dictionary with keys 'hvac_energy', 'battery_action', 'selling_price'
                          and values as [min, max] bounds
        """
        self.action_bounds = action_bounds

        # Define discrete levels for each action type
        # These create 2,100 total discrete actions: 10 × 21 × 10
        self.hvac_levels = 10  # 0%, 10%, 20%, ..., 100%
        self.battery_levels = 21  # -100%, -90%, ..., 0%, ..., 90%, 100%
        self.price_levels = 10  # 0%, 10%, 20%, ..., 100%

        self.total_actions = self.hvac_levels * self.battery_levels * self.price_levels

        # Create discrete action mappings
        self._create_action_mappings()

    def _create_action_mappings(self) -> None:
        """Create mappings between discrete action indices and continuous values."""
        # HVAC energy levels (0 to 1, normalized to action bounds)
        hvac_min, hvac_max = self.action_bounds["hvac_energy"]
        self.hvac_discrete_values = np.linspace(0, 1, self.hvac_levels)
        self.hvac_continuous_values = np.linspace(hvac_min, hvac_max, self.hvac_levels)

        # Battery action levels (-1 to 1, normalized to action bounds)
        battery_min, battery_max = self.action_bounds["battery_action"]
        self.battery_discrete_values = np.linspace(-1, 1, self.battery_levels)
        self.battery_continuous_values = np.linspace(battery_min, battery_max, self.battery_levels)

        # Selling price levels (0 to 1, normalized to action bounds)
        price_min, price_max = self.action_bounds["selling_price"]
        self.price_discrete_values = np.linspace(0, 1, self.price_levels)
        self.price_continuous_values = np.linspace(price_min, price_max, self.price_levels)

        # Create lookup tables for fast conversion
        self._create_lookup_tables()

    def _create_lookup_tables(self) -> None:
        """Create lookup tables for fast action conversion."""
        self.action_to_continuous = {}
        self.continuous_to_action = {}

        action_idx = 0
        for h in range(self.hvac_levels):
            for b in range(self.battery_levels):
                for p in range(self.price_levels):
                    # Map discrete action index to continuous values
                    continuous_action = [
                        self.hvac_continuous_values[h],
                        self.battery_continuous_values[b],
                        self.price_continuous_values[p],
                    ]
                    self.action_to_continuous[action_idx] = continuous_action

                    # Create reverse mapping (approximate for continuous to discrete)
                    key = (h, b, p)
                    self.continuous_to_action[key] = action_idx

                    action_idx += 1

    def discrete_to_continuous(self, discrete_action: int) -> list[float]:
        """
        Convert discrete action index to continuous action values.

        Args:
            discrete_action: Integer action index [0, total_actions-1]

        Returns:
            List of continuous action values [hvac_energy, battery_action, selling_price]

        Raises:
            TypeError: If discrete_action is not an integer
            ValueError: If discrete_action is out of valid range
        """
        # FIXED: Add comprehensive input validation
        if not isinstance(discrete_action, (int, np.integer)):
            raise TypeError(f"Discrete action must be integer, got {type(discrete_action)}")

        discrete_action = int(discrete_action)  # Convert numpy integers

        if discrete_action < 0 or discrete_action >= self.total_actions:
            raise ValueError(
                f"Discrete action {discrete_action} out of range [0, {self.total_actions - 1}]"
            )

        return self.action_to_continuous[discrete_action]

    def continuous_to_discrete(
        self, continuous_action: list[float] | np.ndarray | torch.Tensor
    ) -> int:
        """
        Convert continuous action to nearest discrete action index.

        Args:
            continuous_action: Continuous action values [hvac_energy, battery_action, selling_price]

        Returns:
            Nearest discrete action index

        Raises:
            TypeError: If continuous_action is not a supported type
            ValueError: If continuous_action doesn't have exactly 3 elements
        """
        # FIXED: Add comprehensive input validation
        if isinstance(continuous_action, torch.Tensor):
            if continuous_action.dim() > 1:
                raise ValueError(f"Expected 1D tensor, got {continuous_action.dim()}D")
            continuous_action = continuous_action.detach().cpu().numpy()
        elif isinstance(continuous_action, np.ndarray):
            if continuous_action.ndim > 1:
                raise ValueError(f"Expected 1D array, got {continuous_action.ndim}D")
            continuous_action = continuous_action.tolist()
        elif not isinstance(continuous_action, (list, tuple)):
            raise TypeError(f"Unsupported action type: {type(continuous_action)}")

        if len(continuous_action) != 3:
            raise ValueError(f"Expected 3 action values, got {len(continuous_action)}")

        hvac_val, battery_val, price_val = continuous_action

        # Validate action bounds
        hvac_min, hvac_max = self.action_bounds["hvac_energy"]
        battery_min, battery_max = self.action_bounds["battery_action"]
        price_min, price_max = self.action_bounds["selling_price"]

        if not (hvac_min <= hvac_val <= hvac_max):
            print(
                f"Warning: HVAC value {hvac_val} outside bounds [{hvac_min}, {hvac_max}], clipping"
            )
            hvac_val = np.clip(hvac_val, hvac_min, hvac_max)

        if not (battery_min <= battery_val <= battery_max):
            print(
                f"Warning: Battery value {battery_val} outside bounds [{battery_min}, {battery_max}], clipping"
            )
            battery_val = np.clip(battery_val, battery_min, battery_max)

        if not (price_min <= price_val <= price_max):
            print(
                f"Warning: Price value {price_val} outside bounds [{price_min}, {price_max}], clipping"
            )
            price_val = np.clip(price_val, price_min, price_max)

        # Find nearest discrete indices
        hvac_idx = self._find_nearest_index(hvac_val, self.hvac_continuous_values)
        battery_idx = self._find_nearest_index(battery_val, self.battery_continuous_values)
        price_idx = self._find_nearest_index(price_val, self.price_continuous_values)

        # Convert to single action index
        discrete_action = (
            hvac_idx * self.battery_levels * self.price_levels
            + battery_idx * self.price_levels
            + price_idx
        )

        return discrete_action

    def _find_nearest_index(self, value: float, value_array: np.ndarray) -> int:
        """Find index of nearest value in array."""
        return int(np.argmin(np.abs(value_array - value)))

    def get_action_space_info(self) -> dict[str, int | list[int]]:
        """
        Get information about the discrete action space.

        Returns:
            Dictionary with action space dimensions and structure
        """
        return {
            "total_actions": self.total_actions,
            "hvac_levels": self.hvac_levels,
            "battery_levels": self.battery_levels,
            "price_levels": self.price_levels,
            "action_shape": [self.hvac_levels, self.battery_levels, self.price_levels],
        }

    def sample_random_action(self) -> int:
        """Sample a random discrete action."""
        return np.random.randint(0, self.total_actions)

    def get_action_description(self, discrete_action: int) -> dict[str, float]:
        """
        Get human-readable description of discrete action.

        Args:
            discrete_action: Discrete action index

        Returns:
            Dictionary with action component values and descriptions
        """
        continuous_vals = self.discrete_to_continuous(discrete_action)
        hvac_val, battery_val, price_val = continuous_vals

        return {
            "hvac_energy": hvac_val,
            "hvac_percentage": (hvac_val - self.action_bounds["hvac_energy"][0])
            / (self.action_bounds["hvac_energy"][1] - self.action_bounds["hvac_energy"][0])
            * 100,
            "battery_action": battery_val,
            "battery_percentage": battery_val * 100,  # Already normalized [-1, 1]
            "selling_price": price_val,
            "price_percentage": price_val * 100,  # Already normalized [0, 1]
        }

    def batch_discrete_to_continuous(
        self, discrete_actions: list[int] | np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Convert batch of discrete actions to continuous actions.

        Args:
            discrete_actions: Batch of discrete action indices

        Returns:
            Array of shape (batch_size, 3) with continuous actions
        """
        if isinstance(discrete_actions, torch.Tensor):
            discrete_actions = discrete_actions.detach().cpu().numpy()
        if isinstance(discrete_actions, list):
            discrete_actions = np.array(discrete_actions)

        len(discrete_actions)

        # FIXED: Vectorized batch processing for efficiency
        discrete_actions = discrete_actions.astype(int)

        # Validate all actions at once
        invalid_mask = (discrete_actions < 0) | (discrete_actions >= self.total_actions)
        if np.any(invalid_mask):
            invalid_actions = discrete_actions[invalid_mask]
            raise ValueError(f"Invalid actions found: {invalid_actions}")

        # Vectorized conversion using lookup arrays
        continuous_actions = np.array(
            [self.action_to_continuous[action] for action in discrete_actions]
        )

        return continuous_actions

    def batch_continuous_to_discrete(
        self, continuous_actions: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Convert batch of continuous actions to discrete actions.

        Args:
            continuous_actions: Array of shape (batch_size, 3) with continuous actions

        Returns:
            Array of discrete action indices
        """
        if isinstance(continuous_actions, torch.Tensor):
            continuous_actions = continuous_actions.detach().cpu().numpy()

        batch_size = continuous_actions.shape[0]
        discrete_actions = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            discrete_actions[i] = self.continuous_to_discrete(continuous_actions[i])

        return discrete_actions


def create_discretizer_from_config(config) -> ActionDiscretizer:
    """
    Create ActionDiscretizer from configuration object.

    Args:
        config: Configuration object with action bounds

    Returns:
        Initialized ActionDiscretizer instance
    """
    action_bounds = {
        "hvac_energy": config.get("environment", "hvac_action_bounds"),
        "battery_action": config.get("environment", "battery_action_bounds"),
        "selling_price": config.get("environment", "selling_price_bounds"),
    }

    return ActionDiscretizer(action_bounds)
