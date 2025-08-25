from typing import Tuple, List, Dict, Union, Any, Optional
import numpy as np
import torch

from environment.environment import Environment
from utilities.action_discretizer import ActionDiscretizer, create_discretizer_from_config


class DQNEnvironmentWrapper:
    """
    Environment wrapper for DQN agent interaction.
    
    Wraps the continuous action environment to work with discrete DQN actions,
    handling conversion between discrete action indices and continuous action
    values required by the smart home environment.
    """
    
    def __init__(self, dynamic: bool = False, eval_mode: bool = False):
        """
        Initialize DQN environment wrapper.
        
        Args:
            dynamic: Whether to use dynamic data loading during episodes
            eval_mode: Whether to run in evaluation mode (fixed start point)
        """
        # Initialize the base environment
        self.env = Environment(dynamic=dynamic, eval_mode=eval_mode)
        
        # Create action discretizer from environment configuration
        self.action_discretizer = create_discretizer_from_config(self.env.config)
        
        # Store environment properties
        self.num_houses = self.env.num_houses
        self.action_dim_per_house = 3  # Always 3: hvac, battery, price
        self.state_dim_per_house = self.env.STATE_DIM_PER_HOUSE
        
        # Total dimensions - FIXED: Use independent per-house actions instead of exponential scaling
        self.total_state_dim = self.num_houses * self.state_dim_per_house
        self.actions_per_house = self.action_discretizer.total_actions  # 2,100 actions per house
        
        print(f"DQN Environment Wrapper initialized:")
        print(f"  - Houses: {self.num_houses}")
        print(f"  - State dim per house: {self.state_dim_per_house}")
        print(f"  - Total state dim: {self.total_state_dim}")
        print(f"  - Discrete actions per house: {self.actions_per_house}")
        print(f"  - Architecture: Independent action selection per house")
        
        # Track action usage for analysis
        self.action_usage_stats = {}
        self.reset_action_stats()
        
        # Initialize episode-level metric accumulators (matching DDPG pattern)
        self.episode_metrics = {}
        self.reset_episode_metrics()
    
    def reset_action_stats(self) -> None:
        """Reset action usage statistics."""
        self.action_usage_stats = {
            'discrete_actions': [],
            'continuous_actions': [],
            'action_frequencies': np.zeros(self.action_discretizer.total_actions),
            'step_count': 0
        }
        
    def reset_episode_metrics(self) -> None:
        """Reset episode-level metric accumulators (matching DDPG pattern)."""
        self.episode_metrics = {
            'rewards_per_house': np.zeros(self.num_houses),
            'hvac_consumption_per_house': np.zeros(self.num_houses),
            'depreciation_per_house': np.zeros(self.num_houses),
            'penalty_per_house': np.zeros(self.num_houses),
            'trading_profit_per_house': np.zeros(self.num_houses),
            'energy_bought_p2p_per_house': np.zeros(self.num_houses),
            'selling_prices_per_house': np.zeros(self.num_houses),
            'temperatures_per_house': [],  # List to track temperature at each step
            'step_count': 0
        }
    
    def reset(self) -> np.ndarray:
        """
        Reset environment and return initial state.
        
        Returns:
            Flattened initial state observation
        """
        # Reset base environment
        initial_state = self.env.reset()
        
        # Reset action statistics and episode metrics
        self.reset_action_stats()
        self.reset_episode_metrics()
        
        return self._flatten_state(initial_state)
    
    def step(self, discrete_actions: Union[int, List[int], np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take environment step with discrete actions.
        
        Args:
            discrete_actions: Discrete action(s) for all houses
                            Can be single int (for single-house) or list/array (multi-house)
        
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Handle different input formats - FIXED: Always expect list/array of per-house actions
        if isinstance(discrete_actions, int):
            # Single action for all houses (broadcasting)
            discrete_action_list = [discrete_actions] * self.num_houses
        elif isinstance(discrete_actions, (list, np.ndarray)):
            discrete_action_list = list(discrete_actions)
        else:
            raise ValueError(f"Invalid discrete_actions type: {type(discrete_actions)}")
        
        # Ensure we have actions for all houses
        if len(discrete_action_list) != self.num_houses:
            raise ValueError(f"Expected {self.num_houses} actions, got {len(discrete_action_list)}")
        
        # Convert discrete actions to continuous actions
        continuous_actions = []
        for house_idx, discrete_action in enumerate(discrete_action_list):
            if discrete_action < 0 or discrete_action >= self.action_discretizer.total_actions:
                raise ValueError(f"Discrete action {discrete_action} out of range for house {house_idx}")
            
            continuous_action = self.action_discretizer.discrete_to_continuous(discrete_action)
            continuous_actions.append(continuous_action)
        
        # Flatten continuous actions for environment and convert to torch tensor
        continuous_actions_flat = np.array(continuous_actions).flatten()
        continuous_actions_tensor = torch.FloatTensor(continuous_actions_flat).unsqueeze(0)  # Shape: [1, 30]
        
        # Store action statistics
        self._update_action_stats(discrete_action_list, continuous_actions)
        
        # Step environment with continuous actions (as tensor)
        next_state, rewards, done, info = self.env.step(continuous_actions_tensor)
        
        # Environment is properly providing all required data with correct dimensions
        
        # Convert list of rewards to single reward (sum for multi-agent)
        if isinstance(rewards, list):
            total_reward = sum(rewards)
        else:
            total_reward = rewards
        
        # Accumulate episode metrics (matching DDPG pattern exactly)
        self._accumulate_episode_metrics(rewards, info)
        
        # Add DQN-specific info plus accumulated episode metrics
        info.update({
            'discrete_actions': discrete_action_list,
            'continuous_actions': continuous_actions,
            'action_conversion_successful': True,
            'individual_rewards': rewards if isinstance(rewards, list) else [rewards],
            # Add accumulated episode metrics for episode-end logging
            'house_rewards': self.episode_metrics['rewards_per_house'].tolist(),
            'house_temperatures': self._get_current_temperatures(info),
            'hvac_consumption': self.episode_metrics['hvac_consumption_per_house'].tolist(),
            'depreciation': self.episode_metrics['depreciation_per_house'].tolist(),
            'penalty': self.episode_metrics['penalty_per_house'].tolist(),
            'trading_profit': self.episode_metrics['trading_profit_per_house'].tolist(),
            'energy_bought_p2p': self.episode_metrics['energy_bought_p2p_per_house'].tolist(),
            'selling_prices': self.episode_metrics['selling_prices_per_house'].tolist()
        })
        
        return self._flatten_state(next_state), total_reward, done, info
    
    def _accumulate_episode_metrics(self, rewards: Union[List[float], float], info: Dict[str, Any]) -> None:
        """
        Accumulate episode-level metrics from step info (matching DDPG pattern).
        
        Args:
            rewards: Step rewards (list for multi-house or single value)
            info: Step info dictionary from environment
        """
        self.episode_metrics['step_count'] += 1
        
        # Accumulate rewards
        if isinstance(rewards, list):
            self.episode_metrics['rewards_per_house'] += np.array(rewards)
        else:
            # Single reward - distribute equally across houses
            self.episode_metrics['rewards_per_house'] += rewards / self.num_houses
        
        # Accumulate other metrics from step info (matching DDPG extraction pattern)
        if 'HVAC_energy_cons' in info:
            self.episode_metrics['hvac_consumption_per_house'] += np.array(info['HVAC_energy_cons'])
        
        if 'depreciation' in info:
            self.episode_metrics['depreciation_per_house'] += np.array(info['depreciation'])
            
        if 'penalty' in info:
            self.episode_metrics['penalty_per_house'] += np.array(info['penalty'])
            
        if 'trading_profit' in info:
            self.episode_metrics['trading_profit_per_house'] += np.array(info['trading_profit'])
            
        if 'energy_bought_p2p' in info:
            self.episode_metrics['energy_bought_p2p_per_house'] += np.array(info['energy_bought_p2p'])
        
        # Update selling prices (latest step's prices)
        if 'selling_prices' in info:
            self.episode_metrics['selling_prices_per_house'] = np.array(info['selling_prices'])
            
        # Store current temperatures
        if 'current_temperatures' in info:
            self.episode_metrics['temperatures_per_house'].append(info['current_temperatures'])
            
    def _get_current_temperatures(self, info: Dict[str, Any]) -> List[float]:
        """
        Get current house temperatures from step info.
        
        Args:
            info: Step info dictionary
            
        Returns:
            List of current temperatures for all houses
        """
        if 'current_temperatures' in info:
            return list(info['current_temperatures'])
        elif len(self.episode_metrics['temperatures_per_house']) > 0:
            return self.episode_metrics['temperatures_per_house'][-1]  # Last recorded temperatures
        else:
            return [20.0] * self.num_houses  # Default comfortable temperature
    
    def _flatten_state(self, state: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Flatten multi-house state to single vector.
        
        Args:
            state: Nested state structure from environment
            
        Returns:
            Flattened state array
        """
        if isinstance(state, list):
            state = np.array(state)
        
        return state.flatten()
    
    
    def _update_action_stats(self, discrete_actions: List[int], continuous_actions: List[List[float]]) -> None:
        """Update action usage statistics."""
        self.action_usage_stats['step_count'] += 1
        self.action_usage_stats['discrete_actions'].append(discrete_actions.copy())
        self.action_usage_stats['continuous_actions'].append(continuous_actions.copy())
        
        # Update frequency counts
        for action in discrete_actions:
            if action < len(self.action_usage_stats['action_frequencies']):
                self.action_usage_stats['action_frequencies'][action] += 1
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """
        Get information about the discrete action space.
        
        Returns:
            Dictionary with action space information
        """
        discretizer_info = self.action_discretizer.get_action_space_info()
        
        return {
            'num_houses': self.num_houses,
            'actions_per_house': discretizer_info['total_actions'],
            'total_joint_actions': f"Independent: {self.actions_per_house} per house",
            'action_structure': discretizer_info,
            'state_dim': self.total_state_dim,
            'action_bounds': {
                'hvac_energy': self.env.ACTION_BOUNDS['hvac_energy'],
                'battery_action': self.env.ACTION_BOUNDS['battery_action'],
                'selling_price': self.env.ACTION_BOUNDS['selling_price']
            }
        }
    
    def get_action_usage_stats(self) -> Dict[str, Any]:
        """
        Get action usage statistics for analysis.
        
        Returns:
            Dictionary with action usage patterns
        """
        if self.action_usage_stats['step_count'] == 0:
            return {'message': 'No actions taken yet'}
        
        # Compute action distribution statistics
        total_actions = self.action_usage_stats['step_count'] * self.num_houses
        action_probs = self.action_usage_stats['action_frequencies'] / max(total_actions, 1)
        
        # Compute entropy of action distribution
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        
        # Analyze continuous action patterns
        continuous_actions = np.array(self.action_usage_stats['continuous_actions'])
        if continuous_actions.size > 0:
            continuous_actions = continuous_actions.reshape(-1, 3)  # Flatten to (steps*houses, 3)
            
            continuous_stats = {
                'hvac_mean': np.mean(continuous_actions[:, 0]),
                'hvac_std': np.std(continuous_actions[:, 0]),
                'battery_mean': np.mean(continuous_actions[:, 1]),
                'battery_std': np.std(continuous_actions[:, 1]),
                'price_mean': np.mean(continuous_actions[:, 2]),
                'price_std': np.std(continuous_actions[:, 2])
            }
        else:
            continuous_stats = {}
        
        return {
            'total_steps': self.action_usage_stats['step_count'],
            'total_actions_taken': total_actions,
            'action_entropy': action_entropy,
            'most_used_actions': np.argsort(self.action_usage_stats['action_frequencies'])[::-1][:10],
            'action_frequencies': action_probs,
            'continuous_action_stats': continuous_stats,
            'unique_actions_used': np.sum(self.action_usage_stats['action_frequencies'] > 0)
        }
    
    def sample_random_discrete_action(self) -> int:
        """Sample a random discrete action for single house."""
        return self.action_discretizer.sample_random_action()
    
    def sample_random_house_actions(self) -> List[int]:
        """Sample random discrete actions for all houses."""
        return [self.sample_random_discrete_action() for _ in range(self.num_houses)]
    
    def describe_action(self, discrete_action: int, house_idx: int = 0) -> Dict[str, Any]:
        """
        Get human-readable description of a discrete action.
        
        Args:
            discrete_action: Discrete action index
            house_idx: House index for context
            
        Returns:
            Action description dictionary
        """
        description = self.action_discretizer.get_action_description(discrete_action)
        description['house_idx'] = house_idx
        return description
    
    def get_state_info(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Get information about current state.
        
        Args:
            state: Flattened state array
            
        Returns:
            State information dictionary
        """
        # Reshape to per-house format
        state_per_house = state.reshape(self.num_houses, self.state_dim_per_house)
        
        state_info = {
            'num_houses': self.num_houses,
            'state_dim_per_house': self.state_dim_per_house,
            'total_state_dim': len(state),
            'state_per_house': state_per_house.tolist(),
            'state_statistics': {
                'mean': np.mean(state),
                'std': np.std(state),
                'min': np.min(state),
                'max': np.max(state)
            }
        }
        
        return state_info
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        if hasattr(self.env, 'close'):
            self.env.close()


def create_dqn_environment(dynamic: bool = False, eval_mode: bool = False) -> DQNEnvironmentWrapper:
    """
    Factory function to create DQN environment wrapper.
    
    Args:
        dynamic: Whether to use dynamic data loading
        eval_mode: Whether to run in evaluation mode
        
    Returns:
        Initialized DQNEnvironmentWrapper
    """
    return DQNEnvironmentWrapper(dynamic=dynamic, eval_mode=eval_mode)