import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any

from hyperparameters import Config
from models_code import Actor, Critic
from utilities import Normalizer, Transition, ReplayMemory, OUNoise


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent for Smart Home Energy Management.
    
    Implements a DDPG reinforcement learning agent that can handle continuous
    action spaces for controlling HVAC systems, battery charging/discharging,
    and energy selling price setting in a multi-house environment.
    
    Features:
    - Separate actor and critic networks with target networks
    - Experience replay for sample efficiency
    - Ornstein-Uhlenbeck noise process for exploration
    - Support for multi-house environments
    - Advanced dimension handling with validation
    """

    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        action_bounds: Dict[str, List[float]], 
        config: Config, 
        ckpt: Optional[str] = None
    ):
        """
        Initialize the DDPG Agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            action_bounds: Dictionary of action bounds for each action type
            config: Configuration object with hyperparameters
            ckpt: Optional path to checkpoint file for loading pretrained models
        """
        self.config = config
        
        # Get number of houses from config
        self.num_houses = config.get('environment', 'num_houses')
        
        # Get dimensions from environment or config
        self._initialize_dimensions(state_dim, action_dim)
        
        # Load hyperparameters
        self._load_hyperparameters()

        # Store dimensions and bounds
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds

        # Initialize networks
        self._initialize_networks()

        # Initialize memory and utilities
        self._initialize_utilities()

        # Load checkpoint if provided
        if ckpt:
            self._load_checkpoint(ckpt)
    
    def _initialize_dimensions(self, state_dim: int, action_dim: int) -> None:
        """
        Initialize and validate state and action dimensions.
        
        Args:
            state_dim: Provided state dimension
            action_dim: Provided action dimension
        """
        # Get dimensions from config - FIXED: using self.config instead of config
        self.base_features_per_house = self.config.get('environment', 'state_dim_per_house') - self.num_houses
        self.features_per_house = self.config.get('environment', 'state_dim_per_house')
        self.actions_per_house = self.config.get('environment', 'action_dim_per_house')
        
        # Validate dimensions
        expected_state_dim = self.num_houses * self.features_per_house
        expected_action_dim = self.num_houses * self.actions_per_house
        
        if state_dim != expected_state_dim:
            print(f"Warning: State dimension mismatch. Got {state_dim}, expected {expected_state_dim}")
            print(f"Using dimension from config: {expected_state_dim}")
            self.state_dim = expected_state_dim
        else:
            self.state_dim = state_dim
            
        if action_dim != expected_action_dim:
            print(f"Warning: Action dimension mismatch. Got {action_dim}, expected {expected_action_dim}")
            print(f"Using dimension from config: {expected_action_dim}")
            self.action_dim = expected_action_dim
        else:
            self.action_dim = action_dim
    
    def _load_hyperparameters(self) -> None:
        """Load agent hyperparameters from config."""
        self.gamma = self.config.get('rl_agent', 'gamma')  # Discount factor
        self.tau = self.config.get('rl_agent', 'tau')      # Soft update parameter
        self.batch_size = self.config.get('rl_agent', 'batch_size')  # Batch size for training
        self.memory_size = int(self.config.get('rl_agent', 'memory_size'))  # Replay buffer size
        self.device = torch.device(self.config.get('general', 'device'))  # Device for computation
    
    def _initialize_networks(self) -> None:
        """Initialize actor and critic networks with their target networks."""
        # Initialize actor network and target
        self.actor = Actor(self.state_dim, self.action_dim, self.config).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.config).to(self.device)
        
        # Initialize critic network and target
        self.critic = Critic(self.state_dim, self.action_dim, self.config).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim, self.config).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.get('rl_agent', 'learning_rate_actor')
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.config.get('rl_agent', 'learning_rate_critic'),
            weight_decay=1e-2  # L2 regularization for critic
        )

        # Initialize target networks with same weights as main networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
    
    def _initialize_utilities(self) -> None:
        """Initialize memory buffer, normalizer, and noise process."""
        # Experience replay buffer
        self.memory = ReplayMemory(self.memory_size)
        
        # State normalizer
        self.normalizer = Normalizer(self.state_dim, self.device)
        
        # Exploration noise (only for hvac_energy and battery_action actions)
        num_houses = self.action_dim // self.actions_per_house
        self.noise = OUNoise(num_houses * 2)  # Only for hvac_energy and battery_action actions
    
    def _load_checkpoint(self, ckpt_path: str) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            ckpt_path: Path to the checkpoint file
        """
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.hard_update(self.target_actor, self.actor)
            self.hard_update(self.target_critic, self.critic)
            print(f"Successfully loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    def select_action(self, state: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Select actions for all houses based on current state.
        
        Args:
            state: Current state tensor
            add_noise: Whether to add exploration noise to actions
            
        Returns:
            Tensor of actions for all houses
        """
        self.actor.eval()  # Set actor to evaluation mode
        with torch.no_grad():
            action = self.actor(state).cpu()  # Forward pass through actor network
        self.actor.train()  # Set actor back to training mode
        
        # Ensure action has correct shape [num_houses, actions_per_house]
        num_houses = self.action_dim // self.actions_per_house
        action = self._reshape_action(action, num_houses)
            
        if add_noise:
            # Add exploration noise to actions
            action = self._add_exploration_noise(action, num_houses)
            
        return action
    
    def _reshape_action(self, action: torch.Tensor, num_houses: int) -> torch.Tensor:
        """
        Reshape action tensor to correct dimensions.
        
        Args:
            action: Action tensor from actor network
            num_houses: Number of houses
            
        Returns:
            Reshaped action tensor [num_houses, actions_per_house]
        """
        if action.dim() == 1:  # If action is [action_dim]
            return action.view(num_houses, self.actions_per_house)
        elif action.dim() == 2 and action.shape[0] == 1:  # If action is [1, action_dim]
            return action.view(num_houses, self.actions_per_house)
        return action  # Already in correct shape
    
    def _add_exploration_noise(self, action: torch.Tensor, num_houses: int) -> torch.Tensor:
        """
        Add exploration noise to actions.
        
        Args:
            action: Action tensor
            num_houses: Number of houses
            
        Returns:
            Action tensor with added noise
        """
        # Generate noise only for hvac_energy and battery_action actions
        noise = self.noise.sample()
        
        # Reshape noise to match the first two actions of each house
        noise_reshaped = torch.tensor(noise, dtype=torch.float32).view(num_houses, 2)
        
        # Add noise only to hvac_energy and battery_action
        action[:, :2] += noise_reshaped
        
        # Ensure selling price stays within bounds [0, 1]
        action[:, 2].clamp_(0, 1)
        
        return action

    def optimize_model(self) -> None:
        """
        Perform one step of optimization on both actor and critic networks.
        Uses experience sampled from the replay buffer.
        """
        # Check if we have enough transitions to sample
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Create mask for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        
        # Stack tensors for batch processing
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)  # Shape: [batch_size, state_dim]
        action_batch = torch.stack(batch.action).to(self.device)  # Shape: [batch_size, n_actions]
        action_batch = action_batch.view(self.batch_size, -1)  # Flatten to [batch_size, n_actions]
        reward_batch = torch.stack(batch.reward).to(self.device)  # Shape: [batch_size, 1]

        # Update critic network
        self._update_critic(state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states)
        
        # Update actor network
        self._update_actor(state_batch)
        
        # Soft update target networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
    
    def _update_critic(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        non_final_mask: torch.Tensor,
        non_final_next_states: torch.Tensor
    ) -> None:
        """
        Update critic network using TD learning.
        
        Args:
            state_batch: Batch of current states
            action_batch: Batch of actions taken
            reward_batch: Batch of rewards received
            non_final_mask: Mask for non-terminal states
            non_final_next_states: Batch of next states (excluding terminal states)
        """
        with torch.no_grad():
            # Compute next actions and Q-values for next states
            next_actions = self.target_actor(non_final_next_states)
            next_actions = next_actions.view(next_actions.size(0), -1)  # Flatten actions
            next_q_values = self.target_critic(non_final_next_states, next_actions)
            
            # Prepare Q-targets tensor
            q_targets = torch.zeros((self.batch_size, 1), device=self.device)
            
            # Set target values for non-terminal states
            q_targets[non_final_mask] = reward_batch[non_final_mask] + self.gamma * next_q_values

        # Compute expected Q-values from current policy
        q_expected = self.critic(state_batch, action_batch)

        # Compute critic loss (MSE)
        critic_loss = F.mse_loss(q_expected, q_targets)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def _update_actor(self, state_batch: torch.Tensor) -> None:
        """
        Update actor network using policy gradient.
        
        Args:
            state_batch: Batch of current states
        """
        # Get actions from current policy
        predicted_actions = self.actor(state_batch)
        predicted_actions = predicted_actions.view(predicted_actions.size(0), -1)  # Flatten actions
        
        # Compute actor loss as negative of expected Q-value
        actor_loss = -self.critic(state_batch, predicted_actions).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def soft_update(self, target_net: torch.nn.Module, source_net: torch.nn.Module) -> None:
        """
        Soft update target network parameters using polyak averaging.
        
        θ_target = τ*θ_source + (1 - τ)*θ_target
        
        Args:
            target_net: Target network to update
            source_net: Source network
        """
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self, target_net: torch.nn.Module, source_net: torch.nn.Module) -> None:
        """
        Hard update target network parameters (copy parameters).
        
        θ_target = θ_source
        
        Args:
            target_net: Target network to update
            source_net: Source network
        """
        target_net.load_state_dict(source_net.state_dict())

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get all model parameters as NumPy arrays.
        
        Returns:
            List of parameters from actor and critic networks
        """
        # Return a list of parameters (as NumPy arrays)
        actor_params = [param.cpu().data.numpy() for param in self.actor.parameters()]
        critic_params = [param.cpu().data.numpy() for param in self.critic.parameters()]
        return actor_params + critic_params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of parameters to set in the networks
        """
        # Determine parameter split between actor and critic
        n_actor_params = len(list(self.actor.parameters()))
        actor_params = parameters[:n_actor_params]
        critic_params = parameters[n_actor_params:]

        # Set actor parameters
        for param, new_param in zip(self.actor.parameters(), actor_params):
            param.data = torch.tensor(new_param, dtype=param.data.dtype).to(self.device)
            
        # Set critic parameters
        for param, new_param in zip(self.critic.parameters(), critic_params):
            param.data = torch.tensor(new_param, dtype=param.data.dtype).to(self.device)
            
        # Update target networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)