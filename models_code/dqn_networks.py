import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete action space energy management.
    
    Implements a dueling DQN architecture that separates state value estimation
    from advantage estimation, providing better learning stability for large
    discrete action spaces (2,100 actions in this case).
    """
    
    def __init__(
        self, 
        state_dim: int, 
        num_actions: int,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        fc3_dims: int = 128,
        dueling: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize DQN Network.
        
        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions (2,100 for our case)
            fc1_dims: First hidden layer dimension
            fc2_dims: Second hidden layer dimension  
            fc3_dims: Third hidden layer dimension
            dueling: Whether to use dueling architecture
            device: Device to run network on
        """
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.dueling = dueling
        self.device = device
        
        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        
        if self.dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_fc = nn.Linear(fc2_dims, fc3_dims)
            self.advantage_fc = nn.Linear(fc2_dims, fc3_dims)
            
            self.value_head = nn.Linear(fc3_dims, 1)
            self.advantage_head = nn.Linear(fc3_dims, num_actions)
        else:
            # Standard DQN architecture
            self.fc3 = nn.Linear(fc2_dims, fc3_dims)
            self.q_head = nn.Linear(fc3_dims, num_actions)
        
        # Layer normalization for training stability
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        if not self.dueling:
            self.ln3 = nn.LayerNorm(fc3_dims)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        self.to(device)
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for all actions, shape (batch_size, num_actions)
        """
        # Ensure state is on correct device
        if state.device != self.device:
            state = state.to(self.device)
        
        # Shared feature extraction
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        if self.dueling:
            # Dueling architecture
            value_stream = F.relu(self.value_fc(x))
            advantage_stream = F.relu(self.advantage_fc(x))
            
            # Compute state value and action advantages
            state_value = self.value_head(value_stream)  # Shape: (batch_size, 1)
            advantages = self.advantage_head(advantage_stream)  # Shape: (batch_size, num_actions)
            
            # Combine value and advantages: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # This ensures identifiability of the value function
            q_values = state_value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            # Standard DQN
            x = F.relu(self.ln3(self.fc3(x)))
            q_values = self.q_head(x)
        
        return q_values
    
    def get_action_values(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for specific state-action pairs.
        
        Args:
            state: Input state tensor
            actions: Action indices tensor
            
        Returns:
            Q-values for the specified actions
        """
        q_values = self.forward(state)
        return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    def get_best_action(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Get best action and its Q-value for a single state.
        
        Args:
            state: Single state tensor
            
        Returns:
            Tuple of (best_action_index, max_q_value)
        """
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            max_q_value, best_action = torch.max(q_values, dim=1)
            
        return best_action.item(), max_q_value.item()
    
    def get_batch_best_actions(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get best actions for a batch of states.
        
        Args:
            states: Batch of states
            
        Returns:
            Tuple of (best_actions, max_q_values)
        """
        with torch.no_grad():
            q_values = self.forward(states)
            max_q_values, best_actions = torch.max(q_values, dim=1)
            
        return best_actions, max_q_values


class DoubleDQNNetwork(nn.Module):
    """
    Double DQN implementation to reduce overestimation bias.
    
    Uses two networks: one for action selection and another for value estimation
    during target computation. This helps address the overestimation problem
    common in large discrete action spaces.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        num_actions: int,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
        fc3_dims: int = 128,
        device: str = 'cpu'
    ):
        """Initialize Double DQN with two identical networks."""
        super(DoubleDQNNetwork, self).__init__()
        
        self.num_actions = num_actions
        self.device = device
        
        # Main Q-network for action selection
        self.q_network_1 = DQNNetwork(
            state_dim, num_actions, fc1_dims, fc2_dims, fc3_dims, 
            dueling=True, device=device
        )
        
        # Secondary Q-network for value estimation
        self.q_network_2 = DQNNetwork(
            state_dim, num_actions, fc1_dims, fc2_dims, fc3_dims, 
            dueling=True, device=device
        )
        
        self.to(device)
    
    def forward(self, state: torch.Tensor, network: str = 'main') -> torch.Tensor:
        """
        Forward pass through specified network.
        
        Args:
            state: Input state tensor
            network: Which network to use ('main' or 'secondary')
            
        Returns:
            Q-values from the specified network
        """
        if network == 'main':
            return self.q_network_1(state)
        else:
            return self.q_network_2(state)
    
    def double_q_target(
        self, 
        next_states: torch.Tensor, 
        rewards: torch.Tensor, 
        dones: torch.Tensor, 
        gamma: float = 0.99
    ) -> torch.Tensor:
        """
        Compute Double DQN targets.
        
        Args:
            next_states: Next state batch
            rewards: Reward batch
            dones: Done flags batch
            gamma: Discount factor
            
        Returns:
            Double DQN target values
        """
        with torch.no_grad():
            # Use main network for action selection
            next_q_values_main = self.q_network_1(next_states)
            next_actions = torch.argmax(next_q_values_main, dim=1)
            
            # Use secondary network for value estimation
            next_q_values_secondary = self.q_network_2(next_states)
            next_q_values = next_q_values_secondary.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute target: r + γ * Q_secondary(s', argmax_a Q_main(s', a))
            targets = rewards + (gamma * next_q_values * (~dones))
        
        return targets


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for DQN.
    
    Implements importance sampling to focus learning on transitions
    with high temporal difference error. This helps with sample efficiency
    in large action spaces.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, device: str = 'cpu'):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            device: Device for tensor operations
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        
        # Storage for experiences
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done, td_error: Optional[float] = None):
        """Add experience to buffer with priority."""
        # Use maximum priority for new experiences if td_error not provided
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        priority = max_priority if td_error is None else abs(td_error) + 1e-6
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with importance sampling weights."""
        if self.size < batch_size:
            raise ValueError(f"Buffer size {self.size} < batch size {batch_size}")
        
        # Sample according to priorities
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize for stability
        
        # Extract experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority