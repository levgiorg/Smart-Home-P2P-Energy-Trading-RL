import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, List, Dict, Optional, Union, Any
from collections import deque

from hyperparameters import Config
from models_code.dqn_networks import DQNNetwork, DoubleDQNNetwork, PrioritizedReplayBuffer
from utilities.action_discretizer import ActionDiscretizer, create_discretizer_from_config


class DQNAgent:
    """
    Deep Q-Network Agent for Smart Home Energy Management.
    
    Implements DQN with discrete action space for energy management,
    featuring epsilon-greedy exploration, experience replay, and
    target network updates for stable learning in large action spaces.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_bounds: Dict[str, List[float]],
        config: Config,
        ckpt: Optional[str] = None
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of state space
            action_bounds: Dictionary of action bounds for discretization
            config: Configuration object with hyperparameters
            ckpt: Optional checkpoint path for loading pretrained models
        """
        self.config = config
        self.device = torch.device(config.get('general', 'device'))
        
        # Initialize action discretizer
        self.action_discretizer = ActionDiscretizer(action_bounds)
        self.num_actions = self.action_discretizer.total_actions
        
        print(f"DQN Agent initialized with {self.num_actions} discrete actions")
        print(f"Action space structure: {self.action_discretizer.get_action_space_info()}")
        
        # Network architecture parameters
        self.fc1_dims = config.get('dqn_agent', 'fc1_dims')
        self.fc2_dims = config.get('dqn_agent', 'fc2_dims')
        self.fc3_dims = config.get('dqn_agent', 'fc3_dims')
        
        # Initialize networks
        self.use_double_dqn = config.get('dqn_agent', 'use_double_dqn')
        self.use_dueling = config.get('dqn_agent', 'use_dueling')
        
        if self.use_double_dqn:
            self.q_network = DoubleDQNNetwork(
                state_dim, self.num_actions,
                self.fc1_dims, self.fc2_dims, self.fc3_dims,
                device=self.device
            )
            self.target_network = DoubleDQNNetwork(
                state_dim, self.num_actions,
                self.fc1_dims, self.fc2_dims, self.fc3_dims,
                device=self.device
            )
        else:
            self.q_network = DQNNetwork(
                state_dim, self.num_actions,
                self.fc1_dims, self.fc2_dims, self.fc3_dims,
                dueling=self.use_dueling, device=self.device
            )
            self.target_network = DQNNetwork(
                state_dim, self.num_actions,
                self.fc1_dims, self.fc2_dims, self.fc3_dims,
                dueling=self.use_dueling, device=self.device
            )
        
        # Copy parameters to target network
        self.update_target_network(tau=1.0)
        
        # Optimizer
        self.lr = config.get('dqn_agent', 'learning_rate')
        self.optimizer = optim.Adam(
            self.q_network.parameters() if not self.use_double_dqn 
            else list(self.q_network.q_network_1.parameters()) + list(self.q_network.q_network_2.parameters()),
            lr=self.lr
        )
        
        # Training parameters
        self.gamma = config.get('rl_agent', 'gamma')
        self.batch_size = config.get('rl_agent', 'batch_size')
        self.tau = config.get('rl_agent', 'tau')
        self.target_update_freq = config.get('dqn_agent', 'target_update_freq')
        
        # Exploration parameters
        self.epsilon_start = config.get('dqn_agent', 'epsilon_start')
        self.epsilon_end = config.get('dqn_agent', 'epsilon_end')
        self.epsilon_decay = config.get('dqn_agent', 'epsilon_decay')
        self.epsilon = self.epsilon_start
        
        # Experience replay
        self.memory_size = config.get('rl_agent', 'memory_size')
        self.use_prioritized_replay = config.get('dqn_agent', 'use_prioritized_replay')
        
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                self.memory_size, 
                alpha=config.get('dqn_agent', 'priority_alpha'),
                beta=config.get('dqn_agent', 'priority_beta'),
                device=self.device
            )
        else:
            self.memory = deque(maxlen=self.memory_size)
        
        # Training tracking
        self.learn_step_counter = 0
        self.losses = []
        self.q_values = []
        
        # Load checkpoint if provided
        if ckpt:
            self.load_checkpoint(ckpt)
    
    def select_action(self, state: Union[np.ndarray, torch.Tensor], evaluation: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            evaluation: Whether in evaluation mode (no exploration)
            
        Returns:
            Discrete action index
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        # Epsilon-greedy action selection
        if not evaluation and random.random() < self.epsilon:
            # Random action
            action = self.action_discretizer.sample_random_action()
        else:
            # Greedy action
            with torch.no_grad():
                if self.use_double_dqn:
                    q_values = self.q_network.forward(state.unsqueeze(0), network='main')
                else:
                    q_values = self.q_network.forward(state.unsqueeze(0))
                
                action = torch.argmax(q_values, dim=1).item()
                
                # Store Q-values for analysis
                if len(self.q_values) < 1000:  # Limit storage
                    self.q_values.append(q_values.max().item())
        
        return action
    
    def get_continuous_action(self, state: Union[np.ndarray, torch.Tensor], evaluation: bool = False) -> List[float]:
        """
        Get continuous action for environment interaction.
        
        Args:
            state: Current state observation
            evaluation: Whether in evaluation mode
            
        Returns:
            Continuous action values [hvac_energy, battery_action, selling_price]
        """
        discrete_action = self.select_action(state, evaluation)
        return self.action_discretizer.discrete_to_continuous(discrete_action)
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        td_error: Optional[float] = None
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Discrete action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            td_error: TD error for prioritized replay (optional)
        """
        if self.use_prioritized_replay:
            self.memory.add(state, action, reward, next_state, done, td_error)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def learn(self) -> Optional[float]:
        """
        Perform one learning step using experience replay.
        
        Returns:
            Average loss for the batch, or None if not enough samples
        """
        if self.use_prioritized_replay:
            if self.memory.size < self.batch_size:
                return None
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
        else:
            if len(self.memory) < self.batch_size:
                return None
            
            # Sample random batch
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors - FIXED: Optimize tensor creation for performance
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            dones = torch.tensor(np.array(dones), dtype=torch.bool, device=self.device)
            weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
        
        # Current Q-values
        if self.use_double_dqn:
            current_q_values = self.q_network.forward(states, network='main').gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            current_q_values = self.q_network.get_action_values(states, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN target computation
                next_q_main = self.q_network.forward(next_states, network='main')
                next_actions = torch.argmax(next_q_main, dim=1)
                next_q_target = self.target_network.forward(next_states, network='main')
                next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN target computation
                next_q_values = self.target_network.forward(next_states).max(1)[0]
            
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Compute loss - FIXED: Use consistent Huber loss for both prioritized and standard replay
        td_errors = current_q_values - target_q_values
        if self.use_prioritized_replay:
            # Weighted Huber loss for prioritized replay
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
            # Update priorities
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        else:
            # Huber loss for stability
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters() if not self.use_double_dqn
            else list(self.q_network.q_network_1.parameters()) + list(self.q_network.q_network_2.parameters()),
            max_norm=10.0
        )
        
        self.optimizer.step()
        
        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Store loss for analysis
        loss_val = loss.item()
        self.losses.append(loss_val)
        
        return loss_val
    
    def update_target_network(self, tau: Optional[float] = None) -> None:
        """
        Update target network parameters using soft or hard update.
        
        Args:
            tau: Soft update parameter (if None, uses self.tau)
        """
        if tau is None:
            tau = self.tau
        
        if self.use_double_dqn:
            # Update both networks in Double DQN
            for target_param, main_param in zip(
                self.target_network.q_network_1.parameters(),
                self.q_network.q_network_1.parameters()
            ):
                target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
            
            for target_param, main_param in zip(
                self.target_network.q_network_2.parameters(),
                self.q_network.q_network_2.parameters()
            ):
                target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
        else:
            # Standard DQN update
            for target_param, main_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else None,
            'losses': self.losses,
            'q_values': self.q_values
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.learn_step_counter = checkpoint.get('learn_step_counter', 0)
        self.losses = checkpoint.get('losses', [])
        self.q_values = checkpoint.get('q_values', [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'epsilon': self.epsilon,
            'learn_steps': self.learn_step_counter,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'avg_q_value': np.mean(self.q_values[-100:]) if self.q_values else 0.0,
            'memory_size': self.memory.size if self.use_prioritized_replay else len(self.memory),
            'num_discrete_actions': self.num_actions
        }
    
    def analyze_action_distribution(self, states: np.ndarray, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze the distribution of actions selected by the policy.
        
        Args:
            states: Batch of states to analyze
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with action distribution statistics
        """
        action_counts = np.zeros(self.num_actions)
        continuous_actions = []
        
        # Sample actions for given states
        num_states = min(len(states), num_samples)
        selected_states = states[:num_states]
        
        for state in selected_states:
            action = self.select_action(state, evaluation=True)  # No exploration
            action_counts[action] += 1
            continuous_actions.append(self.action_discretizer.discrete_to_continuous(action))
        
        continuous_actions = np.array(continuous_actions)
        
        return {
            'action_frequencies': action_counts / num_states,
            'most_frequent_actions': np.argsort(action_counts)[::-1][:10],  # Top 10
            'action_entropy': -np.sum((action_counts / num_states) * np.log(action_counts / num_states + 1e-8)),
            'continuous_action_stats': {
                'hvac_mean': np.mean(continuous_actions[:, 0]),
                'hvac_std': np.std(continuous_actions[:, 0]),
                'battery_mean': np.mean(continuous_actions[:, 1]),
                'battery_std': np.std(continuous_actions[:, 1]),
                'price_mean': np.mean(continuous_actions[:, 2]),
                'price_std': np.std(continuous_actions[:, 2])
            }
        }