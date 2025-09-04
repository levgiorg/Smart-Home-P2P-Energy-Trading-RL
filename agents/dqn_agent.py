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


class SingleHouseDQNAgent:
    """
    Single House DQN Agent for independent energy management.
    Each house has its own DQN networks, replay buffer, and learning process.
    """
    
    def __init__(
        self, 
        house_id: int,
        state_dim: int, 
        action_bounds: Dict[str, List[float]], 
        config: Config,
        ckpt: Optional[str] = None
    ):
        """
        Initialize Single House DQN Agent.
        
        Args:
            house_id: Unique identifier for this house
            state_dim: Dimension of single house state space (17)
            action_bounds: Dictionary of action bounds for discretization
            config: Configuration object with hyperparameters
            ckpt: Optional checkpoint path for loading pretrained models
        """
        self.house_id = house_id
        self.config = config
        self.device = torch.device(config.get('general', 'device'))
        
        # Initialize action discretizer for single house
        self.action_discretizer = ActionDiscretizer(action_bounds)
        self.num_actions = self.action_discretizer.total_actions
        
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
        
        # Individual experience replay buffer
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
        """Select action using epsilon-greedy policy for single house."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        # Epsilon-greedy action selection
        if not evaluation and random.random() < self.epsilon:
            action = self.action_discretizer.sample_random_action()
        else:
            with torch.no_grad():
                if self.use_double_dqn:
                    q_values = self.q_network.forward(state.unsqueeze(0), network='main')
                else:
                    q_values = self.q_network.forward(state.unsqueeze(0))
                
                action = torch.argmax(q_values, dim=1).item()
                
                if len(self.q_values) < 1000:
                    self.q_values.append(q_values.max().item())
        
        return action
    
    def get_continuous_action(self, state: Union[np.ndarray, torch.Tensor], evaluation: bool = False) -> List[float]:
        """Get continuous action for single house."""
        discrete_action = self.select_action(state, evaluation)
        return self.action_discretizer.discrete_to_continuous(discrete_action)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, td_error: Optional[float] = None) -> None:
        """Store experience in individual replay buffer."""
        if self.use_prioritized_replay:
            self.memory.add(state, action, reward, next_state, done, td_error)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def learn(self) -> Optional[float]:
        """Perform learning step for individual house."""
        if self.use_prioritized_replay:
            if self.memory.size < self.batch_size:
                return None
            states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
        else:
            if len(self.memory) < self.batch_size:
                return None
            
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
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
        
        # Target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                next_q_main = self.q_network.forward(next_states, network='main')
                next_actions = torch.argmax(next_q_main, dim=1)
                next_q_target = self.target_network.forward(next_states, network='main')
                next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network.forward(next_states).max(1)[0]
            
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Compute loss
        td_errors = current_q_values - target_q_values
        if self.use_prioritized_replay:
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        else:
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
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
        
        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val
    
    def update_target_network(self, tau: Optional[float] = None) -> None:
        """Update target network parameters."""
        if tau is None:
            tau = self.tau
        
        if self.use_double_dqn:
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
            for target_param, main_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


class DQNAgent:
    """
    Multi-Agent DQN wrapper for Smart Home Energy Management.
    
    Manages multiple independent SingleHouseDQNAgent instances,
    one for each house, enabling independent decision-making
    and preventing identical pricing behavior.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_bounds: Dict[str, List[float]],
        config: Config,
        ckpt: Optional[str] = None
    ):
        """
        Initialize Multi-Agent DQN system.
        
        Args:
            state_dim: Total state dimension (170 for all houses)
            action_bounds: Dictionary of action bounds for discretization
            config: Configuration object with hyperparameters
            ckpt: Optional checkpoint path for loading pretrained models
        """
        self.config = config
        self.device = torch.device(config.get('general', 'device'))
        
        # Multi-agent setup
        self.num_houses = config.get('environment', 'num_houses')
        self.state_dim_per_house = config.get('environment', 'state_dim_per_house')
        self.action_dim_per_house = config.get('environment', 'action_dim_per_house')
        
        # Create independent agents for each house
        self.agents = []
        for house_id in range(self.num_houses):
            agent = SingleHouseDQNAgent(
                house_id=house_id,
                state_dim=self.state_dim_per_house,
                action_bounds=action_bounds,
                config=config,
                ckpt=ckpt
            )
            self.agents.append(agent)
        
        print(f"Multi-Agent DQN initialized with {self.num_houses} independent house agents")
        print(f"Each agent: {self.agents[0].num_actions} discrete actions per house")
        
        # Training tracking for all agents
        self.global_learn_steps = 0
    
    def select_action(self, state: Union[np.ndarray, torch.Tensor], evaluation: bool = False) -> List[int]:
        """
        Select actions for all houses using independent agents.
        
        Args:
            state: Full state observation (170-dim for all houses)
            evaluation: Whether in evaluation mode (no exploration)
            
        Returns:
            List of discrete action indices, one per house
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        # Split state into individual house states
        house_states = state.view(self.num_houses, self.state_dim_per_house)
        
        # Get action from each independent agent
        actions = []
        for i, agent in enumerate(self.agents):
            house_action = agent.select_action(house_states[i], evaluation)
            actions.append(house_action)
        
        return actions
    
    def get_continuous_action(self, state: Union[np.ndarray, torch.Tensor], evaluation: bool = False) -> np.ndarray:
        """
        Get continuous actions for all houses.
        
        Args:
            state: Full state observation (170-dim for all houses)
            evaluation: Whether in evaluation mode
            
        Returns:
            Array of continuous action values [num_houses, 3]
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        elif state.device != self.device:
            state = state.to(self.device)
        
        # Split state into individual house states
        house_states = state.view(self.num_houses, self.state_dim_per_house)
        
        # Get continuous action from each independent agent
        all_actions = []
        for i, agent in enumerate(self.agents):
            house_continuous_action = agent.get_continuous_action(house_states[i], evaluation)
            all_actions.append(house_continuous_action)
        
        return np.array(all_actions)  # Shape: [num_houses, 3]
    
    def store_transition(
        self, 
        state: np.ndarray, 
        actions: List[int], 
        next_state: np.ndarray, 
        rewards: List[float], 
        done: bool = False
    ) -> None:
        """
        Store experiences in individual agent replay buffers.
        
        Args:
            state: Full current state (170-dim)
            actions: List of discrete actions, one per house
            next_state: Full next state (170-dim)
            rewards: List of rewards, one per house  
            done: Episode termination flag
        """
        # Split states into individual house states
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        house_states = state_tensor.view(self.num_houses, self.state_dim_per_house).numpy()
        house_next_states = next_state_tensor.view(self.num_houses, self.state_dim_per_house).numpy()
        
        # Store transition for each house agent
        for i, agent in enumerate(self.agents):
            agent.store_transition(
                house_states[i], 
                actions[i], 
                rewards[i], 
                house_next_states[i], 
                done
            )
    
    def learn(self) -> Optional[float]:
        """
        Perform learning step for all house agents.
        
        Returns:
            Average loss across all agents, or None if not enough samples
        """
        total_loss = 0.0
        num_learning_agents = 0
        
        # Each agent learns independently
        for agent in self.agents:
            loss = agent.learn()
            if loss is not None:
                total_loss += loss
                num_learning_agents += 1
        
        self.global_learn_steps += 1
        
        if num_learning_agents > 0:
            return total_loss / num_learning_agents
        return None
    
    def update_target_network(self, tau: Optional[float] = None) -> None:
        """
        Update target networks for all house agents.
        
        Args:
            tau: Soft update parameter (if None, uses agent defaults)
        """
        for agent in self.agents:
            agent.update_target_network(tau)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save checkpoint for all house agents."""
        checkpoint = {
            'num_houses': self.num_houses,
            'global_learn_steps': self.global_learn_steps,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else None,
            'agents': []
        }
        
        # Save each agent's state
        for i, agent in enumerate(self.agents):
            agent_checkpoint = {
                'house_id': agent.house_id,
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'learn_step_counter': agent.learn_step_counter,
                'losses': agent.losses,
                'q_values': agent.q_values
            }
            checkpoint['agents'].append(agent_checkpoint)
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load checkpoint for all house agents."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_learn_steps = checkpoint.get('global_learn_steps', 0)
        
        # Load each agent's state
        if 'agents' in checkpoint:
            for i, agent_checkpoint in enumerate(checkpoint['agents']):
                if i < len(self.agents):
                    agent = self.agents[i]
                    agent.q_network.load_state_dict(agent_checkpoint['q_network_state_dict'])
                    agent.target_network.load_state_dict(agent_checkpoint['target_network_state_dict'])
                    agent.optimizer.load_state_dict(agent_checkpoint['optimizer_state_dict'])
                    agent.epsilon = agent_checkpoint.get('epsilon', agent.epsilon_end)
                    agent.learn_step_counter = agent_checkpoint.get('learn_step_counter', 0)
                    agent.losses = agent_checkpoint.get('losses', [])
                    agent.q_values = agent_checkpoint.get('q_values', [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics for all house agents."""
        # Collect statistics from all agents
        total_memory = 0
        all_losses = []
        all_q_values = []
        avg_epsilon = 0.0
        total_learn_steps = 0
        
        for agent in self.agents:
            total_memory += agent.memory.size if agent.use_prioritized_replay else len(agent.memory)
            all_losses.extend(agent.losses[-100:])  # Last 100 losses per agent
            all_q_values.extend(agent.q_values[-100:])  # Last 100 Q-values per agent
            avg_epsilon += agent.epsilon
            total_learn_steps += agent.learn_step_counter
        
        avg_epsilon /= self.num_houses
        
        return {
            'avg_epsilon': avg_epsilon,
            'total_learn_steps': total_learn_steps,
            'global_learn_steps': self.global_learn_steps,
            'avg_loss': np.mean(all_losses) if all_losses else 0.0,
            'avg_q_value': np.mean(all_q_values) if all_q_values else 0.0,
            'total_memory_size': total_memory,
            'num_houses': self.num_houses,
            'num_discrete_actions_per_house': self.agents[0].num_actions
        }
    
    def analyze_action_distribution(self, states: np.ndarray, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analyze action distribution for all house agents.
        
        Args:
            states: Batch of full states to analyze (170-dim each)
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with action distribution statistics per house
        """
        num_states = min(len(states), num_samples)
        selected_states = states[:num_states]
        
        # Initialize statistics for each house
        house_stats = []
        overall_continuous_actions = []
        
        for house_id, agent in enumerate(self.agents):
            action_counts = np.zeros(agent.num_actions)
            house_continuous_actions = []
            
            # Extract house-specific states
            for state in selected_states:
                state_tensor = torch.FloatTensor(state)
                house_states = state_tensor.view(self.num_houses, self.state_dim_per_house)
                house_state = house_states[house_id]
                
                # Get action for this house
                action = agent.select_action(house_state, evaluation=True)
                action_counts[action] += 1
                continuous_action = agent.get_continuous_action(house_state, evaluation=True)
                house_continuous_actions.append(continuous_action)
                
                if house_id == 0:  # Store overall continuous actions from first house perspective
                    overall_continuous_actions.append(continuous_action)
            
            house_continuous_actions = np.array(house_continuous_actions)
            
            house_stat = {
                'action_frequencies': action_counts / num_states,
                'most_frequent_actions': np.argsort(action_counts)[::-1][:10],
                'action_entropy': -np.sum((action_counts / num_states) * np.log(action_counts / num_states + 1e-8)),
                'continuous_action_stats': {
                    'hvac_mean': np.mean(house_continuous_actions[:, 0]),
                    'hvac_std': np.std(house_continuous_actions[:, 0]),
                    'battery_mean': np.mean(house_continuous_actions[:, 1]),
                    'battery_std': np.std(house_continuous_actions[:, 1]),
                    'price_mean': np.mean(house_continuous_actions[:, 2]),
                    'price_std': np.std(house_continuous_actions[:, 2])
                }
            }
            house_stats.append(house_stat)
        
        overall_continuous_actions = np.array(overall_continuous_actions)
        
        return {
            'num_houses': self.num_houses,
            'house_statistics': house_stats,
            'overall_stats': {
                'hvac_mean': np.mean(overall_continuous_actions[:, 0]),
                'hvac_std': np.std(overall_continuous_actions[:, 0]),
                'battery_mean': np.mean(overall_continuous_actions[:, 1]),
                'battery_std': np.std(overall_continuous_actions[:, 1]),
                'price_mean': np.mean(overall_continuous_actions[:, 2]),
                'price_std': np.std(overall_continuous_actions[:, 2])
            }
        }