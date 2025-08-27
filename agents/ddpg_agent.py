import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any

from hyperparameters import Config
from models_code import Actor, Critic
from utilities import Normalizer, Transition, ReplayMemory, OUNoise


class SingleHouseDDPGAgent:
    """
    Single-house DDPG agent for independent learning.
    
    Each agent manages one house with its own actor/critic networks,
    experience replay, and exploration noise.
    """
    
    def __init__(
        self,
        house_id: int,
        state_dim: int,
        action_dim: int, 
        action_bounds: Dict[str, List[float]],
        config: Config
    ):
        """
        Initialize single-house DDPG agent.
        
        Args:
            house_id: ID of the house this agent controls
            state_dim: House-specific state dimension (17)
            action_dim: House-specific action dimension (3)
            action_bounds: Action bounds for this house
            config: Configuration object
        """
        self.house_id = house_id
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        
        # Load hyperparameters
        self._load_hyperparameters()
        
        # Initialize networks
        self._initialize_networks()
        
        # Initialize utilities  
        self._initialize_utilities()
    
    def _load_hyperparameters(self):
        """Load agent hyperparameters from config."""
        self.gamma = self.config.get('rl_agent', 'gamma')
        self.tau = self.config.get('rl_agent', 'tau')
        self.batch_size = self.config.get('rl_agent', 'batch_size')
        self.memory_size = int(self.config.get('rl_agent', 'memory_size'))
        self.device = torch.device(self.config.get('general', 'device'))
    
    def _initialize_networks(self):
        """Initialize actor and critic networks with targets."""
        # Actor networks
        self.actor = Actor(self.state_dim, self.action_dim, self.config).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.config).to(self.device)
        
        # Critic networks  
        self.critic = Critic(self.state_dim, self.action_dim, self.config).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim, self.config).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config.get('rl_agent', 'learning_rate_actor')
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.get('rl_agent', 'learning_rate_critic'),
            weight_decay=1e-2
        )
        
        # Initialize target networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
    
    def _initialize_utilities(self):
        """Initialize memory, normalizer, and noise."""
        self.memory = ReplayMemory(self.memory_size)
        self.normalizer = Normalizer(self.state_dim, self.device)
        # Noise for hvac_energy and battery_action only (first 2 actions)
        self.noise = OUNoise(2)
    
    def select_action(self, state: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Select action for this house.
        
        Args:
            state: House-specific state tensor [17 dims]
            add_noise: Whether to add exploration noise
            
        Returns:
            Action tensor for this house [3 dims]
        """
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu()
        self.actor.train()
        
        # Ensure correct shape [3]
        if action.dim() == 2:
            action = action.squeeze(0)
            
        if add_noise:
            action = self._add_exploration_noise(action)
            
        return action
    
    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        """Add exploration noise to first 2 actions (hvac, battery)."""
        noise = self.noise.sample()
        noise_tensor = torch.tensor(noise, dtype=torch.float32)
        
        # Add noise to hvac_energy and battery_action only
        action[:2] += noise_tensor
        
        # Clamp selling price to [0, 1]
        action[2] = torch.clamp(action[2], 0, 1)
        
        return action
    
    def optimize_model(self):
        """Optimize this agent's networks."""
        if len(self.memory) < self.batch_size:
            return
            
        # Sample transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create mask for non-final states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        
        # Stack batch data
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device) 
        reward_batch = torch.stack(batch.reward).to(self.device)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(self.device)
        
        # Update critic
        self._update_critic(state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states)
        
        # Update actor
        self._update_actor(state_batch)
        
        # Soft update targets
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
    
    def _update_critic(self, state_batch, action_batch, reward_batch, non_final_mask, non_final_next_states):
        """Update critic network."""
        with torch.no_grad():
            # Compute target Q-values
            next_actions = self.target_actor(non_final_next_states)
            next_q_values = self.target_critic(non_final_next_states, next_actions)
            
            # Compute Q-targets
            q_targets = torch.zeros((self.batch_size, 1), device=self.device)
            q_targets[non_final_mask] = reward_batch[non_final_mask] + self.gamma * next_q_values
            
        # Compute expected Q-values
        q_expected = self.critic(state_batch, action_batch)
        
        # Critic loss
        critic_loss = F.mse_loss(q_expected, q_targets)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def _update_actor(self, state_batch):
        """Update actor network."""
        predicted_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def soft_update(self, target_net, source_net):
        """Soft update target network."""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def hard_update(self, target_net, source_net):
        """Hard update target network."""
        target_net.load_state_dict(source_net.state_dict())


class DDPGAgent:
    """
    Multi-Agent DDPG for Smart Home Energy Management.
    
    Manages multiple independent DDPG agents, one per house.
    Each agent sees house-specific state and controls house-specific actions.
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
        Initialize Multi-Agent DDPG.
        
        Args:
            state_dim: Total state dimension (170 for 10 houses)
            action_dim: Total action dimension (30 for 10 houses)  
            action_bounds: Action bounds
            config: Configuration object
            ckpt: Optional checkpoint path
        """
        self.config = config
        self.num_houses = config.get('environment', 'num_houses')
        self.state_dim_per_house = config.get('environment', 'state_dim_per_house')  # 17
        self.action_dim_per_house = config.get('environment', 'action_dim_per_house')  # 3
        self.device = torch.device(config.get('general', 'device'))
        
        # Validate dimensions
        expected_total_state = self.num_houses * self.state_dim_per_house
        expected_total_action = self.num_houses * self.action_dim_per_house
        
        if state_dim != expected_total_state:
            print(f"Warning: Total state dim mismatch. Got {state_dim}, expected {expected_total_state}")
        if action_dim != expected_total_action:
            print(f"Warning: Total action dim mismatch. Got {action_dim}, expected {expected_total_action}")
        
        # Create independent agents for each house
        self.agents = []
        for house_id in range(self.num_houses):
            agent = SingleHouseDDPGAgent(
                house_id=house_id,
                state_dim=self.state_dim_per_house,
                action_dim=self.action_dim_per_house,
                action_bounds=action_bounds,
                config=config
            )
            self.agents.append(agent)
            
        print(f"Multi-Agent DDPG initialized with {self.num_houses} independent agents")
        print(f"Each agent: state_dim={self.state_dim_per_house}, action_dim={self.action_dim_per_house}")
        
        # Load checkpoint if provided
        if ckpt:
            self._load_checkpoint(ckpt)
    
    def select_action(self, global_state: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """
        Select actions for all houses using independent agents.
        
        Args:
            global_state: Global state tensor [170 dims]
            add_noise: Whether to add exploration noise
            
        Returns:
            Actions tensor [num_houses, 3] 
        """
        # Extract house-specific states
        house_states = self._extract_house_states(global_state)
        
        # Get actions from each agent
        house_actions = []
        for i, (agent, house_state) in enumerate(zip(self.agents, house_states)):
            action = agent.select_action(house_state, add_noise)
            house_actions.append(action)
        
        # Stack to create [num_houses, 3] tensor
        return torch.stack(house_actions)
    
    def _extract_house_states(self, global_state: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract house-specific states from global state.
        
        Args:
            global_state: Global state [170 dims]
            
        Returns:
            List of house-specific states, each [17 dims]
        """
        # Ensure global_state is 1D
        if global_state.dim() > 1:
            global_state = global_state.flatten()
            
        house_states = []
        for house_id in range(self.num_houses):
            start_idx = house_id * self.state_dim_per_house
            end_idx = start_idx + self.state_dim_per_house
            house_state = global_state[start_idx:end_idx]
            house_states.append(house_state)
            
        return house_states
    
    def optimize_model(self):
        """Optimize all agents."""
        for agent in self.agents:
            agent.optimize_model()
    
    def store_transition(self, global_state, actions, global_next_state, rewards):
        """
        Store transitions for all agents.
        
        Args:
            global_state: Global state [170]
            actions: Actions [num_houses, 3]
            global_next_state: Next global state [170] 
            rewards: Rewards for each house [num_houses]
        """
        # Extract house-specific states
        house_states = self._extract_house_states(global_state)
        house_next_states = self._extract_house_states(global_next_state)
        
        # Store transition for each agent
        for i, agent in enumerate(self.agents):
            house_reward = torch.tensor([rewards[i]], dtype=torch.float32, device=self.device)
            
            agent.memory.push(
                house_states[i],
                actions[i] if actions.dim() > 1 else actions,
                house_next_states[i],
                house_reward
            )
    
    @property
    def memory(self):
        """For compatibility - returns first agent's memory."""
        return self.agents[0].memory
    
    def _load_checkpoint(self, ckpt_path: str):
        """Load checkpoint for all agents."""
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            for i, agent in enumerate(self.agents):
                agent.actor.load_state_dict(checkpoint[f'agent_{i}_actor_state_dict'])
                agent.critic.load_state_dict(checkpoint[f'agent_{i}_critic_state_dict'])
                agent.hard_update(agent.target_actor, agent.actor)
                agent.hard_update(agent.target_critic, agent.critic)
            print(f"Successfully loaded multi-agent checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    def save_checkpoint(self, path: str, episode: int, episode_reward: float):
        """Save checkpoint for all agents."""
        checkpoint = {
            'episode': episode,
            'episode_reward': episode_reward,
            'num_houses': self.num_houses,
        }
        
        # Save each agent's state
        for i, agent in enumerate(self.agents):
            checkpoint[f'agent_{i}_actor_state_dict'] = agent.actor.state_dict()
            checkpoint[f'agent_{i}_critic_state_dict'] = agent.critic.state_dict()
            checkpoint[f'agent_{i}_actor_optimizer_state_dict'] = agent.actor_optimizer.state_dict()
            checkpoint[f'agent_{i}_critic_optimizer_state_dict'] = agent.critic_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def get_parameters(self) -> List[np.ndarray]:
        """Get parameters from all agents (for compatibility)."""
        all_params = []
        for agent in self.agents:
            agent_params = [param.cpu().data.numpy() for param in agent.actor.parameters()]
            agent_params += [param.cpu().data.numpy() for param in agent.critic.parameters()]
            all_params.extend(agent_params)
        return all_params
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Set parameters for all agents (for compatibility)."""
        # This is complex for multi-agent - simplified implementation
        print("Warning: set_parameters not fully implemented for multi-agent DDPG")
        pass