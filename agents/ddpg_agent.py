import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from hyperparameters import Config
from models_code import Actor, Critic
from utilities import Normalizer, Transition, ReplayMemory, OUNoise


class DDPGAgent:
    """
    DDPG Agent for continuous control.
    """

    def __init__(self, state_dim, action_dim, action_bounds, config, ckpt=None):
        self.config = config
        self.gamma = config.get('rl_agent', 'gamma')
        self.tau = config.get('rl_agent', 'tau')
        self.batch_size = config.get('rl_agent', 'batch_size')
        self.memory_size = int(config.get('rl_agent', 'memory_size'))
        self.device = torch.device(config.get('general', 'device'))

        # Store dimensions from environment
        self.state_dim = state_dim
        self.action_dim = action_dim  # Now includes selling price action
        self.action_bounds = action_bounds  # Now includes selling price bounds

        # Initialize networks with updated dimensions
        self.actor = Actor(self.state_dim, self.action_dim, config).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, config).to(self.device)
        self.critic = Critic(self.state_dim, self.action_dim, config).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim, config).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=config.get('rl_agent', 'learning_rate_actor')
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=config.get('rl_agent', 'learning_rate_critic'),
            weight_decay=1e-2
        )

        # Initialize memory and utilities
        self.memory = ReplayMemory(self.memory_size)
        self.normalizer = Normalizer(self.state_dim, self.device)
        
        # Initialize noise for action exploration (only for e_t and a_batt)
        # Since we have 3 actions per house now, but only want noise on first 2
        num_houses = self.action_dim // 3
        self.noise = OUNoise(num_houses * 2)  # Only for e_t and a_batt actions

        # Initialize target networks
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

        if ckpt:
            checkpoint = torch.load(ckpt)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])


    def select_action(self, state: torch.Tensor, add_noise: bool = True):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu()  # Shape: [batch_size, n_actions]
        self.actor.train()
        
        # Ensure action has correct shape [num_houses, 3] (e_t, a_batt, selling_price)
        num_houses = self.action_dim // 3  # Since each house now has 3 actions
        if action.dim() == 1:  # If action is [action_dim]
            action = action.view(num_houses, 3)
        elif action.dim() == 2 and action.shape[0] == 1:  # If action is [1, action_dim]
            action = action.view(num_houses, 3)
            
        if add_noise:
            # Generate noise only for e_t and a_batt actions
            noise = self.noise.sample()
            
            # Reshape noise to match the first two actions of each house
            noise_reshaped = torch.tensor(noise, dtype=torch.float32).view(num_houses, 2)
            
            # Add noise only to e_t and a_batt
            action[:, :2] += noise_reshaped
            
            # Ensure selling price stays within bounds [0, 1]
            action[:, 2].clamp_(0, 1)
            
        return action


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)  # Shape: [batch_size, state_dim]
        action_batch = torch.stack(batch.action).to(self.device)  # Shape: [batch_size, n_actions]
        action_batch = action_batch.view(self.batch_size, -1)  # Flatten to [batch_size, n_actions]
        
        # Corrected reward_batch creation
        reward_batch = torch.stack(batch.reward).to(self.device)  # Shape: [batch_size, 1]

        # Critic update
        with torch.no_grad():
            # Compute next actions and Q-values
            next_actions = self.target_actor(non_final_next_states)
            next_actions = next_actions.view(next_actions.size(0), -1)  # Flatten actions
            next_q_values = self.target_critic(non_final_next_states, next_actions)
            # Prepare q_targets tensor
            q_targets = torch.zeros((self.batch_size, 1), device=self.device)
            q_targets[non_final_mask] = reward_batch[non_final_mask] + self.gamma * next_q_values

        # Compute expected Q-values from current policy
        q_expected = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        predicted_actions = self.actor(state_batch)
        predicted_actions = predicted_actions.view(predicted_actions.size(0), -1)  # Flatten actions
        actor_loss = -self.critic(state_batch, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)


    def soft_update(self, target_net: torch.nn.Module, source_net: torch.nn.Module):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self, target_net: torch.nn.Module, source_net: torch.nn.Module):
        target_net.load_state_dict(source_net.state_dict())

    def get_parameters(self):
        # Return a list of parameters (as NumPy arrays)
        actor_params = [param.cpu().data.numpy() for param in self.actor.parameters()]
        critic_params = [param.cpu().data.numpy() for param in self.critic.parameters()]
        return actor_params + critic_params

    def set_parameters(self, parameters):
        # Set the model parameters from a list of NumPy arrays
        n_actor_params = len(list(self.actor.parameters()))
        actor_params = parameters[:n_actor_params]
        critic_params = parameters[n_actor_params:]

        for param, new_param in zip(self.actor.parameters(), actor_params):
            param.data = torch.tensor(new_param, dtype=param.data.dtype).to(self.device)
        for param, new_param in zip(self.critic.parameters(), critic_params):
            param.data = torch.tensor(new_param, dtype=param.data.dtype).to(self.device)