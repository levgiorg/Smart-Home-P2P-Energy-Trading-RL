import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import Config

class Actor(nn.Module):
    """
    Actor network for single-house DDPG agent.
    
    Each actor network handles one house with:
    - Input: House-specific state (own features + all houses' selling prices) [17 dims]
    - Output: House-specific actions (hvac_energy, battery_action, selling_price) [3 dims]
    """
    def __init__(self, input_dims: int, n_actions: int, config: Config) -> None:
        super(Actor, self).__init__()
        
        # For multi-agent DDPG, each actor handles one house
        self.state_dim_per_house = config.get('environment', 'state_dim_per_house')  # 17
        self.actions_per_house = config.get('environment', 'action_dim_per_house')   # 3
        
        # Verify dimensions for single-house agent
        if input_dims != self.state_dim_per_house:
            logging.warning(f"Input dimensions mismatch in single-house Actor. Got {input_dims}, expected {self.state_dim_per_house}")
            input_dims = self.state_dim_per_house
            
        if n_actions != self.actions_per_house:
            logging.warning(f"Action dimensions mismatch in single-house Actor. Got {n_actions}, expected {self.actions_per_house}")
            n_actions = self.actions_per_house
        
        self.input_dims = input_dims  # 17 for single house
        self.fc1_dims = config.get('actor', 'fc1_dims')
        self.fc2_dims = config.get('actor', 'fc2_dims')
        self.n_actions = n_actions    # 3 for single house
        device = config.get('general', 'device')
        self.device = torch.device(device)

        # Network layers
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Ensure state has correct shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        mu = torch.tanh(self.mu(x))
        return mu