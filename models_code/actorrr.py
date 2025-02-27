import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import Config

class Actor(nn.Module):
    """
    Actor network for DDPG agent with dynamic sizing based on number of houses.
    
    The network automatically adjusts its input and output dimensions based on:
    - Number of houses from config
    - Base state features per house (dynamic based on state components)
    - Actions per house (3: hvac_energy, battery_action, selling_price)
    """
    def __init__(self, input_dims, n_actions, config: Config):
        super(Actor, self).__init__()
        # Get number of houses from config
        self.num_houses = config.get('environment', 'num_houses')
        
        # Get dimensions dynamically from config, which are calculated in Environment
        self.base_features_per_house = config.get('environment', 'state_dim_per_house') - self.num_houses
        self.features_per_house = config.get('environment', 'state_dim_per_house')
        self.actions_per_house = config.get('environment', 'action_dim_per_house')
        
        # Verify input dimensions match expected
        expected_input_dims = self.features_per_house * self.num_houses
        expected_n_actions = self.actions_per_house * self.num_houses
        
        if input_dims != expected_input_dims:
            print(f"Warning: Input dimensions mismatch in Actor. Got {input_dims}, expected {expected_input_dims}")
            print(f"Using dimension from config: {expected_input_dims}")
            input_dims = expected_input_dims
            
        if n_actions != expected_n_actions:
            print(f"Warning: Action dimensions mismatch in Actor. Got {n_actions}, expected {expected_n_actions}")
            print(f"Using dimension from config: {expected_n_actions}")
            n_actions = expected_n_actions
        
        self.input_dims = input_dims
        self.fc1_dims = config.get('actor', 'fc1_dims')
        self.fc2_dims = config.get('actor', 'fc2_dims')
        self.n_actions = n_actions
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