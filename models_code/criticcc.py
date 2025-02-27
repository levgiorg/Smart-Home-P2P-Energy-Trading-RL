import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import Config


class Critic(nn.Module):
    """
    Critic network for DDPG agent with dynamic sizing based on number of houses.
    
    The network automatically adjusts its input and output dimensions based on:
    - Number of houses from config
    - Base state features per house (dynamic based on state components)
    - Actions per house (3: e_t, a_batt, selling_price)
    """
    def __init__(self, input_dims, n_actions, config: Config):
        super(Critic, self).__init__()
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
            print(f"Warning: Input dimensions mismatch in Critic. Got {input_dims}, expected {expected_input_dims}")
            print(f"Using dimension from config: {expected_input_dims}")
            input_dims = expected_input_dims
            
        if n_actions != expected_n_actions:
            print(f"Warning: Action dimensions mismatch in Critic. Got {n_actions}, expected {expected_n_actions}")
            print(f"Using dimension from config: {expected_n_actions}")
            n_actions = expected_n_actions
        
        self.input_dims = input_dims
        self.fc1_dims = config.get('critic', 'fc1_dims')
        self.fc2_dims = config.get('critic', 'fc2_dims')
        self.fc3_dims = config.get('critic', 'fc3_dims')
        self.n_actions = n_actions
        device = config.get('general', 'device') or 'cpu'
        self.device = torch.device(device)

        # State pathway
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        # Action pathway
        self.action_fc = nn.Linear(self.n_actions, self.fc3_dims)
        self.action_bn = nn.BatchNorm1d(self.fc3_dims)

        # Combined pathway
        self.fc3 = nn.Linear(self.fc2_dims + self.fc3_dims, self.fc3_dims)
        self.bn3 = nn.BatchNorm1d(self.fc3_dims)
        self.q = nn.Linear(self.fc3_dims, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.action_fc.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.q.weight)
        
        self.to(self.device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Ensure state has correct shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # State pathway
        state_value = F.relu(self.bn1(self.fc1(state)))
        state_value = F.relu(self.bn2(self.fc2(state_value)))

        # Action pathway
        action_value = F.relu(self.action_bn(self.action_fc(action)))

        # Combine state and action pathways
        state_action_value = torch.cat([state_value, action_value], dim=1)
        state_action_value = F.relu(self.bn3(self.fc3(state_action_value)))
        q_value = self.q(state_action_value)
        
        return q_value