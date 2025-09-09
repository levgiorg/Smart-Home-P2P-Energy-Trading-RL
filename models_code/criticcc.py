import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import Config


class Critic(nn.Module):
    """
    Critic network for single-house DDPG agent.
    
    Each critic network handles one house with:
    - State input: House-specific state (own features + all houses' selling prices) [17 dims]
    - Action input: House-specific actions (hvac_energy, battery_action, selling_price) [3 dims]
    - Output: Q-value for this house [1 dim]
    """
    def __init__(self, input_dims: int, n_actions: int, config: Config) -> None:
        super(Critic, self).__init__()
        
        # For multi-agent DDPG, each critic handles one house
        self.state_dim_per_house = config.get('environment', 'state_dim_per_house')  # 17
        self.actions_per_house = config.get('environment', 'action_dim_per_house')   # 3
        
        # Verify dimensions for single-house agent
        if input_dims != self.state_dim_per_house:
            print(f"Warning: Input dimensions mismatch in single-house Critic. Got {input_dims}, expected {self.state_dim_per_house}")
            input_dims = self.state_dim_per_house
            
        if n_actions != self.actions_per_house:
            print(f"Warning: Action dimensions mismatch in single-house Critic. Got {n_actions}, expected {self.actions_per_house}")
            n_actions = self.actions_per_house
        
        self.input_dims = input_dims  # 17 for single house
        self.fc1_dims = config.get('critic', 'fc1_dims')
        self.fc2_dims = config.get('critic', 'fc2_dims')
        self.fc3_dims = config.get('critic', 'fc3_dims')
        self.n_actions = n_actions    # 3 for single house
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