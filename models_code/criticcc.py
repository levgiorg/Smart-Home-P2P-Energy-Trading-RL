import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameters import Config


class Critic(nn.Module):
    """
    Critic network for DDPG agent.

    Attributes:
        input_dims (int): Number of input dimensions.
        fc1_dims (int): Number of neurons in the first hidden layer.
        fc2_dims (int): Number of neurons in the second hidden layer.
        fc3_dims (int): Number of neurons in the third hidden layer.
        n_actions (int): Number of actions.
        device (torch.device): Device to run the model on.
    """
    def __init__(self, input_dims, n_actions, config: Config):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = config.get('critic', 'fc1_dims')
        self.fc2_dims = config.get('critic', 'fc2_dims')
        self.fc3_dims = config.get('critic', 'fc3_dims')
        self.n_actions = n_actions
        device = config.get('general', 'device') or 'cpu'
        self.device = torch.device(device)

        # Define layers
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Action pathway
        self.action_fc = nn.Linear(self.n_actions, self.fc3_dims)

        # Combined pathway
        self.fc3 = nn.Linear(self.fc2_dims + self.fc3_dims, self.fc3_dims)
        self.q = nn.Linear(self.fc3_dims, 1)
        self.to(self.device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))

        action_value = F.relu(self.action_fc(action))

        state_action_value = torch.cat([state_value, action_value], dim=1)
        state_action_value = F.relu(self.fc3(state_action_value))
        q_value = self.q(state_action_value)
        return q_value

