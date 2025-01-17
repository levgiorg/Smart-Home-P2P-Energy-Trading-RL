import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperparameters import Config


class Actor(nn.Module):
    """
    Actor network for DDPG agent.

    Attributes:
        input_dims (int): Number of input dimensions.
        fc1_dims (int): Number of neurons in the first hidden layer.
        fc2_dims (int): Number of neurons in the second hidden layer.
        n_actions (int): Number of actions.
        device (torch.device): Device to run the model on.
    """
    def __init__(self, input_dims, n_actions, config: Config):
        super(Actor, self).__init__()
        self.input_dims = input_dims  # Should be 36 (12 per house × 3 houses)
        self.fc1_dims = config.get('actor', 'fc1_dims')
        self.fc2_dims = config.get('actor', 'fc2_dims')
        self.n_actions = n_actions  # Should be 9 (3 per house × 3 houses)
        device = config.get('general', 'device') or 'cpu'
        self.device = torch.device(device)

        # Define layers
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.to(self.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))
        return mu