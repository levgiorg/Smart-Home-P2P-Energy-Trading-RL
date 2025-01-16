import torch

class Normalizer:
    """
    Normalizes the input data by computing an online variance and mean.
    """

    def __init__(self, num_inputs: int, device: torch.device):
        self.device = device
        self.n = torch.zeros(num_inputs).to(device)
        self.mean = torch.zeros(num_inputs).to(device)
        self.mean_diff = torch.zeros(num_inputs).to(device)
        self.var = torch.zeros(num_inputs).to(device)

    def observe(self, x: torch.Tensor) -> None:
        self.n += 1
        last_mean = self.mean.clone()
        self.mean += (x - last_mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs: torch.Tensor) -> torch.Tensor:
        obs_std = torch.sqrt(self.var).to(self.device)
        return (inputs - self.mean) / obs_std
