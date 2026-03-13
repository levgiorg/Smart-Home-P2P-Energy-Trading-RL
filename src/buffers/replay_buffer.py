from __future__ import annotations

import torch


class ReplayBuffer:
    """GPU-optional off-policy replay buffer.

    Pre-allocates tensors on the first push for efficiency.
    """

    def __init__(self, capacity: int, device: str | torch.device = "cpu") -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self._pos = 0
        self._size = 0
        self._initialized = False

        self._states: torch.Tensor
        self._actions: torch.Tensor
        self._next_states: torch.Tensor
        self._rewards: torch.Tensor
        self._dones: torch.Tensor

    def _init_buffers(self, state_dim: int, action_dim: int) -> None:
        self._states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self._actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self._next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self._rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._dones = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        self._initialized = True

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: float | torch.Tensor,
        done: bool | torch.Tensor,
    ) -> None:
        state = self._to_tensor(state)
        action = self._to_tensor(action)
        next_state = self._to_tensor(next_state)
        reward = self._to_tensor(reward).reshape(1)
        done_t = torch.tensor([[float(done)]], dtype=torch.float32, device=self.device).reshape(1)

        if not self._initialized:
            self._init_buffers(state.numel(), action.numel())

        self._states[self._pos] = state.flatten()
        self._actions[self._pos] = action.flatten()
        self._next_states[self._pos] = next_state.flatten()
        self._rewards[self._pos] = reward
        self._dones[self._pos] = done_t

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size} transitions, need {batch_size}")
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "states": self._states[idx],
            "actions": self._actions[idx],
            "next_states": self._next_states[idx],
            "rewards": self._rewards[idx],
            "dones": self._dones[idx],
        }

    def _to_tensor(self, x: object) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float().to(self.device)
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def __len__(self) -> int:
        return self._size
