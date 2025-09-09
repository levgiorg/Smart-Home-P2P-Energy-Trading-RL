import random
import logging
import torch
import numpy as np
from typing import Optional

from .transition import Transition


class ReplayMemory:
    """
    GPU-optimized replay memory class for maximum VRAM utilization.
    
    Stores transitions directly on GPU to eliminate CPU→GPU transfers
    and maximize A100 GPU memory usage for faster training.
    """

    def __init__(self, capacity: int, device: Optional[torch.device] = None) -> None:
        """
        Initialize GPU-based replay memory.
        
        Args:
            capacity: Maximum number of transitions to store
            device: GPU device to store transitions on
        """
        self.capacity = capacity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.position = 0
        self.size = 0
        
        # Pre-allocate GPU tensors for maximum performance
        # These will be filled as experiences are added
        self.states = None
        self.actions = None  
        self.next_states = None
        self.rewards = None
        self.is_initialized = False
        
    def _initialize_buffers(self, state_dim: int, action_dim: int) -> None:
        """
        Initialize GPU tensor buffers based on first transition dimensions.
        
        Args:
            state_dim: Dimension of state tensors
            action_dim: Dimension of action tensors
        """
        logging.info(f"Initializing GPU replay buffer: {self.capacity} transitions on {self.device}")
        logging.info(f"Buffer dimensions: state={state_dim}, action={action_dim}")
        
        # Pre-allocate all memory on GPU for maximum efficiency
        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device)
        
        self.is_initialized = True
        
        # Calculate memory usage for reporting
        total_elements = self.capacity * (2 * state_dim + action_dim + 1)
        memory_mb = (total_elements * 4) / (1024 * 1024)  # 4 bytes per float32
        logging.info(f"GPU replay buffer allocated: {memory_mb:.1f}MB on {self.device}")

    def push(self, *args) -> None:
        """
        Save a transition tuple directly to GPU memory.
        
        Args:
            *args: (state, action, next_state, reward) tensors
        """
        if len(args) != 4:
            raise ValueError(f"Expected 4 arguments (state, action, next_state, reward), got {len(args)}")
            
        state, action, next_state, reward = args
        
        # Ensure all inputs are tensors on the correct device
        state = self._ensure_gpu_tensor(state)
        action = self._ensure_gpu_tensor(action)
        next_state = self._ensure_gpu_tensor(next_state)  
        reward = self._ensure_gpu_tensor(reward)
        
        # Initialize buffers on first use
        if not self.is_initialized:
            state_dim = state.numel() if state.dim() > 0 else 1
            action_dim = action.numel() if action.dim() > 0 else 1
            self._initialize_buffers(state_dim, action_dim)
        
        # Store transition directly in pre-allocated GPU memory
        self.states[self.position] = state.flatten()
        self.actions[self.position] = action.flatten() 
        self.next_states[self.position] = next_state.flatten()
        self.rewards[self.position] = reward.flatten()
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> list:
        """
        Sample batch directly from GPU memory - zero CPU→GPU transfer cost.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of Transition objects with GPU tensors
        """
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions, only {self.size} available")
            
        # Random sampling from GPU memory
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        # Batch sample - all operations on GPU
        batch_states = self.states[indices]
        batch_actions = self.actions[indices] 
        batch_next_states = self.next_states[indices]
        batch_rewards = self.rewards[indices]
        
        # Return as Transition objects (maintaining compatibility)
        transitions = []
        for i in range(batch_size):
            transitions.append(Transition(
                batch_states[i],
                batch_actions[i], 
                batch_next_states[i],
                batch_rewards[i]
            ))
            
        return transitions

    def _ensure_gpu_tensor(self, data) -> torch.Tensor:
        """
        Convert data to GPU tensor if not already.
        
        Args:
            data: Input data (tensor, numpy array, or scalar)
            
        Returns:
            GPU tensor
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(self.device)
        else:
            return torch.tensor(data, dtype=torch.float32, device=self.device)

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self.size
    
    def get_memory_usage(self) -> dict:
        """
        Get detailed memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        if not self.is_initialized:
            return {"status": "not_initialized", "memory_mb": 0}
            
        total_elements = self.capacity * (
            self.states.shape[1] * 2 +  # states + next_states
            self.actions.shape[1] +      # actions  
            1                            # rewards
        )
        memory_mb = (total_elements * 4) / (1024 * 1024)  # 4 bytes per float32
        
        return {
            "status": "initialized",
            "capacity": self.capacity,
            "size": self.size,
            "utilization": f"{(self.size/self.capacity)*100:.1f}%",
            "memory_mb": memory_mb,
            "device": str(self.device)
        }