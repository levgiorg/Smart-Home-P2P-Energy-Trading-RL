import random

from .transition import Transition


class ReplayMemory:
    """
    Replay memory class.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        """Saves a transition tuple."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """Randomly selects batch_size elements from the memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
