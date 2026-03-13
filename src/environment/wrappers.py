from __future__ import annotations

"""Environment wrapper for DQN agents that need a discrete action wrapper.

Passes continuous actions from the DQNAgent through to the environment unchanged
since DQNAgent already converts discrete → continuous internally.
"""

from .energy_env import EnergyEnv


class DiscreteActionWrapper(EnergyEnv):
    """Identity wrapper kept for API symmetry; DQNAgent handles discretization internally."""

    pass
