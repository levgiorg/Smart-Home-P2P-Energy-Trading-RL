from .base import BaseAgent
from .ddpg import DDPGAgent
from .dqn import DQNAgent
from .ppo import PPOAgent
from .sac import SACAgent
from .td3 import TD3Agent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "ddpg": DDPGAgent,
    "sac": SACAgent,
    "td3": TD3Agent,
    "ppo": PPOAgent,
    "dqn": DQNAgent,
}


def create_agent(name: str, **kwargs) -> BaseAgent:
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name](**kwargs)


__all__ = [
    "BaseAgent",
    "DDPGAgent",
    "DQNAgent",
    "SACAgent",
    "TD3Agent",
    "PPOAgent",
    "AGENT_REGISTRY",
    "create_agent",
]
