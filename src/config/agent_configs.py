from __future__ import annotations

from pydantic import BaseModel, Field


class DDPGConfig(BaseModel):
    gamma: float = 0.99
    tau: float = 0.001
    batch_size: int = 32
    memory_size: int = 838_320
    lr_actor: float = 1e-6
    lr_critic: float = 1e-5
    noise_theta: float = 0.15
    noise_sigma: float = 0.2
    actor_hidden: tuple[int, int] = (400, 300)
    critic_hidden: tuple[int, int, int] = (400, 300, 300)
    device: str = "cpu"

    model_config = {"frozen": True}


class SACConfig(BaseModel):
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    memory_size: int = 1_000_000
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    initial_alpha: float = 0.2
    auto_entropy_tuning: bool = True
    hidden_dims: tuple[int, int] = (256, 256)
    device: str = "cpu"

    model_config = {"frozen": True}


class TD3Config(BaseModel):
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    memory_size: int = 1_000_000
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.1
    hidden_dims: tuple[int, int] = (256, 256)
    device: str = "cpu"

    model_config = {"frozen": True}


class PPOConfig(BaseModel):
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    lr: float = 3e-4
    epochs_per_update: int = 10
    batch_size: int = 64
    rollout_length: int = 2048
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dims: tuple[int, int] = (256, 256)
    device: str = "cpu"

    model_config = {"frozen": True}


class DQNConfig(BaseModel):
    gamma: float = 0.99
    tau: float = 0.001
    batch_size: int = 32
    memory_size: int = 838_320
    lr: float = 1e-5
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.997
    target_update_freq: int = 10
    use_double_dqn: bool = True
    use_dueling: bool = True
    use_prioritized_replay: bool = False
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    gradient_clip: float = 10.0
    fc1_dims: int = 400
    fc2_dims: int = 300
    fc3_dims: int = 300
    hvac_discrete_levels: int = 10
    battery_discrete_levels: int = 21
    price_discrete_levels: int = 10
    device: str = "cpu"

    model_config = {"frozen": True}
