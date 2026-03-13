# Models Codemap

**Last Updated:** 2026-03-13
**Entry Points:** `src/agents/__init__.py`, `src/networks/actors.py`, `src/networks/critics.py`

## Architecture

```
BaseAgent (src/agents/base.py)
    ├── DDPGAgent          → _SingleHouseDDPG × N houses
    │       ├── DeterministicActor  (tanh output)
    │       ├── Critic              (single Q)
    │       ├── ReplayBuffer
    │       └── OUNoise
    │
    ├── SACAgent           → per-house SAC
    │       ├── StochasticActor     (squashed Gaussian)
    │       ├── TwinCritic          (min-Q)
    │       └── ReplayBuffer
    │
    ├── TD3Agent           → per-house TD3
    │       ├── DeterministicActor
    │       ├── TwinCritic
    │       ├── ReplayBuffer
    │       └── GaussianNoise
    │
    ├── PPOAgent           → per-house PPO
    │       ├── StochasticActor     (no squash, clamp)
    │       ├── ValueNetwork        V(s)
    │       └── RolloutBuffer
    │
    └── DQNAgent           → discrete action space
            └── dqn_networks.py (legacy models_code/)

create_agent(name, **kwargs) → BaseAgent   [factory]
```

## Key Modules

| Module | Purpose | Key Exports | Dependencies |
|--------|---------|-------------|--------------|
| `src/agents/base.py` | Abstract interface all agents must implement | `BaseAgent` | `torch` |
| `src/agents/ddpg.py` | Multi-house DDPG wrapper + per-house helper | `DDPGAgent`, `_SingleHouseDDPG` | `DeterministicActor`, `Critic`, `ReplayBuffer`, `OUNoise`, `DDPGConfig` |
| `src/agents/sac.py` | Soft Actor-Critic | `SACAgent` | `StochasticActor`, `TwinCritic`, `ReplayBuffer`, `SACConfig` |
| `src/agents/td3.py` | Twin-Delayed DDPG | `TD3Agent` | `DeterministicActor`, `TwinCritic`, `ReplayBuffer`, `GaussianNoise`, `TD3Config` |
| `src/agents/ppo.py` | Proximal Policy Optimization | `PPOAgent` | `StochasticActor`, `ValueNetwork`, `RolloutBuffer`, `PPOConfig` |
| `src/agents/dqn.py` | Deep Q-Network | `DQNAgent` | `models_code/dqn_networks.py`, `DQNConfig` |
| `src/agents/__init__.py` | Agent registry and factory | `AGENT_REGISTRY`, `create_agent` | All agent modules |
| `src/networks/actors.py` | Actor network implementations | `DeterministicActor`, `StochasticActor` | `torch.nn` |
| `src/networks/critics.py` | Critic network implementations | `Critic`, `TwinCritic` | `torch.nn` |
| `src/networks/value.py` | State-value network for PPO | `ValueNetwork` | `torch.nn` |
| `models_code/dqn_networks.py` | Legacy DQN network (used by DQNAgent) | — | `torch.nn` |

## Network Architectures

### DeterministicActor (DDPG, TD3)
```
Input (state_dim)
  → Linear → ReLU  (hidden[0], default 256)
  → Linear → ReLU  (hidden[1], default 256)
  → Linear         (action_dim)
  → Tanh           → output ∈ (−1, 1)
```
Initialization: Xavier uniform on all Linear layers.

### StochasticActor (SAC, PPO)
```
Input (state_dim)
  → shared trunk (hidden MLP)
  → mean_layer  (action_dim)  ─┐
  → log_std_layer (action_dim) ─┴→ Normal(mean, exp(log_std))
                                    SAC: tanh squash + log_prob correction
                                    PPO: clamp(−1, 1) raw sample
```
Log-std clamped to `[−20, 2]`.

### Critic / TwinCritic (DDPG / SAC, TD3)
```
Input: concat(state, action)
  → Linear → ReLU  (hidden[0])
  → Linear → ReLU  (hidden[1])
  → Linear → scalar Q(s,a)

TwinCritic: two independent copies; forward() returns (Q1, Q2);
            min_q() returns min(Q1, Q2) for target backup.
```
DDPG Critic default hidden: `(400, 300, 300)`.  
SAC/TD3 TwinCritic default hidden: `(256, 256)`.

### ValueNetwork (PPO)
```
Input (state_dim)
  → Linear → ReLU  (256)
  → Linear → ReLU  (256)
  → Linear → scalar V(s)
```

## BaseAgent Interface

```python
class BaseAgent(ABC):
    def select_action(state, deterministic=False) -> Tensor   # [num_houses, action_dim]
    def store_transition(state, action, next_state, reward, done) -> None
    def update() -> dict[str, float]          # returns loss dict
    def ready_to_update() -> bool
    def save(path) -> None
    def load(path) -> None
    @property is_on_policy -> bool            # True=PPO, False=off-policy
```

## Multi-House Design

All agents follow the same pattern: the outer class (`DDPGAgent`, `SACAgent`, …) holds **N independent per-house agents**. On `select_action`, the global state tensor is sliced into per-house chunks (`state[i*sdph : (i+1)*sdph]`), and each sub-agent acts on its own slice. Transitions are stored separately per house.

This means there is **no shared policy or centralized critic** — a fully decentralized multi-agent setup (IDMARL).

## Agent Config Defaults

| Agent | Actor LR | Critic LR | Batch | Memory | Hidden |
|-------|----------|-----------|-------|--------|--------|
| DDPG | 1e-4 | 1e-3 | 64 | 100 000 | (256,256) actor / (400,300,300) critic |
| SAC | 3e-4 | 3e-4 | 256 | 1 000 000 | (256,256) |
| TD3 | 3e-4 | 3e-4 | 256 | 1 000 000 | (256,256) |
| PPO | 3e-4 | 1e-3 | 64 | rollout | (256,256) |
| DQN | 1e-4 | — | 64 | 100 000 | — |

## External Dependencies

- `torch` — Neural networks, optimizers, tensor ops
- `numpy` — Noise sampling, metric aggregation

## Related Areas

- [environment.md](environment.md) — State/action space consumed by agents
- [training.md](training.md) — Training loop that calls `agent.update()`
- [utilities.md](utilities.md) — Replay buffers, noise processes injected into agents
