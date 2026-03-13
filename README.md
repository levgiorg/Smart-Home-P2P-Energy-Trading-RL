# Smart Home Energy Cartel Prevention and Management

A deep reinforcement learning approach to optimize smart home energy usage while preventing cartel-like behavior in peer-to-peer energy markets.

## Abstract

This repository implements a comprehensive framework for smart home energy management with peer-to-peer (P2P) trading capabilities, focusing on the detection and prevention of cartel-like behaviors. We employ Deep Deterministic Policy Gradient (DDPG) reinforcement learning to optimize energy usage, storage, and trading decisions while maintaining market fairness through novel anti-cartel mechanisms.

## Problem Statement

As residential energy systems become increasingly sophisticated, smart homes with battery storage and renewable generation can participate in peer-to-peer energy markets. However, these systems, when optimized individually for maximum profit, can develop cartel-like behaviors that manipulate market prices, leading to unfair outcomes and reduced social welfare. This research addresses the challenge of detecting and preventing such behaviors while maintaining energy efficiency.


### System Model
![System Model](images/Figure.png)
*Configuration of multiple smart homes and the P2P energy exchange framework. The DRL agent of each smart home performs local actions regarding the HVAC and ESS operation, while also setting the price of excess energy in the P2P market.*

## Key Features

- **Smart Home Energy Management**: Optimizes HVAC operation, battery charging/discharging, and price-setting strategies
- **Reinforcement Learning Optimization**: DDPG algorithm with advanced neural network architecture
- **Anti-Cartel Mechanisms**:
  - **Detection-based (referred to as "Reward-Based" in the paper)**: Monitors price patterns and applies penalties when cartel-like behavior is detected
  - **Ceiling-based (referred to as "Threshold-Based" in the paper)**: Enforces a maximum price threshold below the grid price
  - **Baseline (referred to as "No Control Method" in the paper)**: No anti-cartel mechanism applied
- **Comprehensive Environment Simulation**: Realistic modeling of HVAC systems, battery storage, energy generation, and peer-to-peer trading
- **Extensive Evaluation Framework**: Analysis across multiple metrics including energy efficiency, price competitiveness, and trading profits

## Technical Architecture

The system consists of several key components:

1. **Environment** (`environment/environment.py`): Simulates multiple smart homes with:
   - Dynamic temperature control (HVAC)
   - Battery storage with charging/discharging capabilities
   - Solar generation based on weather data
   - Energy consumption patterns derived from real-world data
   - Peer-to-peer energy trading market

2. **Anti-Cartel Mechanisms** (`environment/anti_cartel.py` and `src/market/`):
   - **Detection Mechanism**: Uses statistical methods to identify suspicious price coordination
   - **Ceiling Mechanism**: Implements a dynamic price ceiling based on grid prices
   - **Adaptive Mechanism**: Rolling-window dynamic thresholds (`src/market/adaptive.py`)

3. **RL Agents** (`src/agents/`): Multiple algorithms available:
   - **DDPG** (primary): Deterministic actor-critic with Ornstein-Uhlenbeck exploration
   - **SAC**: Soft Actor-Critic with entropy regularization
   - **TD3**: Twin-Delayed DDPG with target policy smoothing
   - **PPO**: Proximal Policy Optimization (on-policy)
   - **DQN**: Deep Q-Network (discrete actions)

4. **Training Infrastructure** (`src/training/`, `src/experiment/`):
   - Agent-agnostic `Trainer` class
   - `ExperimentRegistry` (SQLite) for run tracking
   - `RunVersioner` for reproducible run directories with git snapshots
   - `BatchOrchestrator` for parallel multi-GPU experiment grids

5. **Analysis Tools** (`src/evaluation/`, `src/plotting/`, `analysis/`):
   - `Evaluator` with multi-seed aggregation and stress-scenario testing
   - `ParameterSweep` for 1D/2D sensitivity analysis
   - Matplotlib plots (convergence, bar comparison, heatmap)
   - TikZ/pgfplots export for publication figures

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Smart-Home-Cartel.git
cd Smart-Home-Cartel

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Support

For containerized execution:

```bash
# Build the container
docker build -t smart-home-cartel .

# Run the container
docker run -it --gpus all smart-home-cartel python main.py
```

## Usage

### Running Experiments

```bash
# Single-run training
python train.py --agent ddpg --mechanism detection --seed 42 --episodes 1000

# Train with a specific algorithm and mechanism
python train.py --agent sac --mechanism ceiling --episodes 2000

# Load config from YAML
python train.py --config experiments/sac_adaptive.yaml

# Parallel batch training (cartesian product over agent × mechanism × seed)
python batch_train.py --batch experiments/full_comparison.yaml --gpus 0,1 --max-concurrent 4

# Check status of all runs
python batch_train.py --status

# Resume failed runs
python batch_train.py --batch experiments/full_comparison.yaml --resume-failed
```

### Evaluating Trained Agents

```bash
# Evaluate a checkpoint
python evaluate.py --checkpoint results/ddpg_detection_42/model_best.pt --scenarios all
```

### Generating Visualizations

```bash
# Standalone analysis scripts
python analysis/analyze_main.py
python analysis/cartel_analyzer.py
```

## Experimental Configuration

The framework supports extensive configuration for different experimental scenarios:

1. **Anti-Cartel Mechanisms**:
   - Detection mechanism ("Reward-Based") with configurable monitoring window and penalties
   - Ceiling mechanism ("Threshold-Based") with adjustable markup limits
   - Baseline with no mechanism ("No Control Method")

2. **Reward Parameters**:
   - Balance between profit optimization and energy efficiency (beta)
   - Temperature comfort penalties
   - Price penalties for market manipulation

3. **Network Architecture**:
   - Various neural network configurations for the actor-critic models
   - Different layer sizes and activation functions

4. **Learning Parameters**:
   - Learning rates for actor and critic
   - Batch sizes and memory capacities
   - Update intervals for target networks

5. **Battery Configurations**:
   - Different capacity ranges and initial charge states
   - Charging/discharging efficiency parameters

6. **Comfort Settings**:
   - Temperature range preferences
   - HVAC efficiency settings
   - Comfort penalty factors

## Results and Findings

Our extensive experiments demonstrate that anti-cartel mechanisms can effectively prevent price manipulation in P2P energy markets while maintaining energy efficiency. Key findings include:

### Market Fairness

The detection-based mechanism successfully identifies and penalizes coordinated pricing strategies, reducing the price ratio (selling price to grid price) by an average of 15% compared to the baseline without significantly impacting trading volume.

### Energy Efficiency

All mechanisms maintain similar levels of HVAC efficiency and temperature control, with the detection-based approach showing a slight advantage (3.2% improvement) in overall energy efficiency.

### Economic Performance

While the ceiling-based mechanism ensures the most competitive pricing (lowest price ratios), it reduces trading profits by approximately 7% compared to the detection-based approach, which offers a better balance between profit and fairness.

### Overall Performance

The detection-based (reward-based) mechanism provides the best overall performance across multiple metrics, with 12% higher cumulative rewards compared to the baseline and 5% higher than the ceiling-based approach.

## Visualizations

### P2P Price Convergence
![P2P Price Convergence](images/p2p_price_convergence.png)
*Price convergence in the peer-to-peer energy market showing how different anti-cartel mechanisms influence price dynamics*

### Temperature Control Performance
![Temperature Comfort Zone](images/temperature_comfort_zone.png)
*Indoor temperature control performance showing how the system maintains temperatures within the comfort zone while optimizing energy usage*

### Battery Management Strategies
![Battery Management](images/battery_management.png)
*Optimal battery charging and discharging strategies under different market conditions*

### Energy Consumption Analysis
![Merged Energy Consumption](images/merged_energy_consumption.png)
*Comprehensive energy consumption analysis showing distribution across different sources and mechanisms*

## Methodology

The project employs a simulated environment with multiple smart homes, each capable of:
- Consuming energy (HVAC and base load)
- Generating energy (solar)
- Storing energy (batteries)
- Trading energy with other homes or the grid

Each home is controlled by a DDPG agent that optimizes:
- HVAC energy usage for temperature control
- Battery charging/discharging
- Energy selling price

Anti-cartel mechanisms monitor and influence the P2P market to prevent price manipulation through:
1. **Detection**: Statistical analysis of price patterns to identify coordination
2. **Ceiling**: Dynamic price thresholds based on grid prices and market conditions

## Dependencies

- PyTorch >= 1.8.0
- NumPy >= 1.19.5
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Pydantic >= 2.0.0
- PyYAML >= 6.0

## Documentation

Detailed code maps are available in [`docs/CODEMAPS/`](docs/CODEMAPS/):

- [`INDEX.md`](docs/CODEMAPS/INDEX.md) — Architecture overview and directory map
- [`environment.md`](docs/CODEMAPS/environment.md) — RL environment, physics, anti-cartel mechanisms
- [`models.md`](docs/CODEMAPS/models.md) — Neural networks and RL agent architectures
- [`training.md`](docs/CODEMAPS/training.md) — Training pipeline and experiment management
- [`utilities.md`](docs/CODEMAPS/utilities.md) — Buffers, configs, market regulators
- [`analysis.md`](docs/CODEMAPS/analysis.md) — Evaluation, plotting, sensitivity analysis

## Citation

If you use this code or methodology in your research, please cite:

```
@article{Levis2025,
  title = {A Peer-to-Peer Energy Management and Exchange Framework in Energy Communities via Actor-Critic Learning},
  url = {http://dx.doi.org/10.36227/techrxiv.174802380.00970908/v1},
  DOI = {10.36227/techrxiv.174802380.00970908/v1},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Levis,  George A. and Spantideas,  Sotirios T and Giannopoulos,  Anastasios E and Trakadas,  Panagiotis},
  year = {2025},
  month = may 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Data sources: Danish energy market data, household consumption patterns from public datasets
- Research inspiration: Recent work on multi-agent reinforcement learning in energy markets and game-theoretic approaches to market manipulation
- Computing resources: [Include if applicable]