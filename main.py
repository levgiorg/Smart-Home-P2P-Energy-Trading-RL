import argparse
import numpy as np
import torch
import random
import json

from hyperparameters import Config
from train import train_ddpg

def run_experiments():
    """
    Enhanced hyperparameter testing focusing on successful comfort parameters
    and incorporating proven hyperparameters from previous runs.
    """
    config = Config()
    original_config = json.loads(json.dumps(Config.config))
    
    def reset_config():
        return Config()

    # Define experiment sets based on successful previous runs and focusing on promising areas
    experiments = {
        'comfort_refined': [
            # (t_min, t_max, comfort_penalty, hvac_efficiency)
            # Based on successful (20.0, 22.0, 5) configuration
            (20.0, 22.0, 5, 1.1),    # Original successful setup with standard HVAC
            (20.0, 22.0, 5.5, 1.15),  # Increased penalty to enhance comfort adherence
            (20.0, 22.0, 4.5, 1.05),  # Reduced penalty for more flexibility
            (19.8, 22.2, 5, 1.1),     # Slightly wider range with same penalty
            (20.2, 21.8, 5, 1.1),     # Narrower range for stricter comfort
        ],
        
        'advanced_comfort_trading': [
            # (t_min, t_max, comfort_penalty, grid_fee, beta)
            (20.0, 22.0, 5, 0.015, 1.3),    # Low grid fee for trading
            (20.0, 22.0, 5, 0.018, 1.2),    # Moderate grid fee
            (20.0, 22.0, 5.5, 0.016, 1.25), # Balanced approach
        ],
        
        'network_comfort_optimized': [
            # (fc1_dims, fc2_dims, fc3_dims)
            (1024, 512, 512),     # Larger network for complex comfort patterns
            (768, 384, 384),      # Proven successful architecture
            (1536, 768, 768),     # Extra capacity for comfort optimization
        ],
        
        'learning_comfort_focused': [
            # (actor_lr, critic_lr, batch_size, memory_size)
            (2e-4, 2e-3, 512, 1200000),   # Current successful values
            (1e-4, 1e-3, 768, 1500000),   # Larger batches for stability
            (3e-4, 3e-3, 384, 1000000),   # Faster learning for comfort
        ],
        
        'comfort_battery_combined': [
            # (t_min, t_max, comfort_penalty, cap_min, cap_max, n_c)
            (20.0, 22.0, 5, 2.0, 250.0, 0.97),    # Higher capacity
            (20.0, 22.0, 5, 1.5, 200.0, 0.98),    # Better efficiency
            (20.0, 22.0, 5, 2.5, 300.0, 0.96),    # Maximum capacity
        ],
        
        'multi_parameter': [
            # (beta, grid_fee, comfort_penalty, price_penalty, n_c, n_d, t_min, t_max)
            (1.3, 0.015, 5, 95, 0.97, 0.97, 20.0, 22.0),    # Reward focused
            (1.2, 0.018, 5, 90, 0.98, 0.98, 20.0, 22.0),    # Trading focused
            (1.25, 0.016, 5, 93, 0.975, 0.975, 20.0, 22.0), # Balanced
        ]
    }

    # Run experiments for each parameter combination
    config = reset_config()
    config.set('environment', 'num_houses', 10)

    # Run multi-parameter experiments first
    for beta, fee, comfort_p, price_p, n_c, n_d, t_min, t_max in experiments['multi_parameter']:
        config = reset_config()
        config.set('environment', 'num_houses', 10)
        config.set('reward', 'beta', beta)
        config.set('environment', 'grid_fee', fee)
        config.set('environment', 'comfort_penalty', comfort_p)
        config.set('cost_model', 'price_penalty', price_p)
        config.set('environment', 'n_c', n_c)
        config.set('environment', 'n_d', n_d)
        config.set('environment', 't_min', t_min)
        config.set('environment', 't_max', t_max)
        print(f"\nRunning multi-parameter experiment with beta={beta}, comfort_penalty={comfort_p}")
        main()

    # Run refined comfort experiments
    for t_min, t_max, comfort_penalty, hvac_eff in experiments['comfort_refined']:
        config = reset_config()
        config.set('environment', 'num_houses', 10)
        config.set('environment', 't_min', t_min)
        config.set('environment', 't_max', t_max)
        config.set('environment', 'comfort_penalty', comfort_penalty)
        config.set('environment', 'eta_hvac', hvac_eff)
        print(f"\nRunning refined comfort experiment with range={t_min}-{t_max}, penalty={comfort_penalty}")
        main()

    # Run advanced comfort-trading experiments
    for t_min, t_max, comfort_penalty, grid_fee, beta in experiments['advanced_comfort_trading']:
        config = reset_config()
        config.set('environment', 'num_houses', 10)
        config.set('environment', 't_min', t_min)
        config.set('environment', 't_max', t_max)
        config.set('environment', 'comfort_penalty', comfort_penalty)
        config.set('environment', 'grid_fee', grid_fee)
        config.set('reward', 'beta', beta)
        print(f"\nRunning comfort-trading experiment with penalty={comfort_penalty}, fee={grid_fee}")
        main()

    # Run network architecture experiments
    for fc1, fc2, fc3 in experiments['network_comfort_optimized']:
        config = reset_config()
        config.set('environment', 'num_houses', 10)
        config.set('actor', 'fc1_dims', fc1)
        config.set('actor', 'fc2_dims', fc2)
        config.set('critic', 'fc1_dims', fc1)
        config.set('critic', 'fc2_dims', fc2)
        config.set('critic', 'fc3_dims', fc3)
        print(f"\nRunning network experiment with architecture: {fc1}, {fc2}, {fc3}")
        main()

    # Run learning parameter experiments
    for actor_lr, critic_lr, batch_size, memory_size in experiments['learning_comfort_focused']:
        config = reset_config()
        config.set('environment', 'num_houses', 10)
        config.set('rl_agent', 'learning_rate_actor', actor_lr)
        config.set('rl_agent', 'learning_rate_critic', critic_lr)
        config.set('rl_agent', 'batch_size', batch_size)
        config.set('rl_agent', 'memory_size', memory_size)
        print(f"\nRunning learning experiment with actor_lr={actor_lr}, batch_size={batch_size}")
        main()

    # Run comfort-battery combined experiments
    for t_min, t_max, comfort_penalty, cap_min, cap_max, n_c in experiments['comfort_battery_combined']:
        config = reset_config()
        config.set('environment', 'num_houses', 10)
        config.set('environment', 't_min', t_min)
        config.set('environment', 't_max', t_max)
        config.set('environment', 'comfort_penalty', comfort_penalty)
        config.set('environment', 'battery_capacity_min', cap_min)
        config.set('environment', 'battery_capacity_max', cap_max)
        config.set('environment', 'n_c', n_c)
        config.set('environment', 'n_d', n_c)
        print(f"\nRunning comfort-battery experiment with cap_max={cap_max}, efficiency={n_c}")
        main()

    # Restore original configuration
    Config.config = original_config
    with open(Config.config_file_path, "w") as f:
        json.dump(original_config, f, indent=4)
    print("\nOriginal configuration has been restored.")

def main():
    parser = argparse.ArgumentParser(description="Train DDPG for Smart Home Energy Management")
    args = parser.parse_args()
    
    config = Config()
    random_seed = config.get('simulation', 'random_seed')
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Using random seed from config: {random_seed}")
    else:
        print("No random seed set in config. Using random initialization.")
    
    train_ddpg()

if __name__ == "__main__":
    run_experiments()