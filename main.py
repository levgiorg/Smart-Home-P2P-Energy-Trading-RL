import argparse
import numpy as np
import torch
import random
import json

from hyperparameters import Config
from train import train_ddpg

def run_experiments():
    """
    Comprehensive hyperparameter testing for the smart grid system.
    Tests multiple parameter combinations that could affect system performance.
    """
    config = Config()
    original_config = json.loads(json.dumps(Config.config))
    
    def reset_config():
        return Config()

    experiments = {
        'reward_beta': [1.0, 1.2, 1.4, 1.6, 1.8, 2.0],  # More granular testing of reward scaling
        
        'grid_fee': [0.02, 0.03, 0.04, 0.05, 0.06],  # Finer granularity for P2P costs
        
        'learning_params': [
            # (actor_lr, critic_lr, batch_size, memory_size)
            (1e-5, 1e-4, 32, 838320),
            (1e-6, 1e-5, 64, 838320),
            (1e-4, 1e-3, 128, 838320),
            (1e-5, 1e-4, 256, 1000000)
        ],
        
        'network_architecture': [
            # (fc1_dims, fc2_dims, fc3_dims)
            (400, 300, 300),
            (512, 512, 512),
            (256, 256, 256),
            (600, 400, 400)
        ],
        
        'comfort_bounds': [
            # (t_min, t_max, comfort_penalty)
            (19.5, 22.5, 5),
            (19.0, 23.0, 4),
            (20.0, 22.0, 6),
            (19.0, 24.0, 3)
        ],
        
        'battery_params': [
            # (capacity_min, capacity_max, n_c, n_d)
            (0.6, 150.0, 0.95, 0.95),
            (1.0, 200.0, 0.90, 0.90),
            (2.0, 250.0, 0.98, 0.98),
            (1.5, 300.0, 0.93, 0.93)
        ],
        
        'rl_params': [
            # (gamma, tau, epsilon_decay, eps_end)
            (0.99, 0.001, 0.997, 0.1),
            (0.95, 0.005, 0.999, 0.05),
            (0.98, 0.002, 0.998, 0.08),
            (0.97, 0.003, 0.996, 0.15)
        ],
        
        'hvac_params': [
            # (eta_hvac, epsilon, e_max)
            (1.0, 0.7, 200),
            (0.9, 0.8, 250),
            (1.1, 0.6, 300),
            (0.95, 0.75, 180)
        ],
        
        'price_params': [
            # (price_penalty, depreciation_coeff)
            (100, 1.0),
            (80, 0.8),
            (120, 1.2),
            (90, 0.9)
        ]
    }

    # Run experiments for each parameter combination
    for beta in experiments['reward_beta']:
        config = reset_config()
        config.set('reward', 'beta', beta)
        print(f"\nRunning experiment with reward beta = {beta}")
        main()

    for fee in experiments['grid_fee']:
        config = reset_config()
        config.set('environment', 'grid_fee', fee)
        print(f"\nRunning experiment with grid fee = {fee}")
        main()

    for actor_lr, critic_lr, batch_size, memory_size in experiments['learning_params']:
        config = reset_config()
        config.set('rl_agent', 'learning_rate_actor', actor_lr)
        config.set('rl_agent', 'learning_rate_critic', critic_lr)
        config.set('rl_agent', 'batch_size', batch_size)
        config.set('rl_agent', 'memory_size', memory_size)
        print(f"\nRunning experiment with learning parameters: actor_lr={actor_lr}, critic_lr={critic_lr}, batch_size={batch_size}")
        main()

    for fc1, fc2, fc3 in experiments['network_architecture']:
        config = reset_config()
        config.set('actor', 'fc1_dims', fc1)
        config.set('actor', 'fc2_dims', fc2)
        config.set('critic', 'fc1_dims', fc1)
        config.set('critic', 'fc2_dims', fc2)
        config.set('critic', 'fc3_dims', fc3)
        print(f"\nRunning experiment with network architecture: {fc1}, {fc2}, {fc3}")
        main()

    for t_min, t_max, comfort_penalty in experiments['comfort_bounds']:
        config = reset_config()
        config.set('environment', 't_min', t_min)
        config.set('environment', 't_max', t_max)
        config.set('environment', 'comfort_penalty', comfort_penalty)
        print(f"\nRunning experiment with comfort parameters: t_min={t_min}, t_max={t_max}, penalty={comfort_penalty}")
        main()

    for cap_min, cap_max, n_c, n_d in experiments['battery_params']:
        config = reset_config()
        config.set('environment', 'battery_capacity_min', cap_min)
        config.set('environment', 'battery_capacity_max', cap_max)
        config.set('environment', 'n_c', n_c)
        config.set('environment', 'n_d', n_d)
        print(f"\nRunning experiment with battery parameters: cap_min={cap_min}, cap_max={cap_max}, n_c={n_c}, n_d={n_d}")
        main()

    for gamma, tau, eps_decay, eps_end in experiments['rl_params']:
        config = reset_config()
        config.set('rl_agent', 'gamma', gamma)
        config.set('rl_agent', 'tau', tau)
        config.set('rl_agent', 'epsilon_decay', eps_decay)
        config.set('rl_agent', 'eps_end', eps_end)
        print(f"\nRunning experiment with RL parameters: gamma={gamma}, tau={tau}, eps_decay={eps_decay}")
        main()

    for eta_hvac, epsilon, e_max in experiments['hvac_params']:
        config = reset_config()
        config.set('environment', 'eta_hvac', eta_hvac)
        config.set('environment', 'epsilon', epsilon)
        config.set('environment', 'e_max', e_max)
        print(f"\nRunning experiment with HVAC parameters: eta_hvac={eta_hvac}, epsilon={epsilon}, e_max={e_max}")
        main()

    for price_penalty, depreciation_coeff in experiments['price_params']:
        config = reset_config()
        config.set('cost_model', 'price_penalty', price_penalty)
        config.set('cost_model', 'depreciation_coeff', depreciation_coeff)
        print(f"\nRunning experiment with price parameters: price_penalty={price_penalty}, depreciation={depreciation_coeff}")
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