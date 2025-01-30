import argparse
import numpy as np
import torch
import random
import json

from hyperparameters import Config
from train import train_ddpg

def run_experiments():
    config = Config()
    original_config = json.loads(json.dumps(Config.config))

    def reset_config():
        return Config()

    anti_cartel_configs = [
        {
            "mechanism_type": "detection",
            "monitoring_window": 100,
            "similarity_threshold": 0.85,
            "penalty_factor": 1.5,
            "markup_limit": 0.2,
            "market_elasticity": -0.5
        },
        {
            "mechanism_type": "ceiling",
            "monitoring_window": 100,
            "similarity_threshold": 0.85,
            "penalty_factor": 1.5,
            "markup_limit": 0.2,
            "market_elasticity": -0.5
        },
        {
            "mechanism_type": None,
            "monitoring_window": 100,
            "similarity_threshold": 0.85,
            "penalty_factor": 1.5,
            "markup_limit": 0.2,
            "market_elasticity": -0.5
        }
    ]

    experiments = {
        'reward_stability': [
            (1.1, 4, 85),    # Balanced approach
            (1.3, 3, 95),    # Reward-focused
            (0.9, 5, 75)     # Stability-focused
        ],
        'trading_optimization': [
            (0.015, 0.7, 0.45),  # Aggressive trading
            (0.022, 0.8, 0.55),  # Conservative trading
            (0.018, 0.75, 0.5)   # Balanced approach
        ],
        'network_advanced': [
            (1024, 768, 512),     # Deeper network
            (2048, 1024, 512),    # Much larger network
            (1536, 768, 384)      # Wide but shallow
        ],
        'learning_advanced': [
            (5e-5, 5e-4, 512, 1500000, 2),    # Larger batches, more memory
            (8e-5, 8e-4, 384, 1200000, 3),    # Faster learning
            (3e-5, 3e-4, 768, 2000000, 1)     # Very large batches
        ],
        'battery_advanced': [
            (2.0, 300.0, 0.98, 0.98, 0.7),    # High capacity, high efficiency
            (1.5, 250.0, 0.99, 0.99, 0.6),    # Maximum efficiency
            (2.5, 350.0, 0.97, 0.97, 0.8)     # Maximum capacity
        ],
        'comfort_advanced': [
            (19.8, 22.2, 4, 1.1),    # Strict comfort
            (19.5, 22.5, 3, 1.2),    # Better HVAC
            (19.2, 22.8, 5, 1.0)     # Flexible comfort
        ],
        'multi_objective': [
            (1.2, 0.018, 4, 90, 0.98, 0.98),   # Trading focused
            (1.1, 0.020, 3, 85, 0.99, 0.99),   # Efficiency focused
            (1.3, 0.015, 5, 95, 0.97, 0.97)    # Reward focused
        ]
    }

    for anti_cartel_config in anti_cartel_configs:
        mechanism_type = anti_cartel_config["mechanism_type"]
        mechanism_name = "No mechanism" if mechanism_type is None else f"{mechanism_type} mechanism"
        print(f"\n{'='*50}")
        print(f"Starting experiments with {mechanism_name}")
        print(f"{'='*50}")

        config = reset_config()
        for key, value in anti_cartel_config.items():
            config.set('anti_cartel', key, value)

        for beta, fee, comfort_p, price_p, n_c, n_d in experiments['multi_objective']:
            config = reset_config()
            for key, value in anti_cartel_config.items():
                config.set('anti_cartel', key, value)
            config.set('environment', 'num_houses', 10)
            config.set('reward', 'beta', beta)
            config.set('environment', 'grid_fee', fee)
            config.set('environment', 'comfort_penalty', comfort_p)
            config.set('cost_model', 'price_penalty', price_p)
            config.set('environment', 'n_c', n_c)
            config.set('environment', 'n_d', n_d)
            print(f"\nRunning multi-objective experiment with {mechanism_name}, beta={beta}, fee={fee}")
            main()

        for fee, init_price, min_price in experiments['trading_optimization']:
            config = reset_config()
            for key, value in anti_cartel_config.items():
                config.set('anti_cartel', key, value)
            config.set('environment', 'num_houses', 10)
            config.set('environment', 'grid_fee', fee)
            config.set('environment', 'initial_selling_price_ratio', init_price)
            config.set('environment', 'min_selling_price', min_price)
            print(f"\nRunning trading optimization with {mechanism_name}, fee={fee}, init_price={init_price}")
            main()

        for fc1, fc2, fc3 in experiments['network_advanced']:
            config = reset_config()
            for key, value in anti_cartel_config.items():
                config.set('anti_cartel', key, value)
            config.set('environment', 'num_houses', 10)
            config.set('actor', 'fc1_dims', fc1)
            config.set('actor', 'fc2_dims', fc2)
            config.set('critic', 'fc1_dims', fc1)
            config.set('critic', 'fc2_dims', fc2)
            config.set('critic', 'fc3_dims', fc3)
            print(f"\nRunning advanced network with {mechanism_name}, architecture: {fc1}, {fc2}, {fc3}")
            main()

        for actor_lr, critic_lr, batch_size, memory_size, update_interval in experiments['learning_advanced']:
            config = reset_config()
            for key, value in anti_cartel_config.items():
                config.set('anti_cartel', key, value)
            config.set('environment', 'num_houses', 10)
            config.set('rl_agent', 'learning_rate_actor', actor_lr)
            config.set('rl_agent', 'learning_rate_critic', critic_lr)
            config.set('rl_agent', 'batch_size', batch_size)
            config.set('rl_agent', 'memory_size', memory_size)
            config.set('rl_agent', 'update_interval', update_interval)
            print(f"\nRunning advanced learning with {mechanism_name}, actor_lr={actor_lr}, batch_size={batch_size}")
            main()

        for cap_min, cap_max, n_c, n_d, initial_charge in experiments['battery_advanced']:
            config = reset_config()
            for key, value in anti_cartel_config.items():
                config.set('anti_cartel', key, value)
            config.set('environment', 'num_houses', 10)
            config.set('environment', 'battery_capacity_min', cap_min)
            config.set('environment', 'battery_capacity_max', cap_max)
            config.set('environment', 'n_c', n_c)
            config.set('environment', 'n_d', n_d)
            config.set('environment', 'initial_battery_charge', initial_charge)
            print(f"\nRunning advanced battery with {mechanism_name}, cap_max={cap_max}, efficiency={n_c}")
            main()

        for t_min, t_max, comfort_penalty, hvac_efficiency in experiments['comfort_advanced']:
            config = reset_config()
            for key, value in anti_cartel_config.items():
                config.set('anti_cartel', key, value)
            config.set('environment', 'num_houses', 10)
            config.set('environment', 't_min', t_min)
            config.set('environment', 't_max', t_max)
            config.set('environment', 'comfort_penalty', comfort_penalty)
            config.set('environment', 'hvac_efficiency', hvac_efficiency)
            print(f"\nRunning advanced comfort with {mechanism_name}, range={t_min}-{t_max}, hvac_eff={hvac_efficiency}")
            main()

        for beta, comfort_penalty, price_penalty in experiments['reward_stability']:
            config = reset_config()
            for key, value in anti_cartel_config.items():
                config.set('anti_cartel', key, value)
            config.set('environment', 'num_houses', 10)
            config.set('reward', 'beta', beta)
            config.set('environment', 'comfort_penalty', comfort_penalty)
            config.set('cost_model', 'price_penalty', price_penalty)
            print(f"\nRunning reward stability with {mechanism_name}, beta={beta}, penalties={comfort_penalty},{price_penalty}")
            main()

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