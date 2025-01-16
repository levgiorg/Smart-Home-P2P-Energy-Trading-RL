import argparse
import numpy as np
import torch
import random

from hyperparameters import Config
from train import train_ddpg

def main():
    parser = argparse.ArgumentParser(description="Train DDPG for Smart Home Energy Management")
    args = parser.parse_args()
    
    config = Config()
    random_seed = config.get('simulation', 'random_seed')
    
    # Set all random seeds if a seed is specified in config
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
    main()