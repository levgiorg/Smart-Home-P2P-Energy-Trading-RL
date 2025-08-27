import os

import torch
import numpy as np

from agents import DDPGAgent
from environment import Environment
from hyperparameters import Config
from utilities import Utilities
from bookkeeper import BookKeeper


def train_ddpg(config_path='hyperparameters.json', model_name='ddpg_', enable_saving=False):
    """
    Train Multi-Agent DDPG with independent agents per house.
    
    Args:
        config_path (str): Path to hyperparameters config file
        model_name (str): Prefix for saved model files
        enable_saving (bool): Whether to save model checkpoints during training
    """
    # 1. Load configuration
    config = Config()
    device = torch.device(config.get('general', 'device'))
    num_episodes = config.get('rl_agent', 'num_episodes')

    # 2. Initialize environment
    env = Environment(dynamic=True)  

    # 3. Get environment info
    action_info = env.get_action_space_info()
    state_info = env.get_state_space_info()
    num_houses = state_info['num_houses']

    # 4. Initialize utilities
    utilities = Utilities(num_houses=num_houses)

    # 5. Initialize BookKeeper
    bookkeeper = BookKeeper(config, model_name=model_name)

    # Create models directory only if saving is enabled
    models_dir = None
    if enable_saving:
        models_dir = os.path.join(bookkeeper.run_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

    # 6. Initialize the DDPG agent with dimensions from environment
    agent = DDPGAgent(
        state_dim=state_info['total_dim'],
        action_dim=action_info['total_dim'],
        action_bounds=action_info['bounds'],
        config=config
    )

    # Initialize accumulators for per-house metrics
    rewards_per_house = np.zeros(num_houses)
    hvac_consumption_per_house = np.zeros(num_houses)
    depreciation_per_house = np.zeros(num_houses)
    penalty_per_house = np.zeros(num_houses)
    trading_profit_per_house = np.zeros(num_houses)  
    energy_bought_p2p_per_house = np.zeros(num_houses)
    selling_prices_per_house = np.zeros(num_houses)  
    
    save_interval = num_episodes // 10

    # 7. Training loop
    for episode in range(num_episodes):
        # Reset environment and get initial state
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        # Reset episode metrics
        rewards_per_house.fill(0)
        hvac_consumption_per_house.fill(0)
        depreciation_per_house.fill(0)
        penalty_per_house.fill(0)
        trading_profit_per_house.fill(0)
        energy_bought_p2p_per_house.fill(0)
        selling_prices_per_house.fill(0)  

        print(f"Episode {episode + 1}/{num_episodes}")

        while not done:
            # Convert current state to tensor (global state)
            state_tensor = torch.FloatTensor(state).to(device)  # [170]
            
            # Select actions from all independent agents
            actions = agent.select_action(state_tensor)  # [num_houses, 3]
            
            # Environment step
            next_state, reward, done, info = env.step(actions)
            
            # Store transitions for all agents using new multi-agent method
            next_state_tensor = torch.FloatTensor(next_state).to(device)  # [170]
            agent.store_transition(state_tensor, actions, next_state_tensor, reward)
            
            # Sum of rewards for episode tracking
            total_reward = sum(reward)
            
            # Optimize all agents
            agent.optimize_model()
            
            # Update accumulators
            episode_reward += total_reward
            rewards_per_house += np.array(reward)
            hvac_consumption_per_house += np.array(info['HVAC_energy_cons'])
            depreciation_per_house += np.array(info['depreciation'])
            penalty_per_house += np.array(info['penalty'])
            trading_profit_per_house += np.array(info['trading_profit'])
            energy_bought_p2p_per_house += np.array(info['energy_bought_p2p'])
            
            # Update selling prices
            if 'selling_prices' in info:
                selling_prices_per_house = np.array(info['selling_prices'])
            
            # Move to the next state
            state = next_state

        # At the end of the episode, calculate the average (per-step) penalty.
        num_steps = config.get('simulation', 'num_hours')
        avg_penalty = penalty_per_house / num_steps
        
        # Log episode metrics
        bookkeeper.log_episode(
            episode=episode,
            score=[episode_reward] * num_houses,
            rewards_per_house=rewards_per_house.tolist(),
            HVAC_energy_cons=hvac_consumption_per_house.tolist(),
            depreciation=depreciation_per_house.tolist(),
            penalty=avg_penalty.tolist(),  # Now logging the average penalty per step
            trading_profit=trading_profit_per_house.tolist(),
            energy_bought_p2p=energy_bought_p2p_per_house.tolist(),
            selling_prices=selling_prices_per_house.tolist(),
            grid_prices=info['grid_prices']
        )

        # Save model checkpoint if enabled (now saves all agents)
        if enable_saving and (episode + 1) % save_interval == 0:
            model_path = os.path.join(models_dir, f'model_checkpoint_{episode + 1}.pt')
            agent.save_checkpoint(model_path, episode + 1, episode_reward)
            bookkeeper.save_metrics()

    # Save final model (always save the final model regardless of enable_saving)
    print("Training complete. Saving final multi-agent model ...")
    final_model_path = os.path.join(bookkeeper.run_dir, f'model_final.pt')
    agent.save_checkpoint(final_model_path, num_episodes, episode_reward)

    # Save final metrics and plots
    bookkeeper.save_metrics()
    bookkeeper.plot_metrics(plot_average_only=True)
    bookkeeper.plot_selling_prices(plot_average_only=True)  

