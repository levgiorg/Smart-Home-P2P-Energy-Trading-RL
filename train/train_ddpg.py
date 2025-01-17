import torch
import numpy as np
from tqdm import tqdm
import os

from agents import DDPGAgent
from environment import Environment
from hyperparameters import Config
from utilities import Utilities
from bookkeeper import BookKeeper

def train_ddpg(config_path='hyperparameters.json', model_name='ddpg_'):
    """
    Train a DDPG agent in a decentralized setting (one agent per house).
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

    # Create models directory
    models_dir = os.path.join(bookkeeper.run_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 6. Initialize the DDPG agent with correct dimensions
    agent = DDPGAgent(
        state_dim=36,  
        action_dim=9,  
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
            # Convert current state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Select an action
            action = agent.select_action(state_tensor)
            
            # Environment step
            next_state, reward, done, info = env.step(action.squeeze(0))
            
            # Sum of rewards (scalar) stored in replay buffer
            total_reward = sum(reward)
            reward_tensor = torch.FloatTensor([total_reward]).to(device)

            # Store transition
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            agent.memory.push(
                state_tensor.squeeze(0),
                action.squeeze(0),
                next_state_tensor.squeeze(0),
                reward_tensor
            )
            
            # Optimize agent
            agent.optimize_model()
            
            # Update accumulators
            episode_reward += total_reward
            rewards_per_house += np.array(reward)
            hvac_consumption_per_house += np.array(info['HVAC_energy_cons'])
            depreciation_per_house += np.array(info['depreciation'])
            penalty_per_house += np.array(info['penalty'])
            trading_profit_per_house += np.array(info['trading_profit'])
            energy_bought_p2p_per_house += np.array(info['energy_bought_p2p'])
            
            # Update selling prices (assuming it's in the info dict)
            if 'selling_prices' in info:
                selling_prices_per_house = np.array(info['selling_prices'])
            
            # Move to the next state
            state = next_state

        # Log episode metrics
        bookkeeper.log_episode(
            episode=episode,
            score=[episode_reward] * num_houses,
            rewards_per_house=rewards_per_house.tolist(),
            HVAC_energy_cons=hvac_consumption_per_house.tolist(),
            depreciation=depreciation_per_house.tolist(),
            penalty=penalty_per_house.tolist(),
            trading_profit=trading_profit_per_house.tolist(),
            energy_bought_p2p=energy_bought_p2p_per_house.tolist(),
            selling_prices=selling_prices_per_house.tolist(),
            grid_prices=info['grid_prices']
        )

        # Save model periodically
        save_interval = config.get('general', 'save_interval')
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(models_dir, f'model_checkpoint_{episode + 1}.pt')
            torch.save({
                'episode': episode + 1,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'episode_reward': episode_reward,
                'trading_metrics': {
                    'trading_profit': trading_profit_per_house.tolist(),
                    'energy_bought_p2p': energy_bought_p2p_per_house.tolist(),
                    'selling_prices': selling_prices_per_house.tolist()  
                }
            }, model_path)
            
            bookkeeper.save_metrics()

    # Save final model
    print("Training complete. Saving final model ...")
    final_model_path = os.path.join(models_dir, f'model_final.pt')
    torch.save({
        'episode': num_episodes,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'episode_reward': episode_reward,
        'trading_metrics': {
            'trading_profit': trading_profit_per_house.tolist(),
            'energy_bought_p2p': energy_bought_p2p_per_house.tolist(),
            'selling_prices': selling_prices_per_house.tolist()  
        }
    }, final_model_path)

    # Save final metrics and plots
    bookkeeper.save_metrics()
    bookkeeper.plot_metrics(plot_average_only=True)
    bookkeeper.plot_selling_prices(plot_average_only=True)  

if __name__ == "__main__":
    train_ddpg()