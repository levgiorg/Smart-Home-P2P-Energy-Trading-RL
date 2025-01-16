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
    Train a DDPG agent in a decentralized setting (one action per house).

    Args:
        config_path (str): Path to JSON configuration file.
        model_name (str): Prefix for saving models and logging.
    """

    # 1. Load configuration
    config = Config()
    device = torch.device(config.get('general', 'device'))
    num_episodes = config.get('rl_agent', 'num_episodes')

    # 2. Initialize environment (decentralized mode only)
    env = Environment(dynamic=True)  

    # 3. Get environment info
    action_info = env.get_action_space_info()
    state_info = env.get_state_space_info()
    num_houses = state_info['num_houses']

    # 4. (Optional) Utilities
    utilities = Utilities(num_houses=num_houses)

    # 5. Initialize BookKeeper
    bookkeeper = BookKeeper(config, model_name=model_name)

    # Create models directory
    models_dir = os.path.join(bookkeeper.run_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 6. Initialize the DDPG agent
    agent = DDPGAgent(
        state_dim=state_info['total_dim'],
        action_dim=action_info['total_dim'],
        action_bounds=action_info['bounds'],
        config=config
    )

    # 7. Training loop
    for episode in tqdm(range(num_episodes), desc="Training DDPG"):
        # Reset environment and get initial state
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        # Initialize accumulators for per-house metrics
        rewards_per_house = np.zeros(num_houses)
        hvac_consumption_per_house = np.zeros(num_houses)
        depreciation_per_house = np.zeros(num_houses)
        penalty_per_house = np.zeros(num_houses)
        temperatures = []  # Track temperatures if needed

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
            
            # Move to the next state
            state = next_state

        # Log episode metrics using existing BookKeeper method
        bookkeeper.log_episode(
            episode=episode,
            score=episode_reward,
            rewards_per_house=rewards_per_house.tolist(),
            temperatures_episode=temperatures if temperatures else None,
            HVAC_energy_cons_per_house=hvac_consumption_per_house.tolist(),
            depreciation_per_house=depreciation_per_house.tolist(),
            penalty_per_house=penalty_per_house.tolist()
        )

        # Save model periodically (directly, not through BookKeeper)
        save_interval = config.get('general', 'save_interval')
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(models_dir, f'model_checkpoint_{episode + 1}.pt')
            torch.save({
                'episode': episode + 1,
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
                'episode_reward': episode_reward
            }, model_path)
            
            # Save metrics at the same time
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
        'episode_reward': episode_reward
    }, final_model_path)

    # Save final metrics and plots
    bookkeeper.save_metrics()
    bookkeeper.plot_metrics(plot_average_only=True)

if __name__ == "__main__":
    train_ddpg()