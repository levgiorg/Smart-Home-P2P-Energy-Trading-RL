import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperparameters import Config
from agents.dqn_agent import DQNAgent
from environment.dqn_environment_wrapper import DQNEnvironmentWrapper
from utilities import Utilities
from bookkeeper import BookKeeper


class DQNTrainer:
    """
    DQN Training Manager for Smart Home Energy Management.
    
    Handles training loop, evaluation, and comparison with DDPG results.
    Implements the same training protocol as DDPG for fair comparison.
    """
    
    def __init__(self, config: Config, run_name: Optional[str] = None):
        """
        Initialize DQN trainer.
        
        Args:
            config: Configuration object
            run_name: Optional name for this training run
        """
        self.config = config
        self.run_name = run_name or f"dqn_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set random seed for reproducibility
        self.random_seed = config.get('simulation', 'random_seed')
        self._set_random_seed()
        
        # Initialize environment
        self.env = DQNEnvironmentWrapper(dynamic=False, eval_mode=False)
        self.eval_env = DQNEnvironmentWrapper(dynamic=False, eval_mode=True)
        
        # Get environment dimensions
        self.state_dim = self.env.total_state_dim
        self.action_bounds = {
            'hvac_energy': config.get('environment', 'hvac_action_bounds'),
            'battery_action': config.get('environment', 'battery_action_bounds'),
            'selling_price': config.get('environment', 'selling_price_bounds')
        }
        
        # Initialize agent
        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_bounds=self.action_bounds,
            config=config
        )
        
        # Training parameters
        self.num_episodes = config.get('rl_agent', 'num_episodes')
        self.eval_interval = max(1, self.num_episodes // 10)  # Evaluate 10 times during training
        self.save_interval = max(1, self.num_episodes // 20)  # Save 20 checkpoints
        
        # Initialize utilities
        self.utilities = Utilities(num_houses=self.env.num_houses)
        
        # Training parameters  
        self.enable_saving = False  # Disable checkpoint saving as requested
        self.save_interval = max(1, self.num_episodes // 10)  # Not used when enable_saving=False
        
        # Keep evaluation scores for progress tracking
        self.evaluation_scores = []
        
        # Setup auto-incrementing run directory like DDPG bookkeeper
        self._setup_dqn_run_directory()
        
        # Initialize bookkeeping after output directory is created (matching DDPG naming)  
        self.bookkeeper = BookKeeper(config=config, model_name='dqn_', run_dir=self.output_dir)
        
        # Add all DDPG metrics for fair comparison
        # Core DDPG metrics
        self.bookkeeper.add_metric('depreciation', 'Battery Depreciation', 'Cost (€)', 'battery_depreciation.png')
        self.bookkeeper.add_metric('penalty', 'Temperature Penalty', 'Penalty (°C)', 'temperature_penalty.png')
        self.bookkeeper.add_metric('energy_bought_p2p', 'P2P Energy Trading', 'Energy (kWh)', 'energy_traded_p2p.png')
        
        # DQN-specific metrics
        self.bookkeeper.add_metric('epsilon', 'Epsilon Values', 'Epsilon', 'epsilon_plot.png')
        self.bookkeeper.add_metric('q_values', 'Q-Values', 'Q-Value', 'q_values_plot.png')
        self.bookkeeper.add_metric('action_entropy', 'Action Entropy', 'Entropy', 'action_entropy_plot.png')
        self.bookkeeper.add_metric('loss', 'Training Loss', 'Loss', 'loss_plot.png')
        
        print(f"DQN Trainer initialized:")
        print(f"  - Run name: {self.run_name}")
        print(f"  - State dimension: {self.state_dim}")
        print(f"  - Houses: {self.agent.num_houses}")
        print(f"  - Discrete actions per house: {self.agent.agents[0].num_actions}")
        print(f"  - Training episodes: {self.num_episodes}")
        print(f"  - Random seed: {self.random_seed}")
        print(f"  - Output directory: {self.output_dir}")
    
    def _set_random_seed(self) -> None:
        """Set random seed for reproducibility with proper error handling."""
        if self.random_seed is not None:
            import random
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)  # Use manual_seed_all for multi-GPU
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    def _setup_dqn_run_directory(self) -> None:
        """
        Set up auto-incrementing run directory for DQN experiments.
        Creates dqn_runs/run_X where X is the next available number.
        """
        runs_dir = 'dqn_runs'
        os.makedirs(runs_dir, exist_ok=True)
        
        # Find existing runs and get next number
        existing_runs = [d for d in os.listdir(runs_dir) 
                        if os.path.isdir(os.path.join(runs_dir, d)) 
                        and d.startswith('run_')]
        run_numbers = [int(d.split('_')[1]) for d in existing_runs 
                     if d.split('_')[1].isdigit()]
        next_run_number = max(run_numbers) + 1 if run_numbers else 1
        
        # Set up directory structure
        self.output_dir = os.path.join(runs_dir, f'run_{next_run_number}')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        
        # Update run name to match directory
        self.run_name = f"run_{next_run_number}"
        
        print(f"Created DQN run directory: {self.output_dir}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary with training results and statistics
        """
        print(f"\nStarting DQN training for {self.num_episodes} episodes...")
        start_time = time.time()
        
        best_eval_score = float('-inf')
        best_episode = 0
        
        for episode in range(self.num_episodes):
            episode_start_time = time.time()
            
            # Run training episode
            episode_metrics = self._run_episode(episode, training=True)
            
            # Log episode data to BookKeeper (matching DDPG pattern)
            self._log_episode_data(episode, episode_metrics)
            
            # Store training metrics
            self._update_training_metrics(episode_metrics, episode_start_time)
            
            # Learn from experience
            if episode > 0:  # Skip first episode to accumulate some experience
                learning_metrics = self._learn_step()
                if learning_metrics:
                    episode_metrics.update(learning_metrics)
            
            # Evaluation
            if episode % self.eval_interval == 0:
                eval_score = self._evaluate(num_episodes=3)
                self.evaluation_scores.append(eval_score)
                
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_episode = episode
                    self._save_best_model()
                
                print(f"Episode {episode}: Eval Score = {eval_score:.3f} (Best: {best_eval_score:.3f} @ {best_episode})")
            
            # Save checkpoint (matching DDPG pattern)
            if self.enable_saving and (episode + 1) % self.save_interval == 0:
                model_path = f"{self.output_dir}/models/dqn__episode_{episode + 1}.pth"
                torch.save({
                    'episode': episode + 1,
                    'q_network_state_dict': self.agent.q_network.state_dict(),
                    'target_network_state_dict': self.agent.target_network.state_dict(),
                    'optimizer_state_dict': self.agent.optimizer.state_dict(),
                    'epsilon': np.mean([agent.epsilon for agent in self.agent.agents]),
                    'memory_size': len(self.agent.memory),
                    'learn_step_counter': self.agent.learn_step_counter,
                    'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
                }, model_path)
                print(f"Saved checkpoint at episode {episode + 1}: {model_path}")
                
                # Save metrics
                self.bookkeeper.save_metrics()
            
            # Progress reporting
            if episode % max(1, self.num_episodes // 20) == 0:
                self._print_progress(episode, episode_metrics)
        
        # Final evaluation and cleanup
        final_eval_score = self._evaluate(num_episodes=10)
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final evaluation score: {final_eval_score:.3f}")
        print(f"Best evaluation score: {best_eval_score:.3f} (Episode {best_episode})")
        
        # Save final model (matching DDPG pattern exactly)
        final_model_path = f"{self.output_dir}/model_final.pt"  # Match DDPG naming
        self.agent.save_checkpoint(final_model_path)
        print(f"Saved final multi-agent model: {final_model_path}")
        
        # Save final metrics and generate plots (matching DDPG pattern)
        self.bookkeeper.save_metrics()
        self.bookkeeper.plot_metrics(plot_average_only=True)
        self.bookkeeper.plot_selling_prices(plot_average_only=True)
        
        # Save final results  
        results = self._compile_results(training_time, final_eval_score, best_eval_score, best_episode)
        self._save_final_results(results)
        
        return results
    
    def _run_episode(self, episode: int, training: bool = True) -> Dict[str, Any]:
        """
        Run a single training or evaluation episode.
        
        Args:
            episode: Episode number
            training: Whether this is a training episode
            
        Returns:
            Episode metrics dictionary
        """
        env = self.env if training else self.eval_env
        state = env.reset()
        
        episode_reward = 0.0
        episode_length = 0
        discrete_actions_taken = []
        
        done = False
        while not done:
            # Multi-agent action selection - get actions for all houses simultaneously
            if training:
                house_discrete_actions = self.agent.select_action(state, evaluation=False)
            else:
                house_discrete_actions = self.agent.select_action(state, evaluation=True)
            
            discrete_actions_taken.append(house_discrete_actions)
            
            # Take step
            next_state, reward, done, info = env.step(house_discrete_actions)
            
            # Store experience for training - multi-agent approach
            if training:
                # Handle rewards - distribute if single value, use directly if per-house
                if hasattr(reward, '__len__') and len(reward) == env.num_houses:
                    house_rewards = reward
                else:
                    house_rewards = [reward / env.num_houses] * env.num_houses
                
                # Store transitions for all houses simultaneously
                self.agent.store_transition(state, house_discrete_actions, next_state, house_rewards, done)
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Compute action entropy for analysis (multi-agent)
        # Flatten all house actions across all steps
        all_actions_flat = []
        for step_actions in discrete_actions_taken:
            if isinstance(step_actions, list):
                all_actions_flat.extend(step_actions)
            else:
                all_actions_flat.append(step_actions)
        
        # Use the discrete actions from the first agent (all have same action space size)
        num_actions = self.agent.agents[0].num_actions
        action_counts = np.bincount(all_actions_flat, minlength=num_actions)
        action_probs = action_counts / max(len(all_actions_flat), 1)
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'action_entropy': action_entropy,
            'discrete_actions': discrete_actions_taken,
            'total_actions_taken': len(all_actions_flat),
            'unique_actions_used': len(np.unique(all_actions_flat)),
            'final_info': info
        }
    
    def _learn_step(self) -> Optional[Dict[str, Any]]:
        """
        Perform learning step.
        
        Returns:
            Learning metrics or None if not enough experience
        """
        loss = self.agent.learn()
        
        if loss is not None:
            return {
                'loss': loss,
                'epsilon': np.mean([agent.epsilon for agent in self.agent.agents]),
                'learn_steps': self.agent.global_learn_steps
            }
        
        return None
    
    def _evaluate(self, num_episodes: int = 5) -> float:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Average evaluation score
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            episode_metrics = self._run_episode(-1, training=False)
            eval_rewards.append(episode_metrics['episode_reward'])
        
        return np.mean(eval_rewards)
    
    def _log_episode_data(self, episode: int, episode_metrics: Dict[str, Any]) -> None:
        """Log episode data to BookKeeper (matching DDPG logging pattern exactly)."""
        info = episode_metrics.get('final_info', {})
        
        # Validate environment data quality
        self._validate_environment_data(info, episode)
        
        # Extract data from environment info 
        num_houses = self.env.num_houses
        episode_reward = episode_metrics['episode_reward']
        
        # Extract properly accumulated environment data (from improved DQN wrapper)
        rewards_per_house = info.get('house_rewards', [episode_reward / num_houses] * num_houses)
        temperatures = info.get('house_temperatures', [20.0] * num_houses)
        hvac_consumption = info.get('hvac_consumption', [0.0] * num_houses)
        trading_profit = info.get('trading_profit', [0.0] * num_houses)
        selling_prices = info.get('selling_prices', [0.1] * num_houses)
        grid_prices = info.get('grid_prices', [0.1] * 24)
        
        # DDPG metrics - now properly accumulated from environment
        depreciation = info.get('depreciation', [0.0] * num_houses)
        penalty = info.get('penalty', [0.0] * num_houses)
        energy_bought_p2p = info.get('energy_bought_p2p', [0.0] * num_houses)
        
        # Log episode data (exactly matching DDPG pattern)
        self.bookkeeper.log_episode(
            episode=episode,
            score=[episode_reward] * num_houses,
            rewards_per_house=rewards_per_house,
            temperatures=temperatures,
            HVAC_energy_cons=hvac_consumption,
            depreciation=depreciation,
            penalty=penalty,
            trading_profit=trading_profit,
            energy_bought_p2p=energy_bought_p2p,
            selling_prices=selling_prices,
            grid_prices=grid_prices,
            # DQN-specific metrics
            epsilon=np.mean([agent.epsilon for agent in self.agent.agents]),
            action_entropy=episode_metrics['action_entropy']
        )

    def _validate_environment_data(self, info: Dict[str, Any], episode: int) -> bool:
        """Validate that environment returns expected data structure."""
        required_keys = ['house_rewards', 'house_temperatures', 'hvac_consumption', 
                        'trading_profit', 'selling_prices', 'grid_prices']
        
        missing_keys = [key for key in required_keys if key not in info]
        if missing_keys and episode % 20 == 0:  # Log every 20 episodes to avoid spam
            print(f"Episode {episode}: Missing environment data: {missing_keys}")
            return False
        return True
        
    def _update_training_metrics(self, episode_metrics: Dict[str, Any], episode_start_time: float) -> None:
        """Update training metrics using BookKeeper (matching DDPG pattern)."""
        episode_time = time.time() - episode_start_time
        
        # Validate episode length for training quality
        if episode_metrics['episode_length'] < 10:  # Expected minimum episode length
            episode = len(self.evaluation_scores) * self.eval_interval  # Approximate episode
            print(f"Warning: Very short episode {episode}: {episode_metrics['episode_length']} steps")
            print(f"  Final reward: {episode_metrics['episode_reward']:.2f}")
            
        # Training metrics are logged through bookkeeper.log_episode()
        pass
    
    def _print_progress(self, episode: int, episode_metrics: Dict[str, Any]) -> None:
        """Print training progress."""
        agent_stats = self.agent.get_statistics()
        
        print(f"Episode {episode:4d} | "
              f"Reward: {episode_metrics['episode_reward']:8.2f} | "
              f"Length: {episode_metrics['episode_length']:4d} | "
              f"ε: {agent_stats['avg_epsilon']:.3f} | "
              f"Loss: {agent_stats['avg_loss']:.4f} | "
              f"Memory: {agent_stats['total_memory_size']:6d}")
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint (maintained for compatibility)."""
        # This method is kept for compatibility but checkpoint saving is now handled in main loop
        pass
    
    def _save_best_model(self) -> None:
        """Save best performing multi-agent model."""
        best_model_path = f"{self.output_dir}/models/dqn__best.pth"
        self.agent.save_checkpoint(best_model_path)
        print(f"Saved best multi-agent model: {best_model_path}")
    
    def _compile_results(self, training_time: float, final_score: float, 
                        best_score: float, best_episode: int) -> Dict[str, Any]:
        """Compile final training results (matching DDPG pattern)."""
        agent_info = self.env.get_action_space_info()
        action_stats = self.env.get_action_usage_stats()
        agent_stats = self.agent.get_statistics()
        
        return {
            'training_info': {
                'algorithm': 'DQN',
                'run_name': self.run_name,
                'training_time': training_time,
                'total_episodes': self.num_episodes,
                'random_seed': self.random_seed,
                'final_epsilon': np.mean([agent.epsilon for agent in self.agent.agents])
            },
            'performance': {
                'final_evaluation_score': final_score,
                'best_evaluation_score': best_score,
                'best_episode': best_episode
            },
            'learning_metrics': {
                'total_learning_steps': agent_stats['total_learn_steps'],
                'final_loss': agent_stats['avg_loss'],
                'average_q_value': agent_stats['avg_q_value']
            },
            'action_space_info': agent_info,
            'action_usage': action_stats,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }
    
    def _save_final_results(self, results: Dict[str, Any]) -> None:
        """Save final results and generate plots."""
        # Save results JSON
        results_path = f"{self.output_dir}/training_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate training plots
        self._generate_training_plots()
        
        print(f"Results saved to: {results_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_training_plots(self) -> None:
        """Generate training visualization plots (now handled by BookKeeper)."""
        # Plotting is now handled by BookKeeper.plot_metrics() and BookKeeper.plot_selling_prices()
        # This method is maintained for compatibility but functionality moved to BookKeeper
        print("Training plots generated via BookKeeper plotting system")


def main():
    """Main training function."""
    # Load configuration
    config = Config()
    
    # Create trainer
    trainer = DQNTrainer(config, run_name="dqn_baseline_comparison")
    
    # Run training
    results = trainer.train()
    
    # Print summary
    print("\n" + "="*60)
    print("DQN TRAINING SUMMARY")
    print("="*60)
    print(f"Algorithm: {results['training_info']['algorithm']}")
    print(f"Training time: {results['training_info']['training_time']:.2f} seconds")
    print(f"Total episodes: {results['training_info']['total_episodes']}")
    print(f"Best evaluation score: {results['performance']['best_evaluation_score']:.3f}")
    print(f"Final evaluation score: {results['performance']['final_evaluation_score']:.3f}")
    print(f"Discrete actions used: {results['action_usage']['unique_actions_used']}")
    print(f"Action space size: {results['action_space_info']['actions_per_house']}")
    print("="*60)


if __name__ == "__main__":
    main()