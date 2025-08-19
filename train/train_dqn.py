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
from bookkeeper import Bookkeeper


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
        self.random_seed = config.get('simulation', 'random_seed', default=42)
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
        self.num_episodes = config.get('rl_agent', 'num_episodes', default=1000)
        self.eval_interval = max(1, self.num_episodes // 10)  # Evaluate 10 times during training
        self.save_interval = max(1, self.num_episodes // 20)  # Save 20 checkpoints
        
        # Initialize utilities and bookkeeping
        self.utilities = Utilities(num_houses=self.env.num_houses)
        self.bookkeeper = Bookkeeper(run_name=self.run_name, agent_type='DQN')
        
        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon_values': [],
            'q_values': [],
            'action_entropies': [],
            'evaluation_scores': [],
            'training_times': []
        }
        
        # Create output directories
        self.output_dir = f"ml-outputs/dqn_runs/{self.run_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/models", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        
        print(f"DQN Trainer initialized:")
        print(f"  - Run name: {self.run_name}")
        print(f"  - State dimension: {self.state_dim}")
        print(f"  - Discrete actions: {self.agent.num_actions}")
        print(f"  - Training episodes: {self.num_episodes}")
        print(f"  - Random seed: {self.random_seed}")
        print(f"  - Output directory: {self.output_dir}")
    
    def _set_random_seed(self) -> None:
        """Set random seed for reproducibility."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_seed)
    
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
                self.training_metrics['evaluation_scores'].append(eval_score)
                
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_episode = episode
                    self._save_best_model()
                
                print(f"Episode {episode}: Eval Score = {eval_score:.3f} (Best: {best_eval_score:.3f} @ {best_episode})")
            
            # Save checkpoint
            if episode % self.save_interval == 0:
                self._save_checkpoint(episode)
            
            # Progress reporting
            if episode % max(1, self.num_episodes // 20) == 0:
                self._print_progress(episode, episode_metrics)
        
        # Final evaluation and cleanup
        final_eval_score = self._evaluate(num_episodes=10)
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final evaluation score: {final_eval_score:.3f}")
        print(f"Best evaluation score: {best_eval_score:.3f} (Episode {best_episode})")
        
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
            # Select action
            if training:
                discrete_action = self.agent.select_action(state, evaluation=False)
            else:
                discrete_action = self.agent.select_action(state, evaluation=True)
            
            discrete_actions_taken.append(discrete_action)
            
            # Convert to per-house actions (assuming single house for simplicity)
            # For multi-house, extend this logic
            house_actions = [discrete_action] if env.num_houses == 1 else [discrete_action] * env.num_houses
            
            # Take step
            next_state, reward, done, info = env.step(house_actions)
            
            # Store experience for training
            if training:
                self.agent.store_transition(state, discrete_action, reward, next_state, done)
            
            # Update for next iteration
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Compute action entropy for analysis
        action_counts = np.bincount(discrete_actions_taken, minlength=self.agent.num_actions)
        action_probs = action_counts / max(len(discrete_actions_taken), 1)
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'action_entropy': action_entropy,
            'discrete_actions': discrete_actions_taken,
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
                'epsilon': self.agent.epsilon,
                'learn_steps': self.agent.learn_step_counter
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
    
    def _update_training_metrics(self, episode_metrics: Dict[str, Any], episode_start_time: float) -> None:
        """Update training metrics storage."""
        self.training_metrics['episode_rewards'].append(episode_metrics['episode_reward'])
        self.training_metrics['episode_lengths'].append(episode_metrics['episode_length'])
        self.training_metrics['action_entropies'].append(episode_metrics['action_entropy'])
        self.training_metrics['training_times'].append(time.time() - episode_start_time)
        
        # Add agent-specific metrics
        agent_stats = self.agent.get_statistics()
        self.training_metrics['epsilon_values'].append(agent_stats['epsilon'])
        if agent_stats['avg_loss'] > 0:
            self.training_metrics['losses'].append(agent_stats['avg_loss'])
        if agent_stats['avg_q_value'] > 0:
            self.training_metrics['q_values'].append(agent_stats['avg_q_value'])
    
    def _print_progress(self, episode: int, episode_metrics: Dict[str, Any]) -> None:
        """Print training progress."""
        recent_rewards = self.training_metrics['episode_rewards'][-10:]
        avg_reward = np.mean(recent_rewards)
        
        agent_stats = self.agent.get_statistics()
        
        print(f"Episode {episode:4d} | "
              f"Reward: {episode_metrics['episode_reward']:8.2f} | "
              f"Avg(10): {avg_reward:8.2f} | "
              f"ε: {agent_stats['epsilon']:.3f} | "
              f"Loss: {agent_stats['avg_loss']:.4f} | "
              f"Memory: {agent_stats['memory_size']:6d}")
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = f"{self.output_dir}/models/checkpoint_episode_{episode}.pth"
        self.agent.save_checkpoint(checkpoint_path)
    
    def _save_best_model(self) -> None:
        """Save best performing model."""
        best_model_path = f"{self.output_dir}/models/best_model.pth"
        self.agent.save_checkpoint(best_model_path)
    
    def _compile_results(self, training_time: float, final_score: float, 
                        best_score: float, best_episode: int) -> Dict[str, Any]:
        """Compile final training results."""
        agent_info = self.env.get_action_space_info()
        action_stats = self.env.get_action_usage_stats()
        
        return {
            'training_info': {
                'algorithm': 'DQN',
                'run_name': self.run_name,
                'training_time': training_time,
                'total_episodes': self.num_episodes,
                'random_seed': self.random_seed,
                'final_epsilon': self.agent.epsilon
            },
            'performance': {
                'final_evaluation_score': final_score,
                'best_evaluation_score': best_score,
                'best_episode': best_episode,
                'average_episode_reward': np.mean(self.training_metrics['episode_rewards']),
                'average_episode_length': np.mean(self.training_metrics['episode_lengths'])
            },
            'learning_metrics': {
                'total_learning_steps': self.agent.learn_step_counter,
                'final_loss': self.training_metrics['losses'][-1] if self.training_metrics['losses'] else 0.0,
                'average_q_value': np.mean(self.training_metrics['q_values']) if self.training_metrics['q_values'] else 0.0
            },
            'action_space_info': agent_info,
            'action_usage': action_stats,
            'training_metrics': self.training_metrics,
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
        """Generate training visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.training_metrics['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        
        # Losses
        if self.training_metrics['losses']:
            axes[0, 2].plot(self.training_metrics['losses'])
            axes[0, 2].set_title('Training Loss')
            axes[0, 2].set_xlabel('Learning Step')
            axes[0, 2].set_ylabel('Loss')
        
        # Epsilon decay
        axes[1, 0].plot(self.training_metrics['epsilon_values'])
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        
        # Q-values
        if self.training_metrics['q_values']:
            axes[1, 1].plot(self.training_metrics['q_values'])
            axes[1, 1].set_title('Q-Values')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Max Q-Value')
        
        # Action entropy
        axes[1, 2].plot(self.training_metrics['action_entropies'])
        axes[1, 2].set_title('Action Entropy')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/training_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Evaluation scores
        if self.training_metrics['evaluation_scores']:
            plt.figure(figsize=(10, 6))
            eval_episodes = [i * self.eval_interval for i in range(len(self.training_metrics['evaluation_scores']))]
            plt.plot(eval_episodes, self.training_metrics['evaluation_scores'], 'o-')
            plt.title('Evaluation Scores During Training')
            plt.xlabel('Episode')
            plt.ylabel('Evaluation Score')
            plt.grid(True)
            plt.savefig(f"{self.output_dir}/plots/evaluation_scores.png", dpi=300, bbox_inches='tight')
            plt.close()


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