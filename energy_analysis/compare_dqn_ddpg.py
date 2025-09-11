"""
DQN vs DDPG Rewards Comparison Script

This script compares the learning performance of DQN and DDPG algorithms
by plotting their average reward curves over training episodes.

Configuration:
- Easily modify episode limit, folder paths, and colors below
- Reuses existing data loading and plotting infrastructure
"""
import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy_analysis.utils import moving_average, save_figure
from energy_analysis.config import configure_matplotlib

# =============================================================================
# CONFIGURATION - Easy to modify
# =============================================================================

# Episode limit - change this to show only first N episodes
MAX_EPISODES = 7000

# Algorithm folder paths
DQN_RUNS_DIR = "dqn_runs"
DDPG_RUNS_DIR = "runs"

# Custom colors as requested
ALGORITHM_COLORS = {
    'ddpg': '#D46600',  # Orange
    'dqn': '#6F6F6F'    # Dark Gray
}

# Display names
ALGORITHM_NAMES = {
    'ddpg': 'DDPG',
    'dqn': 'DQN'
}

# Moving average window for smoothing
SMOOTHING_WINDOW = 100

# Output settings
PLOTS_OUTPUT_DIR = "energy_analysis/ieee_plots"
OUTPUT_FILENAME = "dqn_vs_ddpg_rewards"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_algorithm_rewards(runs_dir, algorithm_name):
    """
    Load reward data from all runs in a directory for a specific algorithm.
    
    Args:
        runs_dir (str): Directory containing run folders
        algorithm_name (str): Name of the algorithm for logging
        
    Returns:
        list: List of reward arrays from all runs
    """
    rewards_data = []
    
    if not os.path.exists(runs_dir):
        print(f"Warning: Directory {runs_dir} does not exist")
        return rewards_data
    
    # Get all run directories
    run_dirs = [d for d in os.listdir(runs_dir) 
                if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith('run_')]
    
    print(f"\nLoading {algorithm_name} runs from {runs_dir}")
    print(f"Found {len(run_dirs)} run directories")
    
    for run_dir in sorted(run_dirs):
        run_path = os.path.join(runs_dir, run_dir)
        
        try:
            # Look for reward files in the data directory
            data_dir = os.path.join(run_path, 'data')
            if not os.path.exists(data_dir):
                print(f"  Warning: No data directory in {run_dir}")
                continue
            
            rewards = _load_rewards_from_run(data_dir, run_dir)
            if rewards is not None:
                # Truncate to MAX_EPISODES if needed
                if len(rewards) > MAX_EPISODES:
                    rewards = rewards[:MAX_EPISODES]
                rewards_data.append(rewards)
                print(f"  ✓ {run_dir}: {len(rewards)} episodes")
            else:
                print(f"  ✗ {run_dir}: No reward data found")
                
        except Exception as e:
            print(f"  ✗ Error loading {run_dir}: {e}")
    
    print(f"Successfully loaded {len(rewards_data)} {algorithm_name} runs")
    return rewards_data


def _load_rewards_from_run(data_dir, run_id):
    """
    Load rewards data from a single run directory.
    Tries multiple possible filenames for compatibility.
    
    Args:
        data_dir (str): Path to the data directory
        run_id (str): Run identifier for logging
        
    Returns:
        np.ndarray or None: Reward data array
    """
    # Possible reward file names (in order of preference)
    reward_filenames = [
        "ddpg__rewards_per_house.pkl",  # DDPG format
        "dqn__rewards_per_house.pkl",   # DQN format  
        "ddpg__score.pkl",              # Alternative DDPG
        "dqn__score.pkl",               # Alternative DQN
        "ddpg__reward.pkl",             # Another DDPG variant
        "dqn__reward.pkl",              # Another DQN variant
        "rewards.pkl",                  # Generic
        "scores.pkl"                    # Generic
    ]
    
    for filename in reward_filenames:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    rewards_data = pickle.load(f)
                
                # Convert to numpy array and handle dimensionality
                rewards_data = np.array(rewards_data)
                
                # If multi-dimensional (multiple agents/houses), take mean across agents
                if rewards_data.ndim > 1:
                    rewards_data = np.mean(rewards_data, axis=1)
                
                # Ensure it's 1D
                rewards_data = rewards_data.flatten()
                
                return rewards_data
                
            except Exception as e:
                print(f"    Warning: Could not load {filename}: {e}")
                continue
    
    return None


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_algorithm_rewards_comparison(ddpg_rewards, dqn_rewards):
    """
    Create a comparison plot of DQN vs DDPG reward curves.
    
    Args:
        ddpg_rewards (list): List of DDPG reward arrays
        dqn_rewards (list): List of DQN reward arrays
        
    Returns:
        str: Path to saved figure
    """
    # Create figure with IEEE formatting
    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
    
    algorithms_data = {
        'ddpg': ddpg_rewards,
        'dqn': dqn_rewards
    }
    
    for algorithm, rewards_list in algorithms_data.items():
        if not rewards_list:
            print(f"Warning: No {algorithm.upper()} data to plot")
            continue
        
        try:
            # Find common length for all runs
            min_length = min(len(rewards) for rewards in rewards_list)
            
            if min_length < SMOOTHING_WINDOW:
                print(f"Warning: {algorithm.upper()} data too short for smoothing (min_length={min_length})")
                continue
            
            # Truncate all runs to common length
            trimmed_rewards = [rewards[:min_length] for rewards in rewards_list]
            
            # Stack arrays and compute mean
            rewards_stack = np.vstack(trimmed_rewards)
            mean_rewards = np.mean(rewards_stack, axis=0)
            
            # Apply moving average smoothing
            smoothed_rewards = moving_average(mean_rewards, SMOOTHING_WINDOW)
            episodes = np.arange(SMOOTHING_WINDOW, min_length + 1)
            
            # Plot the curve
            color = ALGORITHM_COLORS[algorithm]
            label = f"{ALGORITHM_NAMES[algorithm]}"
            
            ax.plot(episodes, smoothed_rewards, color=color, linewidth=2.5, 
                   label=label, alpha=0.9)
            
            print(f"Plotted {algorithm.upper()}: {len(rewards_list)} runs, "
                  f"{min_length} episodes, smoothed with window={SMOOTHING_WINDOW}")
            
        except Exception as e:
            print(f"Error plotting {algorithm.upper()} data: {e}")
    
    # Configure plot appearance
    ax.set_xlabel('Episode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
    ax.set_title('DQN vs DDPG Learning Performance Comparison', fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    
    # Configure legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Set axis limits if we have data
    if MAX_EPISODES < 10000:
        ax.set_xlim(SMOOTHING_WINDOW, MAX_EPISODES)
    
    # Improve tick appearance
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    
    # Save figure
    if not os.path.exists(PLOTS_OUTPUT_DIR):
        os.makedirs(PLOTS_OUTPUT_DIR)
    
    output_path = save_figure(fig, OUTPUT_FILENAME)
    
    plt.close(fig)
    
    print(f"\nRewards comparison plot saved to: {output_path}")
    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to execute the DQN vs DDPG comparison.
    """
    print("=" * 70)
    print("DQN vs DDPG Rewards Comparison")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Max Episodes: {MAX_EPISODES}")
    print(f"  DDPG Runs: {DDPG_RUNS_DIR}")
    print(f"  DQN Runs: {DQN_RUNS_DIR}")
    print(f"  Smoothing Window: {SMOOTHING_WINDOW}")
    print("=" * 70)
    
    # Configure matplotlib for IEEE-compliant plots
    configure_matplotlib()
    
    start_time = time.time()
    
    # Load reward data from both algorithms
    print("\n1. Loading DDPG rewards...")
    ddpg_rewards = load_algorithm_rewards(DDPG_RUNS_DIR, "DDPG")
    
    print("\n2. Loading DQN rewards...")
    dqn_rewards = load_algorithm_rewards(DQN_RUNS_DIR, "DQN")
    
    # Check if we have data for both algorithms
    if not ddpg_rewards and not dqn_rewards:
        print("\nERROR: No reward data found for either algorithm!")
        print("Please check your directory paths and run data.")
        return 1
    
    if not ddpg_rewards:
        print(f"\nWarning: No DDPG data found in {DDPG_RUNS_DIR}")
    
    if not dqn_rewards:
        print(f"\nWarning: No DQN data found in {DQN_RUNS_DIR}")
    
    # Generate comparison plot
    print("\n3. Generating rewards comparison plot...")
    output_path = plot_algorithm_rewards_comparison(ddpg_rewards, dqn_rewards)
    
    elapsed_time = time.time() - start_time
    print(f"\n" + "=" * 70)
    print(f"Comparison complete in {elapsed_time:.2f} seconds")
    print(f"Output saved to: {output_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)