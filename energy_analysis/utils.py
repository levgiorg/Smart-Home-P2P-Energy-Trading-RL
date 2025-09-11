"""
Utility functions for energy mechanism analysis and algorithm comparison.
"""
import numpy as np
import os
import pickle


def moving_average(data, window_size=100):
    """
    Calculate moving average with the specified window size.
    
    Args:
        data (array-like): Input data array
        window_size (int): Size of the moving window
        
    Returns:
        numpy.ndarray: Moving average of the input data
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def  classify_runs_by_mechanism(use_sampling=False, samples_per_mechanism=3):
    """
    Classify runs by their anti-cartel mechanism type.
    
    Args:
        use_sampling (bool): Whether to sample a subset of runs
        samples_per_mechanism (int): Number of runs to sample per mechanism
    
    Returns:
        dict: Dictionary with mechanism types as keys and lists of run IDs as values
    """
    runs_by_mechanism = {
        'detection': [],
        'ceiling': [],
        'null': []
    }
    
    # Use these run ranges based on the specification
    mechanism_ranges = {
        'detection': range(1, 22),
        'ceiling': range(22, 43),
        'null': range(43, 64)
    }
    
    # Apply sampling if requested
    if use_sampling:
        for mechanism, run_range in mechanism_ranges.items():
            # Convert range to list for random sampling
            all_runs = list(run_range)
            # Ensure we don't try to sample more than available
            sample_count = min(samples_per_mechanism, len(all_runs))
            # Randomly select runs
            sampled_runs = np.random.choice(all_runs, size=sample_count, replace=False)
            runs_by_mechanism[mechanism] = sorted(list(sampled_runs))
            print(f"Sampled {sample_count} runs for {mechanism} mechanism: {runs_by_mechanism[mechanism]}")
    else:
        # Use all runs
        for mechanism, run_range in mechanism_ranges.items():
            runs_by_mechanism[mechanism] = list(run_range)
    
    return runs_by_mechanism


def save_figure(fig, filename, formats=None, **kwargs):
    """
    Enhanced save_figure that handles additional parameters.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Base filename without extension
        formats (list, optional): List of formats to save. Defaults to ['pdf'].
        **kwargs: Additional arguments passed to savefig
    """
    from energy_analysis.config import PLOTS_OUTPUT_DIR
    
    # Only save as PDF as requested
    formats = ['pdf']
    
    # Explicitly pop 'caption' from kwargs if it exists, so it's not passed to savefig
    caption_text = kwargs.pop('caption', None) # Safely get and remove caption
    
    # Prepare savefig_kwargs without format or caption
    savefig_kwargs = {k: v for k, v in kwargs.items() if k != 'format'}

    if caption_text:
        print(f"Note: Caption for {filename} was provided but is not directly embeddable in PDF. Caption: {caption_text}")
        # Optionally, save the caption to a .txt file named after the plot
        caption_filename = os.path.join(PLOTS_OUTPUT_DIR, f"{filename}_caption.txt")
        try:
            with open(caption_filename, 'w') as f_cap:
                f_cap.write(caption_text)
            print(f"Caption for {filename} saved to: {caption_filename}")
        except Exception as e:
            print(f"Warning: Could not save caption to file for {filename}. Error: {e}")

    for fmt in formats:
        output_path = os.path.join(PLOTS_OUTPUT_DIR, f"{filename}.{fmt}")
        fig.savefig(output_path, format=fmt, dpi=600, bbox_inches='tight', **savefig_kwargs)
    
    return os.path.join(PLOTS_OUTPUT_DIR, f"{filename}.{formats[0]}")


def load_algorithm_data(runs_dir, metric_name, algorithm_name):
    """
    Load metric data from algorithm runs for comparison plots.
    
    Args:
        runs_dir (str): Directory containing run folders
        metric_name (str): Name of the metric file (without .pkl extension)
        algorithm_name (str): Algorithm name for logging
        
    Returns:
        list: List of metric arrays from all runs
    """
    metric_data = []
    
    if not os.path.exists(runs_dir):
        print(f"Warning: Directory {runs_dir} does not exist")
        return metric_data
    
    # Get all run directories
    run_dirs = [d for d in os.listdir(runs_dir) 
                if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith('run_')]
    
    for run_dir in sorted(run_dirs):
        run_path = os.path.join(runs_dir, run_dir, 'data')
        
        if not os.path.exists(run_path):
            continue
            
        # Look for the specific metric file
        filename = f"{algorithm_name}__{metric_name}.pkl"
        filepath = os.path.join(run_path, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    
                # Convert to numpy array and handle dimensionality
                data = np.array(data)
                
                # If multi-dimensional, take mean across agents/houses
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                    
                metric_data.append(data.flatten())
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
    
    return metric_data


def classify_runs_by_algorithm(ddpg_runs_dir="runs", dqn_runs_dir="dqn_runs"):
    """
    Create data structure for algorithm comparison similar to mechanism classification.
    
    Args:
        ddpg_runs_dir (str): Directory containing DDPG runs
        dqn_runs_dir (str): Directory containing DQN runs
        
    Returns:
        dict: Dictionary with algorithm types as keys and data as values
    """
    algorithm_data = {
        'ddpg': {'runs_dir': ddpg_runs_dir},
        'dqn': {'runs_dir': dqn_runs_dir}
    }
    
    return algorithm_data


# Algorithm colors matching the rewards comparison script
ALGORITHM_COLORS = {
    'ddpg': '#D46600',  # Orange
    'dqn': '#6F6F6F'    # Dark Gray
}

ALGORITHM_NAMES = {
    'ddpg': 'DDPG',
    'dqn': 'DQN'
}