"""
Utility functions for energy mechanism analysis.
"""
import numpy as np
import os


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


def classify_runs_by_mechanism(use_sampling=False, samples_per_mechanism=3):
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
    
    # Handle other kwargs
    savefig_kwargs = {k: v for k, v in kwargs.items() if k != 'format'}
    
    for fmt in formats:
        output_path = os.path.join(PLOTS_OUTPUT_DIR, f"{filename}.{fmt}")
        fig.savefig(output_path, format=fmt, dpi=600, bbox_inches='tight', **savefig_kwargs)
    
    return os.path.join(PLOTS_OUTPUT_DIR, f"{filename}.{formats[0]}")