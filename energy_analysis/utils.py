"""
Utility functions for data analysis and processing.
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


def classify_runs_by_mechanism():
    """
    Classify runs by their anti-cartel mechanism type.
    
    Returns:
        dict: Dictionary with mechanism types as keys and lists of run IDs as values
    """
    runs_by_mechanism = {
        'detection': [],
        'ceiling': [],
        'null': []
    }
    
    # Based on the specification:
    # Runs 1-21 are for detection mechanism
    # Runs 22-42 are for ceiling mechanism
    # Runs 43-63 are for null mechanism
    for run_id in range(1, 22):
        runs_by_mechanism['detection'].append(run_id)
    
    for run_id in range(22, 43):
        runs_by_mechanism['ceiling'].append(run_id)
    
    for run_id in range(43, 64):
        runs_by_mechanism['null'].append(run_id)
    
    return runs_by_mechanism


def save_figure(fig, filename, formats=None):
    """
    Save figure to multiple formats with IEEE-compliant settings.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filename (str): Base filename without extension
        formats (list, optional): List of formats to save. Defaults to ['pdf', 'tiff'].
    """
    from energy_analysis.config import PLOTS_OUTPUT_DIR
    
    if formats is None:
        formats = ['pdf', 'tiff']
    
    for fmt in formats:
        output_path = os.path.join(PLOTS_OUTPUT_DIR, f"{filename}.{fmt}")
        fig.savefig(output_path, format=fmt, dpi=600, bbox_inches='tight')
    
    return output_path