"""
Core performance metrics visualization for energy mechanism analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from energy_analysis.config import MECHANISM_COLORS, MECHANISM_DISPLAY_NAMES
from energy_analysis.utils import moving_average, save_figure


def plot_mechanism_comparison(data_by_mechanism):
    """
    Create separated plots comparing the three anti-cartel mechanisms without error bands.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        list: List of saved figure paths
    """
    # Define subplots to generate
    subplots = [
        {
            'name': 'reward',
            'title': 'Average Reward per Episode',
            'xlabel': 'Episode',
            'ylabel': 'Reward',
            'data_key': 'rewards',
            'y_scale': 1.0  # Scale factor for y values
        },
        {
            'name': 'price_ratio',
            'title': 'Price Ratio (Selling/Grid)',
            'xlabel': 'Episode',
            'ylabel': 'Price Ratio',
            'data_key': 'price_ratios',
            'y_scale': 1.0
        },
        {
            'name': 'trading_profit',
            'title': 'Cumulative Trading Profit',
            'xlabel': 'Episode',
            'ylabel': 'Cumulative Profit',
            'data_key': 'trading_profits',
            'y_scale': 1.0
        },
        {
            'name': 'p2p_energy',
            'title': 'P2P Energy Trading Volume',
            'xlabel': 'Episode',
            'ylabel': 'Energy (kWh)',
            'data_key': 'p2p_energy',
            'y_scale': 1.0
        }
    ]
    
    saved_paths = []
    
    # Create each subplot as a separate figure file
    for subplot in subplots:
        # Create a new figure for this plot
        fig, ax = plt.subplots(figsize=(5, 3.75), dpi=600)
        
        data_key = subplot['data_key']
        
        for mechanism, color in MECHANISM_COLORS.items():
            # Check if we have valid data for this mechanism
            valid_data = [d for d in data_by_mechanism[mechanism][data_key] if (hasattr(d, '__len__') and len(d) > 0)]
            if valid_data:
                try:
                    # Ensure all arrays are numpy arrays
                    valid_data = [np.array(d).flatten() for d in valid_data]
                    
                    # Find shortest common length for alignment
                    min_length = min(len(d) for d in valid_data)
                    trimmed_data = [d[:min_length] for d in valid_data]
                    
                    # Stack arrays for statistics computation
                    data_stack = np.vstack(trimmed_data)
                    data_avg = np.mean(data_stack, axis=0)
                    
                    # Scale if needed
                    data_avg = data_avg * subplot['y_scale']
                    
                    # Calculate moving average for smoother visualization if needed
                    if data_key != 'trading_profits' and len(data_avg) >= 100:  # No smoothing for cumulative profits
                        smoothed_data = moving_average(data_avg, 100)
                        episodes = np.arange(100, min_length + 1)
                        
                        # Plot line only without error band
                        ax.plot(episodes, smoothed_data, color=color, linewidth=1.0, label=f"{MECHANISM_DISPLAY_NAMES[mechanism]}")
                    else:
                        # For trading profits or short data, plot without moving average
                        episodes = np.arange(1, min_length + 1)
                        ax.plot(episodes, data_avg, color=color, linewidth=1.0, label=f"{MECHANISM_DISPLAY_NAMES[mechanism]}")
                
                except Exception as e:
                    print(f"Error plotting {subplot['name']} for {mechanism}: {e}")
        
        # Set titles and labels
        ax.set_title(subplot['title'], fontsize=10)
        ax.set_xlabel(subplot['xlabel'], fontsize=9)
        ax.set_ylabel(subplot['ylabel'], fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save figure
        base_filename = f"mechanism_{subplot['name']}"
        output_path = save_figure(fig, base_filename)
        saved_paths.append(output_path)
        
        plt.close(fig)
    
    # Also create the combined 2x2 grid without error bands for comparison
    fig = plt.figure(figsize=(7.16, 5.37), dpi=600)
    gs = GridSpec(2, 2, figure=fig)
    
    for i, subplot in enumerate(subplots):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        data_key = subplot['data_key']
        
        for mechanism, color in MECHANISM_COLORS.items():
            valid_data = [d for d in data_by_mechanism[mechanism][data_key] if (hasattr(d, '__len__') and len(d) > 0)]
            if valid_data:
                try:
                    valid_data = [np.array(d).flatten() for d in valid_data]
                    min_length = min(len(d) for d in valid_data)
                    trimmed_data = [d[:min_length] for d in valid_data]
                    data_stack = np.vstack(trimmed_data)
                    data_avg = np.mean(data_stack, axis=0) * subplot['y_scale']
                    
                    if data_key != 'trading_profits' and len(data_avg) >= 100:
                        smoothed_data = moving_average(data_avg, 100)
                        episodes = np.arange(100, min_length + 1)
                        ax.plot(episodes, smoothed_data, color=color, linewidth=1.0, label=f"{MECHANISM_DISPLAY_NAMES[mechanism]}")
                    else:
                        episodes = np.arange(1, min_length + 1)
                        ax.plot(episodes, data_avg, color=color, linewidth=1.0, label=f"{MECHANISM_DISPLAY_NAMES[mechanism]}")
                except Exception as e:
                    print(f"Error plotting {subplot['name']} for {mechanism} in grid: {e}")
        
        ax.set_title(subplot['title'], fontsize=10)
        ax.set_xlabel(subplot['xlabel'], fontsize=9)
        ax.set_ylabel(subplot['ylabel'], fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    output_path = save_figure(fig, "mechanism_comparison_grid")
    saved_paths.append(output_path)
    
    plt.close(fig)
    
    print("Mechanism comparison plots generated successfully.")
    return saved_paths