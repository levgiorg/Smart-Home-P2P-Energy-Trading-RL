"""
Parameter sensitivity visualizations for energy mechanism analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from energy_analysis.config import MECHANISMS, IEEE_COLORS, MECHANISM_DISPLAY_NAMES
from energy_analysis.utils import save_figure


def plot_hyperparameter_sensitivity(data_by_mechanism):
    """
    Generate individual plots showing the relationship between key metrics and performance.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        list: List of paths to the saved figures
    """
    # Collect metrics data from all runs
    metrics_data = _collect_metrics_data(data_by_mechanism)
    output_paths = []
    
    # Define the metrics to plot
    metrics = [
        {
            'key': 'trading_efficiency',
            'title': 'P2P Trading Impact on Reward',
            'xlabel': 'P2P Energy Volume',
            'ylabel': 'Reward'
        },
        {
            'key': 'price_competitiveness',
            'title': 'Price Ratio Impact on Reward',
            'xlabel': 'Price Ratio (Selling/Grid)',
            'ylabel': 'Reward'
        },
        {
            'key': 'energy_balance',
            'title': 'Trading Profit to Reward Ratio',
            'xlabel': 'Profit/Reward Ratio',
            'ylabel': 'Reward'
        }
    ]
    
    # Create individual plots for each metric
    for metric in metrics:
        # Create a new figure
        fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
        
        # Plot the metric
        _plot_metric_sensitivity(ax, metrics_data[metric['key']], 
                                metric['title'], metric['xlabel'], metric['ylabel'])
        
        plt.tight_layout()
        
        # Save figure
        filename = f"metric_sensitivity_{metric['key']}"
        output_path = save_figure(fig, filename)
        output_paths.append(output_path)
        
        plt.close(fig)
    
    print("Metric sensitivity plots generated successfully.")
    return output_paths


def plot_beta_grid_fee_analysis(data_by_mechanism):
    """
    Generate individual visualizations comparing the effects of beta and grid fee parameters
    on overall system performance across different mechanisms.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        list: List of paths to the saved figures
    """
    output_paths = []
    
    param_names = [
        ('beta_values', 'Beta Parameter Impact', 'Reward Scaling Factor', 'Average Reward'),
        ('grid_fees', 'Grid Fee Impact', 'Transaction Fee', 'Price Ratio'),
        ('comfort_penalties', 'Comfort Penalty Impact', 'Temperature Penalty Weight', 'Comfort Score')
    ]
    
    for param_key, title, xlabel, ylabel in param_names:
        # Create a new figure
        fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
        
        for mechanism, color in zip(MECHANISMS, [IEEE_COLORS['blue'], IEEE_COLORS['green'], IEEE_COLORS['red']]):
            # Extract parameter data if available
            if param_key in data_by_mechanism[mechanism] and data_by_mechanism[mechanism][param_key]:
                # Sort by parameter value
                sorted_data = sorted(data_by_mechanism[mechanism][param_key], key=lambda x: x[0])
                x_values = [x[0] for x in sorted_data]
                y_values = [x[1] for x in sorted_data]
                
                # Display name based on mechanism mapping
                display_name = MECHANISM_DISPLAY_NAMES[mechanism]
                
                # Plot scatter with trend line
                ax.scatter(x_values, y_values, color=color, s=30, alpha=0.6, label=display_name)
                
                # Add trend line if we have enough points
                if len(x_values) >= 3:
                    z = np.polyfit(x_values, y_values, 1)
                    p = np.poly1d(z)
                    ax.plot(np.unique(x_values), p(np.unique(x_values)), color=color, linewidth=1.0)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"parameter_impact_{param_key}"
        output_path = save_figure(fig, filename)
        output_paths.append(output_path)
        
        plt.close(fig)
    
    print("Parameter impact analysis generated successfully.")
    return output_paths
    
def _collect_metrics_data(data_by_mechanism):
    """
    Collect metrics data for sensitivity analysis.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        dict: Dictionary with metrics data for sensitivity analysis
    """
    metrics_data = {
        'trading_efficiency': {mechanism: [] for mechanism in MECHANISMS},
        'price_competitiveness': {mechanism: [] for mechanism in MECHANISMS},
        'energy_balance': {mechanism: [] for mechanism in MECHANISMS}
    }
    
    for mechanism, data in data_by_mechanism.items():
        valid_runs = min(len(data['rewards']), len(data['trading_profits']), 
                          len(data['price_ratios']), len(data['p2p_energy']))
        
        for i in range(valid_runs):
            # Only process runs with sufficient data
            if (len(data['rewards'][i]) >= 100 and 
                len(data['trading_profits'][i]) >= 100 and
                len(data['price_ratios'][i]) >= 100 and
                len(data['p2p_energy'][i]) >= 100):
                
                # Calculate final reward (last 100 episodes average)
                final_reward = np.mean(data['rewards'][i][-100:])
                
                # 1. Trading Efficiency (P2P energy volume to reward ratio)
                p2p_volume = np.mean(data['p2p_energy'][i][-100:])
                if abs(p2p_volume) > 1e-6:  # Avoid division by zero
                    metrics_data['trading_efficiency'][mechanism].append((p2p_volume, final_reward))
                
                # 2. Price Competitiveness (selling price to grid price ratio)
                price_ratio = np.mean(data['price_ratios'][i][-100:])
                metrics_data['price_competitiveness'][mechanism].append((price_ratio, final_reward))
                
                # 3. Energy Balance (trading profit to total reward ratio)
                if len(data['trading_profits'][i]) > 0:
                    trading_profit = data['trading_profits'][i][-1]  # Final cumulative profit
                    # Normalize by dividing by number of episodes
                    normalized_profit = trading_profit / len(data['trading_profits'][i])
                    if abs(final_reward) > 1e-6:  # Avoid division by zero
                        energy_balance = normalized_profit / abs(final_reward)
                        metrics_data['energy_balance'][mechanism].append((energy_balance, final_reward))
    
    return metrics_data


def _plot_metric_sensitivity(ax, data, title, xlabel, ylabel):
    """
    Create a sensitivity plot for a specific metric.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        data (dict): Dictionary with metrics data for each mechanism
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    for mechanism, color in zip(MECHANISMS, [IEEE_COLORS['blue'], IEEE_COLORS['green'], IEEE_COLORS['red']]):
        if data[mechanism]:
            # Sort by x value
            sorted_data = sorted(data[mechanism], key=lambda x: x[0])
            x_values = [x[0] for x in sorted_data]
            y_values = [x[1] for x in sorted_data]
            
            # Plot scatter with trend line
            ax.scatter(x_values, y_values, color=color, s=15, alpha=0.5, label=MECHANISM_DISPLAY_NAMES[mechanism])
            
            # Add trend line if we have enough points
            if len(x_values) >= 3:
                z = np.polyfit(x_values, y_values, 1)
                p = np.poly1d(z)
                ax.plot(x_values, p(x_values), color=color, linewidth=0.75)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=6)