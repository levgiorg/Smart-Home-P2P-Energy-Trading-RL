"""
Statistical and comparative visualizations for energy mechanism analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from energy_analysis.config import MECHANISMS, IEEE_COLORS, MECHANISM_DISPLAY_NAMES
from energy_analysis.utils import moving_average, save_figure


def plot_per_house_performance(data_by_mechanism):
    """
    Create a multi-panel visualization showing performance metrics for top performing runs.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        str: Path to the saved figure
    """
    # Find top 5 performing runs based on average reward in the last 100 episodes
    all_runs = _find_top_performing_runs(data_by_mechanism)
    
    if not all_runs:
        print("No valid runs found for per-house performance plot")
        return None
    
    # Sort by average reward and get top 5
    all_runs.sort(key=lambda x: x[2], reverse=True)
    top_runs = all_runs[:min(5, len(all_runs))]
    
    print(f"Top performing runs: {top_runs}")
    
    # Create the figure with IEEE dimensions
    fig = plt.figure(figsize=(7.16, 7.16), dpi=600)
    gs = GridSpec(2, 2, figure=fig)
    
    # Define colors for top runs
    colors = [IEEE_COLORS['blue'], IEEE_COLORS['green'], IEEE_COLORS['red'], 
              IEEE_COLORS['orange'], IEEE_COLORS['purple']]
    
    # Create a mapping of subplots
    subplots = [
        {
            'position': (0, 0),
            'title': "Average Reward per House",
            'xlabel': "Episode",
            'ylabel': "Reward",
            'data_key': 'rewards',
            'use_smoothing': True
        },
        {
            'position': (0, 1),
            'title': "Selling Price to Grid Price Ratio",
            'xlabel': "Episode",
            'ylabel': "Price Ratio",
            'data_key': 'price_ratios',
            'use_smoothing': True
        },
        {
            'position': (1, 0),
            'title': "Cumulative Trading Profit per House",
            'xlabel': "Episode",
            'ylabel': "Cumulative Profit",
            'data_key': 'trading_profits',
            'use_smoothing': False
        },
        {
            'position': (1, 1),
            'title': "P2P Energy Trading Percentage",
            'xlabel': "Episode",
            'ylabel': "Energy Trading (%)",
            'data_key': 'p2p_energy',
            'use_smoothing': True
        }
    ]
    
    # Create each subplot
    for subplot in subplots:
        row, col = subplot['position']
        ax = fig.add_subplot(gs[row, col])
        
        data_key = subplot['data_key']
        use_smoothing = subplot['use_smoothing']
        
        for i, (mechanism, run_idx, _) in enumerate(top_runs):
            if i < len(colors):
                try:
                    # Check if data exists for this mechanism and run
                    if (len(data_by_mechanism[mechanism][data_key]) > run_idx):
                        # Get the data and ensure it's a flat numpy array
                        data = data_by_mechanism[mechanism][data_key][run_idx]
                        data_array = np.array(data)
                        if data_array.ndim > 1:
                            data_array = data_array.flatten()
                        
                        if len(data_array) >= 100 and use_smoothing:
                            # Apply smoothing
                            smoothed = moving_average(data_array, 100)
                            episodes = np.arange(100, len(data_array) + 1)
                            ax.plot(episodes, smoothed, color=colors[i], 
                                   label=f"{MECHANISM_DISPLAY_NAMES[mechanism]} Run {run_idx+1}")
                        else:
                            # No smoothing
                            episodes = np.arange(1, len(data_array) + 1)
                            ax.plot(episodes, data_array, color=colors[i], 
                                   label=f"{MECHANISM_DISPLAY_NAMES[mechanism]} Run {run_idx+1}")
                except Exception as e:
                    print(f"Error plotting {data_key} for {mechanism} run {run_idx}: {e}")
                    continue
        
        # Don't add title as requested
        ax.set_xlabel(subplot['xlabel'], fontsize=9)
        ax.set_ylabel(subplot['ylabel'], fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "per_house_performance")
    
    plt.close(fig)
    
    print("Per-house performance visualization generated successfully.")
    return output_path


def plot_comparative_matrix(data_by_mechanism):
    """
    Generate a heatmap visualization comparing all mechanisms across multiple metrics.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        str: Path to the saved figure
    """
    # Create metrics dictionary for each mechanism
    metrics = _calculate_comparison_metrics(data_by_mechanism)
    
    # Create a matrix for the heatmap
    metric_names = ['Final Reward', 'Price Ratio', 'Trading Profit', 'P2P Energy', 'Temp Control', 'HVAC Efficiency']
    matrix_data = []
    
    for metric in metric_names:
        row = []
        for mechanism in MECHANISMS:
            if metric in metrics[mechanism]:
                row.append(metrics[mechanism][metric])
            else:
                row.append(0)  # Default value if metric not available
        
        # Normalize the row to [0, 1] for fair comparison
        row_min, row_max = min(row), max(row)
        if row_max > row_min:
            row = [(x - row_min) / (row_max - row_min) for x in row]
        
        matrix_data.append(row)
    
    # Create the figure with IEEE dimensions
    fig, ax = plt.subplots(figsize=(3.5, 2.625), dpi=600)
    
    # Create heatmap
    sns.heatmap(matrix_data, annot=True, fmt='.2f', cmap='RdBu_r', cbar=True,
                xticklabels=['Reward\nBased', 'Threshold\nBased', 'No Control\nMethod'],
                yticklabels=metric_names, ax=ax)
    
    # No title as requested
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "comparative_matrix")
    
    plt.close(fig)
    
    print("Comparative performance matrix generated successfully.")
    return output_path


def plot_box_plots(data_by_mechanism):
    """
    Create individual box plots comparing the distribution of key metrics across runs for each mechanism.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        list: List of paths to the saved figures
    """
    # Prepare data for box plots
    metrics = ['Price Ratio', 'Trading Profit', 'Energy Efficiency']
    output_paths = []
    
    for metric in metrics:
        # Create a new figure for each metric
        fig, ax = plt.subplots(figsize=(5, 4.5), dpi=600)
        data_to_plot = []
        
        for mechanism in MECHANISMS:
            if metric == 'Price Ratio':
                # Extract price ratios (average of last 100 episodes)
                values = _extract_price_ratios(data_by_mechanism[mechanism])
            
            elif metric == 'Trading Profit':
                # Extract final trading profits
                values = _extract_trading_profits(data_by_mechanism[mechanism])
            
            elif metric == 'Energy Efficiency':
                # Use HVAC energy consumption as inverse proxy for efficiency
                values = _extract_energy_efficiency(data_by_mechanism[mechanism])
            
            data_to_plot.append(values)
        
        # Create box plot
        colors = [IEEE_COLORS['blue'], IEEE_COLORS['green'], IEEE_COLORS['red']]
        bp = ax.boxplot(data_to_plot, patch_artist=True)
        
        # Customize box plot appearance
        for j, box in enumerate(bp['boxes']):
            box.set(color=colors[j], linewidth=0.75)
            box.set(facecolor=colors[j], alpha=0.3)
        
        for j, median in enumerate(bp['medians']):
            median.set(color=colors[j], linewidth=0.75)
        
        for j, whisker in enumerate(bp['whiskers']):
            whisker.set(color=colors[j//2], linewidth=0.5)
        
        # No title as requested
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticklabels(['Reward\nBased', 'Threshold\nBased', 'No Control\nMethod'], fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Add more space at the bottom for x-axis labels
        plt.subplots_adjust(bottom=0.18)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"box_plot_{metric.lower().replace(' ', '_')}"
        output_path = save_figure(fig, filename)
        output_paths.append(output_path)
        
        plt.close(fig)
    
    # Also create a merged figure with all box plots
    merged_path = plot_merged_box_plots(data_by_mechanism)
    if merged_path:
        output_paths.append(merged_path)
    
    print("Box plots for statistical analysis generated successfully.")
    return output_paths


def plot_merged_box_plots(data_by_mechanism):
    """
    Create a merged figure containing all three box plots with (a), (b), (c) labels.
    
    Args:
        data_by_mechanism (dict): Dictionary containing processed data for each mechanism
        
    Returns:
        str: Path to the saved figure
    """
    # Prepare data for box plots
    metrics = ['Price Ratio', 'Trading Profit', 'Energy Efficiency']
    
    # Create a figure with three subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=600)
    
    # Create each box plot in its own subplot
    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        data_to_plot = []
        
        for mechanism in MECHANISMS:
            if metric == 'Price Ratio':
                values = _extract_price_ratios(data_by_mechanism[mechanism])
            elif metric == 'Trading Profit':
                values = _extract_trading_profits(data_by_mechanism[mechanism])
            elif metric == 'Energy Efficiency':
                values = _extract_energy_efficiency(data_by_mechanism[mechanism])
            
            data_to_plot.append(values)
        
        # Create box plot
        colors = [IEEE_COLORS['blue'], IEEE_COLORS['green'], IEEE_COLORS['red']]
        bp = ax.boxplot(data_to_plot, patch_artist=True)
        
        # Customize box plot appearance
        for j, box in enumerate(bp['boxes']):
            box.set(color=colors[j], linewidth=0.75)
            box.set(facecolor=colors[j], alpha=0.3)
        
        for j, median in enumerate(bp['medians']):
            median.set(color=colors[j], linewidth=0.75)
        
        for j, whisker in enumerate(bp['whiskers']):
            whisker.set(color=colors[j//2], linewidth=0.5)
        
        # Add subplot label (a), (b), (c) as title
        ax.set_title(f"({chr(97+i)}) {metric}", fontsize=11)
        
        # Set labels
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticklabels(['Reward\nBased', 'Threshold\nBased', 'No Control\nMethod'], fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "merged_box_plots")
    
    plt.close(fig)
    
    print("Merged box plots generated successfully.")
    return output_path


def _find_top_performing_runs(data_by_mechanism):
    """Find the top performing runs based on average reward."""
    all_runs = []
    for mechanism_type, data in data_by_mechanism.items():
        if data['rewards']:
            for i, rewards in enumerate(data['rewards']):
                # Ensure rewards is a numpy array and flatten if needed
                try:
                    rewards_array = np.array(rewards)
                    if rewards_array.ndim > 1:
                        rewards_array = rewards_array.flatten()
                    
                    if len(rewards_array) >= 100:
                        avg_reward = np.mean(rewards_array[-100:])
                        all_runs.append((mechanism_type, i, avg_reward))
                except Exception as e:
                    print(f"Error processing rewards for mechanism {mechanism_type}, run {i}: {e}")
    
    return all_runs


def _calculate_comparison_metrics(data_by_mechanism):
    """Calculate comparison metrics for the heatmap visualization."""
    metrics = {mechanism: {} for mechanism in MECHANISMS}
    
    # Calculate metrics
    for mechanism, data in data_by_mechanism.items():
        # 1. Final average reward (last 100 episodes)
        metrics[mechanism]['Final Reward'] = _calculate_final_reward(data)
        
        # 2. Average selling price ratio
        metrics[mechanism]['Price Ratio'] = _calculate_price_ratio(data)
        
        # 3. Cumulative trading profit
        metrics[mechanism]['Trading Profit'] = _calculate_trading_profit(data)
        
        # 4. P2P energy trading volume
        metrics[mechanism]['P2P Energy'] = _calculate_p2p_energy(data)
        
        # 5. Temperature maintenance (using penalty as inverse proxy)
        metrics[mechanism]['Temp Control'] = _calculate_temp_control(data)
        
        # 6. HVAC energy efficiency (placeholder)
        metrics[mechanism]['HVAC Efficiency'] = _calculate_hvac_efficiency(data)
    
    return metrics


def _calculate_final_reward(data):
    """Calculate final reward metric."""
    if data['rewards']:
        last_100_rewards = []
        for rewards in data['rewards']:
            if len(rewards) >= 100:
                last_100_rewards.append(np.mean(rewards[-100:]))
        if last_100_rewards:
            return np.mean(last_100_rewards)
    return 0.0


def _calculate_price_ratio(data):
    """Calculate price ratio metric."""
    if data['price_ratios']:
        price_ratios = []
        for ratio in data['price_ratios']:
            if len(ratio) >= 100:
                price_ratios.append(np.mean(ratio[-100:]))
        if price_ratios:
            return np.mean(price_ratios)
    return 0.0


def _calculate_trading_profit(data):
    """Calculate trading profit metric."""
    if data['trading_profits']:
        final_profits = []
        for profit in data['trading_profits']:
            if len(profit) > 0:
                final_profits.append(profit[-1])
        if final_profits:
            return np.mean(final_profits)
    return 0.0


def _calculate_p2p_energy(data):
    """Calculate P2P energy metric."""
    if data['p2p_energy']:
        p2p_values = []
        for p2p in data['p2p_energy']:
            if len(p2p) >= 100:
                p2p_values.append(np.mean(p2p[-100:]))
        if p2p_values:
            return np.mean(p2p_values)
    return 0.0


def _calculate_temp_control(data):
    """Calculate temperature control metric."""
    if data['penalty']:
        penalties = []
        for penalty in data['penalty']:
            if len(penalty) >= 100:
                penalties.append(np.mean(penalty[-100:]))
        if penalties:
            # Lower penalty means better temperature maintenance
            avg_penalty = np.mean(penalties)
            return 1.0 - (avg_penalty / max(avg_penalty, 0.001))
    return 0.0


def _calculate_hvac_efficiency(data):
    """Calculate HVAC efficiency metric."""
    if data['hvac_energy']:
        hvac = []
        for energy in data['hvac_energy']:
            if len(energy) >= 100:
                hvac.append(np.mean(energy[-100:]))
        if hvac:
            avg_energy = np.mean(hvac)
            return 1.0 - (avg_energy / max(avg_energy, 0.001))
    return 0.0


def _extract_price_ratios(mechanism_data):
    """Extract price ratios for box plots."""
    values = []
    for ratios in mechanism_data['price_ratios']:
        if len(ratios) >= 100:
            values.append(np.mean(ratios[-100:]))
    return values


def _extract_trading_profits(mechanism_data):
    """Extract trading profits for box plots."""
    values = []
    for profits in mechanism_data['trading_profits']:
        if len(profits) > 0:
            values.append(profits[-1])
    return values


def _extract_energy_efficiency(mechanism_data):
    """Extract energy efficiency for box plots."""
    values = []
    for energy in mechanism_data['hvac_energy']:
        if len(energy) >= 100:
            # Lower energy consumption means higher efficiency
            values.append(-np.mean(energy[-100:]))
    return values