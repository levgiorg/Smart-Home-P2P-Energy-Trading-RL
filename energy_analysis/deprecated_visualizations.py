"""
DEPRECATED VISUALIZATION FUNCTIONS

This file contains plot generation functions that are no longer actively used.
These functions were moved here to clean up the main visualization codebase while 
preserving them for potential future reference.

Contains functions from:
- core_metrics.py (entire file)
- statistical.py (entire file) 
- sensitivity.py (entire file)
- cartel_penalty_visualization.py (entire file)
- advanced_plots.py (partial - unwanted functions only)
- device_control.py (plot_temperature_control only)

Only the following plots are actively maintained in the main codebase:
1. plot_battery_management (device_control.py)
2. plot_energy_consumption_breakdown (energy_metrics.py) 
3. plot_temperature_comfort_zone (advanced_plots.py)
4. plot_p2p_price_convergence (advanced_plots.py)
"""

# Imports for all deprecated functions
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.sankey import Sankey
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from energy_analysis.config import (
    MECHANISM_COLORS, MECHANISM_DISPLAY_NAMES, MECHANISMS, 
    IEEE_COLORS, PLOTS_OUTPUT_DIR
)
from energy_analysis.utils import moving_average, save_figure


# =============================================================================
# DEPRECATED: CORE METRICS VISUALIZATIONS (from core_metrics.py)
# =============================================================================

def plot_mechanism_comparison(data_by_mechanism):
    """
    DEPRECATED: Create separated plots comparing the three anti-cartel mechanisms without error bands.
    
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
                        ax.plot(episodes, smoothed_data, color=color, linewidth=2.0, label=f"{MECHANISM_DISPLAY_NAMES[mechanism]}")
                    else:
                        # For trading profits or short data, plot without moving average
                        episodes = np.arange(1, min_length + 1)
                        ax.plot(episodes, data_avg, color=color, linewidth=2.0, label=f"{MECHANISM_DISPLAY_NAMES[mechanism]}")
                
                except Exception as e:
                    print(f"Error plotting {subplot['name']} for {mechanism}: {e}")
        
        # Set titles and labels
        ax.set_title(subplot['title'], fontsize=13)
        ax.set_xlabel(subplot['xlabel'], fontsize=12)
        ax.set_ylabel(subplot['ylabel'], fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
        
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
                        ax.plot(episodes, smoothed_data, color=color, linewidth=2.0, label=f"{MECHANISM_DISPLAY_NAMES[mechanism]}")
                    else:
                        episodes = np.arange(1, min_length + 1)
                        ax.plot(episodes, data_avg, color=color, linewidth=2.0, label=f"{MECHANISM_DISPLAY_NAMES[mechanism]}")
                except Exception as e:
                    print(f"Error plotting {subplot['name']} for {mechanism} in grid: {e}")
        
        ax.set_title(subplot['title'], fontsize=13)
        ax.set_xlabel(subplot['xlabel'], fontsize=12)
        ax.set_ylabel(subplot['ylabel'], fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    
    plt.tight_layout()
    output_path = save_figure(fig, "mechanism_comparison_grid")
    saved_paths.append(output_path)
    
    plt.close(fig)
    
    print("DEPRECATED: Mechanism comparison plots generated successfully.")
    return saved_paths


# =============================================================================
# DEPRECATED: STATISTICAL VISUALIZATIONS (from statistical.py)
# =============================================================================

def plot_per_house_performance(data_by_mechanism):
    """
    DEPRECATED: Create a multi-panel visualization showing performance metrics for top performing runs.
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
        ax.set_xlabel(subplot['xlabel'], fontsize=12)
        ax.set_ylabel(subplot['ylabel'], fontsize=12)
        ax.legend(loc='best', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "per_house_performance")
    
    plt.close(fig)
    
    print("DEPRECATED: Per-house performance visualization generated successfully.")
    return output_path


def plot_comparative_matrix(data_by_mechanism):
    """
    DEPRECATED: Generate a heatmap visualization comparing all mechanisms across multiple metrics.
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
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "comparative_matrix")
    
    plt.close(fig)
    
    print("DEPRECATED: Comparative performance matrix generated successfully.")
    return output_path


def plot_box_plots(data_by_mechanism):
    """
    DEPRECATED: Create individual box plots comparing the distribution of key metrics across runs for each mechanism.
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
            box.set(color=colors[j], linewidth=1.5)
            box.set(facecolor=colors[j], alpha=0.3)
        
        for j, median in enumerate(bp['medians']):
            median.set(color=colors[j], linewidth=1.5)
        
        for j, whisker in enumerate(bp['whiskers']):
            whisker.set(color=colors[j//2], linewidth=1.5)
        
        # No title as requested
        ax.set_ylabel('Value', fontsize=13)
        ax.set_xticklabels(['Reward\nBased', 'Threshold\nBased', 'No Control\nMethod'], fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0, axis='y')
        
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
    
    print("DEPRECATED: Box plots for statistical analysis generated successfully.")
    return output_paths


def plot_merged_box_plots(data_by_mechanism):
    """
    DEPRECATED: Create a merged figure containing all three box plots with (a), (b), (c) labels.
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
            box.set(color=colors[j], linewidth=1.5)
            box.set(facecolor=colors[j], alpha=0.3)
        
        for j, median in enumerate(bp['medians']):
            median.set(color=colors[j], linewidth=1.5)
        
        for j, whisker in enumerate(bp['whiskers']):
            whisker.set(color=colors[j//2], linewidth=1.0)
        
        # Add subplot label (a), (b), (c) as title
        ax.set_title(f"({chr(97+i)}) {metric}", fontsize=14)
        
        # Set labels
        ax.set_ylabel('Value', fontsize=13)
        ax.set_xticklabels(['Reward\nBased', 'Threshold\nBased', 'No Control\nMethod'], fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=100, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "merged_box_plots")
    
    plt.close(fig)
    
    print("DEPRECATED: Merged box plots generated successfully.")
    return output_path


# Supporting functions for statistical plots
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


# =============================================================================
# DEPRECATED: SENSITIVITY ANALYSIS VISUALIZATIONS (from sensitivity.py)
# =============================================================================

def plot_hyperparameter_sensitivity(data_by_mechanism):
    """
    DEPRECATED: Generate individual plots showing the relationship between key metrics and performance.
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
                                None, metric['xlabel'], metric['ylabel'])
        
        plt.tight_layout()
        
        # Save figure
        filename = f"metric_sensitivity_{metric['key']}"
        output_path = save_figure(fig, filename)
        output_paths.append(output_path)
        
        plt.close(fig)
    
    # Create merged figure with all sensitivity plots
    merged_path = plot_merged_sensitivity(metrics_data, metrics)
    if merged_path:
        output_paths.append(merged_path)
    
    print("DEPRECATED: Metric sensitivity plots generated successfully.")
    return output_paths


def plot_merged_sensitivity(metrics_data, metrics):
    """
    DEPRECATED: Create a merged figure containing all three sensitivity plots with (a), (b), (c) labels.
    """
    # Create a figure with three subplots in a row
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=600)
    
    # Create each sensitivity plot in its own subplot
    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        # Plot the metric
        _plot_metric_sensitivity(ax, metrics_data[metric['key']], 
                              f"({chr(97+i)}) {metric['title']}", metric['xlabel'], metric['ylabel'])
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "merged_sensitivity_plots")
    
    plt.close(fig)
    
    print("DEPRECATED: Merged sensitivity plots generated successfully.")
    return output_path


def plot_beta_grid_fee_analysis(data_by_mechanism):
    """
    DEPRECATED: Generate individual visualizations comparing the effects of beta and grid fee parameters.
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
                    ax.plot(np.unique(x_values), p(np.unique(x_values)), color=color, linewidth=2.0)
        
        # No title as requested
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
        ax.legend(loc='best', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"parameter_impact_{param_key}"
        output_path = save_figure(fig, filename)
        output_paths.append(output_path)
        
        plt.close(fig)
    
    print("DEPRECATED: Parameter impact analysis generated successfully.")
    return output_paths


# Supporting functions for sensitivity analysis
def _collect_metrics_data(data_by_mechanism):
    """Collect metrics data for sensitivity analysis."""
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
    """Create a sensitivity plot for a specific metric."""
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
                ax.plot(x_values, p(x_values), color=color, linewidth=1.5)
    
    # Only set title if provided (not None)
    if title:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    ax.legend(loc='best', fontsize=9)


# =============================================================================
# DEPRECATED: TEMPERATURE CONTROL (from device_control.py)
# =============================================================================

def plot_temperature_control(data_by_mechanism):
    """
    DEPRECATED: Create temperature control visualization showing penalty trends and temperature violations.
    """
    # Create subplots with IEEE dimensions
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.37), dpi=600)
    
    # Define subplot layout
    subplots = [
        {
            'ax': axes[0, 0],
            'title': 'Temperature Comfort Penalties',
            'ylabel': 'Penalty Value',
            'data_key': 'penalty',
            'use_moving_avg': True
        },
        {
            'ax': axes[0, 1],
            'title': 'Temperature Violation Frequency',
            'ylabel': 'Violations per Episode',
            'data_key': 'penalty',  # We'll derive violations from penalty data
            'use_moving_avg': True,
            'process_as_violations': True
        },
        {
            'ax': axes[1, 0],
            'title': 'HVAC Energy Consumption',
            'ylabel': 'Energy (kWh)',
            'data_key': 'hvac_energy',
            'use_moving_avg': True
        },
        {
            'ax': axes[1, 1],
            'title': 'Energy Efficiency vs. Comfort',
            'ylabel': 'Comfort Score',
            'xlabel': 'HVAC Energy',
            'data_key': 'efficiency_comfort',  # Special scatter plot
            'is_scatter': True
        }
    ]
    
    for subplot in subplots:
        ax = subplot['ax']
        data_key = subplot['data_key']
        
        if data_key == 'efficiency_comfort':
            # Special scatter plot for energy efficiency vs comfort
            for mechanism, color in MECHANISM_COLORS.items():
                hvac_data = data_by_mechanism[mechanism]['hvac_energy']
                penalty_data = data_by_mechanism[mechanism]['penalty']
                
                hvac_values = []
                comfort_scores = []
                
                for i, (hvac, penalty) in enumerate(zip(hvac_data, penalty_data)):
                    if len(hvac) >= 100 and len(penalty) >= 100:
                        avg_hvac = np.mean(hvac[-100:])
                        avg_penalty = np.mean(penalty[-100:])
                        comfort_score = max(0, 1.0 - avg_penalty)  # Convert penalty to comfort score
                        
                        hvac_values.append(avg_hvac)
                        comfort_scores.append(comfort_score)
                
                if hvac_values and comfort_scores:
                    ax.scatter(hvac_values, comfort_scores, color=color, s=30, alpha=0.7, label=MECHANISM_DISPLAY_NAMES[mechanism])
        else:
            # Regular line plots
            for mechanism, color in MECHANISM_COLORS.items():
                mechanism_data = data_by_mechanism[mechanism][data_key]
                valid_data = [d for d in mechanism_data if (hasattr(d, '__len__') and len(d) > 0)]
                
                if valid_data:
                    try:
                        # Process data
                        valid_data = [np.array(d).flatten() for d in valid_data]
                        min_length = min(len(d) for d in valid_data)
                        trimmed_data = [d[:min_length] for d in valid_data]
                        data_stack = np.vstack(trimmed_data)
                        
                        if subplot.get('process_as_violations', False):
                            # Convert penalties to violation counts (binary threshold)
                            violation_threshold = 0.1
                            data_avg = np.mean(data_stack > violation_threshold, axis=0)
                        else:
                            data_avg = np.mean(data_stack, axis=0)
                        
                        if subplot.get('use_moving_avg', False) and len(data_avg) >= 100:
                            smoothed_data = moving_average(data_avg, 100)
                            episodes = np.arange(100, min_length + 1)
                            ax.plot(episodes, smoothed_data, color=color, linewidth=2.0, label=MECHANISM_DISPLAY_NAMES[mechanism])
                        else:
                            episodes = np.arange(1, min_length + 1)
                            ax.plot(episodes, data_avg, color=color, linewidth=2.0, label=MECHANISM_DISPLAY_NAMES[mechanism])
                    
                    except Exception as e:
                        print(f"Error plotting {data_key} for {mechanism}: {e}")
        
        # Set labels and formatting
        ax.set_title(subplot['title'], fontsize=13)
        if 'xlabel' in subplot:
            ax.set_xlabel(subplot['xlabel'], fontsize=12)
        else:
            ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(subplot['ylabel'], fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    
    plt.tight_layout()
    
    # Save figure
    output_path = save_figure(fig, "temperature_control")
    
    plt.close(fig)
    
    print("DEPRECATED: Temperature control visualization generated successfully.")
    return output_path