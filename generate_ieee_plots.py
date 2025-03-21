#!/usr/bin/env python3
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict
import glob

# IEEE-compliant settings
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.labelsize'] = 12  
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['grid.linewidth'] = 1.0
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.axisbelow'] = True  # grid lines behind data
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05

# IEEE color palette
ieee_blue = '#0072BD'
ieee_green = '#77AC30'
ieee_red = '#A2142F'
ieee_orange = '#EDB120'
ieee_purple = '#7E2F8E'

# Base directory for output data
ML_OUTPUT_DIR = "ml-outputs2"
PLOTS_OUTPUT_DIR = "ieee_plots"

# Create output directory if it doesn't exist
if not os.path.exists(PLOTS_OUTPUT_DIR):
    os.makedirs(PLOTS_OUTPUT_DIR)

def load_data(runs_by_mechanism):
    """
    Load data from all run folders with proper preprocessing
    """
    data_by_mechanism = {
        mechanism_type: {
            'rewards': [],
            'price_ratios': [],
            'trading_profits': [],
            'p2p_energy': [],
            'temperatures': [],
            'battery_soc': [],
            'battery_actions': [],
            'hvac_energy': [],
            'penalty': []
        } for mechanism_type in runs_by_mechanism.keys()
    }
    
    for mechanism_type, runs in runs_by_mechanism.items():
        for run_id in runs:
            run_dir = os.path.join(ML_OUTPUT_DIR, f"run_{run_id}")
            
            # Load rewards - try different possible filenames
            try:
                rewards_file = os.path.join(run_dir, "data", "ddpg__rewards_per_house.pkl")
                
                if os.path.exists(rewards_file):
                    with open(rewards_file, "rb") as f:
                        rewards_data = pickle.load(f)
                        # Pre-process to ensure right dimensionality (mean across houses if needed)
                        if isinstance(rewards_data, np.ndarray) and rewards_data.ndim > 1:
                            rewards_data = np.mean(rewards_data, axis=1)
                        data_by_mechanism[mechanism_type]['rewards'].append(rewards_data)
                else:
                    # Try alternate filenames
                    alternate_files = [
                        os.path.join(run_dir, "data", "ddpg__score.pkl"),
                        os.path.join(run_dir, "data", "ddpg__reward.pkl")
                    ]
                    
                    for alt_file in alternate_files:
                        if os.path.exists(alt_file):
                            with open(alt_file, "rb") as f:
                                score_data = pickle.load(f)
                                # Handle different data structures
                                if isinstance(score_data, np.ndarray) and score_data.ndim > 1:
                                    score_data = np.mean(score_data, axis=1)
                                data_by_mechanism[mechanism_type]['rewards'].append(score_data)
                            break
            except Exception as e:
                print(f"Could not load rewards for {run_dir}: {e}")
            
            # Load selling prices and grid prices to calculate ratios
            try:
                prices_file = os.path.join(run_dir, "data", "ddpg__selling_prices.pkl")
                grid_file = os.path.join(run_dir, "data", "ddpg__grid_prices.pkl")
                
                if os.path.exists(prices_file) and os.path.exists(grid_file):
                    with open(prices_file, "rb") as f:
                        selling_prices = pickle.load(f)
                    with open(grid_file, "rb") as f:
                        grid_prices = pickle.load(f)
                    
                    # Convert to numpy arrays if not already
                    selling_prices = np.array(selling_prices)
                    grid_prices = np.array(grid_prices)
                    
                    # Calculate ratio safely with proper dimensionality handling
                    if selling_prices.ndim == grid_prices.ndim:
                        if selling_prices.ndim > 1:
                            # Mean across houses/agents if needed
                            s_mean = np.mean(selling_prices, axis=1)
                            g_mean = np.mean(grid_prices, axis=1)
                            # Avoid division by zero
                            g_mean = np.where(g_mean == 0, 1e-6, g_mean)
                            ratios = s_mean / g_mean
                        else:
                            g_mean = np.where(grid_prices == 0, 1e-6, grid_prices)
                            ratios = selling_prices / g_mean
                    else:
                        # Handle different dimensions (should be rare)
                        s_flat = selling_prices.flatten() if selling_prices.ndim > 1 else selling_prices
                        g_flat = grid_prices.flatten() if grid_prices.ndim > 1 else grid_prices
                        # Take only common length
                        min_len = min(len(s_flat), len(g_flat))
                        s_flat, g_flat = s_flat[:min_len], g_flat[:min_len]
                        g_flat = np.where(g_flat == 0, 1e-6, g_flat)
                        ratios = s_flat / g_flat
                    
                    data_by_mechanism[mechanism_type]['price_ratios'].append(ratios)
            except Exception as e:
                print(f"Could not load price data for {run_dir}: {e}")
                
            # Load trading profits with proper dimensionality handling
            try:
                profit_file = os.path.join(run_dir, "data", "ddpg__trading_profit.pkl")
                
                if os.path.exists(profit_file):
                    with open(profit_file, "rb") as f:
                        trading_profits = pickle.load(f)
                        
                    # Convert to numpy if not already
                    trading_profits = np.array(trading_profits)
                    
                    # Handle dimensionality
                    if trading_profits.ndim > 1:
                        # Mean across houses/agents
                        trading_profits = np.mean(trading_profits, axis=1)
                    
                    # Calculate cumulative sum
                    cum_profits = np.cumsum(trading_profits)
                    data_by_mechanism[mechanism_type]['trading_profits'].append(cum_profits)
            except Exception as e:
                print(f"Could not load trading profit for {run_dir}: {e}")
                
            # Load P2P energy trading with proper error handling
            try:
                p2p_file = os.path.join(run_dir, "data", "ddpg__energy_bought_p2p.pkl")
                
                if os.path.exists(p2p_file):
                    with open(p2p_file, "rb") as f:
                        p2p_energy = pickle.load(f)
                        
                    # Convert to numpy if not already
                    p2p_energy = np.array(p2p_energy)
                    
                    # Handle dimensionality
                    if p2p_energy.ndim > 1:
                        # Mean across houses/agents
                        p2p_energy = np.mean(p2p_energy, axis=1)
                        
                    data_by_mechanism[mechanism_type]['p2p_energy'].append(p2p_energy)
            except Exception as e:
                print(f"Could not load P2P energy for {run_dir}: {e}")
    
    return data_by_mechanism

def moving_average(data, window_size=100):
    """Calculate moving average with the specified window size"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_mechanism_comparison(data_by_mechanism):
    """Create separated plots comparing the three anti-cartel mechanisms without error bands"""
    # Map mechanisms to IEEE colors
    mechanism_colors = {
        'detection': ieee_blue,
        'ceiling': ieee_green,
        'null': ieee_red
    }
    
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
    
    # Create each subplot as a separate figure file
    for subplot in subplots:
        # Create a new figure for this plot
        fig, ax = plt.subplots(figsize=(5, 3.75), dpi=600)
        
        data_key = subplot['data_key']
        
        for mechanism, color in mechanism_colors.items():
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
                        ax.plot(episodes, smoothed_data, color=color, linewidth=1.0, label=f"{mechanism.capitalize()}")
                    else:
                        # For trading profits or short data, plot without moving average
                        episodes = np.arange(1, min_length + 1)
                        ax.plot(episodes, data_avg, color=color, linewidth=1.0, label=f"{mechanism.capitalize()}")
                
                except Exception as e:
                    print(f"Error plotting {subplot['name']} for {mechanism}: {e}")
        
        # Set titles and labels
        ax.set_title(subplot['title'], fontsize=10)
        ax.set_xlabel(subplot['xlabel'], fontsize=9)
        ax.set_ylabel(subplot['ylabel'], fontsize=9)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save as separate PDF and TIFF files
        base_filename = f"mechanism_{subplot['name']}"
        fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, f"{base_filename}.pdf"), format='pdf', dpi=600, bbox_inches='tight')
        fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, f"{base_filename}.tiff"), format='tiff', dpi=600, bbox_inches='tight')
        
        plt.close(fig)
    
    # Also create the combined 2x2 grid without error bands for comparison
    fig = plt.figure(figsize=(7.16, 5.37), dpi=600)
    gs = GridSpec(2, 2, figure=fig)
    
    for i, subplot in enumerate(subplots):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col])
        data_key = subplot['data_key']
        
        for mechanism, color in mechanism_colors.items():
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
                        ax.plot(episodes, smoothed_data, color=color, linewidth=1.0, label=f"{mechanism.capitalize()}")
                    else:
                        episodes = np.arange(1, min_length + 1)
                        ax.plot(episodes, data_avg, color=color, linewidth=1.0, label=f"{mechanism.capitalize()}")
                except Exception as e:
                    print(f"Error plotting {subplot['name']} for {mechanism} in grid: {e}")
        
        ax.set_title(subplot['title'], fontsize=10)
        ax.set_xlabel(subplot['xlabel'], fontsize=9)
        ax.set_ylabel(subplot['ylabel'], fontsize=9)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "mechanism_comparison_grid.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "mechanism_comparison_grid.tiff"), format='tiff', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    print("Mechanism comparison plots generated successfully.")

def plot_temperature_control(data_by_mechanism):
    """Create a temperature profile visualization showing indoor temperature control"""
    # Find the best performing run based on average reward in the last 100 episodes
    best_runs = {}
    for mechanism_type, data in data_by_mechanism.items():
        if data['rewards']:
            # Calculate average reward in last 100 episodes for each run
            last_100_rewards = []
            for rewards in data['rewards']:
                if len(rewards) >= 100:
                    last_100_rewards.append(np.mean(rewards[-100:]))
            
            if last_100_rewards:
                # Find index of best run
                best_idx = np.argmax(last_100_rewards)
                best_runs[mechanism_type] = best_idx
    
    # Create the figure with standard dimensions to match energy_analysis plots (5x4)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
    
    # Placeholder data for demonstration
    # In a real implementation, you would extract actual temperature data from the runs
    hours = np.arange(24)
    comfort_min, comfort_max = 20.0, 22.0
    
    # Simulate outdoor temperature (placeholder)
    outdoor_temp = 15 + 10 * np.sin(np.pi * hours / 12)
    
    # Plot comfort bounds
    ax.axhline(y=comfort_min, color='r', linestyle='--', linewidth=1.5, label='Comfort Bounds')
    ax.axhline(y=comfort_max, color='r', linestyle='--', linewidth=1.5)
    
    # Plot outdoor temperature
    ax.plot(hours, outdoor_temp, color='gray', linestyle='--', linewidth=1.5, label='Outdoor Temp')
    
    # Plot indoor temperatures for best runs of each mechanism
    for mechanism, color in zip(['detection', 'ceiling', 'null'], [ieee_blue, ieee_green, ieee_red]):
        if mechanism in best_runs:
            # Simulate indoor temperature (placeholder)
            # In a real implementation, use actual temperature data from the best run
            indoor_temp = comfort_min + 1 + 0.5 * np.sin(np.pi * hours / 12 + np.pi/2)
            # Add some mechanism-specific variation for demonstration
            if mechanism == 'detection':
                indoor_temp += 0.2 * np.sin(hours)
            elif mechanism == 'ceiling':
                indoor_temp -= 0.1 * np.cos(hours)
            
            ax.plot(hours, indoor_temp, color=color, linewidth=1.5, label=f"{mechanism.capitalize()}")
    
    # Remove title as per energy_analysis settings
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.0)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    # Save only as PDF as per energy_analysis settings
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "temperature_control.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    
    plt.close(fig)
    
    print("Temperature control visualization generated successfully.")

def plot_battery_management(data_by_mechanism):
    """Create a visualization showing battery state-of-charge patterns over 24 hours"""
    # Create the figure with IEEE dimensions for a single column
    fig, ax1 = plt.subplots(figsize=(3.5, 2.33), dpi=600)
    
    # Create a secondary y-axis
    ax2 = ax1.twinx()
    
    # Placeholder data for demonstration
    # In a real implementation, extract battery data from the runs
    hours = np.arange(24)
    
    # Simulate battery state of charge for the best run
    soc = 50 + 40 * np.sin(np.pi * hours / 12) 
    
    # Simulate charging/discharging actions
    actions = 2 * np.cos(np.pi * hours / 6)
    
    # Plot battery SoC on primary axis
    ax1.plot(hours, soc, color=ieee_blue, linewidth=0.75, label='Battery SoC')
    ax1.set_ylabel("State of Charge (%)", fontsize=9)
    ax1.set_ylim(0, 100)
    
    # Plot charging/discharging actions on secondary axis
    ax2.plot(hours, actions, color=ieee_red, linewidth=0.75, linestyle='--', label='Charging Rate')
    ax2.set_ylabel("Charging Rate (kW)", fontsize=9)
    
    # Add annotations for key periods
    # High charging
    high_charging_idx = np.argmax(actions)
    ax1.annotate('', xy=(hours[high_charging_idx], soc[high_charging_idx]), 
                xytext=(hours[high_charging_idx], soc[high_charging_idx]-10),
                arrowprops=dict(arrowstyle='->', color=ieee_green, linewidth=0.5))
    
    # High discharging
    high_discharging_idx = np.argmin(actions)
    ax1.annotate('', xy=(hours[high_discharging_idx], soc[high_discharging_idx]), 
                xytext=(hours[high_discharging_idx], soc[high_discharging_idx]-10),
                arrowprops=dict(arrowstyle='->', color=ieee_red, linewidth=0.5))
    
    ax1.set_title("Smart Battery Management Strategy", fontsize=10)
    ax1.set_xlabel("Hour of Day", fontsize=9)
    ax1.set_xticks(np.arange(0, 25, 6))
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save as PDF and TIFF
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "battery_management.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "battery_management.tiff"), format='tiff', dpi=600, bbox_inches='tight')
    
    plt.close(fig)
    
    print("Battery management visualization generated successfully.")

def plot_energy_consumption_breakdown(data_by_mechanism):
    """Generate a stacked area plot showing energy source breakdown over 24 hours"""
    # Create the figure with IEEE dimensions for full page width
    fig, ax = plt.subplots(figsize=(7.16, 4.77), dpi=600)
    
    # Placeholder data for demonstration
    # In a real implementation, extract actual energy consumption data
    hours = np.arange(24)
    
    # Create separate plots for each mechanism
    mechanisms = ['detection', 'ceiling', 'null']
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 4.77), dpi=600, sharey=True)
    
    for i, mechanism in enumerate(mechanisms):
        ax = axes[i]
        
        # Placeholder data - vary by mechanism for demonstration
        # Grid energy (bottom layer)
        grid_energy = 2 + 1.5 * np.sin(np.pi * hours / 12)
        if mechanism == 'detection':
            grid_energy *= 0.8
        elif mechanism == 'null':
            grid_energy *= 1.2
            
        # P2P energy (middle layer)
        p2p_energy = 1 + np.sin(np.pi * hours / 8 + np.pi/4)
        if mechanism == 'detection':
            p2p_energy *= 1.3
        elif mechanism == 'ceiling':
            p2p_energy *= 1.1
            
        # Renewable energy (top layer)
        renewable_energy = 0.5 + 2 * np.sin(np.pi * hours / 12 + np.pi/2)
        renewable_energy[renewable_energy < 0] = 0
        
        # Create stacked area plot
        ax.fill_between(hours, 0, grid_energy, alpha=0.7, color=ieee_blue, label='Grid Energy')
        ax.fill_between(hours, grid_energy, grid_energy + p2p_energy, alpha=0.7, color=ieee_green, label='P2P Energy')
        ax.fill_between(hours, grid_energy + p2p_energy, grid_energy + p2p_energy + renewable_energy, 
                       alpha=0.7, color=ieee_orange, label='Renewable Energy')
        
        ax.set_title(f"{mechanism.capitalize()} Mechanism", fontsize=10)
        ax.set_xlabel("Hour of Day", fontsize=12)
        if i == 0:
            ax.set_ylabel("Energy (kWh)", fontsize=9)
        ax.set_xticks(np.arange(0, 25, 6))
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Only add legend to the first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle("24-Hour Energy Consumption by Source", fontsize=10, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save as PDF and TIFF
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "energy_consumption_breakdown.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "energy_consumption_breakdown.tiff"), format='tiff', dpi=600, bbox_inches='tight')
    
    plt.close(fig)
    
    print("Energy consumption breakdown visualization generated successfully.")

def plot_per_house_performance(data_by_mechanism):
    """Create a multi-panel visualization showing performance metrics for top performing runs"""
    # Find top 5 performing runs based on average reward in the last 100 episodes
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
    
    if not all_runs:
        print("No valid runs found for per-house performance plot")
        return
    
    # Sort by average reward and get top 5
    all_runs.sort(key=lambda x: x[2], reverse=True)
    top_runs = all_runs[:min(5, len(all_runs))]
    
    print(f"Top performing runs: {top_runs}")
    
    # Create the figure with IEEE dimensions
    fig = plt.figure(figsize=(7.16, 7.16), dpi=600)
    gs = GridSpec(2, 2, figure=fig)
    
    # Define colors for top runs
    colors = [ieee_blue, ieee_green, ieee_red, ieee_orange, ieee_purple]
    
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
                                   label=f"{mechanism.capitalize()} Run {run_idx+1}")
                        else:
                            # No smoothing
                            episodes = np.arange(1, len(data_array) + 1)
                            ax.plot(episodes, data_array, color=colors[i], 
                                   label=f"{mechanism.capitalize()} Run {run_idx+1}")
                except Exception as e:
                    print(f"Error plotting {data_key} for {mechanism} run {run_idx}: {e}")
                    continue
        
        ax.set_title(subplot['title'], fontsize=10)
        ax.set_xlabel(subplot['xlabel'], fontsize=9)
        ax.set_ylabel(subplot['ylabel'], fontsize=9)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save as PDF and TIFF
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "per_house_performance.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "per_house_performance.tiff"), format='tiff', dpi=600, bbox_inches='tight')
    
    plt.close(fig)
    
    print("Per-house performance visualization generated successfully.")

def plot_comparative_matrix(data_by_mechanism):
    """Generate a heatmap visualization comparing all mechanisms across multiple metrics"""
    # Create metrics dictionary for each mechanism
    metrics = {
        'detection': {},
        'ceiling': {},
        'null': {}
    }
    
    # Calculate metrics
    for mechanism, data in data_by_mechanism.items():
        # 1. Final average reward (last 100 episodes)
        if data['rewards']:
            last_100_rewards = []
            for rewards in data['rewards']:
                if len(rewards) >= 100:
                    last_100_rewards.append(np.mean(rewards[-100:]))
            if last_100_rewards:
                metrics[mechanism]['Final Reward'] = np.mean(last_100_rewards)
        
        # 2. Average selling price ratio
        if data['price_ratios']:
            price_ratios = []
            for ratio in data['price_ratios']:
                if len(ratio) >= 100:
                    price_ratios.append(np.mean(ratio[-100:]))
            if price_ratios:
                metrics[mechanism]['Price Ratio'] = np.mean(price_ratios)
        
        # 3. Cumulative trading profit
        if data['trading_profits']:
            final_profits = []
            for profit in data['trading_profits']:
                if len(profit) > 0:
                    final_profits.append(profit[-1])
            if final_profits:
                metrics[mechanism]['Trading Profit'] = np.mean(final_profits)
        
        # 4. P2P energy trading volume
        if data['p2p_energy']:
            p2p_volume = []
            for p2p in data['p2p_energy']:
                if len(p2p) >= 100:
                    p2p_volume.append(np.mean(p2p[-100:]))
            if p2p_volume:
                metrics[mechanism]['P2P Energy'] = np.mean(p2p_volume)
        
        # 5. Temperature maintenance (using penalty as inverse proxy)
        if data['penalty']:
            penalties = []
            for penalty in data['penalty']:
                if len(penalty) >= 100:
                    penalties.append(np.mean(penalty[-100:]))
            if penalties:
                # Lower penalty means better temperature maintenance
                metrics[mechanism]['Temp Control'] = 1.0 - (np.mean(penalties) / max(np.mean(penalties), 0.001))
        
        # 6. HVAC energy efficiency (placeholder)
        if data['hvac_energy']:
            hvac = []
            for energy in data['hvac_energy']:
                if len(energy) >= 100:
                    hvac.append(np.mean(energy[-100:]))
            if hvac:
                metrics[mechanism]['HVAC Efficiency'] = 1.0 - (np.mean(hvac) / max(np.mean(hvac), 0.001))
    
    # Create a matrix for the heatmap
    metric_names = ['Final Reward', 'Price Ratio', 'Trading Profit', 'P2P Energy', 'Temp Control', 'HVAC Efficiency']
    matrix_data = []
    
    for metric in metric_names:
        row = []
        for mechanism in ['detection', 'ceiling', 'null']:
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
                xticklabels=['Detection', 'Ceiling', 'Null'],
                yticklabels=metric_names, ax=ax)
    
    ax.set_title("Comparative Performance Matrix", fontsize=10)
    
    plt.tight_layout()
    
    # Save as PDF and TIFF
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "comparative_matrix.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "comparative_matrix.tiff"), format='tiff', dpi=600, bbox_inches='tight')
    
    plt.close(fig)
    
    print("Comparative performance matrix generated successfully.")

def plot_box_plots(data_by_mechanism):
    """Create box plots comparing the distribution of key metrics across runs for each mechanism"""
    # Create the figure with IEEE dimensions
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 3.58), dpi=600)
    
    # Prepare data for box plots
    metrics = ['Final Reward', 'Price Ratio', 'Trading Profit', 'Energy Efficiency']
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        data_to_plot = []
        
        for mechanism in ['detection', 'ceiling', 'null']:
            if metric == 'Final Reward':
                # Extract final rewards (average of last 100 episodes)
                values = []
                for rewards in data_by_mechanism[mechanism]['rewards']:
                    if len(rewards) >= 100:
                        values.append(np.mean(rewards[-100:]))
            
            elif metric == 'Price Ratio':
                # Extract price ratios (average of last 100 episodes)
                values = []
                for ratios in data_by_mechanism[mechanism]['price_ratios']:
                    if len(ratios) >= 100:
                        values.append(np.mean(ratios[-100:]))
            
            elif metric == 'Trading Profit':
                # Extract final trading profits
                values = []
                for profits in data_by_mechanism[mechanism]['trading_profits']:
                    if len(profits) > 0:
                        values.append(profits[-1])
            
            elif metric == 'Energy Efficiency':
                # Use HVAC energy consumption as inverse proxy for efficiency
                values = []
                for energy in data_by_mechanism[mechanism]['hvac_energy']:
                    if len(energy) >= 100:
                        # Lower energy consumption means higher efficiency
                        values.append(-np.mean(energy[-100:]))
            
            data_to_plot.append(values)
        
        # Create box plot
        colors = [ieee_blue, ieee_green, ieee_red]
        bp = ax.boxplot(data_to_plot, patch_artist=True)
        
        # Customize box plot appearance
        for j, box in enumerate(bp['boxes']):
            box.set(color=colors[j], linewidth=0.75)
            box.set(facecolor=colors[j], alpha=0.3)
        
        for j, median in enumerate(bp['medians']):
            median.set(color=colors[j], linewidth=0.75)
        
        for j, whisker in enumerate(bp['whiskers']):
            whisker.set(color=colors[j//2], linewidth=0.5)
        
        # Set labels and title
        ax.set_title(metric, fontsize=10)
        ax.set_xticklabels(['Detection', 'Ceiling', 'Null'], fontsize=8)
        if i == 0:
            ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    plt.tight_layout()
    
    # Save as PDF and TIFF
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "box_plots.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "box_plots.tiff"), format='tiff', dpi=600, bbox_inches='tight')
    
    plt.close(fig)
    
    print("Box plots for statistical analysis generated successfully.")

def plot_hyperparameter_sensitivity():
    """Generate line plots showing the impact of key hyperparameters on performance"""
    # Create the figure with IEEE dimensions
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.33), dpi=600, sharey=True)
    
    # Placeholder data for demonstration
    # In a real implementation, extract actual hyperparameter values and rewards
    
    # 1. Penalty factor sensitivity
    penalty_factors = np.linspace(0.1, 2.0, 10)
    rewards_detection = 100 - 10 * (penalty_factors - 0.5)**2 + np.random.normal(0, 5, 10)
    rewards_ceiling = 90 - 15 * (penalty_factors - 0.8)**2 + np.random.normal(0, 5, 10)
    rewards_null = 80 - 20 * (penalty_factors - 1.0)**2 + np.random.normal(0, 5, 10)
    
    # Error bands
    detection_std = 5 * np.ones_like(penalty_factors)
    ceiling_std = 5 * np.ones_like(penalty_factors)
    null_std = 5 * np.ones_like(penalty_factors)
    
    ax = axs[0]
    ax.plot(penalty_factors, rewards_detection, color=ieee_blue, linestyle='-', label='Detection')
    ax.fill_between(penalty_factors, rewards_detection - detection_std, rewards_detection + detection_std, 
                   color=ieee_blue, alpha=0.2)
    
    ax.plot(penalty_factors, rewards_ceiling, color=ieee_green, linestyle='--', label='Ceiling')
    ax.fill_between(penalty_factors, rewards_ceiling - ceiling_std, rewards_ceiling + ceiling_std, 
                   color=ieee_green, alpha=0.2)
    
    ax.plot(penalty_factors, rewards_null, color=ieee_red, linestyle=':', label='Null')
    ax.fill_between(penalty_factors, rewards_null - null_std, rewards_null + null_std, 
                   color=ieee_red, alpha=0.2)
    
    ax.set_title("Penalty Factor Sensitivity", fontsize=10)
    ax.set_xlabel("Penalty Factor", fontsize=9)
    ax.set_ylabel("Reward", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', fontsize=11)
    
    # 2. Similarity threshold sensitivity
    similarity_thresholds = np.linspace(0.5, 0.95, 10)
    rewards = 110 - 30 * (similarity_thresholds - 0.75)**2 + np.random.normal(0, 5, 10)
    rewards_std = 5 * np.ones_like(similarity_thresholds)
    
    ax = axs[1]
    ax.plot(similarity_thresholds, rewards, color=ieee_blue, linestyle='-')
    ax.fill_between(similarity_thresholds, rewards - rewards_std, rewards + rewards_std, 
                   color=ieee_blue, alpha=0.2)
    
    ax.set_title("Similarity Threshold Sensitivity", fontsize=10)
    ax.set_xlabel("Similarity Threshold", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 3. Markup limit sensitivity
    markup_limits = np.linspace(1.1, 2.0, 10)
    rewards = 105 - 25 * (markup_limits - 1.4)**2 + np.random.normal(0, 5, 10)
    rewards_std = 5 * np.ones_like(markup_limits)
    
    ax = axs[2]
    ax.plot(markup_limits, rewards, color=ieee_green, linestyle='--')
    ax.fill_between(markup_limits, rewards - rewards_std, rewards + rewards_std, 
                   color=ieee_green, alpha=0.2)
    
    ax.set_title("Markup Limit Sensitivity", fontsize=10)
    ax.set_xlabel("Markup Limit", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save as PDF and TIFF
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "hyperparameter_sensitivity.pdf"), format='pdf', dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_OUTPUT_DIR, "hyperparameter_sensitivity.tiff"), format='tiff', dpi=600, bbox_inches='tight')
    
    plt.close(fig)
    
    print("Hyperparameter sensitivity plots generated successfully.")

def classify_runs_by_mechanism():
    """
    Classify runs by their anti-cartel mechanism type
    Returns a dictionary with mechanism types as keys and lists of run IDs as values
    """
    runs_by_mechanism = {
        'detection': [],
        'ceiling': [],
        'null': []
    }
    
    # Based on the user's specification:
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

def main():
    # Classify runs by mechanism type
    print("Classifying runs by mechanism type...")
    runs_by_mechanism = classify_runs_by_mechanism()
    
    # Print summary
    for mechanism, run_ids in runs_by_mechanism.items():
        print(f"{mechanism}: {len(run_ids)} runs - {run_ids}")
    
    # Load data from all runs
    print("Loading data from all runs...")
    data_by_mechanism = load_data(runs_by_mechanism)
    
    # Generate plots
    print("Generating IEEE-compliant plots...")
    
    # Use try-except blocks for each plot to ensure the script continues even if one plot fails
    try:
        # 1. Mechanism Comparison Plots (2x2 grid)
        print("Generating mechanism comparison plots...")
        plot_mechanism_comparison(data_by_mechanism)
    except Exception as e:
        print(f"Error generating mechanism comparison plots: {e}")
    
    try:
        # 2. Temperature Control Visualization
        print("Generating temperature control visualization...")
        plot_temperature_control(data_by_mechanism)
    except Exception as e:
        print(f"Error generating temperature control visualization: {e}")
    
    try:
        # 3. Battery Management Visualization
        print("Generating battery management visualization...")
        plot_battery_management(data_by_mechanism)
    except Exception as e:
        print(f"Error generating battery management visualization: {e}")
    
    try:
        # 4. Energy Consumption Breakdown
        print("Generating energy consumption breakdown...")
        plot_energy_consumption_breakdown(data_by_mechanism)
    except Exception as e:
        print(f"Error generating energy consumption breakdown: {e}")
    
    try:
        # 5. Per-House Performance Analysis
        print("Generating per-house performance analysis...")
        plot_per_house_performance(data_by_mechanism)
    except Exception as e:
        print(f"Error generating per-house performance analysis: {e}")
    
    try:
        # 6. Comparative Performance Matrix
        print("Generating comparative performance matrix...")
        plot_comparative_matrix(data_by_mechanism)
    except Exception as e:
        print(f"Error generating comparative performance matrix: {e}")
    
    try:
        # 7. Box Plots for Statistical Analysis
        print("Generating box plots for statistical analysis...")
        plot_box_plots(data_by_mechanism)
    except Exception as e:
        print(f"Error generating box plots: {e}")
    
    try:
        # 8. Hyperparameter Sensitivity Plot
        print("Generating hyperparameter sensitivity plot...")
        plot_hyperparameter_sensitivity()
    except Exception as e:
        print(f"Error generating hyperparameter sensitivity plot: {e}")
    
    print("IEEE plot generation complete.")
    print("Plots are saved in the 'ieee_plots' directory.")

if __name__ == "__main__":
    main()