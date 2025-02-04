import os
import json
import shutil
import pickle

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from hyperparameters import Config


class BookKeeper:
    """
    BookKeeper class for tracking and visualizing metrics in decentralized mode.
    Handles multiple houses and various performance metrics.
    """
    
    def __init__(self, config, model_name='ddpg_', run_dir=None):
        
        config = Config()
        self.num_houses = config.get('environment', 'num_houses')
        
        # Set up directories
        self._setup_directories(run_dir)
        
        # Save hyperparameters
        self._save_hyperparameters()
        
        # Define metrics configuration
        self.metrics_config = {
            'rewards_per_house': {
                'title': 'Rewards per House',
                'ylabel': 'Reward (points)',
                'filename': 'rewards_per_house_plot.png'
            },
            'temperatures': {
                'title': 'Temperature Control per House',
                'ylabel': 'Temperature (°C)',
                'filename': 'temperatures_plot.png'
            },
            'HVAC_energy_cons': {
                'title': 'HVAC Energy Consumption per House',
                'ylabel': 'Energy Consumption (kWh)',
                'filename': 'hvac_energy_consumption.png'
            },
            'depreciation': {
                'title': 'Battery Depreciation per House',
                'ylabel': 'Depreciation Cost (€)',
                'filename': 'battery_depreciation.png'
            },
            'penalty': {
                'title': 'Temperature Deviation Penalty per House',
                'ylabel': 'Penalty (Temperature Deviation (°C))',
                'filename': 'temperature_penalty.png'
            },
            'trading_profit': {
                'title': 'Trading Profit per House',
                'ylabel': 'Profit (€)',
                'filename': 'trading_profit.png'
            },
            'energy_bought_p2p': {
                'title': 'P2P Energy Trading Volume per House',
                'ylabel': 'Energy Traded (kWh)',
                'filename': 'energy_traded_p2p.png'
            },
            'grid_prices': {
                'title': 'Grid Energy Prices',
                'ylabel': 'Price (€/kWh)',
                'filename': 'grid_prices.png'
            },
            'selling_prices': {
                'title': 'Energy Selling Prices per House',
                'ylabel': 'Price (€/kWh)',
                'filename': 'selling_prices.png'
            },
            'score': {
                'title': 'Total Episode Score',
                'ylabel': 'Score (points)',
                'filename': 'score.png'
            }
        }
        # Initialize metrics storage
        self.metrics = {metric: [] for metric in self.metrics_config.keys()}
        self.model_name = model_name
        
        # Create output directories
        self.output_dir = os.path.join(self.run_dir, 'data')
        self.plots_dir = os.path.join(self.run_dir, 'plots') 
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)  # Create plots directory

    def _setup_directories(self, run_dir):
        """Set up the directory structure for logging."""
        runs_dir = 'runs'
        os.makedirs(runs_dir, exist_ok=True)
        
        if run_dir is None:
            existing_runs = [d for d in os.listdir(runs_dir) 
                           if os.path.isdir(os.path.join(runs_dir, d)) 
                           and d.startswith('run_')]
            run_numbers = [int(d.split('_')[1]) for d in existing_runs 
                         if d.split('_')[1].isdigit()]
            next_run_number = max(run_numbers) + 1 if run_numbers else 1
            self.run_dir = os.path.join(runs_dir, f'run_{next_run_number}')
        else:
            self.run_dir = run_dir
        
        os.makedirs(self.run_dir, exist_ok=True)

    def _save_hyperparameters(self):
        """Save hyperparameters configuration file."""
        source_path = 'hyperparameters/hyperparameters.json'
        dest_path = os.path.join(self.run_dir, 'hyperparameters.json')
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
        else:
            print('Warning: hyperparameters.json not found.')

    def add_metric(self, metric_name, title, ylabel, filename):
        """
        Add a new metric to track and plot.
        
        Args:
            metric_name (str): Name of the metric to track
            title (str): Title for the plot
            ylabel (str): Label for y-axis
            filename (str): Filename for saving the plot
        """
        self.metrics_config[metric_name] = {
            'title': title,
            'ylabel': ylabel,
            'filename': filename
        }
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

    def log_episode(self, episode, **kwargs):
        """
        Log metrics for an episode.
        
        Args:
            episode (int): Current episode number
            **kwargs: Metric values to log, matching metrics_config keys
        """
        for metric_name, value in kwargs.items():
            if metric_name in self.metrics:
                if value is not None:
                    self.metrics[metric_name].append(value)
                else:
                    print(f"Warning: Received None for metric '{metric_name}'. Skipping.")
            else:
                print(f"Warning: Metric '{metric_name}' not configured.")

    def save_metrics(self):
        """Save all metrics to pickle files."""
        for metric_name, data in self.metrics.items():
            if data:  # Only save if there's data
                filepath = os.path.join(self.output_dir, f'{self.model_name}_{metric_name}.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)

    def plot_metrics(self, plot_average_only=False):
        """Plot all available metrics."""
        markers = ['o', 's', '^', 'D', '*', 'p', 'x', '+', 'v', '<', '>']
        
        for metric_name, config in self.metrics_config.items():
            if not self.metrics[metric_name]:
                continue
            
            try:
                # Convert data to the correct format if it's a single value
                if isinstance(self.metrics[metric_name][0], (float, int, np.float64, np.int64)):
                    # If it's a grid price or similar single value, replicate it for each house
                    formatted_data = [[val] * self.num_houses for val in self.metrics[metric_name]]
                else:
                    formatted_data = self.metrics[metric_name]
                
                data_per_house = list(zip(*formatted_data))
                num_houses = len(data_per_house)
                color_map = cm.get_cmap('tab10', num_houses)
                
                plt.figure(figsize=(12, 8))
                for house_idx in range(num_houses):
                    house_data = np.array(data_per_house[house_idx])
                    average_data = np.cumsum(house_data) / np.arange(1, len(house_data) + 1)
                    color = color_map(house_idx)
                    marker = markers[house_idx % len(markers)]
                    
                    if not plot_average_only:
                        plt.plot(
                            range(1, len(house_data) + 1),
                            house_data,
                            label=f'House {house_idx}',
                            marker=marker,
                            linestyle='-',
                            color=color,
                            markevery=5,
                            linewidth=1
                        )
                    
                    plt.plot(
                        range(1, len(house_data) + 1),
                        average_data,
                        label=f'House {house_idx} Avg' if not plot_average_only else f'House {house_idx}',
                        linestyle='--',
                        color=color,
                        linewidth=2
                    )
                
                plt.xlabel('Episode', fontsize=14)
                plt.ylabel(config['ylabel'], fontsize=14)
                plt.title(f"{config['title']}", fontsize=16)
                plt.legend(fontsize='small', ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                
                plot_path = os.path.join(self.plots_dir, config['filename'])
                plt.savefig(plot_path, bbox_inches='tight')
                plt.close()
            
            except Exception as e:
                print(f"Warning: Could not plot metric '{metric_name}' due to error: {str(e)}")
                continue

    def plot_selling_prices(self, plot_average_only=False):
        """
        Plot selling prices for each house with moving averages and grid price as a line.
        
        Args:
            plot_average_only (bool): If True, only plot averages for each house
        """
        if not self.metrics['selling_prices'] or not self.metrics['grid_prices']:
            return
            
        plt.figure(figsize=(12, 8))
        data_per_house = list(zip(*self.metrics['selling_prices']))
        num_houses = len(data_per_house)
        color_map = cm.get_cmap('tab10', num_houses)
        
        # Plot grid prices as a line
        if self.metrics['grid_prices']:
            grid_prices = np.array(self.metrics['grid_prices'])
            if plot_average_only:
                # Calculate cumulative average for grid prices
                grid_avg = np.cumsum(grid_prices) / np.arange(1, len(grid_prices) + 1)
                plt.plot(
                    range(1, len(grid_prices) + 1),
                    grid_avg,
                    label='Grid Price Average',
                    color='red',
                    linestyle='-',
                    linewidth=2
                )
            else:
                # Plot raw grid prices and their moving average
                window_size = 10
                plt.plot(
                    range(1, len(grid_prices) + 1),
                    grid_prices,
                    label='Grid Price',
                    color='red',
                    alpha=0.5,
                    linewidth=1
                )
                # Add moving average for grid prices
                grid_ma = np.convolve(grid_prices, np.ones(window_size)/window_size, mode='valid')
                plt.plot(
                    range(window_size, len(grid_prices) + 1),
                    grid_ma,
                    label='Grid Price MA',
                    color='red',
                    linewidth=2
                )
        
        # Plot data for each house
        for house_idx in range(num_houses):
            house_data = np.array(data_per_house[house_idx])
            color = color_map(house_idx)
            
            if not plot_average_only:
                # Plot raw data
                plt.plot(
                    range(1, len(house_data) + 1),
                    house_data,
                    label=f'House {house_idx}',
                    color=color,
                    alpha=0.5,
                    linewidth=1
                )
                
                # Plot moving average
                window_size = 10
                moving_avg = np.convolve(house_data, np.ones(window_size)/window_size, mode='valid')
                plt.plot(
                    range(window_size, len(house_data) + 1),
                    moving_avg,
                    label=f'House {house_idx} MA',
                    color=color,
                    linewidth=2
                )
            else:
                # Plot only the cumulative average for each house
                cumulative_avg = np.cumsum(house_data) / np.arange(1, len(house_data) + 1)
                plt.plot(
                    range(1, len(house_data) + 1),
                    cumulative_avg,
                    label=f'House {house_idx} Average',
                    color=color,
                    linewidth=2
                )
        
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Price (per kWh)', fontsize=14)
        plt.title('Energy Prices Comparison', fontsize=16)
        plt.legend(fontsize='small', ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plot_path = os.path.join(self.plots_dir, 'selling_prices.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()