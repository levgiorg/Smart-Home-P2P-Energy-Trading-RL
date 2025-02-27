import os
import json
import shutil
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from hyperparameters import Config


class BookKeeper:
    """
    BookKeeper for tracking, logging, and visualizing metrics during training.
    
    This class handles:
    - Setting up directory structure for logging
    - Saving hyperparameters configuration
    - Tracking various performance metrics
    - Generating plots and visualizations
    - Providing consistent output formatting
    
    The BookKeeper creates standardized project directories and handles 
    multi-house environments with per-house metric tracking.
    """
    
    def __init__(self, config: Config, model_name: str = 'ddpg_', run_dir: Optional[str] = None):
        """
        Initialize the BookKeeper.
        
        Args:
            config: Configuration object with hyperparameters
            model_name: Prefix for saved model and metric files
            run_dir: Optional specific run directory to use (otherwise auto-generated)
        """
        self.config = config
        self.num_houses = config.get('environment', 'num_houses')
        
        # Set up directories
        self._setup_directories(run_dir)
        
        # Save hyperparameters
        self._save_hyperparameters()
        
        # Define metrics configuration
        self._initialize_metrics_config()
        
        # Initialize storage for metrics
        self.metrics = {metric: [] for metric in self.metrics_config.keys()}
        self.model_name = model_name
        
        # Create output directories
        self.output_dir = os.path.join(self.run_dir, 'data')
        self.plots_dir = os.path.join(self.run_dir, 'plots') 
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)  # Create plots directory
        
        print(f"BookKeeper initialized with run directory: {self.run_dir}")

    def _setup_directories(self, run_dir: Optional[str]) -> None:
        """
        Set up the directory structure for logging.
        
        Args:
            run_dir: Optional specific run directory to use
        """
        runs_dir = 'runs'
        os.makedirs(runs_dir, exist_ok=True)
        
        if run_dir is None:
            # Auto-generate run directory with next available number
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
        print(f"Created run directory: {self.run_dir}")

    def _save_hyperparameters(self) -> None:
        """Save hyperparameters configuration file to run directory."""
        source_path = 'hyperparameters/hyperparameters.json'
        dest_path = os.path.join(self.run_dir, 'hyperparameters.json')
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)
            print(f"Saved hyperparameters to {dest_path}")
        else:
            print('Warning: hyperparameters.json not found.')

    def _initialize_metrics_config(self) -> None:
        """Initialize the metrics configuration with titles, labels, and filenames."""
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

    def add_metric(self, metric_name: str, title: str, ylabel: str, filename: str) -> None:
        """
        Add a new metric to track and plot.
        
        Args:
            metric_name: Name of the metric to track
            title: Title for the plot
            ylabel: Label for y-axis
            filename: Filename for saving the plot
        """
        self.metrics_config[metric_name] = {
            'title': title,
            'ylabel': ylabel,
            'filename': filename
        }
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            print(f"Added new metric: {metric_name}")

    def log_episode(self, episode: int, **kwargs) -> None:
        """
        Log metrics for an episode.
        
        Args:
            episode: Current episode number
            **kwargs: Metric values to log, matching metrics_config keys
        """
        metrics_logged = []
        
        for metric_name, value in kwargs.items():
            if metric_name in self.metrics:
                if value is not None:
                    self.metrics[metric_name].append(value)
                    metrics_logged.append(metric_name)
                else:
                    print(f"Warning: Received None for metric '{metric_name}'. Skipping.")
            else:
                print(f"Warning: Metric '{metric_name}' not configured.")
                
        if episode % 100 == 0:
            print(f"Episode {episode}: Logged metrics: {', '.join(metrics_logged)}")

    def save_metrics(self) -> None:
        """Save all collected metrics to pickle files."""
        saved_metrics = 0
        
        for metric_name, data in self.metrics.items():
            if data:  # Only save if there's data
                filepath = os.path.join(self.output_dir, f'{self.model_name}_{metric_name}.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                saved_metrics += 1
        
        print(f"Saved {saved_metrics} metrics to {self.output_dir}")

    def plot_metrics(self, plot_average_only: bool = False) -> None:
        """
        Plot all available metrics.
        
        Args:
            plot_average_only: If True, only plots moving averages not raw data
        """
        metrics_plotted = 0
        markers = ['o', 's', '^', 'D', '*', 'p', 'x', '+', 'v', '<', '>']
        
        for metric_name, config in self.metrics_config.items():
            if not self.metrics[metric_name]:
                continue
            
            try:
                # Format data and create plot
                formatted_data = self._format_metric_data(metric_name)
                self._create_metric_plot(
                    metric_name, 
                    formatted_data, 
                    config, 
                    markers, 
                    plot_average_only
                )
                metrics_plotted += 1
            
            except Exception as e:
                print(f"Warning: Could not plot metric '{metric_name}' due to error: {str(e)}")
                continue
        
        print(f"Generated {metrics_plotted} metric plots in {self.plots_dir}")
    
    def _format_metric_data(self, metric_name: str) -> List[List[float]]:
        """
        Format metric data for plotting.
        
        Args:
            metric_name: Name of the metric to format
            
        Returns:
            Formatted data as list of lists (per house)
        """
        # Convert data to the correct format if it's a single value
        if isinstance(self.metrics[metric_name][0], (float, int, np.float64, np.int64)):
            # If it's a grid price or similar single value, replicate it for each house
            formatted_data = [[val] * self.num_houses for val in self.metrics[metric_name]]
        else:
            formatted_data = self.metrics[metric_name]
        
        return formatted_data
    
    def _create_metric_plot(
        self, 
        metric_name: str, 
        formatted_data: List[List[float]], 
        config: Dict[str, str], 
        markers: List[str], 
        plot_average_only: bool
    ) -> None:
        """
        Create and save a plot for a specific metric.
        
        Args:
            metric_name: Name of the metric to plot
            formatted_data: Formatted data for plotting
            config: Metric configuration (title, labels, etc.)
            markers: List of marker styles to use
            plot_average_only: If True, only plot averages
        """
        data_per_house = list(zip(*formatted_data))
        num_houses = len(data_per_house)
        color_map = cm.get_cmap('tab10', num_houses)
        
        plt.figure(figsize=(12, 8))
        for house_idx in range(num_houses):
            house_data = np.array(data_per_house[house_idx])
            average_data = np.cumsum(house_data) / np.arange(1, len(house_data) + 1)
            color = color_map(house_idx)
            marker = markers[house_idx % len(markers)]
            
            # Plot raw data if requested
            if not plot_average_only:
                plt.plot(
                    range(1, len(house_data) + 1),
                    house_data,
                    label=f'House {house_idx}',
                    marker=marker,
                    linestyle='-',
                    color=color,
                    markevery=5,
                    linewidth=1,
                    alpha=0.5
                )
            
            # Plot average data
            plt.plot(
                range(1, len(house_data) + 1),
                average_data,
                label=f'House {house_idx} Avg' if not plot_average_only else f'House {house_idx}',
                linestyle='--',
                color=color,
                linewidth=2
            )
        
        # Set plot labels and styling
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel(config['ylabel'], fontsize=14)
        plt.title(f"{config['title']}", fontsize=16)
        plt.legend(fontsize='small', ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, config['filename'])
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    def plot_selling_prices(self, plot_average_only: bool = False) -> None:
        """
        Plot selling prices for each house with moving averages and grid price as a line.
        
        Args:
            plot_average_only: If True, only plot averages not raw data
        """
        if not self.metrics['selling_prices'] or not self.metrics['grid_prices']:
            print("Warning: Missing data for selling_prices or grid_prices")
            return
            
        plt.figure(figsize=(12, 8))
        data_per_house = list(zip(*self.metrics['selling_prices']))
        num_houses = len(data_per_house)
        color_map = cm.get_cmap('tab10', num_houses)
        
        # Plot grid prices
        self._plot_grid_prices(plot_average_only)
        
        # Plot selling prices for each house
        self._plot_house_selling_prices(data_per_house, color_map, plot_average_only)
        
        # Add plot styling
        plt.xlabel('Episode', fontsize=14)
        plt.ylabel('Price (per kWh)', fontsize=14)
        plt.title('Energy Prices Comparison', fontsize=16)
        plt.legend(fontsize='small', ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'selling_prices.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"Generated selling prices plot: {plot_path}")
    
    def _plot_grid_prices(self, plot_average_only: bool) -> None:
        """
        Plot grid prices data.
        
        Args:
            plot_average_only: If True, only plot averages not raw data
        """
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
            window_size = min(10, len(grid_prices) // 10) if len(grid_prices) > 10 else 1
            
            plt.plot(
                range(1, len(grid_prices) + 1),
                grid_prices,
                label='Grid Price',
                color='red',
                alpha=0.5,
                linewidth=1
            )
            
            # Add moving average for grid prices if we have enough data points
            if len(grid_prices) > window_size:
                grid_ma = np.convolve(grid_prices, np.ones(window_size)/window_size, mode='valid')
                plt.plot(
                    range(window_size, len(grid_prices) + 1),
                    grid_ma,
                    label='Grid Price MA',
                    color='red',
                    linewidth=2
                )
    
    def _plot_house_selling_prices(
        self, 
        data_per_house: List[List[float]], 
        color_map: cm.ScalarMappable, 
        plot_average_only: bool
    ) -> None:
        """
        Plot selling prices for each house.
        
        Args:
            data_per_house: Per-house selling price data
            color_map: Color map for consistent house colors
            plot_average_only: If True, only plot averages not raw data
        """
        for house_idx in range(len(data_per_house)):
            house_data = np.array(data_per_house[house_idx])
            color = color_map(house_idx)
            
            if not plot_average_only and len(house_data) > 0:
                # Plot raw data
                plt.plot(
                    range(1, len(house_data) + 1),
                    house_data,
                    label=f'House {house_idx}',
                    color=color,
                    alpha=0.5,
                    linewidth=1
                )
                
                # Plot moving average if we have enough data points
                window_size = min(10, len(house_data) // 10) if len(house_data) > 10 else 1
                if len(house_data) > window_size:
                    moving_avg = np.convolve(house_data, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(
                        range(window_size, len(house_data) + 1),
                        moving_avg,
                        label=f'House {house_idx} MA',
                        color=color,
                        linewidth=2
                    )
            else:
                # Plot only the cumulative average
                cumulative_avg = np.cumsum(house_data) / np.arange(1, len(house_data) + 1)
                plt.plot(
                    range(1, len(house_data) + 1),
                    cumulative_avg,
                    label=f'House {house_idx} Average',
                    color=color,
                    linewidth=2
                )