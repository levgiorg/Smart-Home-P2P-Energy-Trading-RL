import os
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class AntiCartelAnalyzer:
    def __init__(self, base_path="ml-outputs"):
        # Convert base_path to Path object
        self.base_path = Path(base_path)
        # Get all run directories (absolute paths)
        self.runs = sorted([str(self.base_path / d) for d in os.listdir(self.base_path) 
                          if d.startswith('run')])
        print(f"Found {len(self.runs)} total runs")
        
        # Group runs by mechanism type based on hyperparameters
        self.mechanism_groups = self._group_runs_by_mechanism()
        
        # Print summary of found runs
        for mech, runs in self.mechanism_groups.items():
            print(f"{mech}: {len(runs)} runs")
            

    def _get_mechanism_type(self, run_path: str) -> str:
        """Read mechanism type from hyperparameters.json"""
        try:
            hyperparams_path = Path(run_path) / 'hyperparameters.json'
            with open(hyperparams_path, 'r') as f:
                hyperparams = json.load(f)
            
            # Check if anti_cartel mechanism is defined
            if 'anti_cartel' in hyperparams:
                mech_type = hyperparams['anti_cartel'].get('mechanism_type')
                # Handle the three possible cases
                if mech_type == "detection":
                    return "detection"
                elif mech_type == "ceiling":
                    return "ceiling"
                else:  # This will catch both null and any other unexpected values
                    return "null"
            return "null"  # Default to null if no anti_cartel section
        except Exception as e:
            print(f"Error reading mechanism type for run {run_path}: {e}")
            return 'unknown'

    def _group_runs_by_mechanism(self) -> Dict[str, List[str]]:
        """Group runs by their anti-cartel mechanism type"""
        groups = {'detection': [], 'ceiling': [], 'null': [], 'unknown': []}
        
        for run_path in self.runs:
            mech_type = self._get_mechanism_type(run_path)
            if mech_type in groups:
                groups[mech_type].append(run_path)
            else:
                print(f"Warning: Unknown mechanism type {mech_type} in {run_path}")
                groups['unknown'].append(run_path)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def load_run_data(self, run_path: str) -> Tuple[dict, dict]:
        """Load data for a specific run"""
        try:
            run_path = Path(run_path)
            
            # Load hyperparameters
            with open(run_path / 'hyperparameters.json', 'r') as f:
                hyperparams = json.load(f)
            
            data = {}
            data_path = run_path / 'data'
            
            for file in os.listdir(data_path):
                if file.endswith('.pkl'):
                    metric_name = file.replace('ddpg__', '').replace('.pkl', '')
                    with open(data_path / file, 'rb') as f:
                        data[metric_name] = pickle.load(f)
            
            return hyperparams, data
            
        except Exception as e:
            print(f"Error loading run {run_path}: {str(e)}")
            return None, None

    def calculate_mechanism_metrics(self, group_runs: List[str]) -> Dict:
        """Calculate aggregated metrics for a group of runs"""
        metrics = {
            'rewards': [],
            'price_ratios': [],
            'trading_profits': [],
            'p2p_energy': [],
            'penalties': []
        }
        
        valid_runs = 0
        for run_path in group_runs:
            hyperparams, data = self.load_run_data(run_path)
            if data is None:
                print(f"Skipping run {run_path} due to loading error")
                continue
                
            try:
                # Extract basic metrics
                if 'rewards_per_house' in data:
                    rewards = np.mean(data['rewards_per_house'], axis=1)
                    metrics['rewards'].append(rewards)
                
                if all(key in data for key in ['selling_prices', 'grid_prices']):
                    price_ratios = np.array(data['selling_prices']) / np.array(data['grid_prices'])[:, None]
                    metrics['price_ratios'].append(np.mean(price_ratios, axis=1))
                
                if 'trading_profit' in data:
                    trading_profits = np.mean(data['trading_profit'], axis=1)
                    metrics['trading_profits'].append(trading_profits)
                
                if 'energy_bought_p2p' in data:
                    p2p_energy = np.mean(data['energy_bought_p2p'], axis=1)
                    metrics['p2p_energy'].append(p2p_energy)
                
                if 'anti_cartel_penalties' in data:
                    penalties = np.array(data['anti_cartel_penalties'])
                    metrics['penalties'].append(penalties)
                
                valid_runs += 1
                
            except Exception as e:
                print(f"Error processing metrics for run {run_path}: {str(e)}")
                continue
        
        print(f"Successfully processed {valid_runs} runs")
        
        # Convert lists to arrays if they contain data
        processed_metrics = {}
        for key, values in metrics.items():
            if values:
                try:
                    processed_metrics[key] = np.array(values)
                except Exception as e:
                    print(f"Error converting {key} to array: {str(e)}")
                    processed_metrics[key] = np.array([])
            else:
                processed_metrics[key] = np.array([])
        
        return processed_metrics

    def compare_mechanisms(self):
        """Compare mechanisms with improved visualization - 2x2 grid"""
        mechanism_data = {
            name: self.calculate_mechanism_metrics(runs)
            for name, runs in self.mechanism_groups.items()
            if name != 'unknown'  # Skip unknown mechanism types
        }
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        fig, axs = plt.subplots(2, 2, figsize=(15, 13))  # Changed from 3x2 to 2x2
        colors = {
            'detection': '#2E86C1',  # Blue
            'ceiling': '#28B463',    # Green
            'null': '#E74C3C'        # Red
        }
        
        labels = {
            'detection': 'Detection Mechanism',
            'ceiling': 'Price Ceiling',
            'null': 'No Mechanism'
        }
        
        # Common plotting parameters
        window = 100  # Window size for moving average
        alpha = 0.15  # Transparency for confidence intervals
        
        # Plot average rewards
        ax = axs[0, 0]
        for name, data in mechanism_data.items():
            if 'rewards' in data and len(data['rewards']) > 0:
                mean_rewards = np.mean(data['rewards'], axis=0)
                std_rewards = np.std(data['rewards'], axis=0)
                episodes = np.arange(len(mean_rewards))
                
                # Plot moving average
                ma_rewards = pd.Series(mean_rewards).rolling(window=window).mean()
                ax.plot(episodes, ma_rewards, label=labels[name], 
                    color=colors[name], linewidth=2)
                ax.fill_between(episodes, 
                            ma_rewards - std_rewards,
                            ma_rewards + std_rewards,
                            alpha=alpha, color=colors[name])
        
        ax.set_title('Average Reward per Episode\n(100-episode moving average)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        
        # Plot price ratios
        ax = axs[0, 1]
        for name, data in mechanism_data.items():
            if 'price_ratios' in data and len(data['price_ratios']) > 0:
                mean_ratios = np.mean(data['price_ratios'], axis=0)
                std_ratios = np.std(data['price_ratios'], axis=0)
                
                # Plot moving average
                ma_ratios = pd.Series(mean_ratios).rolling(window=window).mean()
                ax.plot(episodes, ma_ratios, label=name, color=colors[name], linewidth=2)
                ax.fill_between(episodes,
                            ma_ratios - std_ratios,
                            ma_ratios + std_ratios,
                            alpha=alpha, color=colors[name])
        
        ax.set_title('Price Ratio\n(Selling Price / Grid Price)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Ratio')
        ax.legend()
        
        # Plot cumulative trading profits
        ax = axs[1, 0]
        for name, data in mechanism_data.items():
            if 'trading_profits' in data and len(data['trading_profits']) > 0:
                mean_profits = np.mean(data['trading_profits'], axis=0)
                cum_profits = np.cumsum(mean_profits)
                ax.plot(episodes, cum_profits, label=name, color=colors[name], linewidth=2)
        
        ax.set_title('Cumulative Trading Profit')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Profit')
        ax.legend()
        
        # Plot P2P energy trading
        ax = axs[1, 1]
        for name, data in mechanism_data.items():
            if 'p2p_energy' in data and len(data['p2p_energy']) > 0:
                mean_energy = np.mean(data['p2p_energy'], axis=0)
                std_energy = np.std(data['p2p_energy'], axis=0)
                
                # Plot moving average
                ma_energy = pd.Series(mean_energy).rolling(window=window).mean()
                ax.plot(episodes, ma_energy, label=name, color=colors[name], linewidth=2)
                ax.fill_between(episodes,
                            ma_energy - std_energy,
                            ma_energy + std_energy,
                            alpha=alpha, color=colors[name])
        
        ax.set_title('P2P Energy Trading')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy Amount')
        ax.legend()
        
        plt.tight_layout()
        return fig


    def print_statistical_analysis(self):
        """Print detailed statistical analysis with validation"""
        mechanism_data = {}
        for name, runs in self.mechanism_groups.items():
            print(f"\nAnalyzing {name} mechanism...")
            mechanism_data[name] = self.calculate_mechanism_metrics(runs)
        
        print("\n=== Statistical Analysis of Anti-Cartel Mechanisms ===\n")
        
        metrics = ['rewards', 'price_ratios', 'trading_profits', 'p2p_energy', 'penalties']
        
        for metric in metrics:
            print(f"\n--- {metric.replace('_', ' ').title()} Analysis ---")
            for name, data in mechanism_data.items():
                if metric in data and len(data[metric]) > 0:
                    final_values = np.mean(data[metric][:, -100:], axis=1)
                    print(f"\n{name.title()} Mechanism:")
                    print(f"Mean: {np.mean(final_values):.4f}")
                    print(f"Std: {np.std(final_values):.4f}")
                    print(f"Min: {np.min(final_values):.4f}")
                    print(f"Max: {np.max(final_values):.4f}")
                else:
                    print(f"\n{name.title()} Mechanism: No data available")