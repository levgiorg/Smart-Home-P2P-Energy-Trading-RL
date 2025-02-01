import os
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class ImprovedAntiCartelAnalyzer:
    def __init__(self, base_path="ml-outputs"):
        self.base_path = Path(base_path)
        self.runs = sorted([str(self.base_path / d) for d in os.listdir(base_path) 
                          if d.startswith('run')])
        # Group runs by experiment type and mechanism
        self.grouped_runs = self._group_runs()
        
    def _get_experiment_type(self, hyperparams: dict) -> str:
        """Determine experiment type from hyperparameters"""
        # Check key parameters to identify experiment type
        if 'beta' in hyperparams.get('reward', {}):
            return 'reward_stability'
        elif 'grid_fee' in hyperparams.get('environment', {}):
            if 'initial_selling_price_ratio' in hyperparams.get('environment', {}):
                return 'trading_optimization'
            else:
                return 'multi_objective'
        elif 'fc1_dims' in hyperparams.get('actor', {}):
            return 'network_advanced'
        elif 'batch_size' in hyperparams.get('rl_agent', {}):
            return 'learning_advanced'
        elif 'battery_capacity_min' in hyperparams.get('environment', {}):
            return 'battery_advanced'
        elif 't_min' in hyperparams.get('environment', {}):
            return 'comfort_advanced'
        return 'unknown'

    def _group_runs(self) -> Dict:
        """Group runs by experiment type and mechanism type"""
        groups = {}
        for run_path in self.runs:
            try:
                # Load hyperparameters
                with open(Path(run_path) / 'hyperparameters.json', 'r') as f:
                    hyperparams = json.load(f)
                
                # Get experiment and mechanism types
                exp_type = self._get_experiment_type(hyperparams)
                mech_type = hyperparams.get('anti_cartel', {}).get('mechanism_type', 'null')
                mech_type = 'null' if mech_type is None else mech_type
                
                # Initialize nested structure if needed
                if exp_type not in groups:
                    groups[exp_type] = {}
                if mech_type not in groups[exp_type]:
                    groups[exp_type][mech_type] = []
                
                groups[exp_type][mech_type].append(run_path)
                
            except Exception as e:
                print(f"Error processing run {run_path}: {e}")
                continue
        
        return groups

    def analyze_experiment_type(self, exp_type: str):
        """Analyze performance metrics for a specific experiment type"""
        if exp_type not in self.grouped_runs:
            print(f"No runs found for experiment type: {exp_type}")
            return None
        
        metrics_by_mechanism = {}
        for mech_type, runs in self.grouped_runs[exp_type].items():
            metrics = []
            for run_path in runs:
                try:
                    with open(Path(run_path) / 'data/ddpg__rewards_per_house.pkl', 'rb') as f:
                        rewards = pickle.load(f)
                    with open(Path(run_path) / 'data/ddpg__trading_profit.pkl', 'rb') as f:
                        profits = pickle.load(f)
                    with open(Path(run_path) / 'data/ddpg__selling_prices.pkl', 'rb') as f:
                        prices = pickle.load(f)
                    with open(Path(run_path) / 'data/ddpg__grid_prices.pkl', 'rb') as f:
                        grid_prices = pickle.load(f)
                    
                    metrics.append({
                        'final_reward': np.mean(rewards[:, -100:]),
                        'avg_profit': np.mean(profits),
                        'price_ratio': np.mean(prices[-100:]) / np.mean(grid_prices[-100:]),
                        'convergence': self._calculate_convergence(rewards)
                    })
                except Exception as e:
                    print(f"Error processing metrics for {run_path}: {e}")
                    continue
            
            if metrics:
                metrics_by_mechanism[mech_type] = pd.DataFrame(metrics)
        
        return metrics_by_mechanism

    def _calculate_convergence(self, rewards: np.ndarray, window: int = 100, threshold: float = 0.05) -> int:
        """Calculate episodes until convergence"""
        rolling_mean = pd.DataFrame(rewards).rolling(window=window, min_periods=1).mean()
        rolling_std = pd.DataFrame(rewards).rolling(window=window, min_periods=1).std()
        convergence_point = np.where(rolling_std.mean(axis=1) / np.abs(rolling_mean.mean(axis=1)) < threshold)[0]
        return convergence_point[0] if len(convergence_point) > 0 else len(rewards)

    def plot_experiment_comparison(self, exp_type: str):
        """Create detailed visualization for experiment comparison"""
        metrics = self.analyze_experiment_type(exp_type)
        if not metrics:
            return
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Mechanism Comparison for {exp_type} Experiments')
        
        # Colors for mechanisms
        colors = {'detection': '#2E86C1', 'ceiling': '#28B463', 'null': '#E74C3C'}
        
        # Plot reward distributions
        ax = axs[0, 0]
        data = []
        labels = []
        for mech, df in metrics.items():
            data.append(df['final_reward'].values)
            labels.extend([mech] * len(df))
        sns.boxplot(data=data, ax=ax)
        ax.set_xticklabels(list(metrics.keys()))
        ax.set_title('Final Reward Distribution')
        
        # Plot profit comparison
        ax = axs[0, 1]
        for mech, df in metrics.items():
            sns.kdeplot(data=df['avg_profit'], ax=ax, label=mech, color=colors.get(mech))
        ax.set_title('Trading Profit Distribution')
        ax.legend()
        
        # Plot price ratios
        ax = axs[1, 0]
        for mech, df in metrics.items():
            sns.kdeplot(data=df['price_ratio'], ax=ax, label=mech, color=colors.get(mech))
        ax.set_title('Price Ratio Distribution')
        ax.legend()
        
        # Plot convergence comparison
        ax = axs[1, 1]
        convergence_data = []
        for mech, df in metrics.items():
            convergence_data.append({
                'mechanism': mech,
                'episodes': df['convergence'].mean(),
                'std': df['convergence'].std()
            })
        
        conv_df = pd.DataFrame(convergence_data)
        ax.bar(conv_df['mechanism'], conv_df['episodes'], 
               yerr=conv_df['std'], capsize=5)
        ax.set_title('Average Episodes to Convergence')
        
        plt.tight_layout()
        return fig, metrics

    def print_statistical_summary(self):
        """Print statistical summary for all experiment types"""
        print("\n=== Statistical Summary of Anti-Cartel Mechanisms ===\n")
        
        for exp_type in self.grouped_runs.keys():
            print(f"\n{exp_type.upper()} EXPERIMENTS:")
            print("=" * 50)
            
            metrics = self.analyze_experiment_type(exp_type)
            if not metrics:
                continue
                
            for mech_type, df in metrics.items():
                print(f"\n{mech_type.upper()} Mechanism:")
                print("-" * 30)
                print(f"Number of runs: {len(df)}")
                print("\nMetrics Summary:")
                print(df.describe().round(4))
                print("\n")